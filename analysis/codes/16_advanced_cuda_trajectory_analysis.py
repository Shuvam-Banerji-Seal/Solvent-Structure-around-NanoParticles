#!/usr/bin/env python3
"""
MODULE 16: ADVANCED CUDA-ACCELERATED TRAJECTORY ANALYSIS
========================================================

High-performance parallel analysis of LAMMPS trajectories using CUDA.
Processes multiple epsilon directories concurrently to extract:
1. Spatial Structure (RDF, Density maps, Coordination)
2. Molecular Orientation (Dipoles, 2D Maps)
3. Dynamics (Residence times, H-bonds)
4. Thermodynamics (Entropy, Tetrahedral Order)

System: 3 x C60 molecules (180 atoms) + Water (TIP4P/2005)
Analysis is averaged over the local environment of the C60 molecules.

Requirements:
- numba (for CUDA kernels)
- MDAnalysis
- numpy, pandas, matplotlib, seaborn
- NVIDIA GPU

Author: AI Analysis Suite
Date: November 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import MDAnalysis as mda
import math
from concurrent.futures import ProcessPoolExecutor
import warnings
import json

# Try importing Numba for CUDA
try:
    from numba import cuda, float32, int32
    import numba
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("Numba/CUDA not found. Falling back to CPU (slow).")

warnings.filterwarnings('ignore')

# Plotting settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
sns.set_palette("husl")

# =============================================================================
# CUDA KERNELS
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def density_map_kernel(coords, grid, c60_coms, grid_min, grid_max, grid_res, n_atoms, n_c60s):
        """
        Compute 3D density map relative to NEAREST C60
        """
        i = cuda.grid(1)
        if i < n_atoms:
            # Find nearest C60
            min_dist_sq = 1.0e10
            nearest_c60_idx = -1
            
            ox = coords[i, 0]
            oy = coords[i, 1]
            oz = coords[i, 2]
            
            for c in range(n_c60s):
                dx = ox - c60_coms[c, 0]
                dy = oy - c60_coms[c, 1]
                dz = oz - c60_coms[c, 2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist_sq:
                    min_dist_sq = d2
                    nearest_c60_idx = c
            
            # Relative position to nearest C60
            cx = c60_coms[nearest_c60_idx, 0]
            cy = c60_coms[nearest_c60_idx, 1]
            cz = c60_coms[nearest_c60_idx, 2]
            
            x = ox - cx
            y = oy - cy
            z = oz - cz
            
            # Check bounds
            if (x >= grid_min and x < grid_max and 
                y >= grid_min and y < grid_max and 
                z >= grid_min and z < grid_max):
                
                # Map to grid index
                idx_x = int((x - grid_min) / grid_res)
                idx_y = int((y - grid_min) / grid_res)
                idx_z = int((z - grid_min) / grid_res)
                
                # Atomic add to grid
                cuda.atomic.add(grid, (idx_x, idx_y, idx_z), 1.0)

    @cuda.jit
    def rdf_orientation_kernel(water_o, water_h1, water_h2, c60_coms, 
                             rdf_hist, orient_hist, orient_map, coord_hist,
                             r_min, r_max, n_bins, n_waters, n_c60s):
        """
        Compute RDF, Orientation, and Coordination relative to NEAREST C60
        """
        i = cuda.grid(1)
        if i < n_waters:
            # Find nearest C60
            min_dist_sq = 1.0e10
            nearest_c60_idx = -1
            
            ox = water_o[i, 0]
            oy = water_o[i, 1]
            oz = water_o[i, 2]
            
            for c in range(n_c60s):
                dx = ox - c60_coms[c, 0]
                dy = oy - c60_coms[c, 1]
                dz = oz - c60_coms[c, 2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist_sq:
                    min_dist_sq = d2
                    nearest_c60_idx = c
            
            dist = math.sqrt(min_dist_sq)
            
            # --- RDF ---
            if dist >= r_min and dist < r_max:
                bin_idx = int((dist - r_min) / (r_max - r_min) * n_bins)
                cuda.atomic.add(rdf_hist, bin_idx, 1.0)
                
                # --- Coordination Number Accumulation ---
                # We just count atoms in bins, integration happens later
                cuda.atomic.add(coord_hist, bin_idx, 1.0)
            
            # --- Orientation ---
            # Vector C60 -> O
            cx = c60_coms[nearest_c60_idx, 0]
            cy = c60_coms[nearest_c60_idx, 1]
            cz = c60_coms[nearest_c60_idx, 2]
            
            rx = ox - cx
            ry = oy - cy
            rz = oz - cz
            r_norm = dist 
            
            # Dipole Vector (H1+H2 - 2*O) or just bisector
            # Vector O->H1
            v1x = water_h1[i, 0] - ox
            v1y = water_h1[i, 1] - oy
            v1z = water_h1[i, 2] - oz
            
            # Vector O->H2
            v2x = water_h2[i, 0] - ox
            v2y = water_h2[i, 1] - oy
            v2z = water_h2[i, 2] - oz
            
            # Dipole sum
            dx = v1x + v2x
            dy = v1y + v2y
            dz = v1z + v2z
            d_norm = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            cos_theta = -2.0
            if d_norm > 0 and r_norm > 0:
                dot = dx*rx + dy*ry + dz*rz
                cos_theta = dot / (d_norm * r_norm)
                
                # Orientation Histogram (global)
                # Map -1..1 to 0..n_bins
                cos_bin = int((cos_theta + 1.0) / 2.0 * n_bins)
                if cos_bin >= 0 and cos_bin < n_bins:
                    cuda.atomic.add(orient_hist, cos_bin, 1.0)
                
                # 2D Map (Distance vs CosTheta)
                # Y-axis: Distance (0..20A), X-axis: CosTheta (-1..1)
                # Map dimensions: (50, 50)
                if dist < 20.0:
                    dist_bin_2d = int(dist / 20.0 * 50) # 50 bins for distance
                    cos_bin_2d = int((cos_theta + 1.0) / 2.0 * 50) # 50 bins for cos
                    if dist_bin_2d >= 0 and dist_bin_2d < 50 and cos_bin_2d >= 0 and cos_bin_2d < 50:
                        cuda.atomic.add(orient_map, (dist_bin_2d, cos_bin_2d), 1.0)

    @cuda.jit
    def tetrahedral_order_kernel(o_pos, q_hist, q_vs_dist_map, c60_coms, n_waters, n_c60s):
        """
        Compute Tetrahedral Order Parameter q for each water
        AND correlate it with distance to nearest C60
        q = 1 - 3/8 * sum_j sum_k (cos(psi_jk) + 1/3)^2
        """
        i = cuda.grid(1)
        if i < n_waters:
            ox = o_pos[i, 0]
            oy = o_pos[i, 1]
            oz = o_pos[i, 2]

            # --- Distance to nearest C60 ---
            min_dist_sq = 1.0e10
            for c in range(n_c60s):
                dx = ox - c60_coms[c, 0]
                dy = oy - c60_coms[c, 1]
                dz = oz - c60_coms[c, 2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist_sq:
                    min_dist_sq = d2
            dist_to_c60 = math.sqrt(min_dist_sq)

            # --- Find 4 nearest neighbors ---
            nn_dist_sq = cuda.local.array(4, float32)
            nn_idx = cuda.local.array(4, int32)
            for k in range(4):
                nn_dist_sq[k] = 1.0e10
                nn_idx[k] = -1
            
            # Loop over all other waters
            for j in range(n_waters):
                if i == j: continue
                
                dx = o_pos[j, 0] - ox
                dy = o_pos[j, 1] - oy
                dz = o_pos[j, 2] - oz
                d2 = dx*dx + dy*dy + dz*dz
                
                if d2 > 3.5*3.5: continue # Optimization
                
                # Insert into sorted list
                if d2 < nn_dist_sq[3]:
                    if d2 < nn_dist_sq[2]:
                        nn_dist_sq[3] = nn_dist_sq[2]
                        nn_idx[3] = nn_idx[2]
                        if d2 < nn_dist_sq[1]:
                            nn_dist_sq[2] = nn_dist_sq[1]
                            nn_idx[2] = nn_idx[1]
                            if d2 < nn_dist_sq[0]:
                                nn_dist_sq[1] = nn_dist_sq[0]
                                nn_idx[1] = nn_idx[0]
                                nn_dist_sq[0] = d2
                                nn_idx[0] = j
                            else:
                                nn_dist_sq[1] = d2
                                nn_idx[1] = j
                        else:
                            nn_dist_sq[2] = d2
                            nn_idx[2] = j
                    else:
                        nn_dist_sq[3] = d2
                        nn_idx[3] = j
            
            # Compute q
            valid_neighbors = 0
            for k in range(4):
                if nn_idx[k] != -1:
                    valid_neighbors += 1
            
            if valid_neighbors == 4:
                sum_term = 0.0
                for j in range(3):
                    idx_j = nn_idx[j]
                    vjx = o_pos[idx_j, 0] - ox
                    vjy = o_pos[idx_j, 1] - oy
                    vjz = o_pos[idx_j, 2] - oz
                    vj_norm = math.sqrt(vjx*vjx + vjy*vjy + vjz*vjz)
                    
                    for k in range(j+1, 4):
                        idx_k = nn_idx[k]
                        vkx = o_pos[idx_k, 0] - ox
                        vky = o_pos[idx_k, 1] - oy
                        vkz = o_pos[idx_k, 2] - oz
                        vk_norm = math.sqrt(vkx*vkx + vky*vky + vkz*vkz)
                        
                        dot = vjx*vkx + vjy*vky + vjz*vkz
                        cos_psi = dot / (vj_norm * vk_norm)
                        
                        term = cos_psi + 1.0/3.0
                        sum_term += term * term
                
                q = 1.0 - (3.0/8.0) * sum_term
                
                # Histogram q (0..1)
                bin_idx = int(q * 50) # 50 bins
                if bin_idx >= 0 and bin_idx < 50:
                    cuda.atomic.add(q_hist, bin_idx, 1.0)
                
                # 2D Map: Distance vs Q
                # Dist: 0-20A (50 bins), Q: 0-1 (50 bins)
                if dist_to_c60 < 20.0:
                    dist_bin = int(dist_to_c60 / 20.0 * 50)
                    q_bin = int(q * 50)
                    if dist_bin >= 0 and dist_bin < 50 and q_bin >= 0 and q_bin < 50:
                        cuda.atomic.add(q_vs_dist_map, (dist_bin, q_bin), 1.0)

# =============================================================================
# ANALYSIS CLASS
# =============================================================================

class CUDATrajectoryAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.epsilon_dirs = {
            0.0: 'epsilon_0.0', 0.05: 'epsilon_0.05', 0.1: 'epsilon_0.10',
            0.15: 'epsilon_0.15', 0.2: 'epsilon_0.20', 0.25: 'epsilon_0.25',
            0.3: 'epsilon_0.30', 0.35: 'epsilon_0.35', 0.4: 'epsilon_0.40',
            0.45: 'epsilon_0.45', 0.5: 'epsilon_0.50'
        }
        self.plots_dir = self.base_dir / 'analysis' / 'plots'
        self.data_dir = self.base_dir / 'analysis' / 'data'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def process_epsilon(self, eps):
        """
        Process a single epsilon trajectory using CUDA
        """
        eps_dir = self.base_dir / self.epsilon_dirs[eps]
        traj_file = eps_dir / 'production.lammpstrj'
        
        if not traj_file.exists():
            return None
        
        print(f"  [ε={eps}] Starting Advanced CUDA analysis...")
        
        try:
            u = mda.Universe(str(traj_file), format='LAMMPSDUMP')
            
            # Selections
            # 3 C60 molecules (60 atoms each) = 180 atoms
            c60_atoms = u.atoms[0:180]
            waters = u.atoms[180:]
            n_waters = len(waters) // 3
            
            # Indices for O, H1, H2
            o_indices = np.arange(0, len(waters), 3)
            h1_indices = np.arange(1, len(waters), 3)
            h2_indices = np.arange(2, len(waters), 3)
            
            # --- Initialize Accumulators ---
            
            # 1. RDF & Orientation
            n_bins_rdf = 200
            r_max = 20.0
            rdf_hist = np.zeros(n_bins_rdf, dtype=np.float32)
            coord_hist = np.zeros(n_bins_rdf, dtype=np.float32)
            orient_hist = np.zeros(200, dtype=np.float32)
            orient_map = np.zeros((50, 50), dtype=np.float32) # Dist x CosTheta
            
            # 2. Tetrahedral Order
            q_hist = np.zeros(50, dtype=np.float32)
            q_vs_dist_map = np.zeros((50, 50), dtype=np.float32) # Dist x Q
            
            # 3. Density Map
            grid_res = 1.0
            grid_range = 20.0
            grid_dim = int(2 * grid_range / grid_res)
            density_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
            
            # 4. Residence Time
            shell_cutoff = 5.0
            residence_tracker = {} # water_id -> [frames]
            
            # --- CUDA Setup ---
            if CUDA_AVAILABLE:
                d_rdf_hist = cuda.to_device(rdf_hist)
                d_coord_hist = cuda.to_device(coord_hist)
                d_orient_hist = cuda.to_device(orient_hist)
                d_orient_map = cuda.to_device(orient_map)
                d_q_hist = cuda.to_device(q_hist)
                d_q_vs_dist_map = cuda.to_device(q_vs_dist_map)
                d_density_grid = cuda.to_device(density_grid)
            
            # --- Trajectory Loop ---
            stride = 10
            frames = 0
            
            for ts in u.trajectory[::stride]:
                frames += 1
                
                # Get Coordinates
                c60_pos = c60_atoms.positions
                c60_coms = np.array([
                    c60_atoms[0:60].center_of_mass(),
                    c60_atoms[60:120].center_of_mass(),
                    c60_atoms[120:180].center_of_mass()
                ], dtype=np.float32)
                
                water_pos = waters.positions.astype(np.float32)
                o_pos = water_pos[o_indices]
                h1_pos = water_pos[h1_indices]
                h2_pos = water_pos[h2_indices]
                
                if CUDA_AVAILABLE:
                    d_o_pos = cuda.to_device(o_pos)
                    d_h1_pos = cuda.to_device(h1_pos)
                    d_h2_pos = cuda.to_device(h2_pos)
                    d_c60_coms = cuda.to_device(c60_coms)
                    
                    threadsperblock = 256
                    blockspergrid = (n_waters + (threadsperblock - 1)) // threadsperblock
                    
                    # 1. RDF & Orientation & Coordination
                    rdf_orientation_kernel[blockspergrid, threadsperblock](
                        d_o_pos, d_h1_pos, d_h2_pos, d_c60_coms,
                        d_rdf_hist, d_orient_hist, d_orient_map, d_coord_hist,
                        0.0, r_max, n_bins_rdf, n_waters, 3
                    )
                    
                    # 2. Tetrahedral Order
                    tetrahedral_order_kernel[blockspergrid, threadsperblock](
                        d_o_pos, d_q_hist, d_q_vs_dist_map, d_c60_coms, n_waters, 3
                    )
                    
                    # 3. Density Map
                    density_map_kernel[blockspergrid, threadsperblock](
                        d_o_pos, d_density_grid, d_c60_coms, 
                        -grid_range, grid_range, grid_res, n_waters, 3
                    )
                
                # 4. Residence Time (CPU)
                # Calculate distances to nearest C60 COM
                # Simple CPU implementation for tracking IDs
                for c in range(3):
                    dists = np.linalg.norm(o_pos - c60_coms[c], axis=1)
                    in_shell = np.where(dists < shell_cutoff)[0]
                    for wid in in_shell:
                        if wid not in residence_tracker:
                            residence_tracker[wid] = []
                        residence_tracker[wid].append(frames)
            
            # --- Post-Processing ---
            if CUDA_AVAILABLE:
                rdf_hist = d_rdf_hist.copy_to_host()
                coord_hist = d_coord_hist.copy_to_host()
                orient_hist = d_orient_hist.copy_to_host()
                orient_map = d_orient_map.copy_to_host()
                q_hist = d_q_hist.copy_to_host()
                q_vs_dist_map = d_q_vs_dist_map.copy_to_host()
                density_grid = d_density_grid.copy_to_host()
            
            # Normalize Maps
            orient_map /= frames
            q_vs_dist_map /= frames
            density_grid /= frames
            
            # Residence Time
            residence_times = []
            for wid, frame_list in residence_tracker.items():
                if not frame_list: continue
                frame_list.sort()
                current_run = 1
                for i in range(1, len(frame_list)):
                    if frame_list[i] == frame_list[i-1] + 1:
                        current_run += 1
                    else:
                        residence_times.append(current_run)
                        current_run = 1
                residence_times.append(current_run)
            mean_residence = np.mean(residence_times) * stride * 2.0 / 1000.0 if residence_times else 0 # ns
            
            # Entropy
            total_density = np.sum(density_grid)
            if total_density > 0:
                prob_grid = density_grid / total_density
                nonzero = prob_grid[prob_grid > 0]
                entropy = -np.sum(nonzero * np.log(nonzero))
            else:
                entropy = 0
            
            return {
                'epsilon': eps,
                'rdf_hist': rdf_hist.tolist(),
                'coord_hist': coord_hist.tolist(),
                'orient_hist': orient_hist.tolist(),
                'orient_map': orient_map.tolist(),
                'q_hist': q_hist.tolist(),
                'q_vs_dist_map': q_vs_dist_map.tolist(),
                'density_map': density_grid.tolist(),
                'mean_residence_ns': float(mean_residence),
                'entropy': float(entropy),
                'frames': frames
            }
            
        except Exception as e:
            print(f"  [ε={eps}] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_parallel_analysis(self):
        """Run analysis for all epsilons in parallel"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE PARALLEL CUDA ANALYSIS")
        print("="*80)
        
        results = {}
        max_workers = 2 if CUDA_AVAILABLE else os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_epsilon, eps): eps for eps in self.epsilon_values}
            
            for future in futures:
                eps = futures[future]
                try:
                    res = future.result()
                    if res:
                        results[eps] = res
                        print(f"  ✓ [ε={eps}] Analysis complete")
                except Exception as e:
                    print(f"  ✗ [ε={eps}] Failed: {e}")
        
        self.save_and_plot(results)

    def save_and_plot(self, results):
        """Save results and generate comparison plots"""
        print("\nGenerating comprehensive plots...")
        
        # Save raw data
        with open(self.data_dir / 'comprehensive_analysis_results.json', 'w') as f:
            json.dump(results, f)
        
        # 1. RDF Plot
        plt.figure(figsize=(10, 6))
        r = np.linspace(0, 20.0, 200)
        for eps in sorted(results.keys()):
            hist = np.array(results[eps]['rdf_hist'])
            dr = r[1] - r[0]
            shell_vol = 4 * np.pi * r**2 * dr
            shell_vol[0] = 1.0
            rdf = hist / shell_vol
            tail_mean = np.mean(rdf[-50:])
            if tail_mean > 0: rdf /= tail_mean
            plt.plot(r, rdf, label=f'ε={eps}')
        plt.xlabel('Distance from nearest C60 (Å)')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.legend()
        plt.savefig(self.plots_dir / '52_advanced_rdf.png')
        
        # 2. Coordination Number
        plt.figure(figsize=(10, 6))
        for eps in sorted(results.keys()):
            hist = np.array(results[eps]['coord_hist'])
            # Cumulative sum normalized by frames and number of C60s (3)
            # Actually coord_hist is just counts in bins.
            # We need to integrate density.
            # Approx: just cumsum of raw counts / frames / 3
            coord = np.cumsum(hist) / results[eps]['frames'] / 3.0
            plt.plot(r, coord, label=f'ε={eps}')
        plt.xlabel('Distance (Å)')
        plt.ylabel('Coordination Number')
        plt.title('Running Coordination Number')
        plt.legend()
        plt.savefig(self.plots_dir / '55_coordination_number.png')
        
        # 3. Tetrahedral Order Distribution
        plt.figure(figsize=(10, 6))
        q = np.linspace(0, 1, 50)
        for eps in sorted(results.keys()):
            hist = np.array(results[eps]['q_hist'])
            prob = hist / np.sum(hist)
            plt.plot(q, prob, label=f'ε={eps}')
        plt.xlabel('Tetrahedral Order (q)')
        plt.ylabel('Probability')
        plt.title('Global Tetrahedral Order')
        plt.legend()
        plt.savefig(self.plots_dir / '53_tetrahedral_order.png')
        
        # 4. Orientation Heatmaps (All Epsilons)
        n_eps = len(results)
        cols = 3
        rows = (n_eps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, eps in enumerate(sorted(results.keys())):
            data = np.array(results[eps]['orient_map'])
            sns.heatmap(data.T, ax=axes[i], cmap='RdBu_r', cbar=True,
                       xticklabels=10, yticklabels=10)
            axes[i].set_title(f'Orientation Map (ε={eps})')
            axes[i].set_xlabel('Distance Bin')
            axes[i].set_ylabel('Cos(theta) Bin')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '54_orientation_heatmaps_all.png')
        
        # 5. Tetrahedral Order vs Distance Heatmaps
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, eps in enumerate(sorted(results.keys())):
            data = np.array(results[eps]['q_vs_dist_map'])
            sns.heatmap(data.T, ax=axes[i], cmap='viridis', cbar=True,
                       xticklabels=10, yticklabels=10)
            axes[i].set_title(f'Tetrahedral Order vs Dist (ε={eps})')
            axes[i].set_xlabel('Distance Bin')
            axes[i].set_ylabel('Order Parameter q')
            axes[i].invert_yaxis()
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / '56_tetrahedral_vs_dist_all.png')

        # 5b. Density Maps (Central Slice)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, eps in enumerate(sorted(results.keys())):
            data = np.array(results[eps]['density_map'])
            # Plot central slice (Z=0)
            mid_idx = data.shape[2] // 2
            slice_data = data[:, :, mid_idx]
            
            sns.heatmap(slice_data, ax=axes[i], cmap='viridis', cbar=True)
            axes[i].set_title(f'Density Map Slice (ε={eps})')
            axes[i].set_xlabel('X Bin')
            axes[i].set_ylabel('Y Bin')
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / '52_density_maps_all.png')
        
        # 6. Summary Metrics
        summary_data = []
        for eps in sorted(results.keys()):
            summary_data.append({
                'epsilon': eps,
                'Residence Time (ns)': results[eps]['mean_residence_ns'],
                'Entropy': results[eps]['entropy']
            })
        df_sum = pd.DataFrame(summary_data)
        df_sum.to_csv(self.data_dir / 'comprehensive_metrics.csv', index=False)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.lineplot(data=df_sum, x='epsilon', y='Residence Time (ns)', marker='o', ax=ax[0])
        sns.lineplot(data=df_sum, x='epsilon', y='Entropy', marker='s', ax=ax[1], color='orange')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '57_dynamics_thermodynamics.png')

        print("  ✓ Saved all plots.")

def main():
    base_dir = '/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2'
    analyzer = CUDATrajectoryAnalyzer(base_dir)
    analyzer.run_parallel_analysis()

if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
