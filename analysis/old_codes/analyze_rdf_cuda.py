#!/usr/bin/env python3
"""
GPU-Accelerated Radial Distribution Function Analysis
======================================================

Uses CuPy for GPU acceleration to handle large trajectory files efficiently.
Analyzes water-water and nanoparticle-water structure.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300


class GPURDFAnalyzer:
    """GPU-accelerated RDF calculation"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        print(f"GPU Device: {cp.cuda.Device()}")
        print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")
    
    def read_lammps_dump(self, dump_file, max_frames=100, skip_frames=10):
        """Read LAMMPS dump file efficiently"""
        
        frames = []
        current_frame = []
        in_atoms = False
        box_bounds = None
        frame_count = 0
        
        print(f"  Reading {dump_file.name}...", end='', flush=True)
        
        try:
            with open(dump_file, 'r') as f:
                for line in f:
                    if 'ITEM: NUMBER OF ATOMS' in line:
                        natoms = int(f.readline())
                    elif 'ITEM: BOX BOUNDS' in line:
                        xlo_xhi = [float(x) for x in f.readline().split()]
                        ylo_yhi = [float(x) for x in f.readline().split()]
                        zlo_zhi = [float(x) for x in f.readline().split()]
                        box_bounds = {
                            'x': (xlo_xhi[0], xlo_xhi[1]),
                            'y': (ylo_yhi[0], ylo_yhi[1]),
                            'z': (zlo_zhi[0], zlo_zhi[1])
                        }
                    elif 'ITEM: ATOMS' in line:
                        in_atoms = True
                        current_frame = []
                    elif 'ITEM: TIMESTEP' in line:
                        if current_frame and frame_count % skip_frames == 0:
                            frames.append((current_frame, box_bounds))
                            if len(frames) >= max_frames:
                                break
                        in_atoms = False
                        frame_count += 1
                    elif in_atoms and line.strip():
                        parts = line.split()
                        if len(parts) >= 5:
                            atom_id = int(parts[0])
                            atom_type = int(parts[1])
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            # Handle image flags if present (parts 5-7)
                            if len(parts) >= 8:
                                ix, iy, iz = int(parts[5]), int(parts[6]), int(parts[7])
                            current_frame.append((atom_id, atom_type, x, y, z))
            
            if current_frame and frame_count % skip_frames == 0:
                frames.append((current_frame, box_bounds))
            
            print(f" {len(frames)} frames (every {skip_frames}th frame)")
            return frames
        
        except Exception as e:
            print(f" ERROR: {e}")
            return []
    
    def calculate_rdf_gpu(self, frames, type1, type2, rmax=15.0, nbins=150):
        """GPU-accelerated RDF calculation"""
        
        if not frames:
            return None, None
        
        print(f"    Computing on GPU...", end='', flush=True)
        
        dr = rmax / nbins
        hist = cp.zeros(nbins, dtype=cp.float64)
        r_bins = cp.linspace(dr/2, rmax - dr/2, nbins)
        
        n_frames = len(frames)
        total_pairs = 0
        
        for frame_idx, (atoms, box) in enumerate(frames):
            # Extract coordinates for each type
            coords1_list = [(x, y, z) for aid, atype, x, y, z in atoms if atype == type1]
            coords2_list = [(x, y, z) for aid, atype, x, y, z in atoms if atype == type2]
            
            if len(coords1_list) == 0 or len(coords2_list) == 0:
                continue
            
            # Transfer to GPU
            coords1 = cp.array(coords1_list, dtype=cp.float32)
            coords2 = cp.array(coords2_list, dtype=cp.float32)
            
            # Box dimensions
            Lx = box['x'][1] - box['x'][0]
            Ly = box['y'][1] - box['y'][0]
            Lz = box['z'][1] - box['z'][0]
            L = cp.array([Lx, Ly, Lz], dtype=cp.float32)
            
            # Process in chunks to avoid memory issues
            chunk_size = 500  # Process 500 atoms at a time
            n1 = len(coords1)
            
            for i in range(0, n1, chunk_size):
                chunk_coords1 = coords1[i:min(i+chunk_size, n1)]
                
                # Calculate distances using broadcasting
                # Shape: (chunk_size, 1, 3) - (1, n2, 3) = (chunk_size, n2, 3)
                dx = chunk_coords1[:, cp.newaxis, :] - coords2[cp.newaxis, :, :]
                
                # Apply periodic boundary conditions
                dx = dx - L * cp.round(dx / L)
                
                # Calculate distances
                distances = cp.sqrt(cp.sum(dx**2, axis=2))
                
                # Flatten and histogram
                distances_flat = distances.ravel()
                
                # Filter distances within range (exclude self if same type)
                if type1 == type2:
                    valid_mask = (distances_flat > 0.1) & (distances_flat < rmax)  # Exclude self (d~0)
                else:
                    valid_mask = (distances_flat > 0) & (distances_flat < rmax)
                valid_distances = distances_flat[valid_mask]
                
                # Histogram on GPU
                hist_chunk, _ = cp.histogram(valid_distances, bins=nbins, range=(0, rmax))
                hist += hist_chunk
                
                total_pairs += len(valid_distances)
                
                # Clean up GPU memory
                del dx, distances, distances_flat, valid_mask, valid_distances, hist_chunk
            
            # Clean up
            del coords1, coords2
            cp.cuda.Stream.null.synchronize()
            
            if (frame_idx + 1) % 10 == 0:
                print(f".", end='', flush=True)
        
        # Transfer histogram back to CPU
        hist_cpu = cp.asnumpy(hist)
        r_bins_cpu = cp.asnumpy(r_bins)
        
        # Normalize to get g(r)
        n1 = len([a for a in frames[0][0] if a[1] == type1])
        n2 = len([a for a in frames[0][0] if a[1] == type2])
        
        # Average volume
        V_avg = np.mean([(box['x'][1] - box['x'][0]) * 
                         (box['y'][1] - box['y'][0]) * 
                         (box['z'][1] - box['z'][0]) for _, box in frames])
        
        rho = n2 / V_avg
        
        # Shell volumes
        r_inner = r_bins_cpu - dr/2
        r_outer = r_bins_cpu + dr/2
        shell_vol = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        
        # Normalize
        if type1 == type2:
            # For same type, we counted each pair twice (A-B and B-A)
            # So divide by 2, and total pairs per frame is n1 * n1 (not n1*(n1-1)/2)
            g_r = hist_cpu / (2.0 * n1 * shell_vol * rho * n_frames)
        else:
            # For different types, each pair counted once
            g_r = hist_cpu / (n1 * shell_vol * rho * n_frames)
        
        print(f" Done!")
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        return r_bins_cpu, g_r
    
    def analyze_all_systems(self):
        """Analyze RDFs for all epsilon values"""
        
        print("\n" + "="*80)
        print("GPU-ACCELERATED RDF ANALYSIS")
        print("="*80)
        
        results = {}
        
        for eps in self.epsilon_values:
            if eps == 0.0:
                eps_dir = self.base_dir / "epsilon_0.0"
            else:
                eps_dir = self.base_dir / f"epsilon_{eps:.2f}"
            
            dump_file = eps_dir / "production.lammpstrj"
            
            if not dump_file.exists():
                print(f"\nε={eps:.2f}: Trajectory not found")
                continue
            
            print(f"\nε={eps:.2f}:")
            
            # Load frames (skip every 10th frame, use max 100 frames)
            frames = self.read_lammps_dump(dump_file, max_frames=100, skip_frames=10)
            
            if not frames:
                continue
            
            results[eps] = {}
            
            # Calculate O-O RDF (type 1 - 1)
            print("  O-O RDF:")
            r_oo, g_oo = self.calculate_rdf_gpu(frames, 1, 1, rmax=12.0, nbins=120)
            if r_oo is not None:
                results[eps]['O-O'] = (r_oo, g_oo)
                peak_idx = np.argmax(g_oo[:len(g_oo)//3])
                print(f"    Peak at {r_oo[peak_idx]:.2f} Å, g(r)={g_oo[peak_idx]:.3f}")
            
            # Calculate O-H RDF (type 1 - 2)
            print("  O-H RDF:")
            r_oh, g_oh = self.calculate_rdf_gpu(frames, 1, 2, rmax=8.0, nbins=80)
            if r_oh is not None:
                results[eps]['O-H'] = (r_oh, g_oh)
                peak_idx = np.argmax(g_oh[:len(g_oh)//3])
                print(f"    Peak at {r_oh[peak_idx]:.2f} Å, g(r)={g_oh[peak_idx]:.3f}")
            
            # Calculate C-O RDF (type 3 - 1, nanoparticle-water)
            print("  C-O RDF (Nanoparticle-Water):")
            r_co, g_co = self.calculate_rdf_gpu(frames, 3, 1, rmax=20.0, nbins=200)
            if r_co is not None:
                results[eps]['C-O'] = (r_co, g_co)
                peak_idx = np.argmax(g_co[:len(g_co)//2])
                print(f"    Peak at {r_co[peak_idx]:.2f} Å, g(r)={g_co[peak_idx]:.3f}")
            
            print(f"  GPU Memory after frame: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
        
        print("\n" + "="*80)
        return results
    
    def plot_water_rdfs(self, results):
        """Plot water-water RDFs"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        # O-O RDFs
        ax = axes[0]
        for eps, color in zip(self.epsilon_values, colors):
            if eps in results and 'O-O' in results[eps]:
                r, g = results[eps]['O-O']
                ax.plot(r, g, label=f'ε={eps:.2f}', color=color, lw=2, alpha=0.8)
        
        ax.set_xlabel('r (Å)', fontweight='bold', fontsize=12)
        ax.set_ylabel('g(r)', fontweight='bold', fontsize=12)
        ax.set_title('Water O-O RDF', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # O-H RDFs
        ax = axes[1]
        for eps, color in zip(self.epsilon_values, colors):
            if eps in results and 'O-H' in results[eps]:
                r, g = results[eps]['O-H']
                ax.plot(r, g, label=f'ε={eps:.2f}', color=color, lw=2, alpha=0.8)
        
        ax.set_xlabel('r (Å)', fontweight='bold', fontsize=12)
        ax.set_ylabel('g(r)', fontweight='bold', fontsize=12)
        ax.set_title('Water O-H RDF', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 6)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'water_rdfs_gpu.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: water_rdfs_gpu.png")
        plt.close()
    
    def plot_nanoparticle_water_rdf(self, results):
        """Plot nanoparticle-water RDFs"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        for eps, color in zip(self.epsilon_values, colors):
            if eps in results and 'C-O' in results[eps]:
                r, g = results[eps]['C-O']
                ax.plot(r, g, label=f'ε={eps:.2f}', color=color, lw=2.5, alpha=0.8)
        
        ax.set_xlabel('r (Å)', fontweight='bold', fontsize=12)
        ax.set_ylabel('g(r)', fontweight='bold', fontsize=12)
        ax.set_title('Nanoparticle-Water RDF (C-O)', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'nanoparticle_water_rdf_gpu.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: nanoparticle_water_rdf_gpu.png")
        plt.close()
    
    def analyze_solvation_shell(self, results):
        """Analyze solvation shell structure"""
        
        print("\n" + "="*80)
        print("SOLVATION SHELL ANALYSIS")
        print("="*80)
        
        shell_data = []
        
        for eps in self.epsilon_values:
            if eps not in results or 'C-O' not in results[eps]:
                continue
            
            r, g = results[eps]['C-O']
            
            # Find first peak
            peak_idx = np.argmax(g[:len(g)//2])
            peak_r = r[peak_idx]
            peak_height = g[peak_idx]
            
            # Find first minimum
            after_peak = g[peak_idx:]
            min_idx = np.argmin(after_peak[:len(after_peak)//2])
            first_min = r[peak_idx + min_idx]
            
            # Calculate coordination number (integral up to first minimum)
            rho = 0.0334  # water oxygen density in atoms/Å³ at ~1 g/cm³
            mask = r <= first_min
            r_int = r[mask]
            g_int = g[mask]
            integrand = 4 * np.pi * rho * r_int**2 * g_int
            coord_num = np.trapz(integrand, r_int)
            
            shell_data.append({
                'epsilon': eps,
                'shell_distance': peak_r,
                'shell_intensity': peak_height,
                'shell_boundary': first_min,
                'coordination_number': coord_num
            })
            
            print(f"ε={eps:.2f}: Shell={peak_r:.2f}Å, g(r)={peak_height:.3f}, "
                  f"Boundary={first_min:.2f}Å, CN={coord_num:.1f}")
        
        # Plot solvation shell properties vs epsilon
        if shell_data:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            eps_vals = np.array([d['epsilon'] for d in shell_data])
            shell_dist = np.array([d['shell_distance'] for d in shell_data])
            shell_int = np.array([d['shell_intensity'] for d in shell_data])
            shell_bound = np.array([d['shell_boundary'] for d in shell_data])
            coord_num = np.array([d['coordination_number'] for d in shell_data])
            
            axes[0].plot(eps_vals, shell_dist, 'o-', markersize=10, lw=2.5, color='darkblue')
            axes[0].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[0].set_ylabel('Distance (Å)', fontweight='bold')
            axes[0].set_title('First Solvation Shell Distance', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(eps_vals, shell_int, 'o-', markersize=10, lw=2.5, color='darkgreen')
            axes[1].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[1].set_ylabel('g(r) Peak Height', fontweight='bold')
            axes[1].set_title('Solvation Shell Intensity', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(eps_vals, shell_bound, 'o-', markersize=10, lw=2.5, color='darkred')
            axes[2].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[2].set_ylabel('Distance (Å)', fontweight='bold')
            axes[2].set_title('Solvation Shell Boundary', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(eps_vals, coord_num, 'o-', markersize=10, lw=2.5, color='purple')
            axes[3].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[3].set_ylabel('Coordination Number', fontweight='bold')
            axes[3].set_title('First Shell Coordination Number', fontweight='bold')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.base_dir / 'solvation_shell_analysis_gpu.png', 
                       dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved: solvation_shell_analysis_gpu.png")
            plt.close()
        
        print("="*80)
        
        return shell_data


def main():
    """Main GPU-accelerated RDF analysis workflow"""
    
    print("\n" + "="*80)
    print("GPU-ACCELERATED RADIAL DISTRIBUTION FUNCTION ANALYSIS")
    print("Using CuPy for fast computation on large trajectories")
    print("="*80)
    
    analyzer = GPURDFAnalyzer(base_dir=".")
    
    # Analyze all systems
    results = analyzer.analyze_all_systems()
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    analyzer.plot_water_rdfs(results)
    analyzer.plot_nanoparticle_water_rdf(results)
    
    # Solvation shell analysis
    shell_data = analyzer.analyze_solvation_shell(results)
    
    print("\n" + "="*80)
    print("RDF ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - water_rdfs_gpu.png")
    print("  - nanoparticle_water_rdf_gpu.png")
    print("  - solvation_shell_analysis_gpu.png")
    print("="*80 + "\n")
    
    return analyzer, results, shell_data


if __name__ == "__main__":
    analyzer, results, shell_data = main()
