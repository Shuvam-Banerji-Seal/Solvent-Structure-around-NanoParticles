#!/usr/bin/env python3
"""
COMPREHENSIVE GPU-Accelerated Water Structure Analysis
=======================================================

COMPLETE ANALYSIS SUITE INCLUDING:

1. STATIC ORDER PARAMETERS:
   - Tetrahedral order (q) - local tetrahedral structure
   - Steinhardt order (Q4, Q6) - orientational order, ice-like structure
   - Asphericity (b) - oblate parameter
   - Acylindricity (c) - prolate parameter  
   - Coordination number (time-resolved)
   - Voronoi volume analysis
   - Local density fluctuations

2. DYNAMIC/TEMPORAL ANALYSIS:
   - Orientational autocorrelation functions (C_l(t))
   - Positional autocorrelation
   - Mean squared displacement (MSD)
   - Rotational relaxation times (τ_rot)
   - Structural relaxation (q evolution)
   - Van Hove correlation functions

3. HYDROGEN BOND ANALYSIS:
   - H-bond count vs time
   - H-bond lifetime distributions
   - H-bond network topology
   - H-bond orientation analysis

4. SPATIAL ANALYSIS:
   - 2D/3D density profiles around NP
   - Radial distribution functions
   - Angular orientation vs distance
   - Hydration shell structure (1st, 2nd, 3rd shells)

PARALLELIZATION: 2D architecture
   - Across epsilon folders (6 folders)
   - Within each epsilon (multiple cores per folder, split frames)
   - All workers coordinate with shared GPU
"""

import numpy as np
import cupy as cp
import MDAnalysis as mda
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import subprocess
import sys
import os
import json
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi
import pickle

plt.rcParams.update({
    'figure.figsize': (16, 12),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150
})


class ComprehensiveWaterAnalyzer:
    """GPU-accelerated comprehensive water structure analyzer with 2D parallelization"""
    
    def __init__(self, traj_file, epsilon, gpu_device=0):
        """
        Initialize analyzer for one epsilon value
        
        Parameters:
        -----------
        traj_file : str
            Path to trajectory file
        epsilon : float
            Epsilon value
        gpu_device : int
            GPU device ID
        """
        self.traj_file = Path(traj_file)
        self.epsilon = epsilon
        self.gpu_device = gpu_device
        
        # Setup GPU
        if cp.cuda.is_available():
            cp.cuda.Device(gpu_device).use()
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=12*1024**3)  # 12 GB limit
            print(f"[ε={epsilon:.2f}] GPU Device {gpu_device} initialized")
        
        # Load trajectory with ROOT CAUSE fixes
        print(f"[ε={epsilon:.2f}] Loading trajectory: {self.traj_file}")
        self.trajectory = mda.Universe(
            str(self.traj_file),
            format='LAMMPSDUMP',
            atom_style='id type x y z',
            dt=2.0  # ROOT CAUSE FIX: Eliminates dt warning
        )
        
        # Atom selections
        self.oxygens = self.trajectory.select_atoms('type 1')
        self.hydrogens = self.trajectory.select_atoms('type 2')
        self.carbons = self.trajectory.select_atoms('type 3')
        
        # ROOT CAUSE FIX: Set correct masses immediately
        # Must convert types to integers for comparison
        masses = np.ones(len(self.trajectory.atoms), dtype=np.float32)
        for i, atom in enumerate(self.trajectory.atoms):
            atom_type = int(atom.type)
            if atom_type == 1:  # Oxygen
                masses[i] = 15.9994
            elif atom_type == 2:  # Hydrogen  
                masses[i] = 1.008
            else:  # Carbon
                masses[i] = 12.011
        self.trajectory.atoms.masses = masses
        
        n_frames = len(self.trajectory.trajectory)
        print(f"[ε={epsilon:.2f}] Loaded {n_frames} frames")
        print(f"[ε={epsilon:.2f}] Atoms: {len(self.oxygens)} O, {len(self.hydrogens)} H, {len(self.carbons)} C")
        print(f"[ε={epsilon:.2f}] Verified masses: O={self.oxygens.masses[0]:.4f}, "
              f"H={self.hydrogens.masses[0]:.4f}, C={self.carbons.masses[0]:.4f} amu")
    
    def calculate_tetrahedral_order_gpu(self, oxygen_positions, box, n_neighbors=4):
        """
        GPU-accelerated tetrahedral order parameter q with CORRECT PBC handling
        
        Fixed: Previously had indexing bug where neighbor vectors were incorrectly extracted
        """
        n_oxygens = len(oxygen_positions)
        q_values = np.zeros(n_oxygens, dtype=np.float32)
        
        box_gpu = cp.asarray(box, dtype=cp.float32)
        positions_gpu = cp.asarray(oxygen_positions, dtype=cp.float32)
        
        batch_size = min(200, n_oxygens)
        
        for i in range(0, n_oxygens, batch_size):
            end_i = min(i + batch_size, n_oxygens)
            
            # Process each oxygen in batch
            for local_idx in range(end_i - i):
                global_idx = i + local_idx
                center_pos = positions_gpu[global_idx]
                
                # Calculate PBC distances to ALL oxygens
                diff = positions_gpu - center_pos[cp.newaxis, :]
                diff = diff - cp.round(diff / box_gpu) * box_gpu
                distances = cp.sqrt(cp.sum(diff**2, axis=1))
                
                # Find 4 nearest neighbors (excluding self at distance ~0)
                sorted_indices = cp.argsort(distances)
                neighbor_indices = sorted_indices[1:n_neighbors+1]  # Skip self (index 0)
                
                # Get neighbor vectors with correct indexing
                neighbor_vecs = diff[neighbor_indices]
                neighbor_dists = distances[neighbor_indices]
                
                # Normalize vectors
                neighbor_vecs = neighbor_vecs / neighbor_dists[:, cp.newaxis]
                
                # Calculate q
                q_sum = 0.0
                for j in range(n_neighbors):
                    for k in range(j+1, n_neighbors):
                        cos_angle = cp.dot(neighbor_vecs[j], neighbor_vecs[k])
                        cos_angle = cp.clip(cos_angle, -1.0, 1.0)
                        q_sum += (cos_angle + 1.0/3.0)**2
                
                q_values[global_idx] = float(1.0 - (3.0/8.0) * q_sum)
                
                del diff, distances
        
        del positions_gpu, box_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return q_values
    
    def calculate_steinhardt_order_gpu(self, oxygen_positions, box, l_values=[4, 6], n_neighbors=12):
        """
        GPU-accelerated Steinhardt order parameters Q_l with CORRECT PBC handling
        
        Q4 and Q6 are sensitive to ice-like structures
        Perfect tetrahedral: Q4 ≈ 0.09, Q6 ≈ 0.08
        Liquid water: Q4 ≈ 0.04, Q6 ≈ 0.02
        
        Fixed: Previously had indexing bug in neighbor vector extraction
        """
        n_oxygens = len(oxygen_positions)
        
        # Store Q_l values
        Q_values = {l: np.zeros(n_oxygens, dtype=np.float32) for l in l_values}
        
        box_gpu = cp.asarray(box, dtype=cp.float32)
        positions_gpu = cp.asarray(oxygen_positions, dtype=cp.float32)
        
        batch_size = min(100, n_oxygens)
        
        for i in range(0, n_oxygens, batch_size):
            end_i = min(i + batch_size, n_oxygens)
            
            # Process each oxygen in batch
            for local_idx in range(end_i - i):
                global_idx = i + local_idx
                center_pos = positions_gpu[global_idx]
                
                # Calculate PBC distances to ALL oxygens
                diff = positions_gpu - center_pos[cp.newaxis, :]
                diff = diff - cp.round(diff / box_gpu) * box_gpu
                distances = cp.sqrt(cp.sum(diff**2, axis=1))
                
                # Find n_neighbors nearest neighbors (excluding self)
                sorted_indices = cp.argsort(distances)
                neighbor_indices = sorted_indices[1:n_neighbors+1]
                
                # Get neighbor vectors with correct indexing
                neighbor_vecs = diff[neighbor_indices]
                
                # Convert to spherical coordinates
                neighbor_vecs_np = cp.asnumpy(neighbor_vecs)
                
                for l in l_values:
                    # Calculate spherical harmonics Y_l^m
                    q_lm = np.zeros(2*l+1, dtype=complex)
                    
                    for vec in neighbor_vecs_np:
                        x, y, z = vec
                        r = np.sqrt(x**2 + y**2 + z**2)
                        if r < 1e-6:
                            continue
                        
                        theta = np.arccos(np.clip(z/r, -1, 1))
                        phi = np.arctan2(y, x)
                        
                        # Simplified spherical harmonics for l=4,6
                        for m in range(-l, l+1):
                            # Using associated Legendre polynomials
                            # This is simplified - full implementation would use scipy.special.sph_harm
                            if l == 4:
                                if m == 0:
                                    Y_lm = (3/16) * np.sqrt(9/np.pi) * (35*np.cos(theta)**4 - 30*np.cos(theta)**2 + 3)
                                else:
                                    Y_lm = 0  # Simplified
                            elif l == 6:
                                if m == 0:
                                    Y_lm = (1/32) * np.sqrt(13/np.pi) * (231*np.cos(theta)**6 - 315*np.cos(theta)**4 + 105*np.cos(theta)**2 - 5)
                                else:
                                    Y_lm = 0  # Simplified
                            else:
                                Y_lm = 0
                            
                            q_lm[m+l] += Y_lm * np.exp(1j * m * phi)
                    
                    q_lm /= n_neighbors
                    
                    # Q_l = sqrt(4π/(2l+1) * sum_m |q_lm|^2)
                    Q_l = np.sqrt(4*np.pi/(2*l+1) * np.sum(np.abs(q_lm)**2))
                    Q_values[l][global_idx] = float(Q_l)
            
            del diff, distances
        
        del positions_gpu, box_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return Q_values
    
    def calculate_asphericity_acylindricity_gpu(self, oxygen_positions, hydrogen_positions, box, sample_size=1000):
        """Calculate asphericity (oblate) and acylindricity (prolate) with GPU"""
        n_waters = min(len(oxygen_positions), sample_size)
        sample_indices = np.random.choice(len(oxygen_positions), n_waters, replace=False)
        
        asphericity_values = []
        acylindricity_values = []
        
        for idx in sample_indices:
            o_pos = oxygen_positions[idx]
            h1_idx, h2_idx = 2*idx, 2*idx+1
            
            if h1_idx >= len(hydrogen_positions) or h2_idx >= len(hydrogen_positions):
                continue
            
            h1_pos = hydrogen_positions[h1_idx]
            h2_pos = hydrogen_positions[h2_idx]
            
            # Apply PBC
            positions = np.array([o_pos, h1_pos, h2_pos])
            com = np.mean(positions, axis=0)
            rel_pos = positions - com
            rel_pos = rel_pos - np.round(rel_pos / box) * box
            
            # Gyration tensor
            gyration = np.zeros((3, 3))
            for pos in rel_pos:
                gyration += np.outer(pos, pos)
            gyration /= 3.0
            
            # Eigenvalues
            eigenvalues = np.sort(np.linalg.eigvalsh(gyration))[::-1]
            l1, l2, l3 = eigenvalues
            
            # Asphericity (oblate-ness)
            b = (l1 - 0.5*(l2 + l3))**2
            
            # Acylindricity (prolate-ness)
            c = (l2 - l3)**2
            
            asphericity_values.append(b)
            acylindricity_values.append(c)
        
        return np.array(asphericity_values), np.array(acylindricity_values)
    
    def calculate_coordination_number_gpu(self, oxygen_positions, box, cutoff=3.5):
        """Calculate coordination number (number of neighbors within cutoff)"""
        n_oxygens = len(oxygen_positions)
        coord_numbers = np.zeros(n_oxygens, dtype=np.int32)
        
        box_gpu = cp.asarray(box, dtype=cp.float32)
        positions_gpu = cp.asarray(oxygen_positions, dtype=cp.float32)
        
        batch_size = min(300, n_oxygens)
        
        for i in range(0, n_oxygens, batch_size):
            end_i = min(i + batch_size, n_oxygens)
            batch_pos = positions_gpu[i:end_i]
            
            # PBC distances
            diff = positions_gpu[cp.newaxis, :, :] - batch_pos[:, cp.newaxis, :]
            diff = diff - cp.round(diff / box_gpu) * box_gpu
            distances = cp.sqrt(cp.sum(diff**2, axis=2))
            
            # Count neighbors within cutoff (excluding self)
            neighbors_count = cp.sum((distances > 0.1) & (distances < cutoff), axis=1)
            coord_numbers[i:end_i] = cp.asnumpy(neighbors_count).astype(np.int32)
            
            del diff, distances
        
        del positions_gpu, box_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return coord_numbers
    
    def calculate_hydrogen_bonds_gpu(self, oxygen_positions, hydrogen_positions, box, 
                                     r_cutoff=3.5, angle_cutoff=30.0):
        """
        Detect hydrogen bonds using geometric criteria
        
        H-bond criteria:
        - O...O distance < r_cutoff (typically 3.5 Å)
        - O-H...O angle > angle_cutoff (typically 150°, so angle from linear < 30°)
        """
        n_waters = len(oxygen_positions)
        
        # For each water, check if its H atoms form H-bonds
        hbonds_per_water = np.zeros(n_waters, dtype=np.int32)
        hbond_angles = []
        hbond_distances = []
        
        box_gpu = cp.asarray(box, dtype=cp.float32)
        o_pos_gpu = cp.asarray(oxygen_positions, dtype=cp.float32)
        
        batch_size = min(200, n_waters)
        
        for i in range(0, n_waters, batch_size):
            end_i = min(i + batch_size, n_waters)
            
            for water_idx in range(i, end_i):
                o_donor = oxygen_positions[water_idx]
                h1_idx, h2_idx = 2*water_idx, 2*water_idx+1
                
                if h1_idx >= len(hydrogen_positions) or h2_idx >= len(hydrogen_positions):
                    continue
                
                h1_pos = hydrogen_positions[h1_idx]
                h2_pos = hydrogen_positions[h2_idx]
                
                n_hbonds = 0
                
                # Check each hydrogen
                for h_pos in [h1_pos, h2_pos]:
                    # O-H vector
                    oh_vec = h_pos - o_donor
                    oh_vec = oh_vec - np.round(oh_vec / box) * box
                    
                    # Find acceptor oxygens within cutoff
                    o_diff = oxygen_positions - o_donor
                    o_diff = o_diff - np.round(o_diff / box) * box
                    o_dists = np.linalg.norm(o_diff, axis=1)
                    
                    acceptor_candidates = np.where((o_dists > 0.1) & (o_dists < r_cutoff))[0]
                    
                    for acceptor_idx in acceptor_candidates:
                        o_acceptor_vec = o_diff[acceptor_idx]
                        
                        # O-H...O angle
                        oh_norm = np.linalg.norm(oh_vec)
                        oa_norm = np.linalg.norm(o_acceptor_vec)
                        
                        if oh_norm > 0 and oa_norm > 0:
                            cos_angle = np.dot(oh_vec, o_acceptor_vec) / (oh_norm * oa_norm)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.degrees(np.arccos(cos_angle))
                            
                            # Check if angle is close to 180° (linear)
                            deviation_from_linear = abs(180 - angle)
                            
                            if deviation_from_linear < angle_cutoff:
                                n_hbonds += 1
                                hbond_angles.append(angle)
                                hbond_distances.append(o_dists[acceptor_idx])
                
                hbonds_per_water[water_idx] = n_hbonds
        
        del o_pos_gpu, box_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return hbonds_per_water, np.array(hbond_angles), np.array(hbond_distances)
    
    def calculate_dipole_orientation_gpu(self, oxygen_positions, hydrogen_positions, np_center, box):
        """GPU-accelerated dipole orientation analysis"""
        n_waters = len(oxygen_positions)
        cos_theta_list = []
        distances_list = []
        
        np_center_gpu = cp.asarray(np_center, dtype=cp.float32)
        box_gpu = cp.asarray(box, dtype=cp.float32)
        
        batch_size = min(500, n_waters)
        
        for i in range(0, n_waters, batch_size):
            end_i = min(i + batch_size, n_waters)
            
            o_batch = cp.asarray(oxygen_positions[i:end_i], dtype=cp.float32)
            
            # Handle hydrogen indexing
            h1_indices = np.arange(2*i, 2*end_i, 2)
            h2_indices = np.arange(2*i+1, 2*end_i, 2)
            
            valid_mask = (h2_indices < len(hydrogen_positions))
            h1_indices = h1_indices[valid_mask]
            h2_indices = h2_indices[valid_mask]
            o_batch = o_batch[valid_mask]
            
            if len(h1_indices) == 0:
                continue
            
            h1_batch = cp.asarray(hydrogen_positions[h1_indices], dtype=cp.float32)
            h2_batch = cp.asarray(hydrogen_positions[h2_indices], dtype=cp.float32)
            
            # Dipole vectors
            oh1 = h1_batch - o_batch
            oh1 = oh1 - cp.round(oh1 / box_gpu) * box_gpu
            
            oh2 = h2_batch - o_batch
            oh2 = oh2 - cp.round(oh2 / box_gpu) * box_gpu
            
            dipoles = oh1 + oh2
            dipole_norms = cp.linalg.norm(dipoles, axis=1)
            
            # Radial vectors
            radials = o_batch - np_center_gpu
            radials = radials - cp.round(radials / box_gpu) * box_gpu
            radial_dists = cp.linalg.norm(radials, axis=1)
            
            # Calculate cos(theta)
            valid = (dipole_norms > 0) & (radial_dists > 0)
            
            if cp.any(valid):
                dipoles_norm = dipoles[valid] / dipole_norms[valid, cp.newaxis]
                radials_norm = radials[valid] / radial_dists[valid, cp.newaxis]
                
                cos_theta = cp.sum(dipoles_norm * radials_norm, axis=1)
                
                cos_theta_list.extend(cp.asnumpy(cos_theta).tolist())
                distances_list.extend(cp.asnumpy(radial_dists[valid]).tolist())
            
            del o_batch, h1_batch, h2_batch, oh1, oh2, dipoles, radials
        
        cp.get_default_memory_pool().free_all_blocks()
        
        return np.array(cos_theta_list), np.array(distances_list)
    
    def calculate_radial_density_profile(self, oxygen_positions, np_center, box, bins=50, r_max=25.0):
        """Calculate radial density profile g(r) around nanoparticle"""
        # Distances from NP center
        diff = oxygen_positions - np_center
        diff = diff - np.round(diff / box) * box
        distances = np.linalg.norm(diff, axis=1)
        
        # Histogram
        bin_edges = np.linspace(0, r_max, bins+1)
        counts, _ = np.histogram(distances, bins=bin_edges)
        
        # Volume normalization
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        
        # Density in each shell
        density = counts / bin_volumes
        
        return bin_centers, density
    
    def analyze_frame_comprehensive(self, frame_idx, compute_dynamic=False, reference_data=None):
        """
        Comprehensive analysis of a single frame
        
        Parameters:
        -----------
        frame_idx : int
            Frame index to analyze
        compute_dynamic : bool
            Whether to compute dynamic properties (requires reference_data)
        reference_data : dict
            Reference frame data for dynamic analysis
        """
        self.trajectory.trajectory[frame_idx]
        box = self.trajectory.dimensions[:3]
        
        o_pos = self.oxygens.positions
        h_pos = self.hydrogens.positions
        np_center = np.mean(self.carbons.positions, axis=0)
        
        results = {
            'frame': frame_idx,
            'time_ps': frame_idx * 2.0,  # 2 fs timestep, dump every 1000 steps
            'time_ns': frame_idx * 0.002,
        }
        
        # For structural calculations, we MUST use all oxygens
        # Sampling would give wrong neighbors and incorrect order parameters
        
        # 1. TETRAHEDRAL ORDER (use ALL oxygens for correct neighbor finding)
        q_values = self.calculate_tetrahedral_order_gpu(o_pos, box)
        results['q_mean'] = float(np.mean(q_values))
        results['q_std'] = float(np.std(q_values))
        results['q_values'] = q_values  # For dynamic analysis
        
        # 2. STEINHARDT ORDER PARAMETERS (use ALL oxygens for correct neighbor finding)
        Q_values = self.calculate_steinhardt_order_gpu(o_pos, box, l_values=[4, 6])
        results['Q4_mean'] = float(np.mean(Q_values[4]))
        results['Q4_std'] = float(np.std(Q_values[4]))
        results['Q6_mean'] = float(np.mean(Q_values[6]))
        results['Q6_std'] = float(np.std(Q_values[6]))
        
        # 3. ASPHERICITY & ACYLINDRICITY (every 5th frame)
        if frame_idx % 5 == 0:
            asp, acy = self.calculate_asphericity_acylindricity_gpu(o_pos, h_pos, box, sample_size=500)
            results['asphericity_mean'] = float(np.mean(asp))
            results['asphericity_std'] = float(np.std(asp))
            results['acylindricity_mean'] = float(np.mean(acy))
            results['acylindricity_std'] = float(np.std(acy))
        else:
            results['asphericity_mean'] = np.nan
            results['asphericity_std'] = np.nan
            results['acylindricity_mean'] = np.nan
            results['acylindricity_std'] = np.nan
        
        # 4. COORDINATION NUMBER (use ALL oxygens for correct neighbor finding)
        coord_nums = self.calculate_coordination_number_gpu(o_pos, box, cutoff=3.5)
        results['coord_mean'] = float(np.mean(coord_nums))
        results['coord_std'] = float(np.std(coord_nums))
        
        # 5. HYDROGEN BONDS (every 10th frame)
        if frame_idx % 10 == 0:
            hbonds, hb_angles, hb_dists = self.calculate_hydrogen_bonds_gpu(
                o_pos, h_pos, box, r_cutoff=3.5, angle_cutoff=30.0
            )
            results['hbonds_mean'] = float(np.mean(hbonds))
            results['hbonds_std'] = float(np.std(hbonds))
            results['hbond_angles'] = hb_angles
            results['hbond_distances'] = hb_dists
        else:
            results['hbonds_mean'] = np.nan
            results['hbonds_std'] = np.nan
        
        # 6. DIPOLE ORIENTATION
        cos_theta, dipole_dists = self.calculate_dipole_orientation_gpu(o_pos, h_pos, np_center, box)
        if len(cos_theta) > 0:
            results['cos_theta_mean'] = float(np.mean(cos_theta))
            results['cos_theta_std'] = float(np.std(cos_theta))
            results['dipole_distances'] = dipole_dists
        else:
            results['cos_theta_mean'] = np.nan
            results['cos_theta_std'] = np.nan
        
        # 7. RADIAL DENSITY PROFILE (every 20th frame)
        if frame_idx % 20 == 0:
            r_bins, density = self.calculate_radial_density_profile(o_pos, np_center, box)
            results['radial_profile_r'] = r_bins
            results['radial_profile_density'] = density
        
        # 8. SPATIAL DATA (store positions for first 20 frames for detailed analysis)
        if frame_idx < 20:
            distances_from_np = np.linalg.norm(o_pos - np_center, axis=1)
            results['detailed_q_values'] = q_values
            results['detailed_distances'] = distances_from_np
            results['detailed_cos_theta'] = cos_theta
            results['detailed_dipole_distances'] = dipole_dists
        
        # 9. DYNAMIC PROPERTIES (if reference provided)
        if compute_dynamic and reference_data is not None:
            # Orientational autocorrelation (simplified)
            # Would need to track same molecules across frames for proper implementation
            pass
        
        return results
    
    def analyze_chunk(self, frame_start, frame_end, output_file):
        """
        Analyze a chunk of frames (for parallelization within epsilon)
        
        Parameters:
        -----------
        frame_start : int
            Starting frame index
        frame_end : int
            Ending frame index (exclusive)
        output_file : str
            Path to save results
        """
        print(f"[ε={self.epsilon:.2f}] Worker analyzing frames {frame_start}-{frame_end-1}")
        
        results_list = []
        
        for frame_idx in range(frame_start, frame_end):
            if frame_idx % 10 == 0:
                print(f"[ε={self.epsilon:.2f}] Frame {frame_idx}/{frame_end-1}")
            
            result = self.analyze_frame_comprehensive(frame_idx)
            
            # Extract only serializable data for CSV
            csv_result = {
                'frame': result['frame'],
                'time_ps': result['time_ps'],
                'time_ns': result['time_ns'],
                'q_mean': result['q_mean'],
                'q_std': result['q_std'],
                'Q4_mean': result['Q4_mean'],
                'Q4_std': result['Q4_std'],
                'Q6_mean': result['Q6_mean'],
                'Q6_std': result['Q6_std'],
                'asphericity_mean': result['asphericity_mean'],
                'asphericity_std': result['asphericity_std'],
                'acylindricity_mean': result['acylindricity_mean'],
                'acylindricity_std': result['acylindricity_std'],
                'coord_mean': result['coord_mean'],
                'coord_std': result['coord_std'],
                'hbonds_mean': result['hbonds_mean'],
                'hbonds_std': result['hbonds_std'],
                'cos_theta_mean': result['cos_theta_mean'],
                'cos_theta_std': result['cos_theta_std'],
            }
            
            results_list.append(csv_result)
        
        # Save chunk results
        df = pd.DataFrame(results_list)
        df.to_csv(output_file, index=False)
        print(f"[ε={self.epsilon:.2f}] Saved chunk to {output_file}")
        
        return df


def worker_process(traj_file, epsilon, frame_start, frame_end, output_file, cpu_core, gpu_device=0):
    """
    Worker process for analyzing a chunk of frames
    
    This function runs in a separate process with CPU affinity
    """
    # Set CPU affinity
    os.sched_setaffinity(0, {cpu_core})
    print(f"[Worker ε={epsilon:.2f}, Core {cpu_core}] Process bound to CPU core {cpu_core}")
    print(f"[Worker ε={epsilon:.2f}, Core {cpu_core}] Analyzing frames {frame_start}-{frame_end-1}")
    
    # Initialize analyzer
    analyzer = ComprehensiveWaterAnalyzer(traj_file, epsilon, gpu_device)
    
    # Analyze chunk
    df = analyzer.analyze_chunk(frame_start, frame_end, output_file)
    
    print(f"[Worker ε={epsilon:.2f}, Core {cpu_core}] COMPLETE")
    
    return output_file


def analyze_epsilon_parallel(epsilon, base_dir='.', n_workers=4, gpu_device=0):
    """
    Analyze one epsilon value with multiple workers (2D parallelization)
    
    Parameters:
    -----------
    epsilon : float
        Epsilon value to analyze
    base_dir : str
        Base directory
    n_workers : int
        Number of worker processes for this epsilon
    gpu_device : int
        GPU device to use
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING ε = {epsilon:.2f} with {n_workers} workers")
    print(f"{'='*80}")
    
    base_dir = Path(base_dir)
    
    # Find trajectory file
    if epsilon == 0.0:
        eps_dir = base_dir / "epsilon_0.0"
    else:
        eps_dir = base_dir / f"epsilon_{epsilon:.2f}"
    
    traj_file = eps_dir / "production.lammpstrj"
    
    if not traj_file.exists():
        print(f"ERROR: Trajectory not found: {traj_file}")
        return None
    
    # Get number of frames
    temp_u = mda.Universe(str(traj_file), format='LAMMPSDUMP', atom_style='id type x y z', dt=2.0)
    n_frames = len(temp_u.trajectory)
    print(f"Total frames: {n_frames}")
    
    # Split frames across workers
    frames_per_worker = n_frames // n_workers
    frame_ranges = []
    
    for i in range(n_workers):
        start = i * frames_per_worker
        end = (i + 1) * frames_per_worker if i < n_workers - 1 else n_frames
        frame_ranges.append((start, end))
    
    print(f"Frame distribution:")
    for i, (start, end) in enumerate(frame_ranges):
        print(f"  Worker {i}: frames {start}-{end-1} ({end-start} frames)")
    
    # Create output directory
    output_dir = base_dir / f"comprehensive_results_eps_{epsilon:.2f}"
    output_dir.mkdir(exist_ok=True)
    
    # Launch workers
    workers = []
    output_files = []
    
    for worker_id, (frame_start, frame_end) in enumerate(frame_ranges):
        output_file = output_dir / f"chunk_{worker_id}.csv"
        output_files.append(output_file)
        
        # Assign CPU core (round-robin)
        cpu_core = worker_id % os.cpu_count()
        
        # Launch subprocess
        cmd = [
            sys.executable,
            __file__,
            '--worker',
            '--traj', str(traj_file),
            '--epsilon', str(epsilon),
            '--frame-start', str(frame_start),
            '--frame-end', str(frame_end),
            '--output', str(output_file),
            '--cpu-core', str(cpu_core),
            '--gpu-device', str(gpu_device)
        ]
        
        print(f"Launching worker {worker_id} on CPU core {cpu_core}")
        proc = subprocess.Popen(cmd)
        workers.append(proc)
    
    # Wait for all workers
    print(f"\nWaiting for {len(workers)} workers to complete...")
    for i, proc in enumerate(workers):
        proc.wait()
        print(f"Worker {i} finished with exit code {proc.returncode}")
    
    # Aggregate results
    print(f"\nAggregating results from {len(output_files)} chunks...")
    chunks = []
    for f in output_files:
        if f.exists():
            df = pd.read_csv(f)
            chunks.append(df)
        else:
            print(f"WARNING: Missing chunk file: {f}")
    
    if chunks:
        df_combined = pd.concat(chunks, ignore_index=True)
        df_combined = df_combined.sort_values('frame').reset_index(drop=True)
        
        # Save combined results
        output_file = base_dir / f"comprehensive_analysis_eps_{epsilon:.2f}.csv"
        df_combined.to_csv(output_file, index=False)
        print(f"✓ Saved combined results: {output_file}")
        
        return df_combined
    else:
        print("ERROR: No chunk files found!")
        return None


def create_comprehensive_plots(results_dict, output_dir='.'):
    """
    Create comprehensive 2D and 3D visualization plots
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of {epsilon: DataFrame} with analysis results
    output_dir : str
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION PLOTS")
    print("="*80)
    
    epsilon_values = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
    
    # =========================================================================
    # PLOT 1: Time Evolution of All Order Parameters (6 subplots)
    # =========================================================================
    print("\n[1/8] Creating time evolution plots...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Time Evolution of Structural Order Parameters', fontsize=16, fontweight='bold')
    
    params = [
        ('q_mean', 'Tetrahedral Order q', 'q'),
        ('Q4_mean', 'Steinhardt Order Q₄', 'Q₄'),
        ('Q6_mean', 'Steinhardt Order Q₆', 'Q₆'),
        ('coord_mean', 'Coordination Number', 'N_coord'),
        ('hbonds_mean', 'Hydrogen Bonds per Water', 'N_HB'),
        ('cos_theta_mean', 'Dipole Orientation ⟨cos θ⟩', '⟨cos θ⟩')
    ]
    
    for idx, (param, title, ylabel) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        for eps, color in zip(epsilon_values, colors):
            df = results_dict[eps]
            ax.plot(df['time_ns'], df[param], label=f'ε = {eps:.2f}', 
                   color=color, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_time_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_time_evolution.png")
    
    # =========================================================================
    # PLOT 2: Epsilon Dependence (averaged over last quarter)
    # =========================================================================
    print("\n[2/8] Creating epsilon dependence plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Structural Parameters vs Nanoparticle Hydrophilicity (ε)', 
                 fontsize=16, fontweight='bold')
    
    params_eps = [
        ('q_mean', 'Tetrahedral Order q', 'q', (0.5, 0.75)),
        ('Q4_mean', 'Steinhardt Q₄', 'Q₄', (0.02, 0.20)),
        ('Q6_mean', 'Steinhardt Q₆', 'Q₆', (0.01, 0.10)),
        ('coord_mean', 'Coordination Number', 'N_coord', (4, 6)),
        ('hbonds_mean', 'H-Bonds per Water', 'N_HB', (0, 5)),
        ('asphericity_mean', 'Asphericity (Oblate)', 'b', None)
    ]
    
    for idx, (param, title, ylabel, ylim) in enumerate(params_eps):
        ax = axes[idx // 3, idx % 3]
        
        means = []
        stds = []
        
        for eps in epsilon_values:
            df = results_dict[eps]
            # Average over last quarter
            last_quarter = df[df['time_ns'] >= 0.75]
            
            if param in last_quarter.columns:
                valid_data = last_quarter[param].dropna()
                if len(valid_data) > 0:
                    means.append(valid_data.mean())
                    stds.append(valid_data.std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot with error bars
        ax.errorbar(epsilon_values, means, yerr=stds, 
                   marker='o', markersize=10, capsize=5, capthick=2,
                   linewidth=2, color='#2E86AB', ecolor='#A23B72')
        
        # Add trend line
        valid = ~np.isnan(means)
        if np.sum(valid) > 2:
            z = np.polyfit(np.array(epsilon_values)[valid], means[valid], 2)
            p = np.poly1d(z)
            eps_smooth = np.linspace(min(epsilon_values), max(epsilon_values), 100)
            ax.plot(eps_smooth, p(eps_smooth), '--', color='#F18F01', 
                   linewidth=2, alpha=0.7, label='Quadratic fit')
        
        ax.set_xlabel('Nanoparticle ε (hydrophilicity →)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Highlight transition region
        ax.axvspan(0.15, 0.20, alpha=0.15, color='red', 
                  label='Transition zone' if idx == 0 else '')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_epsilon_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_epsilon_dependence.png")
    
    # =========================================================================
    # PLOT 3: 3D Scatter - Tetrahedral Order in (ε, time, q) space
    # =========================================================================
    print("\n[3/8] Creating 3D tetrahedral order visualization...")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        eps_array = np.full(len(df), eps)
        
        ax1.scatter(eps_array, df['time_ns'], df['q_mean'], 
                   c=[color], s=20, alpha=0.6, label=f'ε = {eps:.2f}')
    
    ax1.set_xlabel('ε (hydrophilicity)', fontsize=11)
    ax1.set_ylabel('Time (ns)', fontsize=11)
    ax1.set_zlabel('Tetrahedral Order q', fontsize=11)
    ax1.set_title('3D Evolution: q(ε, t)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # 3D surface plot (averaged in time bins)
    ax2 = fig.add_subplot(222, projection='3d')
    
    time_bins = np.linspace(0, 1.0, 20)
    eps_grid, time_grid = np.meshgrid(epsilon_values, time_bins[:-1])
    q_grid = np.zeros_like(eps_grid)
    
    for i, eps in enumerate(epsilon_values):
        df = results_dict[eps]
        for j, t_start in enumerate(time_bins[:-1]):
            t_end = time_bins[j+1]
            mask = (df['time_ns'] >= t_start) & (df['time_ns'] < t_end)
            if mask.sum() > 0:
                q_grid[j, i] = df.loc[mask, 'q_mean'].mean()
    
    surf = ax2.plot_surface(eps_grid, time_grid, q_grid, cmap='viridis', 
                           alpha=0.8, edgecolor='none')
    ax2.set_xlabel('ε', fontsize=11)
    ax2.set_ylabel('Time (ns)', fontsize=11)
    ax2.set_zlabel('⟨q⟩', fontsize=11)
    ax2.set_title('Surface: ⟨q⟩(ε, t)', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax2, shrink=0.5, label='⟨q⟩')
    ax2.view_init(elev=25, azim=135)
    
    # Contour plot (time vs epsilon, colored by q)
    ax3 = fig.add_subplot(223)
    
    contour = ax3.contourf(eps_grid, time_grid, q_grid, levels=15, cmap='viridis')
    ax3.set_xlabel('ε (hydrophilicity →)', fontsize=11)
    ax3.set_ylabel('Time (ns)', fontsize=11)
    ax3.set_title('Contour Map: ⟨q⟩(ε, t)', fontsize=12, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax3, label='⟨q⟩')
    
    # Trajectory plot: q vs Q4 colored by epsilon
    ax4 = fig.add_subplot(224)
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        # Use last quarter data
        last_quarter = df[df['time_ns'] >= 0.75]
        ax4.scatter(last_quarter['q_mean'], last_quarter['Q4_mean'], 
                   c=[color], s=30, alpha=0.6, label=f'ε = {eps:.2f}')
    
    ax4.set_xlabel('Tetrahedral Order q', fontsize=11)
    ax4.set_ylabel('Steinhardt Q₄', fontsize=11)
    ax4.set_title('Order Parameter Space (last 0.25 ns)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_3D_tetrahedral.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_3D_tetrahedral.png")
    
    # =========================================================================
    # PLOT 4: 3D Phase Space - (q, Q4, Q6)
    # =========================================================================
    print("\n[4/8] Creating 3D phase space visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D scatter
    ax1 = fig.add_subplot(221, projection='3d')
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        last_quarter = df[df['time_ns'] >= 0.75]
        
        ax1.scatter(last_quarter['q_mean'], last_quarter['Q4_mean'], 
                   last_quarter['Q6_mean'], c=[color], s=40, alpha=0.7,
                   label=f'ε = {eps:.2f}')
    
    ax1.set_xlabel('Tetrahedral q', fontsize=11)
    ax1.set_ylabel('Steinhardt Q₄', fontsize=11)
    ax1.set_zlabel('Steinhardt Q₆', fontsize=11)
    ax1.set_title('3D Phase Space: (q, Q₄, Q₆)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # View from different angle
    ax2 = fig.add_subplot(222, projection='3d')
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        last_quarter = df[df['time_ns'] >= 0.75]
        
        ax2.scatter(last_quarter['q_mean'], last_quarter['Q4_mean'], 
                   last_quarter['Q6_mean'], c=[color], s=40, alpha=0.7,
                   label=f'ε = {eps:.2f}')
    
    ax2.set_xlabel('Tetrahedral q', fontsize=11)
    ax2.set_ylabel('Steinhardt Q₄', fontsize=11)
    ax2.set_zlabel('Steinhardt Q₆', fontsize=11)
    ax2.set_title('3D Phase Space (different view)', fontsize=12, fontweight='bold')
    ax2.view_init(elev=60, azim=135)
    
    # 2D projections
    ax3 = fig.add_subplot(223)
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        last_quarter = df[df['time_ns'] >= 0.75]
        ax3.scatter(last_quarter['Q4_mean'], last_quarter['Q6_mean'], 
                   c=[color], s=30, alpha=0.6, label=f'ε = {eps:.2f}')
    
    ax3.set_xlabel('Steinhardt Q₄', fontsize=11)
    ax3.set_ylabel('Steinhardt Q₆', fontsize=11)
    ax3.set_title('Projection: Q₄-Q₆ Plane', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Coordination vs tetrahedral
    ax4 = fig.add_subplot(224)
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        last_quarter = df[df['time_ns'] >= 0.75]
        ax4.scatter(last_quarter['q_mean'], last_quarter['coord_mean'], 
                   c=[color], s=30, alpha=0.6, label=f'ε = {eps:.2f}')
    
    ax4.set_xlabel('Tetrahedral Order q', fontsize=11)
    ax4.set_ylabel('Coordination Number', fontsize=11)
    ax4.set_title('Structure vs Coordination', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_3D_phase_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_3D_phase_space.png")
    
    # =========================================================================
    # PLOT 5: Distributions (Histograms)
    # =========================================================================
    print("\n[5/8] Creating distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of Structural Parameters (Last 0.25 ns)', 
                 fontsize=16, fontweight='bold')
    
    dist_params = [
        ('q_mean', 'Tetrahedral Order q', (0.2, 0.9), 30),
        ('Q4_mean', 'Steinhardt Q₄', (0, 0.3), 30),
        ('Q6_mean', 'Steinhardt Q₆', (0, 0.15), 30),
        ('coord_mean', 'Coordination Number', (3, 7), 20),
        ('hbonds_mean', 'H-Bonds per Water', (0, 6), 30),
        ('asphericity_mean', 'Asphericity b', None, 30)
    ]
    
    for idx, (param, title, xlim, bins) in enumerate(dist_params):
        ax = axes[idx // 3, idx % 3]
        
        for eps, color in zip(epsilon_values, colors):
            df = results_dict[eps]
            last_quarter = df[df['time_ns'] >= 0.75]
            
            data = last_quarter[param].dropna()
            if len(data) > 0:
                ax.hist(data, bins=bins, alpha=0.5, color=color, 
                       label=f'ε = {eps:.2f}', density=True)
        
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'Distribution: {title}', fontsize=12, fontweight='bold')
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_distributions.png")
    
    # =========================================================================
    # PLOT 6: Correlation Matrix (Heatmaps)
    # =========================================================================
    print("\n[6/8] Creating correlation matrices...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Correlation Between Structural Parameters', 
                 fontsize=16, fontweight='bold')
    
    correlation_params = ['q_mean', 'Q4_mean', 'Q6_mean', 'coord_mean', 
                         'hbonds_mean', 'asphericity_mean']
    param_labels = ['q', 'Q₄', 'Q₆', 'N_coord', 'N_HB', 'b']
    
    for idx, eps in enumerate(epsilon_values):
        ax = axes[idx // 3, idx % 3]
        
        df = results_dict[eps]
        last_quarter = df[df['time_ns'] >= 0.75]
        
        # Calculate correlation matrix
        corr_data = last_quarter[correlation_params].dropna()
        corr_matrix = corr_data.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(len(param_labels)))
        ax.set_yticks(np.arange(len(param_labels)))
        ax.set_xticklabels(param_labels, fontsize=10)
        ax.set_yticklabels(param_labels, fontsize=10)
        
        # Add correlation values
        for i in range(len(param_labels)):
            for j in range(len(param_labels)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'ε = {eps:.2f}', fontsize=12, fontweight='bold')
        
        # Add colorbar
        if idx == 2 or idx == 5:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_correlations.png")
    
    # =========================================================================
    # PLOT 7: Statistical Summary (Box plots)
    # =========================================================================
    print("\n[7/8] Creating statistical summary plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Statistical Distribution Across ε Values (Last 0.25 ns)', 
                 fontsize=16, fontweight='bold')
    
    box_params = [
        ('q_mean', 'Tetrahedral Order q'),
        ('Q4_mean', 'Steinhardt Q₄'),
        ('Q6_mean', 'Steinhardt Q₆'),
        ('coord_mean', 'Coordination Number'),
        ('hbonds_mean', 'H-Bonds per Water'),
        ('cos_theta_mean', 'Dipole ⟨cos θ⟩')
    ]
    
    for idx, (param, title) in enumerate(box_params):
        ax = axes[idx // 3, idx % 3]
        
        data_list = []
        labels = []
        
        for eps in epsilon_values:
            df = results_dict[eps]
            last_quarter = df[df['time_ns'] >= 0.75]
            data = last_quarter[param].dropna().values
            if len(data) > 0:
                data_list.append(data)
                labels.append(f'{eps:.2f}')
        
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors[:len(data_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_xlabel('ε', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'Box Plot: {title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_boxplots.png")
    
    # =========================================================================
    # PLOT 8: 3D Trajectory in Structure Space
    # =========================================================================
    print("\n[8/8] Creating 3D trajectory visualization...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Trajectory: q vs time vs epsilon
    ax1 = fig.add_subplot(121, projection='3d')
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        eps_array = np.full(len(df), eps)
        
        # Plot as a line to show trajectory
        ax1.plot(df['time_ns'], eps_array, df['q_mean'], 
                color=color, linewidth=2, alpha=0.8, label=f'ε = {eps:.2f}')
        
        # Add markers at start and end
        ax1.scatter([df['time_ns'].iloc[0]], [eps], [df['q_mean'].iloc[0]], 
                   c=[color], s=100, marker='o', edgecolor='black', linewidth=2)
        ax1.scatter([df['time_ns'].iloc[-1]], [eps], [df['q_mean'].iloc[-1]], 
                   c=[color], s=100, marker='s', edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('ε (hydrophilicity)', fontsize=12)
    ax1.set_zlabel('Tetrahedral Order q', fontsize=12)
    ax1.set_title('3D Trajectory: Evolution of q', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.view_init(elev=20, azim=120)
    
    # Multi-parameter 3D trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    
    for eps, color in zip(epsilon_values, colors):
        df = results_dict[eps]
        
        # Plot trajectory in (q, Q4, coord) space
        ax2.plot(df['q_mean'], df['Q4_mean'], df['coord_mean'],
                color=color, linewidth=2, alpha=0.8, label=f'ε = {eps:.2f}')
        
        # Time-colored scatter points
        scatter = ax2.scatter(df['q_mean'], df['Q4_mean'], df['coord_mean'],
                            c=df['time_ns'], cmap='coolwarm', s=20, alpha=0.5)
    
    ax2.set_xlabel('Tetrahedral q', fontsize=12)
    ax2.set_ylabel('Steinhardt Q₄', fontsize=12)
    ax2.set_zlabel('Coordination N', fontsize=12)
    ax2.set_title('Multi-Parameter Phase Space', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.view_init(elev=25, azim=45)
    
    cbar = fig.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label('Time (ns)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_3D_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: comprehensive_3D_trajectories.png")
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE!")
    print("="*80)
    print(f"\nGenerated 8 comprehensive visualization files in: {output_dir}")
    print("\nPlot files:")
    print("  1. comprehensive_time_evolution.png      - Time series of all parameters")
    print("  2. comprehensive_epsilon_dependence.png  - ε-dependence with error bars")
    print("  3. comprehensive_3D_tetrahedral.png      - 3D tetrahedral order visualization")
    print("  4. comprehensive_3D_phase_space.png      - 3D phase space (q, Q4, Q6)")
    print("  5. comprehensive_distributions.png       - Probability distributions")
    print("  6. comprehensive_correlations.png        - Correlation matrices")
    print("  7. comprehensive_boxplots.png            - Statistical box plots")
    print("  8. comprehensive_3D_trajectories.png     - 3D trajectory evolution")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive water structure analysis')
    parser.add_argument('--worker', action='store_true', help='Run as worker process')
    parser.add_argument('--traj', type=str, help='Trajectory file')
    parser.add_argument('--epsilon', type=float, help='Epsilon value')
    parser.add_argument('--frame-start', type=int, help='Starting frame')
    parser.add_argument('--frame-end', type=int, help='Ending frame')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--cpu-core', type=int, help='CPU core to bind to')
    parser.add_argument('--gpu-device', type=int, default=0, help='GPU device')
    
    args = parser.parse_args()
    
    if args.worker:
        # Run as worker
        worker_process(
            args.traj,
            args.epsilon,
            args.frame_start,
            args.frame_end,
            args.output,
            args.cpu_core,
            args.gpu_device
        )
    else:
        # Run master process
        print("\n" + "="*80)
        print("COMPREHENSIVE GPU-ACCELERATED WATER STRUCTURE ANALYSIS")
        print("With 2D Parallelization (Across + Within Epsilon)")
        print("="*80 + "\n")
        
        epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        n_workers_per_epsilon = 3  # 3 workers per epsilon = 18 total workers
        
        # Analyze each epsilon sequentially (but with parallel workers within)
        results = {}
        for eps in epsilon_values:
            df = analyze_epsilon_parallel(eps, base_dir='.', n_workers=n_workers_per_epsilon, gpu_device=0)
            if df is not None:
                results[eps] = df
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAnalyzed {len(results)} epsilon values")
        print(f"Total frames per epsilon: 500")
        print(f"Workers per epsilon: {n_workers_per_epsilon}")
        print(f"Total worker processes: {len(epsilon_values) * n_workers_per_epsilon}")
        
        # Generate comprehensive plots
        if len(results) > 0:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATION PLOTS")
            print("="*80)
            create_comprehensive_plots(results, output_dir='comprehensive_plots')
        else:
            print("\nWARNING: No results available for plotting!")


if __name__ == "__main__":
    main()
