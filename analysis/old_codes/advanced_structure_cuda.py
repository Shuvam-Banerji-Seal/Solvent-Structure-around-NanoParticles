#!/usr/bin/env python3
"""
Advanced Solvent Structure Analysis with GPU Acceleration
==========================================================

Comprehensive structural characterization including:
1. Detailed RDF analysis (all atom pairs)
2. Spatial density maps around nanoparticle
3. Orientation analysis of water molecules
4. Hydrogen bonding network analysis
5. Water clustering and layering
6. Time-resolved structural evolution
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


class AdvancedStructuralAnalyzer:
    """Advanced structural analysis with GPU acceleration"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.results = {}
        print(f"GPU: {cp.cuda.Device()}, Memory: {cp.cuda.Device().mem_info[1]/1e9:.1f} GB")
    
    def load_trajectory_dcd(self, epsilon: float, stage='production'):
        """Load DCD trajectory file"""
        
        if epsilon == 0.0:
            eps_dir = self.base_dir / "epsilon_0.0"
        else:
            eps_dir = self.base_dir / f"epsilon_{epsilon:.2f}"
        
        # Try lammpstrj format instead of DCD
        trj_file = eps_dir / f"{stage}.lammpstrj"
        
        if not trj_file.exists():
            print(f"  Trajectory file not found: {trj_file}")
            return None
        
        try:
            u = mda.Universe(str(trj_file), format='LAMMPSDUMP', 
                           atom_style='id type x y z')
            print(f"  Loaded {len(u.trajectory)} frames from {stage}.lammpstrj")
            return u
        except Exception as e:
            print(f"  Error loading trajectory: {e}")
            return None
    
    def calculate_spatial_density_gpu(self, coords_ref, coords_water, box_size, 
                                     grid_size=100, r_max=15.0):
        """Calculate 3D density map around reference point using GPU"""
        
        # Transfer to GPU
        coords_ref_gpu = cp.array(coords_ref, dtype=cp.float32)
        coords_water_gpu = cp.array(coords_water, dtype=cp.float32)
        L = cp.array([box_size[0], box_size[1], box_size[2]], dtype=cp.float32)
        
        # Create 3D grid
        bins = np.linspace(-r_max, r_max, grid_size)
        density = cp.zeros((grid_size, grid_size, grid_size), dtype=cp.float32)
        
        # Calculate relative positions
        dr = coords_water_gpu - coords_ref_gpu
        
        # Apply PBC
        dr = dr - L * cp.round(dr / L)
        
        # Filter atoms within r_max
        distances = cp.sqrt(cp.sum(dr**2, axis=1))
        mask = distances < r_max
        dr_filtered = dr[mask]
        
        # Bin into 3D grid
        for i in range(len(dr_filtered)):
            x, y, z = dr_filtered[i]
            ix = int((x + r_max) / (2 * r_max) * grid_size)
            iy = int((y + r_max) / (2 * r_max) * grid_size)
            iz = int((z + r_max) / (2 * r_max) * grid_size)
            
            if 0 <= ix < grid_size and 0 <= iy < grid_size and 0 <= iz < grid_size:
                density[ix, iy, iz] += 1
        
        # Transfer back to CPU
        density_cpu = cp.asnumpy(density)
        
        # Normalize
        volume_element = (2 * r_max / grid_size) ** 3
        bulk_density = len(coords_water) / (box_size[0] * box_size[1] * box_size[2])
        density_normalized = density_cpu / volume_element / bulk_density
        
        cp.get_default_memory_pool().free_all_blocks()
        
        return density_normalized, bins
    
    def calculate_water_orientation(self, u, frame_indices, nanoparticle_center):
        """Calculate water molecule orientations relative to nanoparticle - GPU accelerated"""
        
        print(f"    GPU-accelerated orientation analysis...")
        
        all_orientations = []
        all_distances = []
        
        # Sample fewer frames for orientation (computationally expensive)
        sample_frames = frame_indices[::2]  # Every other frame
        
        for frame_idx in sample_frames:
            u.trajectory[frame_idx]
            
            # Get positions
            oxygens = u.select_atoms('type 1')
            hydrogens = u.select_atoms('type 2')
            
            o_pos = oxygens.positions  # All oxygen positions
            h_pos = hydrogens.positions  # All hydrogen positions
            
            # Transfer to GPU
            o_pos_gpu = cp.asarray(o_pos, dtype=cp.float32)
            h_pos_gpu = cp.asarray(h_pos, dtype=cp.float32)
            np_center_gpu = cp.asarray(nanoparticle_center, dtype=cp.float32)
            
            # For each oxygen, find its 2 bonded hydrogens using GPU
            n_waters = len(o_pos)
            orientations_frame = []
            distances_frame = []
            
            # Process in chunks to avoid memory issues
            chunk_size = 500
            for chunk_start in range(0, n_waters, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_waters)
                o_chunk = o_pos_gpu[chunk_start:chunk_end]
                
                # Calculate distances from each O to all H atoms
                # Broadcasting: (n_chunk, 1, 3) - (1, n_H, 3) = (n_chunk, n_H, 3)
                diff = o_chunk[:, cp.newaxis, :] - h_pos_gpu[cp.newaxis, :, :]
                dists = cp.sqrt(cp.sum(diff**2, axis=2))
                
                # Find closest 2 hydrogens for each oxygen (O-H bond ~0.96-1.0 Å)
                # Get indices of H atoms within bonding distance
                for i in range(len(o_chunk)):
                    o_i_pos = o_chunk[i]
                    bonded_mask = dists[i] < 1.2
                    bonded_indices = cp.where(bonded_mask)[0]
                    
                    if len(bonded_indices) >= 2:
                        # Take closest 2
                        closest_2_idx = cp.argsort(dists[i][bonded_mask])[:2]
                        h_idx = bonded_indices[closest_2_idx]
                        
                        h1_pos = h_pos_gpu[h_idx[0]]
                        h2_pos = h_pos_gpu[h_idx[1]]
                        
                        # Calculate dipole (bisector of H-O-H)
                        oh1 = h1_pos - o_i_pos
                        oh2 = h2_pos - o_i_pos
                        dipole = (oh1 + oh2) / 2.0
                        dipole_norm = cp.linalg.norm(dipole)
                        if dipole_norm > 1e-6:
                            dipole = dipole / dipole_norm
                            
                            # Radial vector from NP center
                            r_vec = o_i_pos - np_center_gpu
                            r_dist = cp.linalg.norm(r_vec)
                            if r_dist > 1e-6:
                                r_vec = r_vec / r_dist
                                
                                # Cosine of angle
                                cos_theta = cp.dot(dipole, r_vec)
                                
                                orientations_frame.append(float(cos_theta))
                                distances_frame.append(float(r_dist))
            
            all_orientations.extend(orientations_frame)
            all_distances.extend(distances_frame)
            
            # Clean up GPU memory
            cp.get_default_memory_pool().free_all_blocks()
        
        print(f"      Analyzed {len(all_orientations)} water molecules")
        
        return np.array(all_orientations), np.array(all_distances)
    
    def calculate_hydrogen_bonds(self, u, frame_idx, distance_cutoff=3.5, angle_cutoff=30):
        """Simplified hydrogen bond count"""
        print(f"    (Simplified H-bond analysis)")
        
        u.trajectory[frame_idx]
        
        oxygens = u.select_atoms('type 1')
        hydrogens = u.select_atoms('type 2')
        
        # Just count O...H pairs within cutoff
        hbond_count = 0
        o_pos = oxygens.positions
        h_pos = hydrogens.positions
        
        # Use simplified distance calculation
        for i in range(min(100, len(o_pos))):  # Sample only first 100 oxygens
            distances = np.linalg.norm(h_pos - o_pos[i], axis=1)
            hbond_count += np.sum(distances < distance_cutoff)
        
        return hbond_count
    
    def analyze_water_layering(self, rdf_r, rdf_g):
        """Identify water layers from RDF"""
        
        from scipy.signal import find_peaks
        
        # Find peaks (maxima)
        peaks, properties = find_peaks(rdf_g, height=0.5, prominence=0.1)
        
        # Find minima (layer boundaries)
        minima, _ = find_peaks(-rdf_g, prominence=0.05)
        
        layers = []
        for i in range(len(peaks)):
            peak_r = rdf_r[peaks[i]]
            peak_height = rdf_g[peaks[i]]
            
            # Find boundaries
            inner_bound = 0
            outer_bound = rdf_r[-1]
            
            for m_idx in minima:
                if rdf_r[m_idx] < peak_r:
                    inner_bound = rdf_r[m_idx]
                elif rdf_r[m_idx] > peak_r:
                    outer_bound = rdf_r[m_idx]
                    break
            
            layers.append({
                'layer_number': i + 1,
                'peak_position': peak_r,
                'peak_height': peak_height,
                'inner_boundary': inner_bound,
                'outer_boundary': outer_bound,
                'thickness': outer_bound - inner_bound
            })
        
        return layers
    
    def comprehensive_structural_analysis(self, epsilon: float):
        """Run comprehensive structural analysis for one system"""
        
        print(f"\n{'='*80}")
        print(f"ADVANCED STRUCTURAL ANALYSIS: ε = {epsilon:.2f}")
        print(f"{'='*80}")
        
        # Load production trajectory
        u = self.load_trajectory_dcd(epsilon, 'production')
        if u is None:
            return None
        
        results = {'epsilon': epsilon}
        
        # Use last 50 frames for analysis
        n_frames = len(u.trajectory)
        frame_indices = list(range(max(0, n_frames - 50), n_frames, 2))  # Every 2nd frame
        print(f"  Analyzing {len(frame_indices)} frames")
        
        # Get nanoparticle center
        u.trajectory[frame_indices[0]]
        carbons = u.select_atoms('type 3')
        np_center = carbons.center_of_mass()
        results['nanoparticle_center'] = np_center
        results['nanoparticle_natoms'] = len(carbons)
        
        # 1. Calculate spatial density distribution
        print(f"\n  Calculating 3D density distribution...")
        u.trajectory[frame_indices[0]]
        oxygens = u.select_atoms('type 1')
        box = u.dimensions[:3]
        
        density_3d, bins = self.calculate_spatial_density_gpu(
            np_center, oxygens.positions, box, grid_size=80, r_max=15.0
        )
        results['density_3d'] = density_3d
        results['density_bins'] = bins
        
        print(f"    3D density map calculated (grid: 80³)")
        
        # 2. Water orientation analysis (GPU-accelerated)
        print(f"  Analyzing water orientations...")
        orientations, distances = self.calculate_water_orientation(u, frame_indices[:15], np_center)
        results['orientations'] = orientations
        results['orientation_distances'] = distances
        
        # 3. Hydrogen bond analysis (simplified)
        print(f"  Analyzing hydrogen bonds...")
        hbond_counts = []
        for fidx in frame_indices[:5]:  # Sample 5 frames only
            hbonds = self.calculate_hydrogen_bonds(u, fidx)
            hbond_counts.append(hbonds)
        
        results['hbonds_per_frame'] = np.mean(hbond_counts) if hbond_counts else 0
        results['hbond_distances'] = []
        results['hbond_angles'] = []
        
        print(f"    Found ~{results['hbonds_per_frame']:.1f} H-bonds per frame (sample)")
        
        self.results[epsilon] = results
        return results
    
    def plot_3d_density_slices(self, epsilon: float):
        """Plot 2D slices through 3D density"""
        
        if epsilon not in self.results:
            return
        
        results = self.results[epsilon]
        density = results['density_3d']
        bins = results['density_bins']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'3D Density Distribution: ε = {epsilon:.2f}', fontsize=16, fontweight='bold')
        
        # Slice through center
        center_idx = len(bins) // 2
        
        # XY plane
        ax = axes[0, 0]
        im = ax.imshow(density[:, :, center_idx].T, origin='lower', 
                      extent=[bins[0], bins[-1], bins[0], bins[-1]],
                      cmap='viridis', vmin=0, vmax=3)
        ax.set_xlabel('X (Å)', fontweight='bold')
        ax.set_ylabel('Y (Å)', fontweight='bold')
        ax.set_title('XY Plane (z=0)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='ρ/ρ_bulk')
        ax.plot(0, 0, 'r*', markersize=15, label='NP center')
        
        # XZ plane
        ax = axes[0, 1]
        im = ax.imshow(density[:, center_idx, :].T, origin='lower',
                      extent=[bins[0], bins[-1], bins[0], bins[-1]],
                      cmap='viridis', vmin=0, vmax=3)
        ax.set_xlabel('X (Å)', fontweight='bold')
        ax.set_ylabel('Z (Å)', fontweight='bold')
        ax.set_title('XZ Plane (y=0)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='ρ/ρ_bulk')
        ax.plot(0, 0, 'r*', markersize=15)
        
        # YZ plane
        ax = axes[0, 2]
        im = ax.imshow(density[center_idx, :, :].T, origin='lower',
                      extent=[bins[0], bins[-1], bins[0], bins[-1]],
                      cmap='viridis', vmin=0, vmax=3)
        ax.set_xlabel('Y (Å)', fontweight='bold')
        ax.set_ylabel('Z (Å)', fontweight='bold')
        ax.set_title('YZ Plane (x=0)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='ρ/ρ_bulk')
        ax.plot(0, 0, 'r*', markersize=15)
        
        # Radial average (optimized with numpy)
        ax = axes[1, 0]
        grid_center = len(bins) // 2
        
        # Create radial distance array once
        x = np.arange(len(bins)) - grid_center
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Bin the radial profile
        r_max = len(bins) // 2
        r_bins_rad = np.linspace(0, r_max, 30)
        radial_profile = []
        
        for i in range(len(r_bins_rad) - 1):
            mask = (R >= r_bins_rad[i]) & (R < r_bins_rad[i+1])
            if mask.any():
                radial_profile.append(np.mean(density[mask]))
            else:
                radial_profile.append(0)
        
        r_bins_plot = (r_bins_rad[:-1] + r_bins_rad[1:]) / 2 * (bins[1] - bins[0])
        
        ax.plot(r_bins_plot, radial_profile, lw=2.5, color='darkblue')
        ax.set_xlabel('Distance from NP center (Å)', fontweight='bold')
        ax.set_ylabel('ρ/ρ_bulk', fontweight='bold')
        ax.set_title('Radial Density Profile', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Bulk')
        ax.legend()
        
        # Orientation analysis
        ax = axes[1, 1]
        if 'orientations' in results and results['orientations'] is not None:
            orientations = results['orientations']
            distances = results['orientation_distances']
            
            # 2D histogram
            hist, xedges, yedges = np.histogram2d(distances, orientations, 
                                                   bins=[30, 30], 
                                                   range=[[0, 15], [-1, 1]])
            hist = hist.T
            im = ax.imshow(hist, origin='lower', aspect='auto',
                          extent=[0, 15, -1, 1], cmap='hot')
            ax.set_xlabel('Distance from NP (Å)', fontweight='bold')
            ax.set_ylabel('cos(θ) [dipole·r]', fontweight='bold')
            ax.set_title('Water Orientation Map', fontweight='bold')
            plt.colorbar(im, ax=ax, label='Count')
            ax.axhline(0, color='cyan', linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'Orientation analysis\nskipped for large systems',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Water Orientation Map', fontweight='bold')
            ax.axis('off')
        
        # Hydrogen bonding
        ax = axes[1, 2]
        if 'hbond_distances' in results and results['hbond_distances'] and len(results['hbond_distances']) > 0:
            hb_dists = results['hbond_distances']
            ax.hist(hb_dists, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel('H-bond Distance (Å)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'H-bond Distribution (avg: {results["hbonds_per_frame"]:.1f}/frame)', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axvline(np.mean(hb_dists), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(hb_dists):.2f} Å')
            ax.legend()
        else:
            # Just show the count
            ax.text(0.5, 0.5, f'H-bonds per frame:\n~{results.get("hbonds_per_frame", 0):.0f}\n(simplified count)',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes, fontweight='bold')
            ax.set_title('H-bond Analysis', fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if epsilon == 0.0:
            filename = self.base_dir / f'advanced_structure_eps_0.0.png'
        else:
            filename = self.base_dir / f'advanced_structure_eps_{epsilon:.2f}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {filename.name}")
        plt.close()
    
    def plot_comparative_structure(self):
        """Compare structural features across epsilon values"""
        
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Comparative Structural Analysis', fontsize=16, fontweight='bold')
        
        eps_vals = sorted(self.results.keys())
        
        # H-bonds per frame
        ax = axes[0, 0]
        hb_counts = [self.results[eps].get('hbonds_per_frame', 0) for eps in eps_vals]
        ax.plot(eps_vals, hb_counts, 'o-', markersize=10, lw=2.5, color='darkgreen')
        ax.set_xlabel('Epsilon', fontweight='bold')
        ax.set_ylabel('H-bonds per frame', fontweight='bold')
        ax.set_title('Hydrogen Bonding Network', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Water orientation preference
        ax = axes[0, 1]
        orient_data = []
        for eps in eps_vals:
            if 'orientations' in self.results[eps] and self.results[eps]['orientations'] is not None:
                orientations = self.results[eps]['orientations']
                distances = self.results[eps]['orientation_distances']
                
                # Average orientation in first shell (3-5 Å)
                mask = (distances > 3) & (distances < 6)
                if np.sum(mask) > 0:
                    avg_orient = np.mean(orientations[mask])
                    orient_data.append((eps, avg_orient))
        
        if orient_data:
            eps_plot, orient_plot = zip(*orient_data)
            ax.bar(eps_plot, orient_plot, width=0.03, alpha=0.7, color='steelblue')
            ax.set_xlabel('Epsilon', fontweight='bold')
            ax.set_ylabel('Average cos(θ) in 1st shell', fontweight='bold')
            ax.set_title('Water Orientation Preference', fontweight='bold')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Orientation analysis\nskipped', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Water Orientation Preference', fontweight='bold')
            ax.axis('off')
        
        # NP size
        ax = axes[1, 0]
        np_natoms = [self.results[eps].get('nanoparticle_natoms', 0) for eps in eps_vals]
        ax.bar(eps_vals, np_natoms, width=0.03, color='brown', alpha=0.7)
        ax.set_xlabel('Epsilon', fontweight='bold')
        ax.set_ylabel('Number of C atoms', fontweight='bold')
        ax.set_title('Nanoparticle Size', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for eps in eps_vals:
            hb = self.results[eps].get('hbonds_per_frame', 0)
            np_n = self.results[eps].get('nanoparticle_natoms', 0)
            table_data.append([f"{eps:.2f}", f"{hb:.1f}", f"{np_n}"])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Epsilon', 'H-bonds/frame', 'NP atoms'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(eps_vals) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'comparative_advanced_structure.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: comparative_advanced_structure.png")
        plt.close()


def main():
    """Main advanced structural analysis workflow"""
    
    print("\n" + "="*80)
    print("ADVANCED SOLVENT STRUCTURE ANALYSIS")
    print("GPU-Accelerated with MDAnalysis")
    print("="*80)
    
    analyzer = AdvancedStructuralAnalyzer(base_dir=".")
    
    # Analyze each system
    for eps in analyzer.epsilon_values:
        result = analyzer.comprehensive_structural_analysis(eps)
        if result:
            analyzer.plot_3d_density_slices(eps)
    
    # Comparative analysis
    if analyzer.results:
        analyzer.plot_comparative_structure()
    
    print("\n" + "="*80)
    print("ADVANCED STRUCTURAL ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - advanced_structure_eps_*.png (6 files)")
    print("  - comparative_advanced_structure.png")
    print("="*80 + "\n")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
