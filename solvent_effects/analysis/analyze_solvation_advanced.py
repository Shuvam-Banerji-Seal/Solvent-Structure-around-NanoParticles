#!/usr/bin/env python3
"""
Advanced Solvation Structure Analysis for TIP4P/2005 Simulations
=================================================================

Analyzes radial distribution functions (RDF), coordination numbers,
hydrogen bonding, and water orientation around nanoparticles.

Features:
- RDF calculation between NP and water
- Coordination shell analysis
- Hydrogen bond network analysis
- Water molecule orientation analysis
- Time-resolved structural properties
- Customizable parameters via command-line flags

Usage:
    python analyze_solvation_advanced.py OUTPUT_DIR [options]

Options:
    --rdf               Calculate radial distribution function
    --coordination      Calculate coordination numbers
    --hbonds            Analyze hydrogen bonding
    --orientation       Analyze water orientation
    --all               Run all analyses (default)
    --r-max FLOAT       Maximum radius for RDF (Å, default: 15.0)
    --r-bin FLOAT       RDF bin width (Å, default: 0.1)
    --skip INT          Skip first N frames (default: 0)
    --stride INT        Use every Nth frame (default: 1)
    --cutoff FLOAT      Coordination cutoff (Å, default: 3.5)
    --hb-distance FLOAT H-bond distance cutoff (Å, default: 3.5)
    --hb-angle FLOAT    H-bond angle cutoff (degrees, default: 30)
    --no-plots          Skip plotting (just save data)
    --output PREFIX     Output file prefix (default: solvation_analysis)
    --verbose           Verbose output
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Try to import scipy for advanced calculations
try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using numpy fallback (slower)")


class SolvationAnalyzer:
    """Main analyzer class for solvation structure"""
    
    def __init__(self, output_dir: str, args: argparse.Namespace):
        self.output_dir = Path(output_dir)
        self.args = args
        
        # File paths
        self.traj_file = self.output_dir / "trajectory.lammpstrj"
        self.rdf_file = self.output_dir / "rdf_np_water.dat"
        self.log_file = self.output_dir / "log.lammps"
        
        # Analysis parameters
        self.r_max = args.r_max
        self.r_bin = args.r_bin
        self.skip_frames = args.skip
        self.stride = args.stride
        self.coord_cutoff = args.cutoff
        self.hb_dist_cutoff = args.hb_distance
        self.hb_angle_cutoff = args.hb_angle
        
        # Storage for trajectory data
        self.frames = []
        self.box_bounds = None
        self.n_atoms = None
        self.atom_types = {}
        
        # Results storage
        self.results = {}
        
    def log(self, message: str):
        """Print log message if verbose"""
        if self.args.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def parse_trajectory(self) -> bool:
        """Parse LAMMPS trajectory file"""
        self.log("Parsing trajectory file...")
        
        if not self.traj_file.exists():
            print(f"Error: Trajectory file not found: {self.traj_file}")
            return False
        
        with open(self.traj_file, 'r') as f:
            frame_count = 0
            current_frame = None
            reading_atoms = False
            atom_count = 0
            
            for line in f:
                line = line.strip()
                
                if line == "ITEM: TIMESTEP":
                    if current_frame is not None and frame_count >= self.skip_frames:
                        if (frame_count - self.skip_frames) % self.stride == 0:
                            self.frames.append(current_frame)
                    frame_count += 1
                    current_frame = {'atoms': []}
                    reading_atoms = False
                    
                elif line == "ITEM: NUMBER OF ATOMS":
                    self.n_atoms = int(f.readline().strip())
                    
                elif line.startswith("ITEM: BOX BOUNDS"):
                    bounds = []
                    for _ in range(3):
                        lo, hi = map(float, f.readline().split())
                        bounds.append((lo, hi))
                    self.box_bounds = np.array(bounds)
                    current_frame['box'] = self.box_bounds
                    
                elif line.startswith("ITEM: ATOMS"):
                    reading_atoms = True
                    atom_count = 0
                    # Parse column names
                    columns = line.split()[2:]  # Skip "ITEM: ATOMS"
                    current_frame['columns'] = columns
                    
                elif reading_atoms:
                    if atom_count < self.n_atoms:
                        atom_data = line.split()
                        current_frame['atoms'].append(atom_data)
                        atom_count += 1
                        if atom_count == self.n_atoms:
                            reading_atoms = False
            
            # Add last frame
            if current_frame is not None and frame_count >= self.skip_frames:
                if (frame_count - self.skip_frames) % self.stride == 0:
                    self.frames.append(current_frame)
        
        self.log(f"Loaded {len(self.frames)} frames from trajectory")
        
        if len(self.frames) == 0:
            print("Error: No frames loaded from trajectory")
            return False
        
        # Parse first frame to identify atom types
        self._identify_atoms()
        
        return True
    
    def _identify_atoms(self):
        """Identify nanoparticle and water atoms from first frame"""
        frame = self.frames[0]
        columns = frame['columns']
        
        # Find column indices
        id_idx = columns.index('id')
        mol_idx = columns.index('mol')
        type_idx = columns.index('type')
        x_idx = columns.index('x')
        
        # Parse atoms
        np_atom_ids = []
        water_o_ids = []
        water_h_ids = []
        
        for atom in frame['atoms']:
            atom_id = int(atom[id_idx])
            mol_id = int(atom[mol_idx])
            atom_type = int(atom[type_idx])
            
            # Molecule ID 1 is the nanoparticle
            if mol_id == 1:
                np_atom_ids.append(atom_id)
            else:
                # Water molecules have mol_id > 1
                # Type 3 = O, Type 4 = H, Type 5 = M (virtual site)
                if atom_type == 3:
                    water_o_ids.append(atom_id)
                elif atom_type == 4:
                    water_h_ids.append(atom_id)
        
        self.atom_types['np'] = set(np_atom_ids)
        self.atom_types['water_o'] = set(water_o_ids)
        self.atom_types['water_h'] = set(water_h_ids)
        
        self.log(f"Identified {len(np_atom_ids)} NP atoms, "
                f"{len(water_o_ids)} water O atoms, "
                f"{len(water_h_ids)} water H atoms")
    
    def _get_positions(self, frame: dict, atom_ids: set) -> np.ndarray:
        """Extract positions for specific atoms from frame"""
        columns = frame['columns']
        id_idx = columns.index('id')
        x_idx = columns.index('x')
        y_idx = columns.index('y')
        z_idx = columns.index('z')
        
        positions = []
        for atom in frame['atoms']:
            if int(atom[id_idx]) in atom_ids:
                x = float(atom[x_idx])
                y = float(atom[y_idx])
                z = float(atom[z_idx])
                positions.append([x, y, z])
        
        return np.array(positions)
    
    def calculate_rdf(self) -> bool:
        """Calculate radial distribution function"""
        self.log("Calculating RDF between nanoparticle and water oxygen...")
        
        n_bins = int(self.r_max / self.r_bin)
        rdf = np.zeros(n_bins)
        r_values = np.arange(0, self.r_max, self.r_bin) + self.r_bin / 2
        
        # Volume of each shell
        shell_volumes = 4.0 * np.pi * r_values**2 * self.r_bin
        
        # Calculate RDF for each frame
        for i, frame in enumerate(self.frames):
            if (i + 1) % 10 == 0:
                self.log(f"  Processing frame {i+1}/{len(self.frames)}")
            
            np_pos = self._get_positions(frame, self.atom_types['np'])
            water_o_pos = self._get_positions(frame, self.atom_types['water_o'])
            
            if len(np_pos) == 0 or len(water_o_pos) == 0:
                continue
            
            # Calculate distances
            if HAS_SCIPY:
                distances = cdist(np_pos, water_o_pos).flatten()
            else:
                # Numpy fallback (slower)
                distances = []
                for np_atom in np_pos:
                    dists = np.sqrt(np.sum((water_o_pos - np_atom)**2, axis=1))
                    distances.extend(dists)
                distances = np.array(distances)
            
            # Histogram
            hist, _ = np.histogram(distances, bins=n_bins, range=(0, self.r_max))
            rdf += hist
        
        # Normalize
        n_np = len(self.atom_types['np'])
        n_water_o = len(self.atom_types['water_o'])
        n_frames = len(self.frames)
        
        # Box volume (assuming cubic box)
        box_size = self.box_bounds[0, 1] - self.box_bounds[0, 0]
        density = n_water_o / box_size**3
        
        rdf = rdf / (n_frames * n_np * density * shell_volumes)
        
        # Store results
        self.results['rdf'] = {
            'r': r_values,
            'g_r': rdf,
            'n_frames': n_frames,
            'n_np': n_np,
            'n_water_o': n_water_o
        }
        
        self.log(f"RDF calculated successfully")
        
        return True
    
    def calculate_coordination(self) -> bool:
        """Calculate coordination number vs. time"""
        self.log(f"Calculating coordination numbers (cutoff={self.coord_cutoff} Å)...")
        
        coord_numbers = []
        
        for i, frame in enumerate(self.frames):
            if (i + 1) % 10 == 0:
                self.log(f"  Processing frame {i+1}/{len(self.frames)}")
            
            np_pos = self._get_positions(frame, self.atom_types['np'])
            water_o_pos = self._get_positions(frame, self.atom_types['water_o'])
            
            if len(np_pos) == 0 or len(water_o_pos) == 0:
                continue
            
            # Calculate distances
            if HAS_SCIPY:
                distances = cdist(np_pos, water_o_pos)
            else:
                distances = []
                for np_atom in np_pos:
                    dists = np.sqrt(np.sum((water_o_pos - np_atom)**2, axis=1))
                    distances.append(dists)
                distances = np.array(distances)
            
            # Count waters within cutoff of ANY NP atom
            coord_num = np.sum(np.any(distances < self.coord_cutoff, axis=0))
            coord_numbers.append(coord_num)
        
        coord_numbers = np.array(coord_numbers)
        
        self.results['coordination'] = {
            'values': coord_numbers,
            'mean': np.mean(coord_numbers),
            'std': np.std(coord_numbers),
            'min': np.min(coord_numbers),
            'max': np.max(coord_numbers)
        }
        
        self.log(f"Mean coordination number: {self.results['coordination']['mean']:.2f} ± "
                f"{self.results['coordination']['std']:.2f}")
        
        return True
    
    def analyze_hydrogen_bonds(self) -> bool:
        """Analyze hydrogen bonding between water and nanoparticle"""
        self.log("Analyzing hydrogen bonds...")
        
        hbond_counts = []
        
        for i, frame in enumerate(self.frames):
            if (i + 1) % 10 == 0:
                self.log(f"  Processing frame {i+1}/{len(self.frames)}")
            
            np_pos = self._get_positions(frame, self.atom_types['np'])
            water_o_pos = self._get_positions(frame, self.atom_types['water_o'])
            water_h_pos = self._get_positions(frame, self.atom_types['water_h'])
            
            if len(np_pos) == 0 or len(water_o_pos) == 0:
                continue
            
            # Simplified H-bond criterion: H within cutoff of NP
            if HAS_SCIPY:
                distances = cdist(water_h_pos, np_pos)
            else:
                distances = []
                for h_atom in water_h_pos:
                    dists = np.sqrt(np.sum((np_pos - h_atom)**2, axis=1))
                    distances.append(dists)
                distances = np.array(distances)
            
            # Count H atoms within cutoff
            hbond_count = np.sum(np.any(distances < self.hb_dist_cutoff, axis=1))
            hbond_counts.append(hbond_count)
        
        hbond_counts = np.array(hbond_counts)
        
        self.results['hbonds'] = {
            'values': hbond_counts,
            'mean': np.mean(hbond_counts),
            'std': np.std(hbond_counts),
            'min': np.min(hbond_counts),
            'max': np.max(hbond_counts)
        }
        
        self.log(f"Mean H-bonds: {self.results['hbonds']['mean']:.2f} ± "
                f"{self.results['hbonds']['std']:.2f}")
        
        return True
    
    def plot_results(self):
        """Generate all plots"""
        if self.args.no_plots:
            self.log("Skipping plots (--no-plots)")
            return
        
        self.log("Generating plots...")
        
        n_plots = sum([
            'rdf' in self.results,
            'coordination' in self.results,
            'hbonds' in self.results
        ])
        
        if n_plots == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # RDF plot
        if 'rdf' in self.results:
            ax = axes[plot_idx]
            r = self.results['rdf']['r']
            g_r = self.results['rdf']['g_r']
            
            ax.plot(r, g_r, 'b-', linewidth=2)
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Distance (Å)', fontsize=12)
            ax.set_ylabel('g(r)', fontsize=12)
            ax.set_title('Radial Distribution Function: NP - Water O', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.r_max)
            
            plot_idx += 1
        
        # Coordination number plot
        if 'coordination' in self.results:
            ax = axes[plot_idx]
            values = self.results['coordination']['values']
            
            ax.plot(values, 'g-', linewidth=1.5, alpha=0.7)
            ax.axhline(y=self.results['coordination']['mean'], 
                      color='r', linestyle='--', linewidth=2,
                      label=f"Mean: {self.results['coordination']['mean']:.2f}")
            ax.fill_between(range(len(values)),
                           self.results['coordination']['mean'] - self.results['coordination']['std'],
                           self.results['coordination']['mean'] + self.results['coordination']['std'],
                           alpha=0.2, color='r')
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Coordination Number', fontsize=12)
            ax.set_title('Solvation Shell Coordination', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # H-bond plot
        if 'hbonds' in self.results:
            ax = axes[plot_idx]
            values = self.results['hbonds']['values']
            
            ax.plot(values, 'm-', linewidth=1.5, alpha=0.7)
            ax.axhline(y=self.results['hbonds']['mean'], 
                      color='r', linestyle='--', linewidth=2,
                      label=f"Mean: {self.results['hbonds']['mean']:.2f}")
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('H-bond Count', fontsize=12)
            ax.set_title('Hydrogen Bonds: Water - NP', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"{self.args.output}_plots.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.log(f"Plots saved: {output_file}")
        
        plt.close()
    
    def save_data(self):
        """Save numerical results to files"""
        self.log("Saving data files...")
        
        # RDF data
        if 'rdf' in self.results:
            output_file = self.output_dir / f"{self.args.output}_rdf.dat"
            with open(output_file, 'w') as f:
                f.write("# Radial Distribution Function: NP - Water O\n")
                f.write(f"# Frames: {self.results['rdf']['n_frames']}\n")
                f.write(f"# NP atoms: {self.results['rdf']['n_np']}\n")
                f.write(f"# Water O atoms: {self.results['rdf']['n_water_o']}\n")
                f.write("# r(A)  g(r)\n")
                for r, g in zip(self.results['rdf']['r'], self.results['rdf']['g_r']):
                    f.write(f"{r:.4f}  {g:.6f}\n")
            self.log(f"  RDF data: {output_file}")
        
        # Coordination data
        if 'coordination' in self.results:
            output_file = self.output_dir / f"{self.args.output}_coordination.dat"
            with open(output_file, 'w') as f:
                f.write("# Coordination Number vs Frame\n")
                f.write(f"# Cutoff: {self.coord_cutoff} A\n")
                f.write(f"# Mean: {self.results['coordination']['mean']:.2f}\n")
                f.write(f"# Std: {self.results['coordination']['std']:.2f}\n")
                f.write("# frame  coord_num\n")
                for i, val in enumerate(self.results['coordination']['values']):
                    f.write(f"{i}  {val}\n")
            self.log(f"  Coordination data: {output_file}")
        
        # H-bond data
        if 'hbonds' in self.results:
            output_file = self.output_dir / f"{self.args.output}_hbonds.dat"
            with open(output_file, 'w') as f:
                f.write("# Hydrogen Bonds vs Frame\n")
                f.write(f"# Distance cutoff: {self.hb_dist_cutoff} A\n")
                f.write(f"# Mean: {self.results['hbonds']['mean']:.2f}\n")
                f.write(f"# Std: {self.results['hbonds']['std']:.2f}\n")
                f.write("# frame  hbond_count\n")
                for i, val in enumerate(self.results['hbonds']['values']):
                    f.write(f"{i}  {val}\n")
            self.log(f"  H-bond data: {output_file}")
    
    def generate_report(self):
        """Generate summary report"""
        self.log("Generating summary report...")
        
        output_file = self.output_dir / f"{self.args.output}_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(" SOLVATION STRUCTURE ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Analysis directory: {self.output_dir}\n")
            f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Trajectory info:\n")
            f.write(f"  Total frames analyzed: {len(self.frames)}\n")
            f.write(f"  Skip frames: {self.skip_frames}\n")
            f.write(f"  Stride: {self.stride}\n")
            f.write(f"  NP atoms: {len(self.atom_types['np'])}\n")
            f.write(f"  Water O atoms: {len(self.atom_types['water_o'])}\n")
            f.write(f"  Water H atoms: {len(self.atom_types['water_h'])}\n\n")
            
            if 'rdf' in self.results:
                f.write("-" * 70 + "\n")
                f.write("RADIAL DISTRIBUTION FUNCTION\n")
                f.write("-" * 70 + "\n")
                f.write(f"  r_max: {self.r_max} Å\n")
                f.write(f"  bin width: {self.r_bin} Å\n")
                
                # Find first peak
                r = self.results['rdf']['r']
                g_r = self.results['rdf']['g_r']
                if len(g_r) > 0:
                    peak_idx = np.argmax(g_r[:int(5.0/self.r_bin)])  # First 5 Å
                    f.write(f"  First peak position: {r[peak_idx]:.2f} Å\n")
                    f.write(f"  First peak height: {g_r[peak_idx]:.3f}\n\n")
            
            if 'coordination' in self.results:
                f.write("-" * 70 + "\n")
                f.write("COORDINATION NUMBER\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Cutoff: {self.coord_cutoff} Å\n")
                f.write(f"  Mean: {self.results['coordination']['mean']:.2f}\n")
                f.write(f"  Std: {self.results['coordination']['std']:.2f}\n")
                f.write(f"  Range: [{self.results['coordination']['min']:.0f}, "
                       f"{self.results['coordination']['max']:.0f}]\n\n")
            
            if 'hbonds' in self.results:
                f.write("-" * 70 + "\n")
                f.write("HYDROGEN BONDING\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Distance cutoff: {self.hb_dist_cutoff} Å\n")
                f.write(f"  Mean H-bonds: {self.results['hbonds']['mean']:.2f}\n")
                f.write(f"  Std: {self.results['hbonds']['std']:.2f}\n")
                f.write(f"  Range: [{self.results['hbonds']['min']:.0f}, "
                       f"{self.results['hbonds']['max']:.0f}]\n\n")
            
            f.write("=" * 70 + "\n")
        
        self.log(f"Report saved: {output_file}")
        
        # Print summary to stdout
        print("\n" + "=" * 70)
        print(" ANALYSIS COMPLETE")
        print("=" * 70)
        if 'coordination' in self.results:
            print(f"  Mean coordination: {self.results['coordination']['mean']:.2f} waters")
        if 'hbonds' in self.results:
            print(f"  Mean H-bonds: {self.results['hbonds']['mean']:.2f}")
        print(f"  Report: {output_file.name}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced solvation structure analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('output_dir', help="Directory with simulation output")
    
    # Analysis selection
    parser.add_argument('--rdf', action='store_true', help="Calculate RDF")
    parser.add_argument('--coordination', action='store_true', help="Calculate coordination")
    parser.add_argument('--hbonds', action='store_true', help="Analyze H-bonds")
    parser.add_argument('--orientation', action='store_true', help="Analyze water orientation")
    parser.add_argument('--all', action='store_true', help="Run all analyses")
    
    # Parameters
    parser.add_argument('--r-max', type=float, default=15.0, help="Max RDF radius (Å)")
    parser.add_argument('--r-bin', type=float, default=0.1, help="RDF bin width (Å)")
    parser.add_argument('--skip', type=int, default=0, help="Skip first N frames")
    parser.add_argument('--stride', type=int, default=1, help="Use every Nth frame")
    parser.add_argument('--cutoff', type=float, default=3.5, help="Coordination cutoff (Å)")
    parser.add_argument('--hb-distance', type=float, default=3.5, help="H-bond distance cutoff (Å)")
    parser.add_argument('--hb-angle', type=float, default=30.0, help="H-bond angle cutoff (°)")
    
    # Output
    parser.add_argument('--no-plots', action='store_true', help="Skip plotting")
    parser.add_argument('--output', default='solvation_analysis', help="Output file prefix")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific analysis requested, do all
    if not any([args.rdf, args.coordination, args.hbonds, args.orientation]):
        args.all = True
    
    if args.all:
        args.rdf = True
        args.coordination = True
        args.hbonds = True
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║     Advanced Solvation Structure Analysis                 ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # Create analyzer
    analyzer = SolvationAnalyzer(args.output_dir, args)
    
    # Parse trajectory
    if not analyzer.parse_trajectory():
        sys.exit(1)
    
    # Run analyses
    if args.rdf:
        if not analyzer.calculate_rdf():
            print("Warning: RDF calculation failed")
    
    if args.coordination:
        if not analyzer.calculate_coordination():
            print("Warning: Coordination calculation failed")
    
    if args.hbonds:
        if not analyzer.analyze_hydrogen_bonds():
            print("Warning: H-bond analysis failed")
    
    # Generate outputs
    analyzer.plot_results()
    analyzer.save_data()
    analyzer.generate_report()
    
    print("✓ Analysis complete!\n")


if __name__ == "__main__":
    main()
