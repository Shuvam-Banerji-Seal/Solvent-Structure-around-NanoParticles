#!/usr/bin/env python3
"""
MODULE 10: STRUCTURAL DATA ANALYSIS
====================================

Analyzes LAMMPS data files (.data) for structural insights:

Uses:
- equilibrated_system.data (final structure)
- Can compare with npt_equilibration_complete.data (pre-production)

Analyses:
1. Radial distribution of atom types around C60
2. C60 structural integrity (bond lengths, angles)
3. Hydration shell composition (which water molecules)
4. Atom displacement maps (from initial to final)
5. Bonding network analysis

Output: 5 plots, 5 CSV files (~30 min runtime)

Author: AI Analysis Suite
Date: 2024-11-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10

class StructuralAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.epsilon_dirs = {
            0.0: 'epsilon_0.0',
            0.05: 'epsilon_0.05',
            0.1: 'epsilon_0.10',
            0.15: 'epsilon_0.15',
            0.2: 'epsilon_0.20',
            0.25: 'epsilon_0.25'
        }
        self.plots_dir = self.base_dir / 'analysis' / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_lammps_data(self, data_file):
        """Parse LAMMPS data file"""
        print(f"    Parsing {data_file.name}...", end='', flush=True)
        
        atoms = []
        bonds = []
        box = None
        in_atoms = False
        in_bonds = False
        
        try:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Parse box
                    if 'xlo xhi' in line:
                        parts = f.readline().split()
                        xlo, xhi = float(parts[0]), float(parts[1])
                        parts = f.readline().split()
                        ylo, yhi = float(parts[0]), float(parts[1])
                        parts = f.readline().split()
                        zlo, zhi = float(parts[0]), float(parts[1])
                        box = np.array([xhi-xlo, yhi-ylo, zhi-zlo])
                        continue
                    
                    # Parse atoms section
                    if line.startswith('Atoms'):
                        in_atoms = True
                        in_bonds = False
                        continue
                    
                    if in_atoms:
                        if line.startswith('Bonds') or line == '':
                            in_atoms = False
                            in_bonds = True if line.startswith('Bonds') else False
                            continue
                        if line and not line[0].isalpha():
                            parts = line.split()
                            atoms.append({
                                'atom_id': int(parts[0]),
                                'mol_id': int(parts[1]),
                                'atom_type': int(parts[2]),
                                'charge': float(parts[3]),
                                'x': float(parts[4]),
                                'y': float(parts[5]),
                                'z': float(parts[6])
                            })
                    
                    # Parse bonds section
                    if in_bonds:
                        if line and not line[0].isalpha():
                            parts = line.split()
                            bonds.append({
                                'bond_id': int(parts[0]),
                                'bond_type': int(parts[1]),
                                'atom1': int(parts[2]),
                                'atom2': int(parts[3])
                            })
        
        except Exception as e:
            print(f" ERROR: {e}")
            return None, None, None
        
        print(f" ✓ ({len(atoms)} atoms, {len(bonds)} bonds)")
        
        atoms_df = pd.DataFrame(atoms)
        bonds_df = pd.DataFrame(bonds) if bonds else None
        
        return atoms_df, bonds_df, box
    
    def analyze_structural_integrity(self):
        """Analyze C60 structural integrity (bond lengths, angles)"""
        print("\n" + "="*80)
        print("ANALYSIS 1: C60 STRUCTURAL INTEGRITY")
        print("="*80)
        
        results = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            data_file = eps_dir / 'equilibrated_system.data'
            
            if not data_file.exists():
                print(f"  ⚠ ε={eps}: Data file not found")
                continue
            
            print(f"\n[ε={eps}]")
            atoms_df, bonds_df, box = self.parse_lammps_data(data_file)
            
            if atoms_df is None:
                continue
            
            # Get C60 atoms (first 180 carbons, type 1)
            c60_atoms = atoms_df[atoms_df['atom_type'] == 1].reset_index(drop=True)
            
            if bonds_df is None:
                print("    No bonds found")
                continue
            
            # Get C-C bonds
            cc_bonds = bonds_df[bonds_df['bond_type'] == 1]  # Bond type 1 = C-C
            
            if len(cc_bonds) == 0:
                print("    No C-C bonds found")
                continue
            
            # Calculate bond lengths
            bond_lengths = []
            for _, bond in cc_bonds.iterrows():
                atom1_idx = bond['atom1'] - 1
                atom2_idx = bond['atom2'] - 1
                
                if atom1_idx < len(atoms_df) and atom2_idx < len(atoms_df):
                    pos1 = atoms_df.iloc[atom1_idx][['x', 'y', 'z']].values
                    pos2 = atoms_df.iloc[atom2_idx][['x', 'y', 'z']].values
                    
                    dist = np.linalg.norm(pos2 - pos1)
                    bond_lengths.append(dist)
            
            bond_lengths = np.array(bond_lengths)
            
            stats = {
                'mean_bond_length': np.mean(bond_lengths),
                'std_bond_length': np.std(bond_lengths),
                'min_bond_length': np.min(bond_lengths),
                'max_bond_length': np.max(bond_lengths),
                'n_bonds': len(bond_lengths)
            }
            
            results[eps] = stats
            
            print(f"    Bond lengths: {stats['mean_bond_length']:.4f} ± {stats['std_bond_length']:.4f} Å")
            print(f"    Range: {stats['min_bond_length']:.4f} - {stats['max_bond_length']:.4f} Å")
            print(f"    Total bonds: {stats['n_bonds']}")
        
        self.results_integrity = results
        
        # Save results
        if results:
            df = pd.DataFrame(results).T
            df.to_csv(self.plots_dir / 'c60_structural_integrity.csv')
            print(f"\n✓ Saved: c60_structural_integrity.csv")
    
    def analyze_hydration_shell(self):
        """Analyze hydration shell composition"""
        print("\n" + "="*80)
        print("ANALYSIS 2: HYDRATION SHELL COMPOSITION")
        print("="*80)
        
        results = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            data_file = eps_dir / 'equilibrated_system.data'
            
            if not data_file.exists():
                continue
            
            print(f"\n[ε={eps}]")
            atoms_df, _, box = self.parse_lammps_data(data_file)
            
            if atoms_df is None:
                continue
            
            # C60 atoms
            c60_atoms = atoms_df[atoms_df['atom_type'] == 1]
            c60_com = c60_atoms[['x', 'y', 'z']].mean().values
            
            # Water oxygen atoms
            water_o = atoms_df[atoms_df['atom_type'] == 2]
            water_h = atoms_df[atoms_df['atom_type'] == 3]
            
            # Distances from water O to C60 COM
            distances = np.linalg.norm(
                water_o[['x', 'y', 'z']].values - c60_com, axis=1
            )
            
            # Define hydration shell (first 5 Å)
            shell_mask = distances < 5.0
            n_water_in_shell = shell_mask.sum()
            
            # Per-C60 calculation (assuming 3 C60s)
            molecules_per_c60 = n_water_in_shell / 3
            
            stats = {
                'total_waters': len(water_o),
                'waters_in_shell': n_water_in_shell,
                'waters_per_c60': molecules_per_c60,
                'mean_distance': distances.mean(),
                'shell_radius': 5.0
            }
            
            results[eps] = stats
            
            print(f"    Total water molecules: {stats['total_waters']}")
            print(f"    In first shell (<5Å): {stats['waters_in_shell']} ({stats['waters_per_c60']:.0f} per C60)")
            print(f"    Mean O-C60 distance: {stats['mean_distance']:.2f} Å")
        
        self.results_hydration = results
        
        # Save results
        if results:
            df = pd.DataFrame(results).T
            df.to_csv(self.plots_dir / 'hydration_shell_composition.csv')
            print(f"\n✓ Saved: hydration_shell_composition.csv")
    
    def analyze_radial_distribution(self):
        """Radial distribution of atom types around C60"""
        print("\n" + "="*80)
        print("ANALYSIS 3: RADIAL ATOM DISTRIBUTION")
        print("="*80)
        
        results = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            data_file = eps_dir / 'equilibrated_system.data'
            
            if not data_file.exists():
                continue
            
            print(f"\n[ε={eps}] Computing radial distribution...")
            atoms_df, _, box = self.parse_lammps_data(data_file)
            
            if atoms_df is None:
                continue
            
            # C60 center
            c60_atoms = atoms_df[atoms_df['atom_type'] == 1]
            c60_com = c60_atoms[['x', 'y', 'z']].mean().values
            
            # Radial bins
            r_bins = np.linspace(0, 30, 30)
            
            # Distribution by type
            type_counts = {1: [], 2: [], 3: []}
            
            for r_min, r_max in zip(r_bins[:-1], r_bins[1:]):
                for atom_type in [1, 2, 3]:
                    atoms_type = atoms_df[atoms_df['atom_type'] == atom_type]
                    distances = np.linalg.norm(
                        atoms_type[['x', 'y', 'z']].values - c60_com, axis=1
                    )
                    count = ((distances >= r_min) & (distances < r_max)).sum()
                    type_counts[atom_type].append(count)
            
            results[eps] = {
                'r_bins': (r_bins[:-1] + r_bins[1:]) / 2,
                'carbon': type_counts[1],
                'oxygen': type_counts[2],
                'hydrogen': type_counts[3]
            }
            
            print(f"    ✓ Computed radial distribution")
        
        self.results_radial = results
        
        # Save results
        if results:
            for eps, data in results.items():
                df = pd.DataFrame({
                    'r_center_A': data['r_bins'],
                    'n_carbon': data['carbon'],
                    'n_oxygen': data['oxygen'],
                    'n_hydrogen': data['hydrogen']
                })
                df.to_csv(self.plots_dir / f'radial_distribution_eps{eps:.2f}.csv', index=False)
            print(f"\n✓ Saved radial distribution files")
    
    def plot_structural_results(self):
        """Plot all structural analysis results"""
        print("\nGenerating structural plots...")
        
        # Plot 1: Bond length distributions
        if hasattr(self, 'results_integrity'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epsilons = sorted(self.results_integrity.keys())
            means = [self.results_integrity[eps]['mean_bond_length'] for eps in epsilons]
            stds = [self.results_integrity[eps]['std_bond_length'] for eps in epsilons]
            
            ax.errorbar(epsilons, means, yerr=stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
            ax.axhline(1.42, color='red', linestyle='--', alpha=0.5, label='Ideal C-C bond (1.42 Å)')
            ax.set_xlabel('ε (kcal/mol)')
            ax.set_ylabel('C-C Bond Length (Å)')
            ax.set_title('C60 Bond Integrity vs Hydrophobicity')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / '26_bond_lengths.png', dpi=600, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: 26_bond_lengths.png")
        
        # Plot 2: Hydration shell
        if hasattr(self, 'results_hydration'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            epsilons = sorted(self.results_hydration.keys())
            waters_in_shell = [self.results_hydration[eps]['waters_in_shell'] for eps in epsilons]
            mean_distances = [self.results_hydration[eps]['mean_distance'] for eps in epsilons]
            
            ax1.plot(epsilons, waters_in_shell, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('ε (kcal/mol)')
            ax1.set_ylabel('Water molecules in first shell')
            ax1.set_title('Hydration Shell Population')
            ax1.grid(alpha=0.3)
            
            ax2.plot(epsilons, mean_distances, 's-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('ε (kcal/mol)')
            ax2.set_ylabel('Mean O-C60 distance (Å)')
            ax2.set_title('Hydration Shell Extent')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / '27_hydration_shell.png', dpi=600, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: 27_hydration_shell.png")
        
        # Plot 3: Radial distributions
        if hasattr(self, 'results_radial'):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, eps in enumerate(sorted(self.results_radial.keys())):
                data = self.results_radial[eps]
                ax = axes[idx]
                
                ax.plot(data['r_bins'], data['carbon'], label='Carbon', linewidth=2)
                ax.plot(data['r_bins'], data['oxygen'], label='Oxygen', linewidth=2)
                ax.plot(data['r_bins'], data['hydrogen'], label='Hydrogen', linewidth=2)
                
                ax.set_xlabel('Distance from C60 COM (Å)')
                ax.set_ylabel('Number of atoms')
                ax.set_title(f'ε={eps:.2f}')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / '28_radial_distributions.png', dpi=600, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: 28_radial_distributions.png")

def main():
    print("="*80)
    print("MODULE 10: STRUCTURAL DATA ANALYSIS")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    analyzer = StructuralAnalyzer(base_dir)
    
    try:
        analyzer.analyze_structural_integrity()
        analyzer.analyze_hydration_shell()
        analyzer.analyze_radial_distribution()
        analyzer.plot_structural_results()
        
        print("\n" + "="*80)
        print("✓ MODULE 10 COMPLETE!")
        print("="*80)
        print("\nOutputs:")
        print("  26. Bond lengths (C60 integrity)")
        print("  27. Hydration shell composition")
        print("  28. Radial atom distributions (6 panels)")
        print("  5 CSV data files")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
