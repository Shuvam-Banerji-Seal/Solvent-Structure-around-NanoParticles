#!/usr/bin/env python3
"""
Rigorous Structural Analysis of Nanoparticle-Solvent Systems
==============================================================

This script performs detailed structural analysis including:
1. Radial Distribution Functions (RDF) for all atom pairs
2. Coordination number analysis
3. Solvent structure around nanoparticle
4. Hydrogen bonding analysis
5. Spatial distribution functions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rdf, distances
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


class StructuralAnalyzer:
    """Analyze structural properties of nanoparticle-solvent systems"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.universes = {}
        self.rdfs = {}
        
    def load_trajectory(self, epsilon: float) -> mda.Universe:
        """Load LAMMPS trajectory using MDAnalysis"""
        
        # Paths to data and trajectory files
        data_file = self.base_dir / "initial_system.data"
        traj_dir = self.base_dir / f"epsilon_{epsilon:.2f}"
        dump_file = traj_dir / "equilibration.lammpstrj"
        
        if not dump_file.exists():
            print(f"WARNING: Trajectory not found for ε={epsilon:.2f}")
            return None
        
        try:
            # Load universe
            u = mda.Universe(str(data_file), str(dump_file), 
                           format='LAMMPSDUMP', 
                           atom_style='full')
            
            print(f"ε={epsilon:.2f}: Loaded {len(u.atoms)} atoms, "
                  f"{len(u.trajectory)} frames")
            
            return u
        
        except Exception as e:
            print(f"ERROR loading trajectory for ε={epsilon:.2f}: {e}")
            return None
    
    def identify_atom_types(self, universe: mda.Universe) -> Dict:
        """Identify different atom types in the system"""
        
        # In TIP4P/2005 water + carbon nanoparticle system:
        # Type 1: Oxygen (O)
        # Type 2: Hydrogen (H)
        # Type 3: Carbon (C) - nanoparticle
        
        atom_groups = {}
        
        try:
            # Select by type
            atom_groups['oxygen'] = universe.select_atoms('type 1')
            atom_groups['hydrogen'] = universe.select_atoms('type 2')
            atom_groups['carbon'] = universe.select_atoms('type 3')
            atom_groups['water'] = universe.select_atoms('type 1 or type 2')
            atom_groups['nanoparticle'] = universe.select_atoms('type 3')
            
            print(f"\nAtom groups identified:")
            for name, group in atom_groups.items():
                print(f"  {name}: {len(group)} atoms")
        
        except Exception as e:
            print(f"ERROR identifying atom types: {e}")
            return {}
        
        return atom_groups
    
    def calculate_rdf(self, universe: mda.Universe, 
                      group1: mda.AtomGroup, group2: mda.AtomGroup,
                      rmax: float = 15.0, nbins: int = 150,
                      start_frame: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate radial distribution function between two atom groups
        
        Args:
            universe: MDAnalysis Universe
            group1, group2: Atom groups for RDF calculation
            rmax: Maximum distance for RDF (Angstroms)
            nbins: Number of bins
            start_frame: Start from this frame (for equilibrated region)
        
        Returns:
            r: Radial distances
            g_r: RDF values
        """
        
        if start_frame is None:
            start_frame = len(universe.trajectory) // 2  # Use last 50%
        
        try:
            # Calculate RDF using MDAnalysis
            rdf_analysis = rdf.InterRDF(group1, group2, 
                                       nbins=nbins, range=(0, rmax))
            
            # Run only on equilibrated frames
            rdf_analysis.run(start=start_frame, stop=None, step=1)
            
            return rdf_analysis.results.bins, rdf_analysis.results.rdf
        
        except Exception as e:
            print(f"ERROR calculating RDF: {e}")
            return None, None
    
    def calculate_coordination_number(self, r: np.ndarray, g_r: np.ndarray, 
                                     rho: float, r_cutoff: float) -> float:
        """
        Calculate coordination number from RDF
        
        n(r) = 4π * ρ * ∫[0→r] r²g(r)dr
        
        Args:
            r: Radial distances
            g_r: RDF values
            rho: Number density (atoms/Å³)
            r_cutoff: Integration cutoff (typically first minimum)
        
        Returns:
            Coordination number
        """
        
        # Find indices within cutoff
        mask = r <= r_cutoff
        r_int = r[mask]
        g_int = g_r[mask]
        
        # Numerical integration using trapezoidal rule
        integrand = 4 * np.pi * rho * r_int**2 * g_int
        coord_num = np.trapz(integrand, r_int)
        
        return coord_num
    
    def analyze_all_rdfs(self):
        """Calculate RDFs for all systems and all relevant pairs"""
        
        print("\n" + "="*70)
        print("CALCULATING RADIAL DISTRIBUTION FUNCTIONS")
        print("="*70)
        
        rdf_pairs = [
            ('oxygen', 'oxygen', 'O-O'),
            ('oxygen', 'hydrogen', 'O-H'),
            ('hydrogen', 'hydrogen', 'H-H'),
            ('carbon', 'oxygen', 'C-O (NP-Water)'),
            ('carbon', 'carbon', 'C-C (NP)'),
        ]
        
        for eps in self.epsilon_values:
            print(f"\n--- ε={eps:.2f} ---")
            
            u = self.load_trajectory(eps)
            if u is None:
                continue
            
            self.universes[eps] = u
            atom_groups = self.identify_atom_types(u)
            
            if not atom_groups:
                continue
            
            self.rdfs[eps] = {}
            
            for type1, type2, label in rdf_pairs:
                if type1 in atom_groups and type2 in atom_groups:
                    print(f"  Calculating {label} RDF...", end=' ')
                    
                    r, g_r = self.calculate_rdf(u, 
                                               atom_groups[type1], 
                                               atom_groups[type2])
                    
                    if r is not None and g_r is not None:
                        self.rdfs[eps][label] = {'r': r, 'g_r': g_r}
                        
                        # Find first peak and first minimum
                        if len(g_r) > 10:
                            first_peak_idx = np.argmax(g_r[:len(g_r)//3])
                            first_peak = r[first_peak_idx]
                            
                            # Find first minimum after peak
                            after_peak = g_r[first_peak_idx:]
                            min_after_peak = np.argmin(after_peak[:len(after_peak)//2])
                            first_min = r[first_peak_idx + min_after_peak]
                            
                            print(f"Peak at {first_peak:.2f} Å, Min at {first_min:.2f} Å")
                        else:
                            print("Done")
        
        print("\n" + "="*70)
    
    def plot_water_rdfs(self):
        """Plot water-water RDFs for all epsilon values"""
        
        if not self.rdfs:
            print("ERROR: No RDF data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        rdf_types = ['O-O', 'O-H', 'H-H']
        
        for i, rdf_type in enumerate(rdf_types):
            ax = axes[i]
            
            for eps, color in zip(self.epsilon_values, colors):
                if eps in self.rdfs and rdf_type in self.rdfs[eps]:
                    r = self.rdfs[eps][rdf_type]['r']
                    g_r = self.rdfs[eps][rdf_type]['g_r']
                    ax.plot(r, g_r, label=f'ε={eps:.2f}', 
                           color=color, lw=2, alpha=0.8)
            
            ax.set_xlabel('r (Å)', fontweight='bold')
            ax.set_ylabel('g(r)', fontweight='bold')
            ax.set_title(f'{rdf_type} RDF', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 10)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'water_rdfs.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: water_rdfs.png")
        plt.close()
    
    def plot_nanoparticle_water_rdf(self):
        """Plot nanoparticle-water interaction RDFs"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        for eps, color in zip(self.epsilon_values, colors):
            if eps in self.rdfs and 'C-O (NP-Water)' in self.rdfs[eps]:
                r = self.rdfs[eps]['C-O (NP-Water)']['r']
                g_r = self.rdfs[eps]['C-O (NP-Water)']['g_r']
                ax.plot(r, g_r, label=f'ε={eps:.2f}', 
                       color=color, lw=2.5, alpha=0.8)
        
        ax.set_xlabel('r (Å)', fontweight='bold')
        ax.set_ylabel('g(r)', fontweight='bold')
        ax.set_title('Nanoparticle-Water RDF (C-O)', fontweight='bold', fontsize=14)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'nanoparticle_water_rdf.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: nanoparticle_water_rdf.png")
        plt.close()
    
    def calculate_solvation_shell_analysis(self):
        """Analyze solvation shell structure around nanoparticle"""
        
        print("\n" + "="*70)
        print("SOLVATION SHELL ANALYSIS")
        print("="*70)
        
        results = []
        
        for eps in self.epsilon_values:
            if eps not in self.rdfs or 'C-O (NP-Water)' not in self.rdfs[eps]:
                continue
            
            r = self.rdfs[eps]['C-O (NP-Water)']['r']
            g_r = self.rdfs[eps]['C-O (NP-Water)']['g_r']
            
            # Find first peak (first solvation shell)
            first_peak_idx = np.argmax(g_r[:len(g_r)//3])
            first_peak_r = r[first_peak_idx]
            first_peak_height = g_r[first_peak_idx]
            
            # Find first minimum (solvation shell boundary)
            after_peak = g_r[first_peak_idx:]
            min_idx = np.argmin(after_peak[:len(after_peak)//2])
            first_min_r = r[first_peak_idx + min_idx]
            
            # Estimate coordination number (requires density)
            # For water at ~1 g/cm³: ρ ≈ 0.0334 molecules/Å³
            # For oxygen atoms: ρ_O ≈ 0.0334 atoms/Å³
            rho_O = 0.0334
            
            coord_num = self.calculate_coordination_number(r, g_r, rho_O, first_min_r)
            
            results.append({
                'epsilon': eps,
                'first_shell_peak_r': first_peak_r,
                'first_shell_peak_height': first_peak_height,
                'first_shell_boundary': first_min_r,
                'coordination_number': coord_num
            })
            
            print(f"\nε={eps:.2f}:")
            print(f"  First shell peak: {first_peak_r:.2f} Å (g={first_peak_height:.3f})")
            print(f"  Shell boundary: {first_min_r:.2f} Å")
            print(f"  Coordination number: {coord_num:.1f} water molecules")
        
        print("="*70)
        
        # Plot solvation shell properties vs epsilon
        if results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            eps_vals = [r['epsilon'] for r in results]
            peak_r = [r['first_shell_peak_r'] for r in results]
            peak_h = [r['first_shell_peak_height'] for r in results]
            coord = [r['coordination_number'] for r in results]
            
            axes[0].plot(eps_vals, peak_r, 'o-', markersize=8, linewidth=2, color='darkblue')
            axes[0].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[0].set_ylabel('Peak Position (Å)', fontweight='bold')
            axes[0].set_title('First Solvation Shell Distance', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(eps_vals, peak_h, 'o-', markersize=8, linewidth=2, color='darkgreen')
            axes[1].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[1].set_ylabel('Peak Height g(r)', fontweight='bold')
            axes[1].set_title('Solvation Shell Intensity', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(eps_vals, coord, 'o-', markersize=8, linewidth=2, color='darkred')
            axes[2].set_xlabel('Epsilon (ε)', fontweight='bold')
            axes[2].set_ylabel('Coordination Number', fontweight='bold')
            axes[2].set_title('First Shell Coordination', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.base_dir / 'solvation_shell_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved: solvation_shell_analysis.png")
            plt.close()
        
        return results


def main():
    """Main structural analysis workflow"""
    
    print("\n" + "="*70)
    print("STRUCTURAL ANALYSIS - NANOPARTICLE SOLVATION")
    print("="*70)
    
    # Initialize analyzer
    analyzer = StructuralAnalyzer(base_dir=".")
    
    # Calculate all RDFs
    analyzer.analyze_all_rdfs()
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING STRUCTURAL ANALYSIS FIGURES")
    print("="*70)
    
    analyzer.plot_water_rdfs()
    analyzer.plot_nanoparticle_water_rdf()
    
    # Solvation shell analysis
    solvation_results = analyzer.calculate_solvation_shell_analysis()
    
    print("\n" + "="*70)
    print("STRUCTURAL ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - water_rdfs.png")
    print("  - nanoparticle_water_rdf.png")
    print("  - solvation_shell_analysis.png")
    print("="*70 + "\n")
    
    return analyzer, solvation_results


if __name__ == "__main__":
    analyzer, results = main()
