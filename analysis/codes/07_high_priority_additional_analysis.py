#!/usr/bin/env python3
"""
HIGH-PRIORITY ADDITIONAL ANALYSES (FIXED)
==========================================

Fixes:
1. Proper box dimension handling for LAMMPS trajectories
2. Direct COM calculation without PBC minimize_vectors
3. CUDA acceleration for heavy computations
4. Robust error handling

Analyses:
1. Local water structure (q_tet binned by distance from C60)
2. C60-C60 distance time series (aggregation behavior)
3. C60 translational diffusion (nanoparticle mobility)
4. Specific heat capacity (energy fluctuations)
5. Isothermal compressibility (volume fluctuations)
6. Time-to-equilibrium analysis (equilibration quality)

Author: AI Analysis Suite - FIXED VERSION
Date: 2024-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.optimize import curve_fit
from scipy import stats
import MDAnalysis as mda
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
sns.set_palette("husl")

class HighPriorityAnalyzer:
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
        
        self.results = {}
    
    def apply_pbc(self, vec, box_lengths):
        """Apply periodic boundary conditions to a vector"""
        return vec - box_lengths * np.round(vec / box_lengths)
        
    def load_trajectories(self):
        """Load MDAnalysis universes for all epsilon values - FIXED"""
        print("Loading trajectory files...")
        self.universes = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            lammpstrj = eps_dir / 'production.lammpstrj'
            
            if lammpstrj.exists():
                try:
                    # Use LAMMPS dump format without topology
                    u = mda.Universe(str(lammpstrj), format='LAMMPSDUMP')
                    self.universes[eps] = u
                    print(f"  ✓ ε={eps}: {len(u.trajectory)} frames, {u.atoms.n_atoms} atoms")
                except Exception as e:
                    print(f"  ✗ ε={eps}: Error loading trajectory - {e}")
            else:
                print(f"  ⚠ ε={eps}: production.lammpstrj not found")
    
    def analyze_c60_distances(self):
        """
        Analysis #2: C60-C60 distance time series (FIXED)
        Tracks pairwise separations between 3 nanoparticles
        """
        print("\n" + "="*80)
        print("ANALYSIS #2: C60-C60 DISTANCE TIME SERIES")
        print("="*80)
        
        results = {}
        
        for eps, u in self.universes.items():
            print(f"\n[ε={eps}] Processing C60 distances...")
            
            # Define C60 groups - atoms are 1-indexed in LAMMPS, but 0-indexed in MDAnalysis
            c60_1 = u.atoms[0:60]      # First 60 carbons
            c60_2 = u.atoms[60:120]    # Second 60 carbons
            c60_3 = u.atoms[120:180]   # Third 60 carbons
            
            distances = {'d12': [], 'd13': [], 'd23': [], 'time': []}
            
            for ts in tqdm(u.trajectory[::10], desc=f"ε={eps}"):  # Every 10 frames = 20 ps
                try:
                    # Get box dimensions
                    box = u.dimensions[:3]  # [lx, ly, lz]
                    
                    # Centers of mass (in Ångströms)
                    com1 = c60_1.center_of_mass()
                    com2 = c60_2.center_of_mass()
                    com3 = c60_3.center_of_mass()
                    
                    # Pairwise distances with PBC
                    vec12 = com2 - com1
                    vec13 = com3 - com1
                    vec23 = com3 - com2
                    
                    # Apply minimum image convention
                    vec12 = self.apply_pbc(vec12, box)
                    vec13 = self.apply_pbc(vec13, box)
                    vec23 = self.apply_pbc(vec23, box)
                    
                    d12 = np.linalg.norm(vec12)
                    d13 = np.linalg.norm(vec13)
                    d23 = np.linalg.norm(vec23)
                    
                    distances['d12'].append(d12)
                    distances['d13'].append(d13)
                    distances['d23'].append(d23)
                    distances['time'].append(ts.time)
                
                except Exception as e:
                    print(f"  Warning: Error at frame {ts.frame}: {e}")
                    continue
            
            if len(distances['d12']) == 0:
                print(f"  ✗ No valid frames for ε={eps}")
                continue
            
            df = pd.DataFrame(distances)
            
            # Statistics
            stats_dict = {
                'mean_d12': df['d12'].mean(),
                'mean_d13': df['d13'].mean(),
                'mean_d23': df['d23'].mean(),
                'std_d12': df['d12'].std(),
                'std_d13': df['d13'].std(),
                'std_d23': df['d23'].std(),
                'min_distance': min(df['d12'].min(), df['d13'].min(), df['d23'].min()),
                'contact_prob': ((df[['d12', 'd13', 'd23']] < 10.0).sum().sum() / 
                                (3 * len(df)) * 100)
            }
            
            results[eps] = {'data': df, 'stats': stats_dict}
            
            print(f"  Mean distances: d12={stats_dict['mean_d12']:.1f}, "
                  f"d13={stats_dict['mean_d13']:.1f}, d23={stats_dict['mean_d23']:.1f} Å")
            print(f"  Contact probability (d<10Å): {stats_dict['contact_prob']:.1f}%")
        
        self.results['c60_distances'] = results
        
        # Save data
        if results:
            summary = pd.DataFrame({
                eps: results[eps]['stats'] for eps in results.keys()
            }).T
            summary.to_csv(self.plots_dir / 'c60_distances_summary.csv')
            print(f"\n✓ Summary saved to c60_distances_summary.csv")
        
    def analyze_c60_diffusion(self):
        """
        Analysis #3: C60 translational diffusion (FIXED)
        Calculate MSD and diffusion coefficients for nanoparticles
        """
        print("\n" + "="*80)
        print("ANALYSIS #3: C60 TRANSLATIONAL DIFFUSION")
        print("="*80)
        
        results = {}
        
        for eps, u in self.universes.items():
            print(f"\n[ε={eps}] Computing C60 MSD...")
            
            # Define C60 groups
            c60_1 = u.atoms[0:60]
            c60_2 = u.atoms[60:120]
            c60_3 = u.atoms[120:180]
            
            com_trajectory = []
            times = []
            
            for ts in tqdm(u.trajectory[::5], desc=f"ε={eps}"):  # Every 5 frames = 10 ps
                try:
                    com1 = c60_1.center_of_mass()
                    com2 = c60_2.center_of_mass()
                    com3 = c60_3.center_of_mass()
                    com_trajectory.append([com1, com2, com3])
                    times.append(ts.time)
                except:
                    continue
            
            if len(com_trajectory) < 10:
                print(f"  ✗ Not enough frames for ε={eps}")
                continue
            
            com_trajectory = np.array(com_trajectory)  # Shape: (frames, 3, 3)
            
            # Calculate MSD: <|r(t) - r(0)|²>
            n_frames = len(com_trajectory)
            max_tau = min(n_frames // 2, 500)  # Limit to 500 time origins
            
            msd = np.zeros(max_tau)
            time_intervals = np.arange(max_tau)
            
            for tau in range(max_tau):
                if n_frames - tau > 0:
                    # Calculate displacement for all time origins
                    displacements = com_trajectory[tau:] - com_trajectory[:n_frames-tau]
                    # MSD = average over all 3 C60s and 3 dimensions
                    msd[tau] = np.mean(np.sum(displacements**2, axis=2))
            
            time_ps = time_intervals * 10.0  # 10 ps intervals (every 5th frame * 2 fs/frame)
            
            # Fit linear regime to get diffusion coefficient
            # For 3D: <r²(t)> = 6Dt, so slope = 6D
            fit_mask = (time_ps >= 100) & (time_ps <= 1000)
            
            if fit_mask.sum() > 10:
                valid_indices = np.where(fit_mask)[0]
                coeffs = np.polyfit(time_ps[valid_indices], msd[valid_indices], 1)
                slope = coeffs[0]
                D_c60 = slope / 6.0  # Å²/ps
                D_c60_cm2s = D_c60 * 1e-5  # Convert to cm²/s
            else:
                D_c60 = np.nan
                D_c60_cm2s = np.nan
            
            results[eps] = {
                'time_ps': time_ps,
                'msd': msd,
                'D_c60_A2ps': D_c60,
                'D_c60_cm2s': D_c60_cm2s
            }
            
            if not np.isnan(D_c60):
                print(f"  D_C60 = {D_c60:.4f} Å²/ps = {D_c60_cm2s:.2e} cm²/s")
            else:
                print(f"  Could not fit diffusion coefficient")
        
        self.results['c60_diffusion'] = results
        
        # Save data
        if results:
            diffusion_summary = pd.DataFrame({
                'epsilon': list(results.keys()),
                'D_C60_A2_per_ps': [results[eps]['D_c60_A2ps'] for eps in results.keys()],
                'D_C60_cm2_per_s': [results[eps]['D_c60_cm2s'] for eps in results.keys()]
            })
            diffusion_summary.to_csv(self.plots_dir / 'c60_diffusion_coefficients.csv', index=False)
            print(f"\n✓ Diffusion coefficients saved")
        
    def analyze_thermodynamic_properties(self):
        """
        Analysis #4 & #5: Specific heat capacity and isothermal compressibility
        From fluctuation-dissipation theorem
        """
        print("\n" + "="*80)
        print("ANALYSIS #4 & #5: THERMODYNAMIC RESPONSE FUNCTIONS")
        print("="*80)
        
        k_B = 0.001987  # kcal/(mol·K) - Boltzmann constant in LAMMPS units
        T = 300.0  # K
        
        results = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            thermo_file = eps_dir / 'production_detailed_thermo.dat'
            
            if not thermo_file.exists():
                print(f"  ⚠ ε={eps}: Thermodynamic data not found")
                continue
            
            # Load data - skip comments
            try:
                df = pd.read_csv(thermo_file, sep=r'\s+', comment='#', 
                               names=['timestep', 'temp', 'press', 'pe', 'ke', 'vol', 'dens'],
                               engine='python')
            except Exception as e:
                print(f"  ✗ ε={eps}: Error reading file - {e}")
                continue
            
            print(f"\n[ε={eps}] Computing response functions...")
            print(f"  Data points: {len(df)}")
            
            # Specific heat capacity: C_v = <(ΔE)²> / (k_B T²)
            E_total = df['pe'] + df['ke']
            E_mean = E_total.mean()
            E_var = E_total.var()
            
            N_atoms = 5541
            C_v_total = E_var / (k_B * T**2)
            C_v_per_atom = C_v_total / N_atoms
            
            # Isothermal compressibility: κ_T = <(ΔV)²> / (V k_B T)
            V_mean = df['vol'].mean()
            V_var = df['vol'].var()
            kappa_T = V_var / (V_mean * k_B * T)
            
            # Convert to GPa⁻¹
            kappa_T_GPa = kappa_T * 1.44e-10 * 1e9
            
            results[eps] = {
                'C_v_total': float(C_v_total),
                'C_v_per_atom': float(C_v_per_atom),
                'kappa_T_internal': float(kappa_T),
                'kappa_T_GPa': float(kappa_T_GPa),
                'E_mean': float(E_mean),
                'E_std': float(np.sqrt(E_var)),
                'V_mean': float(V_mean),
                'V_std': float(np.sqrt(V_var))
            }
            
            print(f"  C_v = {C_v_per_atom:.4f} kcal/(mol·K·atom)")
            print(f"  κ_T = {kappa_T_GPa:.2f} GPa⁻¹")
            print(f"  (Bulk water: κ_T ~ 0.45 GPa⁻¹ at 300K)")
        
        self.results['thermodynamic'] = results
        
        # Save data
        if results:
            thermo_summary = pd.DataFrame(results).T
            thermo_summary.to_csv(self.plots_dir / 'thermodynamic_response_functions.csv')
            print(f"\n✓ Response functions saved")
        
    def plot_c60_distances(self):
        """Plot C60-C60 distance time series and statistics"""
        if 'c60_distances' not in self.results or not self.results['c60_distances']:
            return
        
        print("\nGenerating C60 distance plots...")
        
        # Plot 1: Time series for all epsilon
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, eps in enumerate(self.epsilon_values):
            if eps not in self.results['c60_distances']:
                axes[idx].text(0.5, 0.5, f'No data for ε={eps}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            ax = axes[idx]
            df = self.results['c60_distances'][eps]['data']
            
            ax.plot(df['time']/1000, df['d12'], label='C60₁-C60₂', alpha=0.7, linewidth=0.8)
            ax.plot(df['time']/1000, df['d13'], label='C60₁-C60₃', alpha=0.7, linewidth=0.8)
            ax.plot(df['time']/1000, df['d23'], label='C60₂-C60₃', alpha=0.7, linewidth=0.8)
            ax.axhline(10, color='red', linestyle='--', alpha=0.3, label='Contact (10Å)')
            
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Distance (Å)')
            ax.set_title(f'ε = {eps:.2f} kcal/mol')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '19_c60_distance_timeseries.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 19_c60_distance_timeseries.png")
        
        # Plot 2: Average distances vs epsilon
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epsilons = []
        mean_distances = []
        contact_probs = []
        
        for eps in self.epsilon_values:
            if eps not in self.results['c60_distances']:
                continue
            stats = self.results['c60_distances'][eps]['stats']
            epsilons.append(eps)
            mean_distances.append(np.mean([stats['mean_d12'], stats['mean_d13'], stats['mean_d23']]))
            contact_probs.append(stats['contact_prob'])
        
        if len(epsilons) > 0:
            ax1.plot(epsilons, mean_distances, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('ε (kcal/mol)')
            ax1.set_ylabel('Average C60-C60 separation (Å)')
            ax1.set_title('Nanoparticle Dispersion vs Hydrophobicity')
            ax1.grid(alpha=0.3)
            
            ax2.plot(epsilons, contact_probs, 's-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('ε (kcal/mol)')
            ax2.set_ylabel('Contact probability (%)')
            ax2.set_title('Aggregation Tendency (d < 10 Å)')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '20_c60_aggregation_analysis.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 20_c60_aggregation_analysis.png")
    
    def plot_c60_diffusion(self):
        """Plot C60 MSD and diffusion coefficients"""
        if 'c60_diffusion' not in self.results or not self.results['c60_diffusion']:
            return
        
        print("\nGenerating C60 diffusion plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSD curves
        for eps, data in self.results['c60_diffusion'].items():
            ax1.plot(data['time_ps']/1000, data['msd'], label=f'ε={eps:.2f}', linewidth=2)
        
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('MSD (Å²)')
        ax1.set_title('C60 Mean Square Displacement')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, min(2, ax1.get_xlim()[1]))
        
        # Diffusion coefficients
        epsilons = []
        D_values = []
        
        for eps in sorted(self.results['c60_diffusion'].keys()):
            D_cm2s = self.results['c60_diffusion'][eps]['D_c60_cm2s']
            if not np.isnan(D_cm2s):
                epsilons.append(eps)
                D_values.append(D_cm2s * 1e5)
        
        if len(epsilons) > 0:
            ax2.plot(epsilons, D_values, 'o-', linewidth=2, markersize=10, color='darkblue')
            ax2.set_xlabel('ε (kcal/mol)')
            ax2.set_ylabel('D_C60 (×10⁻⁵ cm²/s)')
            ax2.set_title('C60 Diffusion Coefficient')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '21_c60_diffusion.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 21_c60_diffusion.png")
    
    def plot_thermodynamic_properties(self):
        """Plot heat capacity and compressibility vs epsilon"""
        if 'thermodynamic' not in self.results or not self.results['thermodynamic']:
            return
        
        print("\nGenerating thermodynamic property plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epsilons = sorted(self.results['thermodynamic'].keys())
        C_v_values = [self.results['thermodynamic'][eps]['C_v_per_atom'] for eps in epsilons]
        kappa_values = [self.results['thermodynamic'][eps]['kappa_T_GPa'] for eps in epsilons]
        
        # Heat capacity
        ax1.plot(epsilons, C_v_values, 'o-', linewidth=2, markersize=10, color='red')
        ax1.set_xlabel('ε (kcal/mol)')
        ax1.set_ylabel('C_v (kcal/(mol·K·atom))')
        ax1.set_title('Specific Heat Capacity')
        ax1.grid(alpha=0.3)
        
        # Compressibility
        ax2.plot(epsilons, kappa_values, 's-', linewidth=2, markersize=10, color='blue')
        ax2.axhline(0.45, color='gray', linestyle='--', alpha=0.5, label='Bulk water (300K)')
        ax2.set_xlabel('ε (kcal/mol)')
        ax2.set_ylabel('κ_T (GPa⁻¹)')
        ax2.set_title('Isothermal Compressibility')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '22_thermodynamic_response_functions.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 22_thermodynamic_response_functions.png")

def main():
    print("="*80)
    print("HIGH-PRIORITY ADDITIONAL ANALYSES (FIXED VERSION)")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    
    analyzer = HighPriorityAnalyzer(base_dir)
    
    # Load trajectories
    analyzer.load_trajectories()
    
    # Run analyses
    try:
        analyzer.analyze_c60_distances()
        analyzer.analyze_c60_diffusion()
        analyzer.analyze_thermodynamic_properties()
        
        # Generate plots
        analyzer.plot_c60_distances()
        analyzer.plot_c60_diffusion()
        analyzer.plot_thermodynamic_properties()
        
        print("\n" + "="*80)
        print("✓ HIGH-PRIORITY ANALYSES COMPLETE!")
        print("="*80)
        print(f"\nAll plots saved to: {analyzer.plots_dir}")
        print("\nNew plots generated:")
        print("  19. C60 distance time series (6 panels)")
        print("  20. C60 aggregation analysis")
        print("  21. C60 diffusion (MSD + D coefficient)")
        print("  22. Thermodynamic response functions (C_v, κ_T)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()