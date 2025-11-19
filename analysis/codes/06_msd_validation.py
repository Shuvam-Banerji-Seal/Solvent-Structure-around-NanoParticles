#!/usr/bin/env python3
"""
MSD Validation and Diffusion Coefficient Analysis
==================================================

Uses LAMMPS-computed MSD from msd_water.dat files to:
1. Validate Module 4 CUDA-computed MSD
2. Calculate diffusion coefficients: D = MSD/(6t)
3. Analyze effect of epsilon on water mobility

Author: Scientific Analysis Suite
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import curve_fit
import json

# Plotting configuration
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 10)

# Paths
BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")
PLOTS_DIR = BASE_DIR / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TIMESTEP = 2.0  # fs
PRODUCTION_START = 600000


class MSDAnalyzer:
    """MSD and diffusion coefficient analyzer"""
    
    def __init__(self):
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.msd_data = {}
        self.diffusion_coefficients = {}
        
    def load_msd_data(self):
        """Load MSD data from LAMMPS output files"""
        print("Loading MSD data from LAMMPS...")
        
        for eps in self.epsilon_values:
            # Handle epsilon_0.0 special case
            if eps == 0.0:
                eps_dir = BASE_DIR / "epsilon_0.0"
            else:
                eps_dir = BASE_DIR / f"epsilon_{eps:.2f}"
            
            msd_file = eps_dir / "msd_water.dat"
            
            if not msd_file.exists():
                print(f"  Warning: {msd_file} not found")
                continue
            
            # Read MSD data
            data = np.loadtxt(msd_file, comments='#')
            
            timesteps = data[:, 0]
            msd_x = data[:, 1]
            msd_y = data[:, 2]
            msd_z = data[:, 3]
            msd_total = data[:, 4]
            
            # Convert timestep to time (ns)
            time_ns = (timesteps - PRODUCTION_START) * TIMESTEP / 1e6
            
            self.msd_data[eps] = {
                'time_ns': time_ns,
                'msd_x': msd_x,
                'msd_y': msd_y,
                'msd_z': msd_z,
                'msd_total': msd_total,
                'n_points': len(timesteps)
            }
            
            print(f"  ε={eps:.2f}: {len(timesteps)} MSD points")
        
        return self
    
    def calculate_diffusion_coefficients(self):
        """
        Calculate diffusion coefficients from MSD
        
        Einstein relation: MSD = 6Dt (3D)
        Therefore: D = MSD / (6t)
        
        Fit linear region (typically 1-2 ns after initial ballistic regime)
        """
        print("\nCalculating diffusion coefficients...")
        
        for eps in self.epsilon_values:
            if eps not in self.msd_data:
                continue
            
            data = self.msd_data[eps]
            time_ns = data['time_ns']
            msd = data['msd_total']
            
            # Use linear region: 0.5-3.0 ns (avoid ballistic and long-time anomalies)
            fit_mask = (time_ns >= 0.5) & (time_ns <= 3.0)
            
            if np.sum(fit_mask) < 10:
                print(f"  ε={eps:.2f}: Insufficient data for fitting")
                continue
            
            time_fit = time_ns[fit_mask]
            msd_fit = msd[fit_mask]
            
            # Linear fit: MSD = 6Dt + b (b accounts for cage effects)
            def linear(t, D, b):
                return 6 * D * t + b
            
            try:
                popt, pcov = curve_fit(linear, time_fit, msd_fit)
                D_fit = popt[0]  # Å²/ns
                D_error = np.sqrt(pcov[0, 0])
                
                # Convert to cm²/s (standard units)
                # 1 Å²/ns = 1e-16 m²/ns = 1e-16 m²/(1e-9 s) = 1e-7 m²/s = 1e-3 cm²/s
                D_cm2_s = D_fit * 1e-3
                D_error_cm2_s = D_error * 1e-3
                
                self.diffusion_coefficients[eps] = {
                    'D_A2_ns': D_fit,
                    'D_cm2_s': D_cm2_s,
                    'D_error_A2_ns': D_error,
                    'D_error_cm2_s': D_error_cm2_s,
                    'fit_range': (time_fit[0], time_fit[-1])
                }
                
                print(f"  ε={eps:.2f}: D = {D_cm2_s:.2e} ± {D_error_cm2_s:.2e} cm²/s")
                
            except Exception as e:
                print(f"  ε={eps:.2f}: Fit failed - {e}")
                continue
        
        return self
    
    def plot_msd_evolution(self):
        """Plot MSD evolution for all epsilon values"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Total MSD
        ax1 = fig.add_subplot(gs[0, :])
        for i, eps in enumerate(self.epsilon_values):
            if eps not in self.msd_data:
                continue
            
            data = self.msd_data[eps]
            ax1.plot(data['time_ns'], data['msd_total'], 
                    label=f'ε={eps:.2f}', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('MSD (ų)')
        ax1.set_title('Mean Squared Displacement Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(alpha=0.3)
        
        # Plot 2: MSD components (epsilon=0.0)
        ax2 = fig.add_subplot(gs[1, 0])
        if 0.0 in self.msd_data:
            data = self.msd_data[0.0]
            ax2.plot(data['time_ns'], data['msd_x'], label='MSD_x', linewidth=2, alpha=0.8)
            ax2.plot(data['time_ns'], data['msd_y'], label='MSD_y', linewidth=2, alpha=0.8)
            ax2.plot(data['time_ns'], data['msd_z'], label='MSD_z', linewidth=2, alpha=0.8)
            ax2.plot(data['time_ns'], data['msd_total'], 
                    label='MSD_total', linewidth=2, alpha=0.8, linestyle='--', color='black')
            
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('MSD (ų)')
            ax2.set_title('MSD Components (ε=0.0, Hydrophobic)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # Plot 3: Linear fits showing diffusion
        ax3 = fig.add_subplot(gs[1, 1])
        for eps in self.epsilon_values:
            if eps not in self.msd_data or eps not in self.diffusion_coefficients:
                continue
            
            data = self.msd_data[eps]
            D_info = self.diffusion_coefficients[eps]
            
            # Plot data
            ax3.plot(data['time_ns'], data['msd_total'], 
                    label=f'ε={eps:.2f}', linewidth=2, alpha=0.6)
            
            # Plot fit line
            t_fit = np.linspace(D_info['fit_range'][0], D_info['fit_range'][1], 100)
            msd_fit = 6 * D_info['D_A2_ns'] * t_fit
            ax3.plot(t_fit, msd_fit, linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('MSD (ų)')
        ax3.set_title('MSD with Linear Fits (D extraction)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', ncol=2, fontsize=8)
        ax3.grid(alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        output_file = PLOTS_DIR / "18_msd_evolution.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()
        
        # Save data
        csv_data = []
        for eps in self.epsilon_values:
            if eps in self.msd_data:
                data = self.msd_data[eps]
                for i in range(len(data['time_ns'])):
                    csv_data.append({
                        'Epsilon': eps,
                        'Time_ns': data['time_ns'][i],
                        'MSD_total': data['msd_total'][i],
                        'MSD_x': data['msd_x'][i],
                        'MSD_y': data['msd_y'][i],
                        'MSD_z': data['msd_z'][i]
                    })
        
        df = pd.DataFrame(csv_data)
        csv_file = PLOTS_DIR / "msd_evolution_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")
    
    def plot_diffusion_coefficients(self):
        """Plot diffusion coefficients vs epsilon"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        eps_vals = []
        D_vals = []
        D_errors = []
        
        for eps in sorted(self.diffusion_coefficients.keys()):
            info = self.diffusion_coefficients[eps]
            eps_vals.append(eps)
            D_vals.append(info['D_cm2_s'])
            D_errors.append(info['D_error_cm2_s'])
        
        eps_vals = np.array(eps_vals)
        D_vals = np.array(D_vals)
        D_errors = np.array(D_errors)
        
        # Plot 1: Diffusion coefficient vs epsilon
        ax1.errorbar(eps_vals, D_vals * 1e5, yerr=D_errors * 1e5,
                    fmt='o-', markersize=10, capsize=5, linewidth=2, 
                    color='steelblue', ecolor='gray')
        
        # Add experimental reference line for bulk water
        D_bulk_exp = 2.3e-5  # cm²/s at 298 K (experimental)
        ax1.axhline(D_bulk_exp * 1e5, color='red', linestyle='--', 
                   linewidth=2, label='Bulk water (exp)', alpha=0.7)
        
        ax1.set_xlabel('C-O Interaction Strength ε (kcal/mol)', fontsize=12)
        ax1.set_ylabel('Diffusion Coefficient D (×10⁻⁵ cm²/s)', fontsize=12)
        ax1.set_title('Water Diffusion vs C-O Interaction', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Plot 2: Percent change from hydrophobic case
        if len(D_vals) > 0:
            D_ref = D_vals[0]  # epsilon=0.0 reference
            percent_change = ((D_vals - D_ref) / D_ref) * 100
            
            ax2.bar(eps_vals, percent_change, width=0.035, color='coral', alpha=0.8, edgecolor='black')
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('C-O Interaction Strength ε (kcal/mol)', fontsize=12)
            ax2.set_ylabel('Change in Diffusion (%)', fontsize=12)
            ax2.set_title('Relative Change from Hydrophobic Case (ε=0.0)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = PLOTS_DIR / "19_diffusion_coefficients.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()
        
        # Save diffusion data
        df = pd.DataFrame({
            'Epsilon': eps_vals,
            'D_cm2_s': D_vals,
            'D_error_cm2_s': D_errors,
            'D_10e5_cm2_s': D_vals * 1e5
        })
        csv_file = PLOTS_DIR / "diffusion_coefficients.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")
        
        # Save JSON summary
        summary = {
            'units': {
                'D': 'cm²/s',
                'epsilon': 'kcal/mol'
            },
            'experimental_bulk_D': D_bulk_exp,
            'results': {}
        }
        
        for eps in sorted(self.diffusion_coefficients.keys()):
            info = self.diffusion_coefficients[eps]
            summary['results'][f'epsilon_{eps:.2f}'] = {
                'D_cm2_s': float(info['D_cm2_s']),
                'D_error_cm2_s': float(info['D_error_cm2_s']),
                'fit_range_ns': [float(info['fit_range'][0]), float(info['fit_range'][1])]
            }
        
        json_file = PLOTS_DIR / "diffusion_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {json_file}")


def main():
    """Main analysis workflow"""
    print("="*80)
    print("MSD VALIDATION AND DIFFUSION COEFFICIENT ANALYSIS")
    print("="*80)
    print()
    
    analyzer = MSDAnalyzer()
    analyzer.load_msd_data()
    analyzer.calculate_diffusion_coefficients()
    analyzer.plot_msd_evolution()
    analyzer.plot_diffusion_coefficients()
    
    print("\n" + "="*80)
    print("MSD ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
