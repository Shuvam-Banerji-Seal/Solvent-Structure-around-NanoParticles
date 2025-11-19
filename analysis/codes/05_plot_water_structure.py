#!/usr/bin/env python3
"""
Visualization Script for Comprehensive Water Structure Analysis
================================================================

Creates publication-quality plots (600 DPI) for all water structure metrics:
- Tetrahedral order parameter evolution
- Steinhardt order parameters (Q4, Q6)
- Asphericity and acylindricity (shape parameters)
- Coordination numbers
- Hydrogen bond analysis
- Radial density profiles
- Mean squared displacement and diffusion

All plots saved with corresponding CSV/JSON data.

Author: Scientific Analysis Suite  
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
from pathlib import Path
from scipy.optimize import curve_fit

# Plotting configuration - 600 DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)

# Paths
BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")
DATA_DIR = BASE_DIR / "analysis" / "plots"
PLOTS_DIR = DATA_DIR

# Color palette
COLORS = sns.color_palette("husl", 6)


class WaterStructurePlotter:
    """Comprehensive plotting for water structure analysis"""
    
    def __init__(self):
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load analysis results for all epsilon values"""
        print("Loading water structure data...")
        
        for eps in self.epsilon_values:
            json_file = DATA_DIR / f"water_structure_epsilon_{eps:.2f}.json"
            csv_file = DATA_DIR / f"water_structure_epsilon_{eps:.2f}.csv"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    self.data[eps] = json.load(f)
                print(f"  Loaded ε={eps:.2f}")
            elif csv_file.exists():
                df = pd.read_csv(csv_file)
                self.data[eps] = df.to_dict(orient='list')
                print(f"  Loaded ε={eps:.2f} (CSV)")
            else:
                print(f"  Warning: No data for ε={eps:.2f}")
    
    def plot_tetrahedral_order(self):
        """Plot tetrahedral order parameter evolution and distribution"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Evolution over time
        ax1 = fig.add_subplot(gs[0, :])
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            q_vals = self.data[eps]['tetrahedral_order']
            ax1.plot(times, q_vals, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Time (ns)', fontsize=11)
        ax1.set_ylabel('Tetrahedral Order Parameter q', fontsize=11)
        ax1.set_title('Tetrahedral Order Evolution (q=1: perfect tetrahedron, q=0: random)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # Mean values
        ax2 = fig.add_subplot(gs[1, 0])
        valid_eps = [eps for eps in self.epsilon_values if eps in self.data]
        means = [np.mean(self.data[eps]['tetrahedral_order']) for eps in valid_eps]
        stds = [np.std(self.data[eps]['tetrahedral_order']) for eps in valid_eps]
        
        x_pos = np.arange(len(valid_eps))
        ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=COLORS[:len(valid_eps)])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax2.set_xlabel('Epsilon (kcal/mol)', fontsize=11)
        ax2.set_ylabel('Mean Tetrahedral Order', fontsize=11)
        ax2.set_title('Average Tetrahedral Structure', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            q_vals = self.data[eps]['tetrahedral_order']
            ax3.hist(q_vals, bins=30, alpha=0.5, label=f'ε={eps:.2f}', density=True)
        
        ax3.set_xlabel('Tetrahedral Order q', fontsize=11)
        ax3.set_ylabel('Probability Density', fontsize=11)
        ax3.set_title('Distribution of Tetrahedral Order', fontsize=11, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.savefig(PLOTS_DIR / '13_tetrahedral_order_analysis.png', dpi=600, bbox_inches='tight')
        print("  Saved: 13_tetrahedral_order_analysis.png")
        
        # Save data
        summary = pd.DataFrame({
            'Epsilon': valid_eps,
            'Mean_q': means,
            'Std_q': stds
        })
        summary.to_csv(PLOTS_DIR / 'tetrahedral_order_summary.csv', index=False)
        plt.close()
    
    def plot_steinhardt_order(self):
        """Plot Steinhardt Q4 and Q6 order parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Steinhardt Bond-Orientational Order Parameters', fontsize=14, fontweight='bold')
        
        # Q4 evolution
        ax = axes[0, 0]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            q4 = self.data[eps]['steinhardt_q4']
            ax.plot(times, q4, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Q4')
        ax.set_title('Q4 Evolution (Ice-like Structure Indicator)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Q6 evolution
        ax = axes[0, 1]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            q6 = self.data[eps]['steinhardt_q6']
            ax.plot(times, q6, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Q6')
        ax.set_title('Q6 Evolution (Long-range Order)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Mean Q4
        ax = axes[1, 0]
        valid_eps = [eps for eps in self.epsilon_values if eps in self.data]
        means_q4 = [np.mean(self.data[eps]['steinhardt_q4']) for eps in valid_eps]
        stds_q4 = [np.std(self.data[eps]['steinhardt_q4']) for eps in valid_eps]
        
        x_pos = np.arange(len(valid_eps))
        ax.errorbar(x_pos, means_q4, yerr=stds_q4, fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax.set_xlabel('Epsilon (kcal/mol)')
        ax.set_ylabel('Mean Q4')
        ax.set_title('Q4 vs C-O Interaction Strength')
        ax.grid(True, alpha=0.3)
        
        # Mean Q6
        ax = axes[1, 1]
        means_q6 = [np.mean(self.data[eps]['steinhardt_q6']) for eps in valid_eps]
        stds_q6 = [np.std(self.data[eps]['steinhardt_q6']) for eps in valid_eps]
        
        ax.errorbar(x_pos, means_q6, yerr=stds_q6, fmt='o-', capsize=5, linewidth=2, markersize=8, color='C1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax.set_xlabel('Epsilon (kcal/mol)')
        ax.set_ylabel('Mean Q6')
        ax.set_title('Q6 vs C-O Interaction Strength')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(PLOTS_DIR / '14_steinhardt_order_parameters.png', dpi=600, bbox_inches='tight')
        print("  Saved: 14_steinhardt_order_parameters.png")
        
        # Save data
        summary = pd.DataFrame({
            'Epsilon': valid_eps,
            'Mean_Q4': means_q4,
            'Std_Q4': stds_q4,
            'Mean_Q6': means_q6,
            'Std_Q6': stds_q6
        })
        summary.to_csv(PLOTS_DIR / 'steinhardt_order_summary.csv', index=False)
        plt.close()
    
    def plot_shape_parameters(self):
        """Plot asphericity and acylindricity (oblate/prolate parameters)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Water Cluster Shape Parameters: Asphericity (Oblate) & Acylindricity (Prolate)', 
                     fontsize=14, fontweight='bold')
        
        # Asphericity evolution
        ax = axes[0, 0]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            asp = self.data[eps]['asphericity']
            ax.plot(times, asp, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Asphericity b (Oblate Parameter)')
        ax.set_title('Disk-like Structure Evolution (b → 1: more oblate)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Acylindricity evolution
        ax = axes[0, 1]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            acy = self.data[eps]['acylindricity']
            ax.plot(times, acy, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Acylindricity c (Prolate Parameter)')
        ax.set_title('Rod-like Structure Evolution (c → 1: more prolate)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Mean values comparison
        ax = axes[1, 0]
        valid_eps = [eps for eps in self.epsilon_values if eps in self.data]
        asp_means = [np.mean(self.data[eps]['asphericity']) for eps in valid_eps]
        acy_means = [np.mean(self.data[eps]['acylindricity']) for eps in valid_eps]
        
        x_pos = np.arange(len(valid_eps))
        width = 0.35
        ax.bar(x_pos - width/2, asp_means, width, label='Asphericity (Oblate)', alpha=0.8)
        ax.bar(x_pos + width/2, acy_means, width, label='Acylindricity (Prolate)', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax.set_xlabel('Epsilon (kcal/mol)')
        ax.set_ylabel('Shape Parameter')
        ax.set_title('Mean Shape Parameters vs Epsilon')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Scatter plot: asphericity vs acylindricity
        ax = axes[1, 1]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            asp = self.data[eps]['asphericity']
            acy = self.data[eps]['acylindricity']
            ax.scatter(asp, acy, label=f'ε={eps:.2f}', alpha=0.6, s=20)
        ax.set_xlabel('Asphericity (Oblate)')
        ax.set_ylabel('Acylindricity (Prolate)')
        ax.set_title('Shape Parameter Phase Space')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(PLOTS_DIR / '15_shape_parameters_oblate_prolate.png', dpi=600, bbox_inches='tight')
        print("  Saved: 15_shape_parameters_oblate_prolate.png")
        
        # Save data
        summary = pd.DataFrame({
            'Epsilon': valid_eps,
            'Mean_Asphericity': asp_means,
            'Mean_Acylindricity': acy_means
        })
        summary.to_csv(PLOTS_DIR / 'shape_parameters_summary.csv', index=False)
        plt.close()
    
    def plot_coordination_hbonds(self):
        """Plot coordination numbers and hydrogen bond analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Water Coordination and Hydrogen Bonding Analysis', fontsize=14, fontweight='bold')
        
        # Coordination number evolution
        ax = axes[0, 0]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            coord = self.data[eps]['coordination_numbers']
            ax.plot(times, coord, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Coordination Number')
        ax.set_title('Water Coordination Around C60 Nanoparticles')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # H-bond count evolution
        ax = axes[0, 1]
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            times = self.data[eps]['timestamps']
            hbonds = self.data[eps]['hbond_count']
            ax.plot(times, hbonds, label=f'ε={eps:.2f}', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Total H-bond Count')
        ax.set_title('Hydrogen Bond Network Evolution')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Mean coordination
        ax = axes[1, 0]
        valid_eps = [eps for eps in self.epsilon_values if eps in self.data]
        coord_means = [np.mean(self.data[eps]['coordination_numbers']) for eps in valid_eps]
        coord_stds = [np.std(self.data[eps]['coordination_numbers']) for eps in valid_eps]
        
        x_pos = np.arange(len(valid_eps))
        ax.errorbar(x_pos, coord_means, yerr=coord_stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax.set_xlabel('Epsilon (kcal/mol)')
        ax.set_ylabel('Mean Coordination Number')
        ax.set_title('Hydration Layer Thickness vs Epsilon')
        ax.grid(True, alpha=0.3)
        
        # Mean H-bonds
        ax = axes[1, 1]
        hbond_means = [np.mean(self.data[eps]['hbond_count']) for eps in valid_eps]
        hbond_stds = [np.std(self.data[eps]['hbond_count']) for eps in valid_eps]
        
        ax.errorbar(x_pos, hbond_means, yerr=hbond_stds, fmt='o-', capsize=5, linewidth=2, markersize=8, color='C1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax.set_xlabel('Epsilon (kcal/mol)')
        ax.set_ylabel('Mean H-bond Count')
        ax.set_title('H-bond Network Strength vs Epsilon')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(PLOTS_DIR / '16_coordination_hbond_analysis.png', dpi=600, bbox_inches='tight')
        print("  Saved: 16_coordination_hbond_analysis.png")
        
        # Save data
        summary = pd.DataFrame({
            'Epsilon': valid_eps,
            'Mean_Coordination': coord_means,
            'Std_Coordination': coord_stds,
            'Mean_HBonds': hbond_means,
            'Std_HBonds': hbond_stds
        })
        summary.to_csv(PLOTS_DIR / 'coordination_hbond_summary.csv', index=False)
        plt.close()
    
    def plot_msd_diffusion(self):
        """Plot mean squared displacement and calculate diffusion coefficients"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Water Diffusion Analysis via Mean Squared Displacement', fontsize=14, fontweight='bold')
        
        # MSD curves
        ax = axes[0]
        diffusion_coeffs = []
        valid_eps = []
        
        for eps in self.epsilon_values:
            if eps not in self.data or 'msd_time' not in self.data[eps]:
                continue
            
            times = np.array(self.data[eps]['msd_time'])
            msd = np.array(self.data[eps]['msd_values'])
            
            ax.plot(times, msd, 'o-', label=f'ε={eps:.2f}', linewidth=2, markersize=4, alpha=0.8)
            
            # Fit linear regime (Einstein relation: MSD = 6Dt)
            if len(times) > 10:
                # Use middle portion for fitting
                fit_start = len(times) // 4
                fit_end = 3 * len(times) // 4
                
                def linear(t, D):
                    return 6 * D * t
                
                try:
                    popt, _ = curve_fit(linear, times[fit_start:fit_end], msd[fit_start:fit_end])
                    D = popt[0]
                    diffusion_coeffs.append(D)
                    valid_eps.append(eps)
                    
                    # Plot fit
                    ax.plot(times, linear(times, D), '--', alpha=0.5, linewidth=1)
                except:
                    pass
        
        ax.set_xlabel('Time Lag (ps)', fontsize=11)
        ax.set_ylabel('MSD (Å²)', fontsize=11)
        ax.set_title('Mean Squared Displacement of Water', fontsize=11, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Diffusion coefficients
        ax = axes[1]
        if diffusion_coeffs:
            x_pos = np.arange(len(valid_eps))
            ax.bar(x_pos, diffusion_coeffs, alpha=0.7, color=COLORS[:len(valid_eps)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
            ax.set_xlabel('Epsilon (kcal/mol)', fontsize=11)
            ax.set_ylabel('Diffusion Coefficient (Å²/ps)', fontsize=11)
            ax.set_title('Self-Diffusion Coefficient vs Epsilon', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Save diffusion data
            diff_df = pd.DataFrame({
                'Epsilon': valid_eps,
                'Diffusion_Coeff_A2_per_ps': diffusion_coeffs
            })
            diff_df.to_csv(PLOTS_DIR / 'diffusion_coefficients.csv', index=False)
        
        plt.savefig(PLOTS_DIR / '17_msd_diffusion_analysis.png', dpi=600, bbox_inches='tight')
        print("  Saved: 17_msd_diffusion_analysis.png")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all water structure plots"""
        print("\n" + "="*80)
        print("GENERATING WATER STRUCTURE PLOTS")
        print("="*80 + "\n")
        
        if not self.data:
            print("ERROR: No data loaded. Run analysis first.")
            return
        
        self.plot_tetrahedral_order()
        self.plot_steinhardt_order()
        self.plot_shape_parameters()
        self.plot_coordination_hbonds()
        self.plot_msd_diffusion()
        
        print("\n" + "="*80)
        print("ALL WATER STRUCTURE PLOTS COMPLETE!")
        print(f"Saved to: {PLOTS_DIR}")
        print("="*80)


def main():
    """Main plotting workflow"""
    plotter = WaterStructurePlotter()
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
