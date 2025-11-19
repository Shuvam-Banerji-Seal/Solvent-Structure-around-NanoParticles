#!/usr/bin/env python3
"""
Comprehensive Thermodynamic Analysis for C60 Nanoparticle Solvation Study
==========================================================================

This script analyzes thermodynamic properties across different epsilon values
to understand how C-O interaction strength affects system equilibration and stability.

System Details:
- 3 C60 nanoparticles (180 carbon atoms)
- ~2000 TIP4P/2005 water molecules
- Box: 40×40×40 Å³
- Atom types: 1=C (carbon), 2=O (oxygen), 3=H (hydrogen)
- Epsilon range: 0.0 to 0.25 kcal/mol

Author: Scientific Analysis Suite
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from matplotlib.gridspec import GridSpec

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define paths
BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")

# Setup epsilon directories (handle epsilon_0.0 special case)
epsilon_values_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
EPSILON_DIRS = []
for eps in epsilon_values_list:
    if eps == 0.0:
        EPSILON_DIRS.append(BASE_DIR / "epsilon_0.0")
    else:
        EPSILON_DIRS.append(BASE_DIR / f"epsilon_{eps:.2f}")

PLOTS_DIR = BASE_DIR / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# System parameters
TIMESTEP = 2.0  # fs
TARGET_TEMP = 300.0  # K
TARGET_PRESS = 1.0  # atm
PRODUCTION_START = 600000  # step where production begins
TOTAL_STEPS = 2600000  # total simulation steps

class ThermodynamicAnalyzer:
    """Analyzer for thermodynamic properties across epsilon values"""
    
    def __init__(self, epsilon_dirs, epsilon_values):
        self.epsilon_dirs = epsilon_dirs
        self.epsilon_values = epsilon_values
        self.data = {}
        
    def load_production_data(self):
        """Load production run thermodynamic data for all epsilon values"""
        print("Loading production thermodynamic data...")
        
        for eps, eps_dir in zip(self.epsilon_values, self.epsilon_dirs):
            thermo_file = eps_dir / "production_detailed_thermo.dat"
            
            if not thermo_file.exists():
                print(f"  Warning: {thermo_file} not found")
                continue
                
            # Read thermodynamic data
            df = pd.read_csv(thermo_file, sep=r'\s+', comment='#',
                           names=['TimeStep', 'Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens'])
            
            # Convert timestep to time in ns
            df['Time_ns'] = (df['TimeStep'] + PRODUCTION_START) * TIMESTEP / 1e6
            
            self.data[eps] = df
            print(f"  Loaded ε={eps}: {len(df)} data points")
            
        return self
    
    def load_equilibration_data(self):
        """Load NPT equilibration data"""
        print("Loading equilibration thermodynamic data...")
        
        for eps, eps_dir in zip(self.epsilon_values, self.epsilon_dirs):
            thermo_file = eps_dir / "npt_equilibration_thermo.dat"
            
            if not thermo_file.exists():
                continue
                
            df = pd.read_csv(thermo_file, sep=r'\s+', comment='#',
                           names=['TimeStep', 'Epsilon', 'Temp', 'Press', 'Vol', 'PE', 'KE', 'Etotal', 'Dens'])
            
            df['Time_ns'] = df['TimeStep'] * TIMESTEP / 1e6
            
            if eps not in self.data:
                self.data[eps] = {}
            self.data[eps]['equilibration'] = df
            print(f"  Loaded equilibration for ε={eps}: {len(df)} points")
            
        return self
    
    def compute_statistics(self):
        """Compute statistical properties for each epsilon"""
        stats_data = []
        
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
                
            df = self.data[eps]
            
            # Compute averages and standard deviations
            stats_entry = {
                'Epsilon': eps,
                'Temp_mean': df['Temp'].mean(),
                'Temp_std': df['Temp'].std(),
                'Press_mean': df['Press'].mean(),
                'Press_std': df['Press'].std(),
                'Dens_mean': df['Dens'].mean(),
                'Dens_std': df['Dens'].std(),
                'PE_mean': df['PE'].mean(),
                'PE_std': df['PE'].std(),
                'Vol_mean': df['Vol'].mean(),
                'Vol_std': df['Vol'].std(),
                'N_samples': len(df)
            }
            
            stats_data.append(stats_entry)
            
        self.stats_df = pd.DataFrame(stats_data)
        
        # Save statistics to CSV
        stats_file = PLOTS_DIR / "thermodynamic_statistics.csv"
        self.stats_df.to_csv(stats_file, index=False, float_format='%.6f')
        print(f"\nStatistics saved to {stats_file}")
        
        return self
    
    def plot_temperature_evolution(self):
        """Plot temperature evolution for all epsilon values"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Individual trajectories with mean lines
        ax1 = axes[0]
        colors = sns.color_palette("husl", len(self.epsilon_values))
        color_dict = {eps: colors[i] for i, eps in enumerate(self.epsilon_values)}
        
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax1.plot(df['Time_ns'], df['Temp'], label=f'ε={eps}', alpha=0.4, linewidth=0.5, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_temp = df['Temp'].mean()
                ax1.axhline(mean_temp, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax1.axhline(TARGET_TEMP, color='red', linestyle='--', linewidth=2.5, label='Target (300 K)', zorder=10)
        ax1.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Temperature (K)', fontsize=11, fontweight='bold')
        ax1.set_title('Temperature Evolution During Production Run (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean ± std for each epsilon with mean lines
        ax2 = axes[1]
        valid_eps = self.stats_df['Epsilon'].values
        x_pos = np.arange(len(valid_eps))
        means = self.stats_df['Temp_mean'].values
        stds = self.stats_df['Temp_std'].values
        
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                color=sns.color_palette("husl", len(valid_eps)))
        ax2.axhline(TARGET_TEMP, color='red', linestyle='--', linewidth=2.5, label='Target', zorder=10)
        
        # Add horizontal lines for each bar's mean value
        for i, (eps, mean) in enumerate(zip(valid_eps, means)):
            ax2.hlines(mean, i-0.35, i+0.35, colors='black', linestyles='solid', linewidth=2.5, alpha=0.9, zorder=5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eps:.2f}' for eps in valid_eps], fontsize=10)
        ax2.set_xlabel('Epsilon (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Mean Temperature (K)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Temperature by Epsilon Value (Black lines: mean values)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "01_temperature_evolution.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 01_temperature_evolution.png")
        
    def plot_pressure_evolution(self):
        """Plot pressure evolution and distribution"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Pressure time series with mean lines
        ax1 = fig.add_subplot(gs[0, :])
        colors = sns.color_palette("husl", len(self.epsilon_values))
        color_dict = {eps: colors[i] for i, eps in enumerate(self.epsilon_values)}
        
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax1.plot(df['Time_ns'], df['Press'], label=f'ε={eps}', alpha=0.4, linewidth=0.5, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_press = df['Press'].mean()
                ax1.axhline(mean_press, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax1.axhline(TARGET_PRESS, color='red', linestyle='--', linewidth=2.5, label='Target (1 atm)', zorder=10)
        ax1.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Pressure (atm)', fontsize=11, fontweight='bold')
        ax1.set_title('Pressure Evolution During Production Run (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=3, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pressure distribution (histogram) with mean lines
        ax2 = fig.add_subplot(gs[1, 0])
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax2.hist(df['Press'], bins=50, alpha=0.5, label=f'ε={eps}', density=True, color=color_dict[eps])
                # Add vertical line for mean (no label to avoid cluttering legend)
                mean_press = df['Press'].mean()
                ax2.axvline(mean_press, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.9)
        
        ax2.axvline(TARGET_PRESS, color='red', linestyle='--', linewidth=2.5, label='Target', zorder=10)
        ax2.set_xlabel('Pressure (atm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax2.set_title('Pressure Distribution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean pressure by epsilon with mean lines
        ax3 = fig.add_subplot(gs[1, 1])
        valid_eps = self.stats_df['Epsilon'].values
        x_pos = np.arange(len(valid_eps))
        means = self.stats_df['Press_mean'].values
        stds = self.stats_df['Press_std'].values
        
        ax3.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8, color='navy')
        ax3.axhline(TARGET_PRESS, color='red', linestyle='--', linewidth=2.5, label='Target', zorder=10)
        
        # Add horizontal lines for each point's mean value
        for i, (eps, mean) in enumerate(zip(valid_eps, means)):
            ax3.hlines(mean, i-0.25, i+0.25, colors='black', linestyles='solid', linewidth=2.5, alpha=0.9, zorder=5)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{eps:.2f}' for eps in valid_eps], fontsize=10)
        ax3.set_xlabel('Epsilon (kcal/mol)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Mean Pressure (atm)', fontsize=11, fontweight='bold')
        ax3.set_title('Average Pressure vs Epsilon (Black lines: mean values)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_pressure_analysis.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 02_pressure_analysis.png")
        
    def plot_density_analysis(self):
        """Plot density evolution and epsilon dependence"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Density time series with mean lines
        ax1 = axes[0, 0]
        colors = sns.color_palette("husl", len(self.epsilon_values))
        color_dict = {eps: colors[i] for i, eps in enumerate(self.epsilon_values)}
        
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax1.plot(df['Time_ns'], df['Dens'], label=f'ε={eps}', alpha=0.4, linewidth=0.8, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_dens = df['Dens'].mean()
                ax1.axhline(mean_dens, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax1.axhline(1.0, color='red', linestyle='--', linewidth=2.5, label='Water (1.0 g/cm³)', zorder=10)
        ax1.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Density (g/cm³)', fontsize=11, fontweight='bold')
        ax1.set_title('Density Evolution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean density vs epsilon with mean lines
        ax2 = axes[0, 1]
        valid_eps = self.stats_df['Epsilon'].values
        x_pos = np.arange(len(valid_eps))
        means = self.stats_df['Dens_mean'].values
        stds = self.stats_df['Dens_std'].values
        
        ax2.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8,
                    color='darkblue')
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=2.5, label='Pure water', zorder=10)
        
        # Add horizontal lines for each point's mean value
        for i, (eps, mean) in enumerate(zip(valid_eps, means)):
            ax2.hlines(mean, i-0.25, i+0.25, colors='black', linestyles='solid', linewidth=2.5, alpha=0.9, zorder=5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eps:.2f}' for eps in valid_eps], fontsize=10)
        ax2.set_xlabel('Epsilon (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Density (g/cm³)', fontsize=11, fontweight='bold')
        ax2.set_title('Mean Density vs C-O Interaction Strength (Black lines: mean values)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volume evolution with mean lines
        ax3 = axes[1, 0]
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax3.plot(df['Time_ns'], df['Vol'], label=f'ε={eps}', alpha=0.4, linewidth=0.8, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_vol = df['Vol'].mean()
                ax3.axhline(mean_vol, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax3.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Volume (Å³)', fontsize=11, fontweight='bold')
        ax3.set_title('Box Volume Evolution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', ncol=2, fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Density distribution with mean lines
        ax4 = axes[1, 1]
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax4.hist(df['Dens'], bins=30, alpha=0.5, label=f'ε={eps}', density=True, color=color_dict[eps])
                # Add vertical line for mean (no label to avoid cluttering legend)
                mean_dens = df['Dens'].mean()
                ax4.axvline(mean_dens, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.9)
        
        ax4.axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Pure water', zorder=10)
        ax4.set_xlabel('Density (g/cm³)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax4.set_title('Density Distribution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "03_density_analysis.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 03_density_analysis.png")
        
    def plot_energy_analysis(self):
        """Plot potential energy analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: PE time series with mean lines
        ax1 = axes[0, 0]
        colors = sns.color_palette("husl", len(self.epsilon_values))
        color_dict = {eps: colors[i] for i, eps in enumerate(self.epsilon_values)}
        
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax1.plot(df['Time_ns'], df['PE'], label=f'ε={eps}', alpha=0.4, linewidth=0.6, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_pe = df['PE'].mean()
                ax1.axhline(mean_pe, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax1.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Potential Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax1.set_title('Potential Energy Evolution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean PE vs epsilon with mean lines
        ax2 = axes[0, 1]
        valid_eps = self.stats_df['Epsilon'].values
        x_pos = np.arange(len(valid_eps))
        means = self.stats_df['PE_mean'].values
        stds = self.stats_df['PE_std'].values
        
        ax2.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8, color='navy')
        
        # Add horizontal lines for each point's mean value
        for i, (eps, mean) in enumerate(zip(valid_eps, means)):
            ax2.hlines(mean, i-0.25, i+0.25, colors='black', linestyles='solid', linewidth=2.5, alpha=0.9, zorder=5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eps:.2f}' for eps in valid_eps], fontsize=10)
        ax2.set_xlabel('Epsilon (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Mean PE (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Potential Energy vs C-O Interaction (Black lines: mean values)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: PE distribution with mean lines
        ax3 = axes[1, 0]
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax3.hist(df['PE'], bins=40, alpha=0.5, label=f'ε={eps}', density=True, color=color_dict[eps])
                # Add vertical line for mean (no label to avoid cluttering legend)
                mean_pe = df['PE'].mean()
                ax3.axvline(mean_pe, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.9)
        
        ax3.set_xlabel('Potential Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax3.set_title('PE Distribution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: KE evolution with mean lines
        ax4 = axes[1, 1]
        for eps in self.epsilon_values:
            if eps in self.data:
                df = self.data[eps]
                ax4.plot(df['Time_ns'], df['KE'], label=f'ε={eps}', alpha=0.4, linewidth=0.6, color=color_dict[eps])
                # Add mean line for this epsilon (no label to avoid cluttering legend)
                mean_ke = df['KE'].mean()
                ax4.axhline(mean_ke, color=color_dict[eps], linestyle='-', linewidth=2.5, alpha=0.8)
        
        ax4.set_xlabel('Time (ns)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Kinetic Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax4.set_title('Kinetic Energy Evolution (Solid lines: mean per epsilon)', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', ncol=2, fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_energy_analysis.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 04_energy_analysis.png")
        
    def plot_comparison_matrix(self):
        """Create comprehensive comparison matrix"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        properties = ['Temp', 'Press', 'Dens', 'PE', 'Vol']
        titles = ['Temperature (K)', 'Pressure (atm)', 'Density (g/cm³)', 
                 'Potential Energy (kcal/mol)', 'Volume (Å³)']
        
        for idx, (prop, title) in enumerate(zip(properties, titles)):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Plot mean ± std
            valid_eps = self.stats_df['Epsilon'].values
            means = self.stats_df[f'{prop}_mean'].values
            stds = self.stats_df[f'{prop}_std'].values
            x_pos = np.arange(len(valid_eps))
            
            ax.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5, 
                       linewidth=2, markersize=8, color=f'C{idx}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
            ax.set_xlabel('Epsilon (kcal/mol)')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs C-O Interaction')
            ax.grid(True, alpha=0.3)
            
        # Summary statistics table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        table_data = []
        for eps in self.epsilon_values:
            row = self.stats_df[self.stats_df['Epsilon'] == eps].iloc[0]
            table_data.append([
                f"{eps:.2f}",
                f"{row['Temp_mean']:.2f}±{row['Temp_std']:.2f}",
                f"{row['Press_mean']:.1f}±{row['Press_std']:.1f}",
                f"{row['Dens_mean']:.4f}±{row['Dens_std']:.4f}",
                f"{row['PE_mean']:.1f}±{row['PE_std']:.1f}"
            ])
        
        table = ax_table.table(cellText=table_data,
                              colLabels=['ε', 'Temp (K)', 'Press (atm)', 'Dens (g/cm³)', 'PE (kcal/mol)'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('Comprehensive Thermodynamic Comparison Across Epsilon Values',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(PLOTS_DIR / "05_comparison_matrix.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 05_comparison_matrix.png")
        
    def export_summary_json(self):
        """Export summary statistics to JSON"""
        summary = {
            'system_info': {
                'n_c60_particles': 3,
                'n_carbon_atoms': 180,
                'n_water_molecules': '~2000',
                'box_size': '40×40×40 Å³',
                'water_model': 'TIP4P/2005',
                'timestep_fs': TIMESTEP,
                'target_temp_K': TARGET_TEMP,
                'target_press_atm': TARGET_PRESS
            },
            'epsilon_statistics': {}
        }
        
        for eps in self.epsilon_values:
            if eps not in self.data:
                continue
            row = self.stats_df[self.stats_df['Epsilon'] == eps].iloc[0]
            
            summary['epsilon_statistics'][f'eps_{eps:.2f}'] = {
                'temperature': {
                    'mean_K': float(row['Temp_mean']),
                    'std_K': float(row['Temp_std']),
                    'deviation_from_target': float(row['Temp_mean'] - TARGET_TEMP)
                },
                'pressure': {
                    'mean_atm': float(row['Press_mean']),
                    'std_atm': float(row['Press_std']),
                    'deviation_from_target': float(row['Press_mean'] - TARGET_PRESS)
                },
                'density': {
                    'mean_g_cm3': float(row['Dens_mean']),
                    'std_g_cm3': float(row['Dens_std'])
                },
                'potential_energy': {
                    'mean_kcal_mol': float(row['PE_mean']),
                    'std_kcal_mol': float(row['PE_std'])
                },
                'n_samples': int(row['N_samples'])
            }
        
        json_file = PLOTS_DIR / "thermodynamic_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary JSON saved to {json_file}")

def main():
    """Main analysis workflow"""
    print("="*70)
    print("THERMODYNAMIC ANALYSIS - C60 NANOPARTICLE SOLVATION STUDY")
    print("="*70)
    
    epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    analyzer = ThermodynamicAnalyzer(EPSILON_DIRS, epsilon_values)
    
    # Load data
    analyzer.load_production_data()
    analyzer.compute_statistics()
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Generate all plots
    analyzer.plot_temperature_evolution()
    analyzer.plot_pressure_evolution()
    analyzer.plot_density_analysis()
    analyzer.plot_energy_analysis()
    analyzer.plot_comparison_matrix()
    
    # Export summary
    analyzer.export_summary_json()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"All plots saved to: {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
