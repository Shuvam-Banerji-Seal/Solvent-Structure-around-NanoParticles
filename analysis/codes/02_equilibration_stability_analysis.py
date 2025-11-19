#!/usr/bin/env python3
"""
Equilibration and Stability Analysis for C60 Nanoparticle Solvation Study
==========================================================================

This script analyzes the equilibration quality and stability of simulations
across different stages (NVT, NPT, Production) and epsilon values.

Key analyses:
1. Equilibration time estimation using block averaging
2. Autocorrelation analysis for effective sample size
3. Convergence assessment for all thermodynamic properties
4. Stability metrics (drift, variance)

Author: Scientific Analysis Suite
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats, signal
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10

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

TIMESTEP = 2.0  # fs
TARGET_TEMP = 300.0  # K
TARGET_PRESS = 1.0  # atm

class EquilibrationAnalyzer:
    """Analyzer for equilibration and stability assessment"""
    
    def __init__(self, epsilon_dirs, epsilon_values):
        self.epsilon_dirs = epsilon_dirs
        self.epsilon_values = epsilon_values
        self.data = {}
        self.equilibration_metrics = {}
        
    def load_all_stages(self):
        """Load data from all simulation stages"""
        print("Loading simulation data from all stages...")
        
        for eps, eps_dir in zip(self.epsilon_values, self.epsilon_dirs):
            self.data[eps] = {}
            
            # Load NPT equilibration data
            npt_file = eps_dir / "npt_equilibration_thermo.dat"
            if npt_file.exists():
                df = pd.read_csv(npt_file, sep=r'\s+', comment='#',
                               names=['TimeStep', 'Epsilon', 'Temp', 'Press', 'Vol', 'PE', 'KE', 'Etotal', 'Dens'])
                df['Time_ns'] = df['TimeStep'] * TIMESTEP / 1e6
                df['Stage'] = 'NPT_Equilibration'
                self.data[eps]['npt'] = df
                print(f"  ε={eps}: Loaded NPT equilibration ({len(df)} points)")
            
            # Load production data
            prod_file = eps_dir / "production_detailed_thermo.dat"
            if prod_file.exists():
                df = pd.read_csv(prod_file, sep=r'\s+', comment='#',
                               names=['TimeStep', 'Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens'])
                df['Time_ns'] = (df['TimeStep'] + 600000) * TIMESTEP / 1e6
                df['Stage'] = 'Production'
                self.data[eps]['production'] = df
                print(f"  ε={eps}: Loaded production ({len(df)} points)")
                
        return self
    
    def compute_autocorrelation(self, data, max_lag=1000):
        """Compute autocorrelation function"""
        mean = np.mean(data)
        var = np.var(data)
        data_centered = data - mean
        
        # Use FFT for efficiency
        autocorr = signal.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (var * len(data))
        
        return autocorr[:min(max_lag, len(autocorr))]
    
    def estimate_correlation_time(self, data):
        """Estimate correlation time from autocorrelation"""
        acf = self.compute_autocorrelation(data, max_lag=min(5000, len(data)//2))
        
        # Find first zero crossing
        zero_crossing = np.where(acf < 0)[0]
        if len(zero_crossing) > 0:
            tau = zero_crossing[0]
        else:
            # Integrate autocorrelation until it decays to 1/e
            threshold = 1.0 / np.e
            below_threshold = np.where(acf < threshold)[0]
            tau = below_threshold[0] if len(below_threshold) > 0 else len(acf)
        
        # Integrated correlation time
        tau_int = 0.5 + np.sum(acf[:tau])
        
        return tau, tau_int
    
    def block_average_analysis(self, data, property_name, max_blocks=20):
        """Perform block averaging to assess equilibration"""
        n_data = len(data)
        block_sizes = np.logspace(0, np.log10(n_data//4), max_blocks, dtype=int)
        block_sizes = np.unique(block_sizes)
        
        means = []
        stds = []
        
        for block_size in block_sizes:
            n_blocks = n_data // block_size
            if n_blocks < 2:
                continue
                
            blocks = [data[i*block_size:(i+1)*block_size].mean() 
                     for i in range(n_blocks)]
            means.append(np.mean(blocks))
            stds.append(np.std(blocks) / np.sqrt(n_blocks))
        
        return block_sizes[:len(means)], means, stds
    
    def analyze_equilibration_quality(self):
        """Analyze equilibration quality for each epsilon"""
        print("\nAnalyzing equilibration quality...")
        
        metrics_list = []
        
        for eps in self.epsilon_values:
            if eps not in self.data or 'production' not in self.data[eps]:
                continue
            
            df = self.data[eps]['production']
            
            # Temperature analysis
            temp_data = df['Temp'].values
            temp_tau, temp_tau_int = self.estimate_correlation_time(temp_data)
            temp_eff_samples = len(temp_data) / (2 * temp_tau_int)
            
            # Pressure analysis  
            press_data = df['Press'].values
            press_tau, press_tau_int = self.estimate_correlation_time(press_data)
            press_eff_samples = len(press_data) / (2 * press_tau_int)
            
            # Density analysis
            dens_data = df['Dens'].values
            dens_tau, dens_tau_int = self.estimate_correlation_time(dens_data)
            dens_eff_samples = len(dens_data) / (2 * dens_tau_int)
            
            # Compute drift (linear trend)
            time_points = np.arange(len(temp_data))
            temp_slope, _, _, _, _ = stats.linregress(time_points, temp_data)
            press_slope, _, _, _, _ = stats.linregress(time_points, press_data)
            dens_slope, _, _, _, _ = stats.linregress(time_points, dens_data)
            
            metrics = {
                'Epsilon': eps,
                'Temp_corr_time': temp_tau * TIMESTEP / 1000,  # ps
                'Temp_eff_samples': temp_eff_samples,
                'Temp_drift_K_per_ns': temp_slope * 1000,  # K/ns
                'Press_corr_time': press_tau * TIMESTEP / 1000,  # ps
                'Press_eff_samples': press_eff_samples,
                'Press_drift_atm_per_ns': press_slope * 1000,  # atm/ns
                'Dens_corr_time': dens_tau * TIMESTEP / 1000,  # ps
                'Dens_eff_samples': dens_eff_samples,
                'Dens_drift_per_ns': dens_slope * 1000,  # g/cm³/ns
                'N_total_samples': len(temp_data)
            }
            
            metrics_list.append(metrics)
            self.equilibration_metrics[eps] = metrics
            
        self.metrics_df = pd.DataFrame(metrics_list)
        
        # Save metrics
        metrics_file = PLOTS_DIR / "equilibration_metrics.csv"
        self.metrics_df.to_csv(metrics_file, index=False, float_format='%.6f')
        print(f"  Metrics saved to {metrics_file}")
        
        return self
    
    def plot_autocorrelation_functions(self):
        """Plot autocorrelation functions for temperature, pressure, density"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        properties = ['Temp', 'Press', 'Dens']
        prop_names = ['Temperature', 'Pressure', 'Density']
        
        max_lag = 2000
        
        for col_idx, (prop, name) in enumerate(zip(properties, prop_names)):
            # Plot 1: ACF for all epsilon values
            ax1 = axes[0, col_idx]
            for eps in self.epsilon_values:
                if eps in self.data and 'production' in self.data[eps]:
                    df = self.data[eps]['production']
                    data = df[prop].values
                    acf = self.compute_autocorrelation(data, max_lag=max_lag)
                    time_lag = np.arange(len(acf)) * TIMESTEP / 1000  # ps
                    ax1.plot(time_lag, acf, label=f'ε={eps}', alpha=0.7)
            
            ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax1.axhline(1/np.e, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='1/e')
            ax1.set_xlabel('Lag Time (ps)')
            ax1.set_ylabel('Autocorrelation')
            ax1.set_title(f'{name} Autocorrelation Function')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Correlation times
            ax2 = axes[1, col_idx]
            corr_times = [self.equilibration_metrics[eps][f'{prop}_corr_time'] 
                         for eps in self.epsilon_values if eps in self.equilibration_metrics]
            
            ax2.bar(range(len(corr_times)), corr_times, 
                   color=sns.color_palette("husl", len(corr_times)))
            ax2.set_xticks(range(len(self.epsilon_values)))
            ax2.set_xticklabels([f'{eps:.2f}' for eps in self.epsilon_values])
            ax2.set_xlabel('Epsilon (kcal/mol)')
            ax2.set_ylabel('Correlation Time (ps)')
            ax2.set_title(f'{name} Correlation Time')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "06_autocorrelation_analysis.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 06_autocorrelation_analysis.png")
    
    def plot_block_averaging(self):
        """Plot block averaging analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        properties = ['Temp', 'Press', 'Dens']
        prop_names = ['Temperature (K)', 'Pressure (atm)', 'Density (g/cm³)']
        
        for col_idx, (prop, name) in enumerate(zip(properties, prop_names)):
            # Select two representative epsilon values
            eps_to_plot = [0.0, 0.25]
            
            for row_idx, eps in enumerate(eps_to_plot):
                if eps not in self.data or 'production' not in self.data[eps]:
                    continue
                
                ax = axes[row_idx, col_idx]
                df = self.data[eps]['production']
                data = df[prop].values
                
                block_sizes, means, stds = self.block_average_analysis(data, prop)
                
                ax.errorbar(block_sizes, means, yerr=stds, fmt='o-', 
                           capsize=3, label=f'ε={eps}')
                ax.axhline(np.mean(data), color='red', linestyle='--', 
                          linewidth=2, label='Overall mean')
                ax.set_xscale('log')
                ax.set_xlabel('Block Size')
                ax.set_ylabel(name)
                ax.set_title(f'{name} Block Averaging (ε={eps})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "07_block_averaging.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 07_block_averaging.png")
    
    def plot_running_averages(self):
        """Plot running averages to show convergence"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        properties = ['Temp', 'Press', 'Dens']
        prop_names = ['Temperature (K)', 'Pressure (atm)', 'Density (g/cm³)']
        targets = [TARGET_TEMP, TARGET_PRESS, 1.0]
        
        for row_idx, (prop, name, target) in enumerate(zip(properties, prop_names, targets)):
            # Plot 1: Running average for all epsilon
            ax1 = axes[row_idx, 0]
            for eps in self.epsilon_values:
                if eps in self.data and 'production' in self.data[eps]:
                    df = self.data[eps]['production']
                    data = df[prop].values
                    running_avg = np.cumsum(data) / (np.arange(len(data)) + 1)
                    time_ns = df['Time_ns'].values
                    ax1.plot(time_ns, running_avg, label=f'ε={eps}', alpha=0.7)
            
            if target is not None:
                ax1.axhline(target, color='red', linestyle='--', linewidth=2, label='Target')
            ax1.set_xlabel('Time (ns)')
            ax1.set_ylabel(name)
            ax1.set_title(f'{name} - Running Average')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Running standard deviation
            ax2 = axes[row_idx, 1]
            for eps in self.epsilon_values:
                if eps in self.data and 'production' in self.data[eps]:
                    df = self.data[eps]['production']
                    data = df[prop].values
                    # Compute running std
                    running_std = [np.std(data[:i+1]) for i in range(len(data))]
                    time_ns = df['Time_ns'].values
                    ax2.plot(time_ns, running_std, label=f'ε={eps}', alpha=0.7)
            
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel(f'{name} - Std Dev')
            ax2.set_title(f'{name} - Running Standard Deviation')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "08_running_averages.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 08_running_averages.png")
    
    def plot_stability_metrics(self):
        """Plot comprehensive stability metrics"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Effective sample sizes
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Get valid epsilon values
        valid_eps = [eps for eps in self.epsilon_values if eps in self.equilibration_metrics]
        x_pos = np.arange(len(valid_eps))
        width = 0.25
        
        temp_eff = [self.equilibration_metrics[eps]['Temp_eff_samples'] 
                   for eps in valid_eps]
        press_eff = [self.equilibration_metrics[eps]['Press_eff_samples'] 
                    for eps in valid_eps]
        dens_eff = [self.equilibration_metrics[eps]['Dens_eff_samples'] 
                   for eps in valid_eps]
        
        ax1.bar(x_pos - width, temp_eff, width, label='Temperature', alpha=0.8)
        ax1.bar(x_pos, press_eff, width, label='Pressure', alpha=0.8)
        ax1.bar(x_pos + width, dens_eff, width, label='Density', alpha=0.8)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax1.set_xlabel('Epsilon (kcal/mol)')
        ax1.set_ylabel('Effective Sample Size')
        ax1.set_title('Statistical Independence of Samples')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Drift analysis (temperature)
        ax2 = fig.add_subplot(gs[0, 1])
        temp_drift = [abs(self.equilibration_metrics[eps]['Temp_drift_K_per_ns']) 
                     for eps in valid_eps]
        
        ax2.bar(x_pos, temp_drift, color='coral', alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax2.set_xlabel('Epsilon (kcal/mol)')
        ax2.set_ylabel('|Temperature Drift| (K/ns)')
        ax2.set_title('Temperature Stability (Lower is Better)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Drift analysis (pressure)
        ax3 = fig.add_subplot(gs[1, 0])
        press_drift = [abs(self.equilibration_metrics[eps]['Press_drift_atm_per_ns']) 
                      for eps in valid_eps]
        
        ax3.bar(x_pos, press_drift, color='skyblue', alpha=0.8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax3.set_xlabel('Epsilon (kcal/mol)')
        ax3.set_ylabel('|Pressure Drift| (atm/ns)')
        ax3.set_title('Pressure Stability (Lower is Better)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Drift analysis (density)
        ax4 = fig.add_subplot(gs[1, 1])
        dens_drift = [abs(self.equilibration_metrics[eps]['Dens_drift_per_ns']) 
                     for eps in valid_eps]
        
        ax4.bar(x_pos, dens_drift, color='lightgreen', alpha=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{eps:.2f}' for eps in valid_eps])
        ax4.set_xlabel('Epsilon (kcal/mol)')
        ax4.set_ylabel('|Density Drift| (g/cm³/ns)')
        ax4.set_title('Density Stability (Lower is Better)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Summary table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for eps in self.epsilon_values:
            if eps not in self.equilibration_metrics:
                continue
            m = self.equilibration_metrics[eps]
            table_data.append([
                f"{eps:.2f}",
                f"{m['Temp_corr_time']:.1f}",
                f"{m['Temp_eff_samples']:.0f}",
                f"{abs(m['Temp_drift_K_per_ns']):.2e}",
                f"{m['Press_corr_time']:.1f}",
                f"{abs(m['Press_drift_atm_per_ns']):.2e}",
                f"{m['Dens_corr_time']:.1f}",
                f"{abs(m['Dens_drift_per_ns']):.2e}"
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['ε', 'τ_T (ps)', 'N_eff_T', '|Drift_T|',
                                   'τ_P (ps)', '|Drift_P|', 'τ_ρ (ps)', '|Drift_ρ|'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('Equilibration and Stability Analysis', fontsize=16, fontweight='bold')
        plt.savefig(PLOTS_DIR / "09_stability_metrics.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 09_stability_metrics.png")
    
    def export_equilibration_report(self):
        """Export detailed equilibration report"""
        report = {
            'analysis_summary': {
                'purpose': 'Assess equilibration quality and statistical reliability',
                'methods': [
                    'Autocorrelation analysis for correlation times',
                    'Block averaging for error estimation',
                    'Running averages for convergence assessment',
                    'Drift analysis for stability evaluation'
                ]
            },
            'epsilon_metrics': {}
        }
        
        for eps in self.epsilon_values:
            if eps not in self.equilibration_metrics:
                continue
            
            m = self.equilibration_metrics[eps]
            report['epsilon_metrics'][f'eps_{eps:.2f}'] = {
                'temperature': {
                    'correlation_time_ps': float(m['Temp_corr_time']),
                    'effective_samples': float(m['Temp_eff_samples']),
                    'drift_K_per_ns': float(m['Temp_drift_K_per_ns']),
                    'stability_assessment': 'Good' if abs(m['Temp_drift_K_per_ns']) < 0.1 else 'Fair'
                },
                'pressure': {
                    'correlation_time_ps': float(m['Press_corr_time']),
                    'effective_samples': float(m['Press_eff_samples']),
                    'drift_atm_per_ns': float(m['Press_drift_atm_per_ns']),
                    'stability_assessment': 'Good' if abs(m['Press_drift_atm_per_ns']) < 10 else 'Fair'
                },
                'density': {
                    'correlation_time_ps': float(m['Dens_corr_time']),
                    'effective_samples': float(m['Dens_eff_samples']),
                    'drift_per_ns': float(m['Dens_drift_per_ns']),
                    'stability_assessment': 'Good' if abs(m['Dens_drift_per_ns']) < 0.001 else 'Fair'
                },
                'total_samples': int(m['N_total_samples'])
            }
        
        json_file = PLOTS_DIR / "equilibration_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nEquilibration report saved to {json_file}")

def main():
    """Main analysis workflow"""
    print("="*70)
    print("EQUILIBRATION AND STABILITY ANALYSIS")
    print("="*70)
    
    epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    analyzer = EquilibrationAnalyzer(EPSILON_DIRS, epsilon_values)
    
    # Load data and analyze
    analyzer.load_all_stages()
    analyzer.analyze_equilibration_quality()
    
    print("\n" + "="*70)
    print("GENERATING EQUILIBRATION PLOTS")
    print("="*70)
    
    # Generate plots
    analyzer.plot_autocorrelation_functions()
    analyzer.plot_block_averaging()
    analyzer.plot_running_averages()
    analyzer.plot_stability_metrics()
    
    # Export report
    analyzer.export_equilibration_report()
    
    print("\n" + "="*70)
    print("EQUILIBRATION ANALYSIS COMPLETE!")
    print(f"All plots saved to: {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
