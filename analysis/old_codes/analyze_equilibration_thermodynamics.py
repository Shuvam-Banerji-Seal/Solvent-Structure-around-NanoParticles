#!/usr/bin/env python3
"""
Rigorous Thermodynamic Analysis of LAMMPS Equilibration Results
================================================================

This script analyzes the equilibration trajectories for multiple epsilon values,
extracting thermodynamic properties and validating equilibration quality.

Analysis includes:
1. Time series analysis of all thermodynamic properties
2. Statistical equilibration detection using block averaging
3. Equilibration quality metrics (fluctuations, stability)
4. Comparative analysis across epsilon values
5. Publication-quality visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


class EquilibrationAnalyzer:
    """Analyze LAMMPS equilibration trajectories"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.data = {}
        self.stats = {}
        
    def load_thermodynamic_data(self, epsilon: float) -> pd.DataFrame:
        """Load thermodynamic data from equilibration log"""
        log_file = self.base_dir / f"epsilon_{epsilon:.2f}" / "equilibration.log"
        
        if not log_file.exists():
            print(f"WARNING: Log file not found for ε={epsilon:.2f}")
            return pd.DataFrame()
        
        # Read log file and extract thermodynamic data
        data_lines = []
        in_thermo_section = False
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Detect thermo output header
                if line.startswith("Step") and "Temp" in line and "Press" in line:
                    in_thermo_section = True
                    headers = line.split()
                    continue
                
                # End of thermo section
                if in_thermo_section and (line.startswith("Loop") or 
                                          line.startswith("WARNING") or
                                          line.startswith("Neighbor") or
                                          line.startswith("Per MPI") or
                                          line == ""):
                    in_thermo_section = False
                    continue
                
                # Extract data lines
                if in_thermo_section and line and line[0].isdigit():
                    try:
                        values = [float(x) for x in line.split()]
                        data_lines.append(values)
                    except ValueError:
                        continue
        
        if not data_lines:
            print(f"ERROR: No data extracted for ε={epsilon:.2f}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=headers)
        
        # Calculate time in nanoseconds (assuming 2 fs timestep)
        df['Time_ns'] = df['Step'] * 2e-6  # 2 fs * 1e-9 s/fs * 1e9 ns/s
        
        print(f"ε={epsilon:.2f}: Loaded {len(df)} data points, "
              f"time range: {df['Time_ns'].min():.3f} - {df['Time_ns'].max():.3f} ns")
        
        return df
    
    def detect_equilibration(self, series: np.ndarray, block_size: int = 100) -> Tuple[int, float]:
        """
        Detect equilibration point using block averaging method
        
        Returns:
            equilibration_index: Index where system is equilibrated
            stability_metric: Measure of stability (lower is better)
        """
        n_points = len(series)
        n_blocks = n_points // block_size
        
        if n_blocks < 5:
            return 0, np.std(series) / np.mean(np.abs(series))
        
        # Calculate block averages
        block_means = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block_means.append(np.mean(series[start:end]))
        
        block_means = np.array(block_means)
        
        # Find where standard deviation stabilizes
        window_size = 5
        std_evolution = []
        for i in range(window_size, n_blocks):
            std_evolution.append(np.std(block_means[i-window_size:i]))
        
        # Equilibration is where std becomes relatively constant
        if len(std_evolution) > 10:
            std_gradient = np.abs(np.gradient(std_evolution))
            equilibration_block = np.argmin(std_gradient) + window_size
            equilibration_index = equilibration_block * block_size
        else:
            equilibration_index = n_points // 2  # Conservative estimate
        
        # Stability metric (coefficient of variation in last 50%)
        equilibrated_data = series[equilibration_index:]
        stability_metric = np.std(equilibrated_data) / np.mean(np.abs(equilibrated_data))
        
        return equilibration_index, stability_metric
    
    def calculate_statistics(self, df: pd.DataFrame, epsilon: float) -> Dict:
        """Calculate comprehensive statistics for equilibrated region"""
        
        # Detect equilibration for density (most critical property)
        if 'Density' in df.columns and len(df) > 100:
            eq_idx, stability = self.detect_equilibration(df['Density'].values)
        else:
            eq_idx = len(df) // 2  # Use last 50% if detection fails
            stability = np.nan
        
        # Use equilibrated portion (conservative: last 50%)
        eq_idx = max(eq_idx, len(df) // 2)
        df_eq = df.iloc[eq_idx:]
        
        stats_dict = {
            'epsilon': epsilon,
            'total_points': len(df),
            'equilibration_index': eq_idx,
            'equilibration_time_ns': df.iloc[eq_idx]['Time_ns'] if len(df) > 0 else 0,
            'equilibrated_points': len(df_eq),
            'stability_metric': stability
        }
        
        # Calculate mean ± std for all properties
        for col in ['Temp', 'Press', 'Volume', 'Density', 'E_pair', 'E_long', 'TotEng']:
            if col in df_eq.columns:
                values = df_eq[col].values
                stats_dict[f'{col}_mean'] = np.mean(values)
                stats_dict[f'{col}_std'] = np.std(values)
                stats_dict[f'{col}_sem'] = stats.sem(values)
                stats_dict[f'{col}_min'] = np.min(values)
                stats_dict[f'{col}_max'] = np.max(values)
                
                # Autocorrelation time (simplified)
                if len(values) > 10:
                    acf = np.correlate(values - np.mean(values), 
                                      values - np.mean(values), 
                                      mode='full')
                    acf = acf[len(acf)//2:]
                    acf /= acf[0]
                    
                    # Find where ACF drops below 0.1
                    tau_indices = np.where(acf < 0.1)[0]
                    if len(tau_indices) > 0:
                        stats_dict[f'{col}_autocorr_time'] = tau_indices[0]
                    else:
                        stats_dict[f'{col}_autocorr_time'] = len(acf)
        
        return stats_dict
    
    def analyze_all_systems(self):
        """Load and analyze all epsilon values"""
        print("\n" + "="*70)
        print("LOADING THERMODYNAMIC DATA FOR ALL SYSTEMS")
        print("="*70)
        
        for eps in self.epsilon_values:
            df = self.load_thermodynamic_data(eps)
            if not df.empty:
                self.data[eps] = df
                self.stats[eps] = self.calculate_statistics(df, eps)
        
        print(f"\nSuccessfully loaded {len(self.data)} systems")
    
    def plot_thermodynamic_evolution(self):
        """Plot time evolution of all thermodynamic properties"""
        
        if not self.data:
            print("ERROR: No data to plot")
            return
        
        properties = ['Temp', 'Press', 'Volume', 'Density', 'TotEng']
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        for i, prop in enumerate(properties):
            ax = axes[i]
            
            for eps, color in zip(self.epsilon_values, colors):
                if eps in self.data and prop in self.data[eps].columns:
                    df = self.data[eps]
                    ax.plot(df['Time_ns'], df[prop], 
                           label=f'ε={eps:.2f}', color=color, alpha=0.7, lw=1.5)
                    
                    # Mark equilibration point
                    if eps in self.stats:
                        eq_time = self.stats[eps]['equilibration_time_ns']
                        ax.axvline(eq_time, color=color, linestyle='--', alpha=0.3, lw=1)
            
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel(prop, fontweight='bold')
            ax.set_title(f'{prop} Evolution', fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'thermodynamic_evolution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: thermodynamic_evolution.png")
        plt.close()
    
    def plot_equilibrated_distributions(self):
        """Plot probability distributions of equilibrated properties"""
        
        properties = ['Temp', 'Press', 'Density', 'TotEng']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.epsilon_values)))
        
        for i, prop in enumerate(properties):
            ax = axes[i]
            
            for eps, color in zip(self.epsilon_values, colors):
                if eps in self.data and prop in self.data[eps].columns:
                    df = self.data[eps]
                    eq_idx = self.stats[eps]['equilibration_index']
                    df_eq = df.iloc[eq_idx:]
                    
                    # Plot histogram
                    ax.hist(df_eq[prop], bins=30, alpha=0.5, color=color, 
                           label=f'ε={eps:.2f}', density=True)
                    
                    # Add mean line
                    mean_val = self.stats[eps][f'{prop}_mean']
                    ax.axvline(mean_val, color=color, linestyle='--', lw=2, alpha=0.8)
            
            ax.set_xlabel(prop, fontweight='bold')
            ax.set_ylabel('Probability Density', fontweight='bold')
            ax.set_title(f'{prop} Distribution (Equilibrated)', fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'equilibrated_distributions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: equilibrated_distributions.png")
        plt.close()
    
    def plot_epsilon_dependence(self):
        """Plot how properties depend on epsilon value"""
        
        if not self.stats:
            print("ERROR: No statistics available")
            return
        
        eps_values = []
        properties = {}
        
        for eps in self.epsilon_values:
            if eps in self.stats:
                eps_values.append(eps)
                for key, val in self.stats[eps].items():
                    if key.endswith('_mean'):
                        prop_name = key.replace('_mean', '')
                        if prop_name not in properties:
                            properties[prop_name] = {'mean': [], 'std': []}
                        properties[prop_name]['mean'].append(val)
                        properties[prop_name]['std'].append(self.stats[eps][f'{prop_name}_std'])
        
        eps_values = np.array(eps_values)
        
        # Plot key properties
        plot_props = ['Density', 'Press', 'Volume', 'TotEng', 'Temp']
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, prop in enumerate(plot_props):
            if prop in properties:
                ax = axes[i]
                means = np.array(properties[prop]['mean'])
                stds = np.array(properties[prop]['std'])
                
                ax.errorbar(eps_values, means, yerr=stds, 
                           marker='o', markersize=8, capsize=5, 
                           linewidth=2, color='darkblue', label='Mean ± Std')
                
                ax.set_xlabel('Epsilon (ε)', fontweight='bold')
                ax.set_ylabel(prop, fontweight='bold')
                ax.set_title(f'{prop} vs Epsilon', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'epsilon_dependence.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: epsilon_dependence.png")
        plt.close()
    
    def generate_statistics_table(self):
        """Generate comprehensive statistics table"""
        
        if not self.stats:
            print("ERROR: No statistics available")
            return
        
        # Create summary DataFrame
        summary_data = []
        for eps in self.epsilon_values:
            if eps in self.stats:
                s = self.stats[eps]
                summary_data.append({
                    'Epsilon': eps,
                    'Eq_Time_ns': f"{s.get('equilibration_time_ns', 0):.3f}",
                    'Temp_K': f"{s.get('Temp_mean', 0):.2f} ± {s.get('Temp_std', 0):.2f}",
                    'Press_atm': f"{s.get('Press_mean', 0):.1f} ± {s.get('Press_std', 0):.1f}",
                    'Density_g/cm3': f"{s.get('Density_mean', 0):.4f} ± {s.get('Density_std', 0):.4f}",
                    'Volume_A3': f"{s.get('Volume_mean', 0):.0f} ± {s.get('Volume_std', 0):.0f}",
                    'TotEng_kcal/mol': f"{s.get('TotEng_mean', 0):.1f} ± {s.get('TotEng_std', 0):.1f}",
                    'Stability': f"{s.get('stability_metric', 0):.4f}"
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_file = self.base_dir / 'equilibration_statistics.csv'
        df_summary.to_csv(csv_file, index=False)
        print(f"\n✓ Saved: equilibration_statistics.csv")
        
        # Print formatted table
        print("\n" + "="*120)
        print("EQUILIBRATION STATISTICS SUMMARY")
        print("="*120)
        print(df_summary.to_string(index=False))
        print("="*120)
        
        return df_summary
    
    def validate_equilibration_quality(self):
        """Validate that equilibration was successful"""
        
        print("\n" + "="*70)
        print("EQUILIBRATION QUALITY VALIDATION")
        print("="*70)
        
        validation_results = []
        
        for eps in self.epsilon_values:
            if eps not in self.stats:
                continue
            
            s = self.stats[eps]
            result = {'epsilon': eps, 'passed': True, 'issues': []}
            
            # Check 1: Temperature control (should be 300 ± 5 K)
            temp_mean = s.get('Temp_mean', 0)
            if not (295 < temp_mean < 305):
                result['passed'] = False
                result['issues'].append(f"Temperature {temp_mean:.1f} K outside 295-305 K")
            
            # Check 2: Pressure control (should be near 1 atm)
            press_mean = s.get('Press_mean', 0)
            press_std = s.get('Press_std', 0)
            if abs(press_mean) > 500:  # Allow ±500 atm
                result['passed'] = False
                result['issues'].append(f"Pressure {press_mean:.1f} atm far from target")
            
            # Check 3: Density (should be liquid water ~1.0 g/cm³)
            dens_mean = s.get('Density_mean', 0)
            if not (0.90 < dens_mean < 1.10):
                result['passed'] = False
                result['issues'].append(f"Density {dens_mean:.3f} g/cm³ outside liquid range")
            
            # Check 4: Stability (fluctuations should be small)
            stability = s.get('stability_metric', 1.0)
            if stability > 0.1:
                result['passed'] = False
                result['issues'].append(f"High fluctuations (stability={stability:.4f})")
            
            # Check 5: Sufficient equilibration time
            eq_time = s.get('equilibration_time_ns', 0)
            if eq_time < 0.1:
                result['passed'] = False
                result['issues'].append(f"Insufficient equilibration time ({eq_time:.3f} ns)")
            
            validation_results.append(result)
            
            # Print results
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            print(f"\nε={eps:.2f}: {status}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"  - {issue}")
            else:
                print(f"  All quality checks passed!")
                print(f"  T={temp_mean:.2f}K, P={press_mean:.1f}atm, ρ={dens_mean:.4f}g/cm³")
        
        print("="*70)
        
        return validation_results


def main():
    """Main analysis workflow"""
    
    print("\n" + "="*70)
    print("LAMMPS EQUILIBRATION ANALYSIS")
    print("Nanoparticle Solvation Study - Epsilon Dependence")
    print("="*70)
    
    # Initialize analyzer
    analyzer = EquilibrationAnalyzer(base_dir=".")
    
    # Load all data
    analyzer.analyze_all_systems()
    
    # Generate statistics table
    summary_df = analyzer.generate_statistics_table()
    
    # Validate equilibration quality
    validation_results = analyzer.validate_equilibration_quality()
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70)
    
    analyzer.plot_thermodynamic_evolution()
    analyzer.plot_equilibrated_distributions()
    analyzer.plot_epsilon_dependence()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - equilibration_statistics.csv")
    print("  - thermodynamic_evolution.png")
    print("  - equilibrated_distributions.png")
    print("  - epsilon_dependence.png")
    print("="*70 + "\n")
    
    return analyzer, summary_df, validation_results


if __name__ == "__main__":
    analyzer, summary, validation = main()
