#!/usr/bin/env python3
"""
MODULE 15: THERMAL TRAJECTORY ANALYSIS
======================================

Analyzes thermalization dynamics and energy conservation across stages.
Tracks system approach to equilibrium through temperature/energy trajectories.

Analyses:
- NVT thermalization: Cooling/heating to 300K
- Pre-equilibration: Thermal relaxation
- Pressure ramp: Isothermal-isobaric transition
- Thermal equilibrium: NPT production stability

Output: 3 plots, 1 CSV file

Author: AI Analysis Suite
Date: 2024-11-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

class ThermalTrajectoryAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.epsilon_dirs = {
            0.0: 'epsilon_0.0',
            0.05: 'epsilon_0.05',
            0.1: 'epsilon_0.10',
            0.15: 'epsilon_0.15',
            0.2: 'epsilon_0.20',
            0.25: 'epsilon_0.25',
            0.3: 'epsilon_0.30',
            0.35: 'epsilon_0.35',
            0.4: 'epsilon_0.40',
            0.45: 'epsilon_0.45',
            0.5: 'epsilon_0.50'
        }
        self.plots_dir = self.base_dir / 'analysis' / 'plots'
        self.data_dir = self.base_dir / 'analysis' / 'data'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_thermo_file(self, thermo_file):
        """Parse LAMMPS thermo file (handles both thermo_style and fix ave/time formats)"""
        try:
            with open(thermo_file, 'r') as f:
                lines = f.readlines()
            
            # Find header line - could be 'Step Temp ...' or '# TimeStep v_epsilon_co v_temp ...'
            header_idx = None
            header = None
            for i, line in enumerate(lines):
                # Check for fix ave/time header format
                if line.startswith('# TimeStep') or line.startswith('#TimeStep'):
                    header_parts = line.strip('#').strip().split()
                    # Map v_temp -> Temp, v_press -> Press, etc.
                    header = []
                    for col in header_parts:
                        if col.startswith('v_'):
                            col_name = col[2:].capitalize()
                            if col_name == 'Temp': col_name = 'Temp'
                            elif col_name == 'Press': col_name = 'Press'
                            elif col_name == 'Pe': col_name = 'PotEng'
                            elif col_name == 'Ke': col_name = 'KinEng'
                            elif col_name == 'Etotal': col_name = 'TotEng'
                            elif col_name == 'Vol': col_name = 'Volume'
                            elif col_name == 'Dens': col_name = 'Density'
                            elif col_name == 'Epsilon_co': col_name = 'Epsilon'
                            header.append(col_name)
                        else:
                            header.append(col)
                    header_idx = i
                    break
                # Check for thermo_style custom format
                elif 'Step' in line and 'Temp' in line:
                    header = line.strip().split()
                    header_idx = i
                    break
            
            if header_idx is None or not header:
                return None
            
            # Parse data lines
            data = []
            for line in lines[header_idx + 1:]:
                if line.strip() and not line.startswith('#'):
                    try:
                        values = list(map(float, line.strip().split()))
                        if len(values) == len(header):
                            data.append(values)
                    except ValueError:
                        continue
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=header)
            if 'TimeStep' in df.columns and 'Step' not in df.columns:
                df.rename(columns={'TimeStep': 'Step'}, inplace=True)
            return df
        except Exception as e:
            return None
    
    def analyze_nvt_thermalization(self):
        """Analyze NVT thermalization stage"""
        print("\n" + "="*80)
        print("ANALYZING NVT THERMALIZATION STAGE")
        print("="*80)
        
        eps = 0.0
        eps_dir = self.base_dir / self.epsilon_dirs[eps]
        thermo_file = eps_dir / 'nvt_thermalization_thermo.dat'
        
        print(f"\n[ε={eps}] NVT Stage: {thermo_file.name}", end='', flush=True)
        
        if not thermo_file.exists():
            print(" ✗ Not found")
            return None
        
        df = self.parse_thermo_file(thermo_file)
        if df is None:
            print(" ✗ Parse error")
            return None
        
        print(" ✓")
        
        if 'Temp' not in df.columns:
            return None
        
        temps = df['Temp'].values
        steps = df['Step'].values
        times = (steps - steps[0]) / 1000.0  # Convert to ps
        
        # Calculate thermalization metrics
        target_t = 300.0
        initial_t = temps[0]
        final_t = temps[-1]
        
        # Find time to reach 95% of target
        diff_from_target = np.abs(temps - target_t)
        within_5pct = temps[-100:].std() / temps[-100:].mean() < 0.05
        
        convergence_time = None
        for i, temp in enumerate(temps):
            if np.abs(temp - target_t) < 0.05 * target_t:
                convergence_time = times[i]
                break
        
        print(f"  Initial T: {initial_t:.1f} K")
        print(f"  Final T: {final_t:.1f} K")
        print(f"  Target T: {target_t:.1f} K")
        if convergence_time:
            print(f"  Convergence time (5%): {convergence_time:.2f} ps")
        print(f"  Final std dev: {temps[-100:].std():.2f} K")
        
        return {
            'times': times,
            'temps': temps,
            'target_t': target_t,
            'convergence_time': convergence_time
        }
    
    def analyze_thermal_convergence(self):
        """Analyze thermal convergence across all stages"""
        print("\n" + "="*80)
        print("ANALYZING THERMAL CONVERGENCE")
        print("="*80)
        
        stages = {
            'NVT': 'nvt_thermalization_thermo.dat',
            'Pre-Eq': 'pre_equilibration_thermo.dat',
            'Pressure': 'pressure_ramp_thermo.dat',
            'NPT': 'npt_equilibration_thermo.dat'
        }
        
        eps = 0.0
        eps_dir = self.base_dir / self.epsilon_dirs[eps]
        
        stage_data = {}
        
        print(f"\n[ε={eps}] Parsing stages...")
        for stage_name, filename in stages.items():
            thermo_file = eps_dir / filename
            print(f"  {stage_name}: ", end='', flush=True)
            # Flexible discovery if exact filename missing
            if not thermo_file.exists():
                candidates = []
                if stage_name == 'NPT':
                    candidates = list(eps_dir.glob('npt*thermo*.dat'))
                else:
                    stage_key = stage_name.lower().replace('-', '_').replace(' ', '_')
                    candidates = list(eps_dir.glob(f'*{stage_key}*thermo*.dat'))
                
                # Filter out production files
                candidates = [p for p in candidates if p.is_file() and 'production' not in p.name.lower()]
                
                if candidates:
                    thermo_file = candidates[0]
                    print(f"(using {thermo_file.name}) ", end='', flush=True)
                else:
                    print("✗ Not found")
                    continue
            
            df = self.parse_thermo_file(thermo_file)
            if df is None:
                print("✗ Parse error")
                continue
            
            print(f"✓ ({len(df)} steps)")
            stage_data[stage_name] = df
        
        if not stage_data:
            return None
        
        # Create multi-panel thermal convergence plot
        self._plot_thermal_convergence(stage_data, eps)
        self._plot_energy_conservation(stage_data, eps)
        self._plot_thermalization_metrics(stage_data, eps)
        
        return stage_data
    
    def _plot_thermal_convergence(self, stage_data, eps):
        """Plot thermal convergence through all stages"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        stage_names = list(stage_data.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            if 'Temp' not in df.columns:
                continue
            
            temps = df['Temp'].values
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            # Plot temperature trajectory
            ax.plot(times, temps, color=colors[idx], linewidth=2, label='Temperature')
            
            # Add target temperature band
            if stage_name == 'Pressure':
                ax.axhline(y=300, color='k', linestyle='--', alpha=0.3)
            else:
                ax.fill_between(times, 295, 305, alpha=0.15, color='gray', label='Target region')
            
            # Add running average (window=100)
            if len(temps) > 100:
                running_avg = pd.Series(temps).rolling(window=100).mean().values
                ax.plot(times, running_avg, color=colors[idx], linewidth=1.5, 
                       linestyle='--', alpha=0.7, label='Running avg (100 steps)')
            
            ax.set_xlabel('Time (ps)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Temperature (K)', fontsize=11, fontweight='bold')
            ax.set_title(f'{stage_name} Thermalization', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            
            # Add statistics box
            t_final = temps[-1]
            t_std = temps[-100:].std() if len(temps) > 100 else temps.std()
            t_mean = temps[-100:].mean() if len(temps) > 100 else temps.mean()
            
            stats_text = f'Final: {t_final:.1f} K\nMean(last100): {t_mean:.1f} K\nStd: {t_std:.2f} K'
            ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, 
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.suptitle(f'Thermal Convergence Through Equilibration Stages: eps={eps:.2f}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.plots_dir / f'42_thermal_convergence_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 42_thermal_convergence_eps{eps:.2f}.png")
    
    def _plot_energy_conservation(self, stage_data, eps):
        """Plot energy conservation during equilibration"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            # Total energy
            if 'TotEng' in df.columns:
                tot_eng = df['TotEng'].values
                ax.plot(times, tot_eng, color=colors[idx], linewidth=2, label='Total Energy', alpha=0.8)
                
                # Add trend
                if len(times) > 10:
                    z = np.polyfit(times, tot_eng, 2)
                    p = np.poly1d(z)
                    ax.plot(times, p(times), color=colors[idx], linestyle='--', 
                           linewidth=1.5, alpha=0.5, label='Trend')
            
            # Potential energy if available
            if 'PotEng' in df.columns:
                pot_eng = df['PotEng'].values
                ax2 = ax.twinx()
                ax2.plot(times, pot_eng, color='purple', linewidth=1.5, 
                        linestyle=':', label='Potential Energy', alpha=0.7)
                ax2.set_ylabel('Potential Energy (kcal/mol)', fontsize=10, fontweight='bold', color='purple')
            
            ax.set_xlabel('Time (ps)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Total Energy (kcal/mol)', fontsize=11, fontweight='bold')
            ax.set_title(f'{stage_name} Energy Conservation', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
            
            # Add energy drift
            if 'TotEng' in df.columns:
                e_initial = tot_eng[0]
                e_final = tot_eng[-1]
                e_drift = (e_final - e_initial) / np.abs(e_initial) * 100
                drift_text = f'Energy drift: {e_drift:.2f}%'
                ax.text(0.98, 0.95, drift_text, transform=ax.transAxes,
                       fontsize=9, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle(f'Energy Conservation During Equilibration: eps={eps:.2f}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.plots_dir / f'43_energy_conservation_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 43_energy_conservation_eps{eps:.2f}.png")
    
    def _plot_thermalization_metrics(self, stage_data, eps):
        """Plot thermalization metrics summary"""
        fig = plt.figure(figsize=(12, 8))
        
        # Create grid for various plots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Temperature evolution normalized
        ax = fig.add_subplot(gs[0, 0])
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            if 'Temp' not in df.columns:
                continue
            
            temps = df['Temp'].values
            # Normalize 0-1 over stage
            t_norm = np.linspace(0, 1, len(temps))
            
            ax.plot(t_norm, temps, label=stage_name, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Normalized Stage Progress', fontsize=11, fontweight='bold')
        ax.set_ylabel('Temperature (K)', fontsize=11, fontweight='bold')
        ax.set_title('Temperature Evolution (Normalized)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(y=300, color='k', linestyle='--', alpha=0.2)
        
        # Plot 2: Temperature stability (std dev over time)
        ax = fig.add_subplot(gs[0, 1])
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            if 'Temp' not in df.columns:
                continue
            
            temps = df['Temp'].values
            window = max(50, len(temps)//20)
            
            # Calculate rolling std dev
            std_devs = []
            for i in range(len(temps) - window):
                std_devs.append(temps[i:i+window].std())
            
            t_norm = np.linspace(0, 1, len(std_devs))
            ax.plot(t_norm, std_devs, label=stage_name, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Normalized Stage Progress', fontsize=11, fontweight='bold')
        ax.set_ylabel('Temperature Std Dev (K)', fontsize=11, fontweight='bold')
        ax.set_title('Temperature Stability', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Plot 3: Comparison table
        ax = fig.add_subplot(gs[1, :])
        ax.axis('off')
        
        # Build metrics table
        metrics_data = []
        for stage_name, df in stage_data.items():
            row = [stage_name]
            
            if 'Temp' in df.columns:
                temps = df['Temp'].values
                t_final = temps[-1]
                t_std = temps[-100:].std() if len(temps) > 100 else temps.std()
                row.extend([f'{t_final:.1f}', f'{t_std:.2f}'])
            
            if 'Press' in df.columns:
                press = df['Press'].values
                p_final = press[-1]
                p_std = press[-100:].std() if len(press) > 100 else press.std()
                row.extend([f'{p_final:.2f}', f'{p_std:.2f}'])
            
            metrics_data.append(row)
        
        table = ax.table(cellText=metrics_data,
                        colLabels=['Stage', 'T_final (K)', 'T_std (K)', 'P_final (atm)', 'P_std (atm)'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color code table
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[i])):
                cell = table[(i+1, j)]
                if j > 0:  # Data cells
                    cell.set_facecolor('#E8F4F8')
        
        ax.text(0.5, 0.95, 'Thermalization Metrics Summary',
               transform=ax.transAxes, ha='center', fontweight='bold', fontsize=12)
        
        plt.suptitle(f'Thermalization Dynamics Analysis: eps={eps:.2f}',
                    fontsize=13, fontweight='bold')
        
        plot_file = self.plots_dir / f'44_thermalization_metrics_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 44_thermalization_metrics_eps{eps:.2f}.png")
    
    def save_thermal_analysis_data(self, stage_data, eps):
        """Save thermal analysis data to CSV"""
        analysis_metrics = []
        
        for stage_name, df in stage_data.items():
            metric = {'Stage': stage_name}
            
            if 'Temp' in df.columns:
                temps = df['Temp'].values
                metric['T_initial'] = temps[0]
                metric['T_final'] = temps[-1]
                metric['T_mean_last100'] = temps[-100:].mean() if len(temps) > 100 else temps.mean()
                metric['T_std_last100'] = temps[-100:].std() if len(temps) > 100 else temps.std()
            
            if 'Press' in df.columns:
                press = df['Press'].values
                metric['P_initial'] = press[0]
                metric['P_final'] = press[-1]
                metric['P_mean_last100'] = press[-100:].mean() if len(press) > 100 else press.mean()
            
            if 'TotEng' in df.columns:
                eng = df['TotEng'].values
                metric['E_initial'] = eng[0]
                metric['E_final'] = eng[-1]
                metric['E_drift_pct'] = (eng[-1] - eng[0]) / np.abs(eng[0]) * 100
            
            analysis_metrics.append(metric)
        
        df_metrics = pd.DataFrame(analysis_metrics)
        csv_file = self.data_dir / f'thermal_analysis_metrics_eps{eps:.2f}.csv'
        df_metrics.to_csv(csv_file, index=False)
        print(f"  ✓ Saved: thermal_analysis_metrics_eps{eps:.2f}.csv")

def main():
    print("="*80)
    print("MODULE 15: THERMAL TRAJECTORY ANALYSIS")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    analyzer = ThermalTrajectoryAnalyzer(base_dir)
    
    try:
        nvt_data = analyzer.analyze_nvt_thermalization()
        stage_data = analyzer.analyze_thermal_convergence()
        
        if stage_data:
            analyzer.save_thermal_analysis_data(stage_data, 0.0)
        
        print("\n" + "="*80)
        print("✓ MODULE 15 COMPLETE!")
        print("="*80)
        print(f"\n📊 Generated plots: 42-44 (3 plots)")
        print(f"📊 Generated data: thermal_analysis_metrics_*.csv")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
