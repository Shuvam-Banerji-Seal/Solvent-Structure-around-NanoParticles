#!/usr/bin/env python3
"""
MODULE 12: EQUILIBRATION CONVERGENCE ANALYSIS
==============================================

Tracks thermodynamic convergence through all 4 equilibration stages.
Analyzes temperature, pressure, density, and energy stabilization.

Stages:
1. NVT Thermalization (50 ps): Heating to 300K
2. Pre-Equilibration (50 ps): Structure relaxation
3. Pressure Ramp (100 ps): P = 0 -> 1 atm
4. NPT Equilibration (1000 ps): Density stabilization

Output: 6 plots, 1 CSV file

Author: AI Analysis Suite
Date: 2024-11-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

class EquilibrationConvergenceAnalyzer:
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
        self.data_dir = self.base_dir / 'analysis' / 'data'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_thermo_file(self, thermo_file):
        """Parse LAMMPS thermo output file"""
        try:
            with open(thermo_file, 'r') as f:
                lines = f.readlines()
            
            # Find header line
            header_idx = None
            for i, line in enumerate(lines):
                if 'Step' in line and 'Temp' in line:
                    header_idx = i
                    break
            
            if header_idx is None:
                return None
            
            # Parse header to get column names
            header = lines[header_idx].strip().split()
            
            # Parse data lines
            data = []
            for line in lines[header_idx + 1:]:
                if line.strip() and not line.startswith('#'):
                    try:
                        values = list(map(float, line.strip().split()))
                        data.append(values)
                    except ValueError:
                        continue
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=header)
            return df
        except Exception as e:
            print(f"      Error parsing {thermo_file.name}: {e}")
            return None
    
    def analyze_equilibration_stages(self):
        """Analyze convergence of all 4 stages for one epsilon"""
        print("\n" + "="*80)
        print("ANALYZING EQUILIBRATION CONVERGENCE")
        print("="*80)
        
        eps = 0.0  # Use epsilon=0.0 as reference
        eps_dir = self.base_dir / self.epsilon_dirs[eps]
        
        stages = {
            'NVT': 'nvt_thermalization_thermo.dat',
            'Pre-Eq': 'pre_equilibration_thermo.dat',
            'Pressure': 'pressure_ramp_thermo.dat',
            'NPT': 'npt_equilibration_thermo.dat'
        }
        
        stage_data = {}
        
        print(f"\n[ε={eps}] Parsing thermodynamic files...")
        for stage_name, filename in stages.items():
            thermo_file = eps_dir / filename
            print(f"  {stage_name}: ", end='', flush=True)
            
            if not thermo_file.exists():
                print(f"✗ Not found")
                continue
            
            df = self.parse_thermo_file(thermo_file)
            if df is None:
                print(f"✗ Parse error")
                continue
            
            stage_data[stage_name] = df
            print(f"✓ ({len(df)} steps)")
        
        if not stage_data:
            print("  ✗ No valid thermodynamic data found")
            return
        
        # Create comprehensive plots
        self._plot_temperature_evolution(stage_data, eps)
        self._plot_pressure_evolution(stage_data, eps)
        self._plot_density_evolution(stage_data, eps)
        self._plot_energy_evolution(stage_data, eps)
        self._plot_convergence_comparison(stage_data, eps)
        self._plot_kinetic_vs_potential(stage_data, eps)
        
        # Save convergence metrics
        self._save_convergence_metrics(stage_data, eps)
        
    def _plot_temperature_evolution(self, stage_data, eps):
        """Plot temperature convergence through stages"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        stage_names = list(stage_data.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            if 'Temp' not in df.columns:
                continue
            
            temps = df['Temp'].values
            steps = df['Step'].values
            
            # Convert steps to time (assuming 1 fs/step)
            times = (steps - steps[0]) / 1000.0  # Convert to ps
            
            ax.plot(times, temps, color=colors[idx], linewidth=2, label=stage_name)
            ax.axhline(y=300, color='k', linestyle='--', alpha=0.3, label='Target: 300 K')
            ax.fill_between(times, 295, 305, alpha=0.1, color='gray')
            
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Temperature (K)')
            ax.set_title(f'{stage_name} Thermalization')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Calculate convergence metrics
            t_avg = temps[-100:].mean() if len(temps) > 100 else temps.mean()
            t_std = temps[-100:].std() if len(temps) > 100 else temps.std()
            ax.text(0.98, 0.05, f'Final: {t_avg:.1f}±{t_std:.1f} K',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Temperature Evolution: ε={eps:.2f} kcal/mol', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'31_temperature_convergence_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 31_temperature_convergence_eps{eps:.2f}.png")
    
    def _plot_pressure_evolution(self, stage_data, eps):
        """Plot pressure convergence through stages"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            if 'Press' not in df.columns:
                continue
            
            press = df['Press'].values
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            ax.plot(times, press, color=colors[idx], linewidth=2, label=stage_name)
            if stage_name in ['Pressure', 'NPT']:
                ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Target: 1 atm')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Pressure (atm)')
            ax.set_title(f'{stage_name} Pressure')
            ax.legend()
            ax.grid(alpha=0.3)
            
            p_avg = press[-100:].mean() if len(press) > 100 else press.mean()
            p_std = press[-100:].std() if len(press) > 100 else press.std()
            ax.text(0.98, 0.05, f'Final: {p_avg:.2f}±{p_std:.2f} atm',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(f'Pressure Evolution: ε={eps:.2f} kcal/mol', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'32_pressure_convergence_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 32_pressure_convergence_eps{eps:.2f}.png")
    
    def _plot_density_evolution(self, stage_data, eps):
        """Plot density convergence through stages"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            if 'Volume' not in df.columns:
                continue
            
            volume = df['Volume'].values
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            # Calculate density (5541 atoms, MW of system)
            # Approximate density from volume
            density = 5541.0 / volume * 1.66054  # g/cm^3 (rough conversion)
            
            ax.plot(times, density, color=colors[idx], linewidth=2, label=stage_name)
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Water density')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Density (g/cm³)')
            ax.set_title(f'{stage_name} Density')
            ax.legend()
            ax.grid(alpha=0.3)
            
            rho_avg = density[-100:].mean() if len(density) > 100 else density.mean()
            rho_std = density[-100:].std() if len(density) > 100 else density.std()
            ax.text(0.98, 0.05, f'Final: {rho_avg:.3f}±{rho_std:.3f} g/cm³',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.suptitle(f'Density Evolution: ε={eps:.2f} kcal/mol', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'33_density_convergence_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 33_density_convergence_eps{eps:.2f}.png")
    
    def _plot_energy_evolution(self, stage_data, eps):
        """Plot total energy convergence"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            if 'TotEng' not in df.columns:
                continue
            
            energy = df['TotEng'].values
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            ax.plot(times, energy, color=colors[idx], linewidth=2, label=stage_name)
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Total Energy (kcal/mol)')
            ax.set_title(f'{stage_name} Total Energy')
            ax.legend()
            ax.grid(alpha=0.3)
            
            e_avg = energy[-100:].mean() if len(energy) > 100 else energy.mean()
            e_std = energy[-100:].std() if len(energy) > 100 else energy.std()
            ax.text(0.98, 0.05, f'Final: {e_avg:.1f}±{e_std:.1f}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.suptitle(f'Total Energy Evolution: ε={eps:.2f} kcal/mol', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'34_energy_convergence_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 34_energy_convergence_eps{eps:.2f}.png")
    
    def _plot_convergence_comparison(self, stage_data, eps):
        """Plot all stages on single figure for comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        stage_colors = {'NVT': '#1f77b4', 'Pre-Eq': '#ff7f0e', 
                       'Pressure': '#2ca02c', 'NPT': '#d62728'}
        
        # Temperature comparison
        ax = axes[0, 0]
        for stage_name, df in stage_data.items():
            if 'Temp' in df.columns:
                temps = df['Temp'].values
                # Normalize time within stage
                times = np.linspace(0, 1, len(temps))
                ax.plot(times, temps, color=stage_colors[stage_name], 
                       linewidth=2, label=stage_name, alpha=0.7)
        ax.set_xlabel('Normalized Stage Progress')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Temperature Convergence (Normalized)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Pressure comparison
        ax = axes[0, 1]
        for stage_name, df in stage_data.items():
            if 'Press' in df.columns:
                press = df['Press'].values
                times = np.linspace(0, 1, len(press))
                ax.plot(times, press, color=stage_colors[stage_name],
                       linewidth=2, label=stage_name, alpha=0.7)
        ax.set_xlabel('Normalized Stage Progress')
        ax.set_ylabel('Pressure (atm)')
        ax.set_title('Pressure Convergence (Normalized)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Energy comparison
        ax = axes[1, 0]
        for stage_name, df in stage_data.items():
            if 'TotEng' in df.columns:
                energy = df['TotEng'].values
                times = np.linspace(0, 1, len(energy))
                ax.plot(times, energy, color=stage_colors[stage_name],
                       linewidth=2, label=stage_name, alpha=0.7)
        ax.set_xlabel('Normalized Stage Progress')
        ax.set_ylabel('Total Energy (kcal/mol)')
        ax.set_title('Energy Convergence (Normalized)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Convergence summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_data = []
        for stage_name, df in stage_data.items():
            row = [stage_name]
            if 'Temp' in df.columns:
                t_final = df['Temp'].values[-1]
                t_avg = df['Temp'].values[-100:].mean() if len(df) > 100 else df['Temp'].values.mean()
                row.append(f'{t_avg:.1f} K')
            if 'Press' in df.columns:
                p_avg = df['Press'].values[-100:].mean() if len(df) > 100 else df['Press'].values.mean()
                row.append(f'{p_avg:.2f} atm')
            summary_data.append(row)
        
        table = ax.table(cellText=summary_data, 
                        colLabels=['Stage', 'Avg T', 'Avg P'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Equilibration Summary', pad=20, fontweight='bold')
        
        plt.suptitle(f'Equilibration Convergence Overview: ε={eps:.2f} kcal/mol', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'35_convergence_comparison_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 35_convergence_comparison_eps{eps:.2f}.png")
    
    def _plot_kinetic_vs_potential(self, stage_data, eps):
        """Plot kinetic and potential energy separately"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (stage_name, df) in enumerate(stage_data.items()):
            ax = axes.flatten()[idx]
            
            has_ke = 'KinEng' in df.columns
            has_pe = 'PotEng' in df.columns
            
            if not (has_ke or has_pe):
                continue
            
            steps = df['Step'].values
            times = (steps - steps[0]) / 1000.0
            
            if has_ke:
                ke = df['KinEng'].values
                ax.plot(times, ke, color=colors[idx], linewidth=2, label='Kinetic', alpha=0.7)
            
            if has_pe:
                pe = df['PotEng'].values
                ax.plot(times, pe, color=colors[idx], linewidth=2, linestyle='--', 
                       label='Potential', alpha=0.7)
            
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Energy (kcal/mol)')
            ax.set_title(f'{stage_name} K.E. vs P.E.')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Kinetic vs Potential Energy: ε={eps:.2f} kcal/mol', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / f'36_kinetic_vs_potential_eps{eps:.2f}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 36_kinetic_vs_potential_eps{eps:.2f}.png")
    
    def _save_convergence_metrics(self, stage_data, eps):
        """Save convergence metrics to CSV"""
        metrics = []
        
        for stage_name, df in stage_data.items():
            metric = {'Stage': stage_name}
            
            # Temperature metrics
            if 'Temp' in df.columns:
                temps = df['Temp'].values
                metric['T_initial'] = temps[0]
                metric['T_final'] = temps[-1]
                metric['T_avg_last100'] = temps[-100:].mean() if len(temps) > 100 else temps.mean()
                metric['T_std_last100'] = temps[-100:].std() if len(temps) > 100 else temps.std()
            
            # Pressure metrics
            if 'Press' in df.columns:
                press = df['Press'].values
                metric['P_initial'] = press[0]
                metric['P_final'] = press[-1]
                metric['P_avg_last100'] = press[-100:].mean() if len(press) > 100 else press.mean()
                metric['P_std_last100'] = press[-100:].std() if len(press) > 100 else press.std()
            
            # Energy metrics
            if 'TotEng' in df.columns:
                energy = df['TotEng'].values
                metric['E_initial'] = energy[0]
                metric['E_final'] = energy[-1]
                metric['E_avg_last100'] = energy[-100:].mean() if len(energy) > 100 else energy.mean()
            
            metrics.append(metric)
        
        df_metrics = pd.DataFrame(metrics)
        csv_file = self.data_dir / f'convergence_metrics_eps{eps:.2f}.csv'
        df_metrics.to_csv(csv_file, index=False)
        print(f"  ✓ Saved metrics: convergence_metrics_eps{eps:.2f}.csv")

def main():
    print("="*80)
    print("MODULE 12: EQUILIBRATION CONVERGENCE ANALYSIS")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    analyzer = EquilibrationConvergenceAnalyzer(base_dir)
    
    try:
        analyzer.analyze_equilibration_stages()
        
        print("\n" + "="*80)
        print("✓ MODULE 12 COMPLETE!")
        print("="*80)
        print(f"\n📊 Generated plots: 31-36 (6 plots)")
        print(f"📊 Generated data: convergence_metrics_*.csv")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
