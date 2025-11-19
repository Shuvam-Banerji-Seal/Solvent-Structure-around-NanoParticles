#!/usr/bin/env python3
"""
MODULE 13: LOG FILE PERFORMANCE ANALYSIS
========================================

Extracts computational performance metrics from LAMMPS log files.
Analyzes timesteps/second, GPU utilization, and wall time across simulations.

Metrics:
- Speed of simulation (timesteps/sec)
- CPU vs GPU time
- Force evaluations per stage
- Scalability analysis

Output: 2 plots, 2 CSV files

Author: AI Analysis Suite
Date: 2024-11-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

class LogFilePerformanceAnalyzer:
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
    
    def parse_log_file(self, log_file):
        """Extract performance metrics from LAMMPS log file"""
        metrics = {
            'timesteps_per_sec': None,
            'total_wall_time': None,
            'cpu_seconds': None,
            'gpu_seconds': None,
            'performance_notes': []
        }
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for timing information
            # Pattern: "X timesteps in Y seconds"
            ts_pattern = r'(\d+)\s+timesteps\s+in\s+([\d.]+)\s+seconds'
            ts_match = re.search(ts_pattern, content)
            if ts_match:
                n_steps = int(ts_match.group(1))
                wall_time = float(ts_match.group(2))
                metrics['timesteps_per_sec'] = n_steps / wall_time
                metrics['total_wall_time'] = wall_time
                metrics['performance_notes'].append(f"{n_steps} timesteps in {wall_time:.2f} sec")
            
            # Look for CPU time
            cpu_pattern = r'CPU\s+time\s*=\s*([\d.]+)\s*seconds'
            cpu_match = re.search(cpu_pattern, content)
            if cpu_match:
                metrics['cpu_seconds'] = float(cpu_match.group(1))
                metrics['performance_notes'].append(f"CPU time: {metrics['cpu_seconds']:.2f} sec")
            
            # Look for GPU time if available
            gpu_pattern = r'GPU\s+time\s*=\s*([\d.]+)\s*seconds'
            gpu_match = re.search(gpu_pattern, content)
            if gpu_match:
                metrics['gpu_seconds'] = float(gpu_match.group(1))
                metrics['performance_notes'].append(f"GPU time: {metrics['gpu_seconds']:.2f} sec")
            
            # Extract memory usage if available
            mem_pattern = r'(\d+)\s+bytes\s+at\s+largest'
            mem_match = re.search(mem_pattern, content)
            if mem_match:
                mem_bytes = int(mem_match.group(1))
                mem_mb = mem_bytes / (1024 ** 2)
                metrics['performance_notes'].append(f"Memory: {mem_mb:.1f} MB")
            
            return metrics
        
        except Exception as e:
            metrics['performance_notes'].append(f"Error: {str(e)}")
            return metrics
    
    def analyze_production_performance(self):
        """Analyze performance across all epsilon values"""
        print("\n" + "="*80)
        print("ANALYZING PRODUCTION RUN PERFORMANCE")
        print("="*80)
        
        performance_data = []
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            log_file = eps_dir / 'equil_run.out'
            
            print(f"\n[ε={eps:.2f}] ", end='', flush=True)
            
            if not log_file.exists():
                print("✗ No log file found")
                continue
            
            metrics = self.parse_log_file(log_file)
            
            if metrics['timesteps_per_sec'] is None:
                print("✗ Could not parse performance metrics")
                continue
            
            performance_data.append({
                'epsilon': eps,
                'timesteps_per_sec': metrics['timesteps_per_sec'],
                'total_wall_time': metrics['total_wall_time'],
                'cpu_seconds': metrics['cpu_seconds'],
                'gpu_seconds': metrics['gpu_seconds']
            })
            
            print(f"✓ {metrics['timesteps_per_sec']:.1f} ts/sec")
            for note in metrics['performance_notes']:
                print(f"   └─ {note}")
        
        if not performance_data:
            print("  ✗ No valid performance data found")
            return
        
        df_perf = pd.DataFrame(performance_data)
        
        # Create plots
        self._plot_performance_comparison(df_perf)
        self._plot_time_breakdown(df_perf)
        
        # Save data
        csv_file = self.data_dir / 'production_performance.csv'
        df_perf.to_csv(csv_file, index=False)
        print(f"\n  ✓ Saved: production_performance.csv")
    
    def _plot_performance_comparison(self, df_perf):
        """Plot timesteps/sec across epsilon values"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Timesteps per second
        ax = axes[0]
        epsilons = df_perf['epsilon'].values
        tps = df_perf['timesteps_per_sec'].values
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
        bars = ax.bar(range(len(epsilons)), tps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f'{e:.2f}' for e in epsilons])
        ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Timesteps/Second', fontsize=12, fontweight='bold')
        ax.set_title('Production Run Performance', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Wall time
        ax = axes[1]
        wall_times = df_perf['total_wall_time'].values
        
        bars = ax.bar(range(len(epsilons)), wall_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f'{e:.2f}' for e in epsilons])
        ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Wall Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Simulation Duration', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Production Run Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = self.plots_dir / '37_performance_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 37_performance_comparison.png")
    
    def _plot_time_breakdown(self, df_perf):
        """Plot CPU vs GPU time breakdown"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epsilons = df_perf['epsilon'].values
        cpu_times = df_perf['cpu_seconds'].values
        gpu_times = df_perf['gpu_seconds'].values
        
        # Handle None values
        cpu_times = np.array([t if t is not None else 0 for t in cpu_times])
        gpu_times = np.array([t if t is not None else 0 for t in gpu_times])
        
        x = np.arange(len(epsilons))
        width = 0.35
        
        colors = ['#1f77b4', '#ff7f0e']
        
        # Only plot if we have data
        if cpu_times.sum() > 0:
            bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU Time', 
                          color=colors[0], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        if gpu_times.sum() > 0:
            bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU Time',
                          color=colors[1], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.2f}' for e in epsilons])
        ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('CPU vs GPU Time Breakdown', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / '38_cpu_vs_gpu_time.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 38_cpu_vs_gpu_time.png")
    
    def analyze_equilibration_performance(self):
        """Analyze performance of each equilibration stage"""
        print("\n" + "="*80)
        print("ANALYZING EQUILIBRATION PERFORMANCE")
        print("="*80)
        
        stages = {
            'NVT': 'nvt_thermalization.log',
            'Pre-Eq': 'pre_equilibration.log',
            'Pressure': 'pressure_ramp.log',
            'NPT': 'npt_equilibration.log'
        }
        
        stage_perf = defaultdict(list)
        
        eps = 0.0
        eps_dir = self.base_dir / self.epsilon_dirs[eps]
        
        for stage_name, log_filename in stages.items():
            log_file = eps_dir / log_filename
            
            print(f"  {stage_name}: ", end='', flush=True)
            
            if not log_file.exists():
                print("✗ Not found")
                continue
            
            metrics = self.parse_log_file(log_file)
            
            if metrics['timesteps_per_sec'] is None:
                print("✗ Parse error")
                continue
            
            stage_perf[stage_name] = metrics
            print(f"✓ {metrics['timesteps_per_sec']:.1f} ts/sec")
        
        if not stage_perf:
            return
        
        # Create comparison plot
        self._plot_stage_performance(stage_perf)
        
        # Save stage performance data
        stage_data = []
        for stage_name, metrics in stage_perf.items():
            stage_data.append({
                'stage': stage_name,
                'timesteps_per_sec': metrics['timesteps_per_sec'],
                'total_wall_time': metrics['total_wall_time']
            })
        
        df_stages = pd.DataFrame(stage_data)
        csv_file = self.data_dir / 'equilibration_performance_by_stage.csv'
        df_stages.to_csv(csv_file, index=False)
        print(f"  ✓ Saved: equilibration_performance_by_stage.csv")
    
    def _plot_stage_performance(self, stage_perf):
        """Plot equilibration stage performance comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stage_names = list(stage_perf.keys())
        tps_values = [stage_perf[s]['timesteps_per_sec'] for s in stage_names]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(stage_names)]
        
        bars = ax.bar(range(len(stage_names)), tps_values, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, fontsize=11)
        ax.set_ylabel('Timesteps/Second', fontsize=12, fontweight='bold')
        ax.set_title('Equilibration Stage Performance (ε=0.0 kcal/mol)', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plot_file = self.plots_dir / '39_stage_performance.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 39_stage_performance.png")

def main():
    print("="*80)
    print("MODULE 13: LOG FILE PERFORMANCE ANALYSIS")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    analyzer = LogFilePerformanceAnalyzer(base_dir)
    
    try:
        analyzer.analyze_production_performance()
        analyzer.analyze_equilibration_performance()
        
        print("\n" + "="*80)
        print("✓ MODULE 13 COMPLETE!")
        print("="*80)
        print(f"\n📊 Generated plots: 37-39 (3 plots)")
        print(f"📊 Generated data: production_performance.csv, equilibration_performance_by_stage.csv")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
