#!/usr/bin/env python3
"""
Radial Distribution Function (RDF) and Structural Analysis
===========================================================

This script analyzes radial distribution functions and structural properties:
1. C-C, C-O, and O-O RDF analysis
2. Coordination number calculations
3. Hydration shell structure analysis
4. RDF peak analysis and comparison across epsilon values

Uses pre-computed RDF data from LAMMPS production runs.

Author: Scientific Analysis Suite
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import integrate, signal
from matplotlib.gridspec import GridSpec

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")

# Setup epsilon directories (handle epsilon_0.0 special case)
epsilon_values_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
EPSILON_DIRS = []
for eps in epsilon_values_list:
    if eps == 0.0:
        EPSILON_DIRS.append(BASE_DIR / "epsilon_0.0")
    else:
        EPSILON_DIRS.append(BASE_DIR / f"epsilon_{eps:.2f}")

PLOTS_DIR = BASE_DIR / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class RDFAnalyzer:
    """Analyzer for radial distribution functions"""
    
    def __init__(self, epsilon_dirs, epsilon_values):
        self.epsilon_dirs = epsilon_dirs
        self.epsilon_values = epsilon_values
        self.rdf_data = {}
        self.coordination_numbers = {}
        
    def load_rdf_data(self):
        """Load RDF data for C-C, C-O, and O-O pairs"""
        print("Loading RDF data...")
        
        rdf_types = ['CC', 'CO', 'OO']
        
        for eps, eps_dir in zip(self.epsilon_values, self.epsilon_dirs):
            self.rdf_data[eps] = {}
            
            for rdf_type in rdf_types:
                rdf_file = eps_dir / f"rdf_{rdf_type}.dat"
                
                if not rdf_file.exists():
                    print(f"  Warning: {rdf_file} not found")
                    continue
                
                # Read RDF data - skip header lines
                with open(rdf_file, 'r') as f:
                    lines = f.readlines()
                    
                # Find where actual data starts
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('# Row'):
                        data_start = i + 2  # Skip the row count line
                        break
                
                # Parse data - collect ALL timesteps then average
                all_r = []
                all_gr = []
                current_r = []
                current_gr = []
                
                for line in lines[data_start:]:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) == 2:  # Timestep line (e.g., "610000 150")
                            # Save previous timestep if exists
                            if current_r:
                                all_r.append(np.array(current_r))
                                all_gr.append(np.array(current_gr))
                                current_r = []
                                current_gr = []
                        elif len(parts) >= 3:  # Data line
                            current_r.append(float(parts[1]))  # r
                            current_gr.append(float(parts[2]))  # g(r)
                
                # Add last timestep
                if current_r:
                    all_r.append(np.array(current_r))
                    all_gr.append(np.array(current_gr))
                
                # Average over all timesteps
                if all_r:
                    r_avg = np.mean(all_r, axis=0)
                    gr_avg = np.mean(all_gr, axis=0)
                    n_timesteps = len(all_r)
                    n_bins = len(r_avg)
                else:
                    r_avg = np.array([])
                    gr_avg = np.array([])
                    n_timesteps = 0
                    n_bins = 0
                
                self.rdf_data[eps][rdf_type] = {
                    'r': r_avg,
                    'g_r': gr_avg,
                    'n_timesteps': n_timesteps,
                    'n_bins': n_bins
                }
                
                print(f"  ε={eps}, {rdf_type}: {n_bins} bins × {n_timesteps} timesteps (averaged)")
        
        return self
    
    def compute_coordination_numbers(self, cutoff_distances={'CC': 5.0, 'CO': 5.0, 'OO': 3.5}):
        """Compute coordination numbers by integrating RDF"""
        print("\nComputing coordination numbers...")
        
        # System parameters
        BOX_VOLUME = 53000.0  # ų (average)
        N_CARBON = 180  # C60 atoms
        N_OXYGEN = 1787  # Water oxygens
        
        # Number densities (atoms/ų)
        rho_C = N_CARBON / BOX_VOLUME
        rho_O = N_OXYGEN / BOX_VOLUME
        
        # Density mapping for each RDF type
        density_map = {
            'CC': rho_C,  # Carbon-Carbon
            'CO': rho_O,  # Carbon-Oxygen (use O density)
            'OO': rho_O   # Oxygen-Oxygen
        }
        
        for eps in self.epsilon_values:
            if eps not in self.rdf_data:
                continue
            
            self.coordination_numbers[eps] = {}
            
            for rdf_type, cutoff in cutoff_distances.items():
                if rdf_type not in self.rdf_data[eps]:
                    continue
                
                data = self.rdf_data[eps][rdf_type]
                r = data['r']
                g_r = data['g_r']
                
                # Skip if no data
                if len(r) == 0:
                    continue
                
                # Find indices within cutoff
                idx = r <= cutoff
                r_cut = r[idx]
                g_r_cut = g_r[idx]
                
                # Get proper density for this pair type
                rho = density_map.get(rdf_type, 0.033)
                
                # Compute coordination number: N = 4π ρ ∫ r² g(r) dr
                integrand = 4 * np.pi * rho * r_cut**2 * g_r_cut
                coord_num = integrate.simpson(integrand, r_cut)
                
                self.coordination_numbers[eps][rdf_type] = coord_num
                print(f"  ε={eps}, {rdf_type}: N_coord = {coord_num:.2f} (r < {cutoff} Å)")
        
        return self
    
    def find_rdf_peaks(self, rdf_type='CO', prominence=0.1):
        """Find peaks in RDF to identify hydration shells"""
        peak_data = []
        
        for eps in self.epsilon_values:
            if eps not in self.rdf_data or rdf_type not in self.rdf_data[eps]:
                continue
            
            data = self.rdf_data[eps][rdf_type]
            r = data['r']
            g_r = data['g_r']
            
            # Find peaks
            peaks, properties = signal.find_peaks(g_r, prominence=prominence, distance=5)
            
            if len(peaks) > 0:
                first_peak_r = r[peaks[0]]
                first_peak_g = g_r[peaks[0]]
                
                peak_data.append({
                    'Epsilon': eps,
                    'First_peak_r': first_peak_r,
                    'First_peak_g': first_peak_g,
                    'N_peaks': len(peaks)
                })
                
                print(f"  ε={eps}: First {rdf_type} peak at r={first_peak_r:.2f} Å, g(r)={first_peak_g:.2f}")
        
        self.peak_df = pd.DataFrame(peak_data)
        
        # Save peak data
        peak_file = PLOTS_DIR / f"rdf_{rdf_type}_peaks.csv"
        self.peak_df.to_csv(peak_file, index=False)
        
        return self
    
    def plot_rdf_comparison(self):
        """Plot RDF comparison for all epsilon values"""
        rdf_types = ['CC', 'CO', 'OO']
        titles = ['C-C (Nanoparticle-Nanoparticle)', 
                 'C-O (Nanoparticle-Water)', 
                 'O-O (Water-Water)']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (rdf_type, title) in enumerate(zip(rdf_types, titles)):
            ax = axes[idx]
            
            for eps in self.epsilon_values:
                if eps in self.rdf_data and rdf_type in self.rdf_data[eps]:
                    data = self.rdf_data[eps][rdf_type]
                    r = data['r']
                    g_r = data['g_r']
                    
                    ax.plot(r, g_r, label=f'ε={eps}', linewidth=2, alpha=0.8)
            
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Distance r (Å)')
            ax.set_ylabel('g(r)')
            ax.set_title(title)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 12)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "10_rdf_comparison.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 10_rdf_comparison.png")
    
    def plot_co_rdf_detailed(self):
        """Detailed C-O RDF analysis showing hydration shells"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: C-O RDF with enhanced view of first shell
        ax1 = fig.add_subplot(gs[0, :])
        for eps in self.epsilon_values:
            if eps in self.rdf_data and 'CO' in self.rdf_data[eps]:
                data = self.rdf_data[eps]['CO']
                r = data['r']
                g_r = data['g_r']
                ax1.plot(r, g_r, label=f'ε={eps}', linewidth=2.5, alpha=0.8)
        
        ax1.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Bulk water')
        ax1.set_xlabel('Distance from C60 Surface (Å)')
        ax1.set_ylabel('g(r)')
        ax1.set_title('Nanoparticle-Water Radial Distribution Function')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 12)
        
        # Plot 2: First hydration shell (zoomed)
        ax2 = fig.add_subplot(gs[1, 0])
        for eps in self.epsilon_values:
            if eps in self.rdf_data and 'CO' in self.rdf_data[eps]:
                data = self.rdf_data[eps]['CO']
                r = data['r']
                g_r = data['g_r']
                mask = r <= 6.0
                ax2.plot(r[mask], g_r[mask], label=f'ε={eps}', linewidth=2.5, alpha=0.8)
        
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Distance (Å)')
        ax2.set_ylabel('g(r)')
        ax2.set_title('First Hydration Shell (Detailed View)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: First peak position vs epsilon
        ax3 = fig.add_subplot(gs[1, 1])
        if hasattr(self, 'peak_df') and not self.peak_df.empty:
            x_pos = np.arange(len(self.peak_df))
            ax3.plot(self.peak_df['Epsilon'], self.peak_df['First_peak_r'], 
                    'o-', markersize=10, linewidth=2.5, color='darkblue')
            ax3.set_xlabel('Epsilon (kcal/mol)')
            ax3.set_ylabel('First Peak Position (Å)')
            ax3.set_title('Hydration Shell Distance vs C-O Interaction Strength')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "11_co_rdf_detailed.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 11_co_rdf_detailed.png")
    
    def plot_coordination_analysis(self):
        """Plot coordination number analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        rdf_types = ['CC', 'CO', 'OO']
        titles = ['C-C Coordination', 'C-O Coordination', 'O-O Coordination']
        
        for idx, (rdf_type, title) in enumerate(zip(rdf_types, titles)):
            ax = axes[idx]
            
            # Extract coordination numbers
            coords = []
            eps_list = []
            for eps in self.epsilon_values:
                if eps in self.coordination_numbers and rdf_type in self.coordination_numbers[eps]:
                    coords.append(self.coordination_numbers[eps][rdf_type])
                    eps_list.append(eps)
            
            if coords:
                x_pos = np.arange(len(eps_list))
                bars = ax.bar(x_pos, coords, color=f'C{idx}', alpha=0.8, edgecolor='black')
                
                # Add value labels on bars
                for bar, coord in zip(bars, coords):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{coord:.1f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'{eps:.2f}' for eps in eps_list])
                ax.set_xlabel('Epsilon (kcal/mol)')
                ax.set_ylabel('Coordination Number')
                ax.set_title(title)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "12_coordination_numbers.png", dpi=600, bbox_inches='tight')
        plt.close()
        print("  Saved: 12_coordination_numbers.png")
    
    def export_rdf_summary(self):
        """Export RDF analysis summary to JSON"""
        summary = {
            'analysis_info': {
                'rdf_types': ['CC', 'CO', 'OO'],
                'description': 'Radial distribution functions and hydration structure',
                'cutoffs': {'CC': '5.0 Å', 'CO': '5.0 Å', 'OO': '3.5 Å'}
            },
            'coordination_numbers': {},
            'hydration_shell_peaks': {}
        }
        
        # Add coordination numbers
        for eps in self.epsilon_values:
            if eps in self.coordination_numbers:
                summary['coordination_numbers'][f'eps_{eps:.2f}'] = {
                    rdf_type: float(coord) 
                    for rdf_type, coord in self.coordination_numbers[eps].items()
                }
        
        # Add peak positions
        if hasattr(self, 'peak_df') and not self.peak_df.empty:
            for _, row in self.peak_df.iterrows():
                eps = row['Epsilon']
                summary['hydration_shell_peaks'][f'eps_{eps:.2f}'] = {
                    'first_peak_position_A': float(row['First_peak_r']),
                    'first_peak_height': float(row['First_peak_g']),
                    'num_peaks': int(row['N_peaks'])
                }
        
        json_file = PLOTS_DIR / "rdf_analysis_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nRDF summary saved to {json_file}")

def main():
    """Main RDF analysis workflow"""
    print("="*70)
    print("RADIAL DISTRIBUTION FUNCTION ANALYSIS")
    print("="*70)
    
    epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    analyzer = RDFAnalyzer(EPSILON_DIRS, epsilon_values)
    
    # Load and analyze RDF data
    analyzer.load_rdf_data()
    analyzer.compute_coordination_numbers()
    analyzer.find_rdf_peaks(rdf_type='CO', prominence=0.1)
    
    print("\n" + "="*70)
    print("GENERATING RDF PLOTS")
    print("="*70)
    
    # Generate plots
    analyzer.plot_rdf_comparison()
    analyzer.plot_co_rdf_detailed()
    analyzer.plot_coordination_analysis()
    
    # Export summary
    analyzer.export_rdf_summary()
    
    print("\n" + "="*70)
    print("RDF ANALYSIS COMPLETE!")
    print(f"All plots saved to: {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
