#!/usr/bin/env python3
"""
Simplified TIP4P/2005 Solvation Analysis
=========================================

Analyzes thermodynamic properties from LAMMPS log file.
For trajectory analysis with proper NP identification, use specialized tools.

Usage:
    python analyze_tip4p_simple.py <output_directory>
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_lammps_log(log_file):
    """Parse LAMMPS log file and extract thermodynamic data"""
    print(f"Parsing: {log_file}")
    
    # Handle potential encoding issues in LAMMPS log files
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    data = {
        'Step': [], 'Temp': [], 'Press': [], 'PotEng': [],
        'KinEng': [], 'TotEng': [], 'Volume': [], 'Density': []
    }
    
    in_thermo = False
    headers = []
    
    for line in lines:
        line = line.strip()
        
        if 'Step' in line and 'Temp' in line:
            headers = line.split()
            in_thermo = True
            continue
        
        if in_thermo and line and not line.startswith('#'):
            parts = line.split()
            if parts and parts[0].replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                try:
                    for i, header in enumerate(headers):
                        if header in data and i < len(parts):
                            data[header].append(float(parts[i]))
                except (ValueError, IndexError):
                    continue
    
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"✓ Loaded {len(data['Step'])} timesteps")
    return data

def analyze_simulation(data):
    """Analyze simulation data"""
    print(f"\n{'='*80}")
    print(f"SIMULATION ANALYSIS")
    print(f"{'='*80}\n")
    
    if len(data['Step']) == 0:
        print("❌ No data found")
        return
    
    steps = data['Step']
    time_ps = steps * 0.001
    
    # Temperature
    temp_avg = np.mean(data['Temp'])
    temp_std = np.std(data['Temp'])
    print(f"Temperature:")
    print(f"  Average: {temp_avg:.2f} ± {temp_std:.2f} K")
    print(f"  Target:  300.0 K")
    print(f"  Status:  {'✓ GOOD' if abs(temp_avg - 300) < 20 else '⚠ NEEDS CHECK'}")
    
    # Density
    dens_avg = np.mean(data['Density'])
    dens_std = np.std(data['Density'])
    print(f"\nDensity:")
    print(f"  Average: {dens_avg:.4f} ± {dens_std:.4f} g/cm³")
    print(f"  Pure water: 0.997 g/cm³")
    
    # Pressure
    press_avg = np.mean(data['Press'])
    press_std = np.std(data['Press'])
    print(f"\nPressure:")
    print(f"  Average: {press_avg:.1f} ± {press_std:.1f} bar")
    print(f"  1 atm: 1.01 bar")
    
    # Energy
    tot_eng_avg = np.mean(data['TotEng'])
    pot_eng_avg = np.mean(data['PotEng'])
    kin_eng_avg = np.mean(data['KinEng'])
    
    print(f"\nEnergy:")
    print(f"  Total:     {tot_eng_avg:.2f} kcal/mol")
    print(f"  Potential: {pot_eng_avg:.2f} kcal/mol")
    print(f"  Kinetic:   {kin_eng_avg:.2f} kcal/mol")
    
    # Energy drift
    if len(data['TotEng']) > 1:
        drift = abs(data['TotEng'][-1] - data['TotEng'][0])
        drift_pct = drift / abs(np.mean(data['TotEng'])) * 100 if np.mean(data['TotEng']) != 0 else 0
        print(f"  Drift:     {drift:.2f} kcal/mol ({drift_pct:.2f}%)")
        print(f"  Status:    {'✓ GOOD' if drift_pct < 5 else '⚠ MODERATE' if drift_pct < 10 else '❌ HIGH'}")
    
    # Volume
    vol_avg = np.mean(data['Volume'])
    print(f"\nVolume:")
    print(f"  Average: {vol_avg:.1f} ų")
    
    # Simulation time
    sim_time = time_ps[-1] if len(time_ps) > 0 else 0
    print(f"\nSimulation Time: {sim_time:.2f} ps")

def plot_analysis(data, output_dir):
    """Create analysis plots"""
    print(f"\n{'='*80}")
    print(f"Creating plots...")
    print(f"{'='*80}\n")
    
    steps = data['Step']
    time_ps = steps * 0.001
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TIP4P/2005 Simulation Analysis', fontsize=16, fontweight='bold')
    
    # Temperature
    ax = axes[0, 0]
    ax.plot(time_ps, data['Temp'], 'b-', linewidth=0.8, alpha=0.7)
    ax.axhline(300, color='r', linestyle='--', linewidth=2, label='Target')
    ax.fill_between(time_ps, 280, 320, alpha=0.2, color='green', label='±20 K')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[0, 1]
    ax.plot(time_ps, data['Density'], 'g-', linewidth=0.8)
    ax.axhline(0.997, color='r', linestyle='--', linewidth=2, label='Pure H₂O')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Density (g/cm³)')
    ax.set_title('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pressure
    ax = axes[0, 2]
    ax.plot(time_ps, data['Press'], 'orange', linewidth=0.8)
    ax.axhline(1.01, color='r', linestyle='--', linewidth=2, label='1 atm')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total Energy
    ax = axes[1, 0]
    ax.plot(time_ps, data['TotEng'], 'purple', linewidth=0.8)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Total Energy (kcal/mol)')
    ax.set_title('Total Energy')
    ax.grid(True, alpha=0.3)
    
    # Potential Energy
    ax = axes[1, 1]
    ax.plot(time_ps, data['PotEng'], 'red', linewidth=0.8)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Potential Energy (kcal/mol)')
    ax.set_title('Potential Energy')
    ax.grid(True, alpha=0.3)
    
    # Energy Drift
    ax = axes[1, 2]
    if len(data['TotEng']) > 0:
        drift = (data['TotEng'] - data['TotEng'][0]) / abs(data['TotEng'][0]) * 100 if data['TotEng'][0] != 0 else data['TotEng'] - data['TotEng'][0]
        ax.plot(time_ps, drift, 'brown', linewidth=0.8)
        ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy Drift (%)')
        ax.set_title('Relative Energy Drift')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()

def generate_report(data, output_dir):
    """Generate text report"""
    print(f"\n{'='*80}")
    print(f"Generating report...")
    print(f"{'='*80}\n")
    
    steps = data['Step']
    time_ps = steps * 0.001
    
    report = []
    report.append("="*80)
    report.append("TIP4P/2005 SIMULATION ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Simulation info
    report.append("SIMULATION:")
    report.append(f"  Total steps: {int(steps[-1])}")
    report.append(f"  Simulation time: {time_ps[-1]:.2f} ps")
    report.append(f"  Data points: {len(steps)}")
    report.append("")
    
    # Temperature
    temp_avg = np.mean(data['Temp'])
    temp_std = np.std(data['Temp'])
    report.append("TEMPERATURE:")
    report.append(f"  Average: {temp_avg:.2f} ± {temp_std:.2f} K")
    report.append(f"  Target:  300.0 K")
    report.append(f"  Status:  {'✓ GOOD' if abs(temp_avg - 300) < 20 else '⚠ CHECK'}")
    report.append("")
    
    # Density
    dens_avg = np.mean(data['Density'])
    dens_std = np.std(data['Density'])
    report.append("DENSITY:")
    report.append(f"  Average: {dens_avg:.4f} ± {dens_std:.4f} g/cm³")
    report.append(f"  Reference (pure water): 0.997 g/cm³")
    report.append("")
    
    # Pressure
    press_avg = np.mean(data['Press'])
    press_std = np.std(data['Press'])
    report.append("PRESSURE:")
    report.append(f"  Average: {press_avg:.1f} ± {press_std:.1f} bar")
    report.append(f"  Reference (1 atm): 1.01 bar")
    report.append("")
    
    # Energy
    tot_avg = np.mean(data['TotEng'])
    pot_avg = np.mean(data['PotEng'])
    kin_avg = np.mean(data['KinEng'])
    drift = abs(data['TotEng'][-1] - data['TotEng'][0])
    drift_pct = drift / abs(np.mean(data['TotEng'])) * 100 if np.mean(data['TotEng']) != 0 else 0
    
    report.append("ENERGY:")
    report.append(f"  Total:     {tot_avg:.2f} kcal/mol")
    report.append(f"  Potential: {pot_avg:.2f} kcal/mol")
    report.append(f"  Kinetic:   {kin_avg:.2f} kcal/mol")
    report.append(f"  Drift:     {drift:.2f} kcal/mol ({drift_pct:.2f}%)")
    report.append(f"  Status:    {'✓ GOOD' if drift_pct < 5 else '⚠ MODERATE' if drift_pct < 10 else '❌ HIGH'}")
    report.append("")
    
    # Volume
    vol_avg = np.mean(data['Volume'])
    report.append("VOLUME:")
    report.append(f"  Average: {vol_avg:.1f} ų")
    report.append("")
    
    report.append("="*80)
    
    # Save report
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: {report_file}")
    
    # Print to console
    print("\n" + '\n'.join(report))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python analyze_tip4p_simple.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    log_file = os.path.join(output_dir, 'log.lammps')
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    # Parse log
    data = parse_lammps_log(log_file)
    
    if len(data['Step']) == 0:
        print("❌ No data found in log file")
        sys.exit(1)
    
    # Analyze
    analyze_simulation(data)
    
    # Plot
    plot_analysis(data, output_dir)
    
    # Report
    generate_report(data, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
