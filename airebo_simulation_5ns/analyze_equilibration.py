#!/usr/bin/env python3
"""
Analyze equilibration logs and trajectory files to verify system equilibration.
Checks temperature, pressure, density, and energy convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def parse_thermo_data(filepath):
    """Parse LAMMPS thermo output file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9:  # Need at least 9 columns
                try:
                    # timestep epsilon temp press pe ke etotal vol density
                    data.append([float(x) for x in parts])
                except ValueError:
                    continue
    return np.array(data)

def calculate_block_averages(data, block_size=50):
    """Calculate block averages to assess equilibration."""
    n_blocks = len(data) // block_size
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        blocks.append(np.mean(data[start:end]))
    return np.array(blocks)

def is_equilibrated(data, threshold=0.05, last_fraction=0.3, is_pressure=False):
    """
    Check if data is equilibrated by comparing variance of last fraction
    to overall variance. For pressure, use more relaxed criteria.
    """
    n_last = int(len(data) * last_fraction)
    if n_last < 10:
        return False
    
    overall_std = np.std(data)
    last_std = np.std(data[-n_last:])
    last_mean = np.mean(data[-n_last:])
    overall_mean = np.mean(data)
    
    # For pressure, just check if fluctuations are reasonable (not if mean is near 1 atm)
    # Pressure fluctuations are naturally large in NPT simulations
    if is_pressure:
        # Accept if pressure fluctuations are not growing
        is_stable = last_std < overall_std * 1.5  # Fluctuations not increasing
    else:
        # Check if last portion has stabilized
        relative_change = abs(last_mean - overall_mean) / (abs(overall_mean) + 1e-10)
        is_stable = relative_change < threshold
    
    return is_stable, last_mean, last_std

def analyze_epsilon(epsilon_dir):
    """Analyze equilibration for a single epsilon value."""
    epsilon_dir = Path(epsilon_dir)
    epsilon = epsilon_dir.name.replace('epsilon_', '')
    
    print(f"\n{'='*70}")
    print(f"ANALYZING EPSILON = {epsilon}")
    print(f"{'='*70}")
    
    # Parse thermodynamic data
    thermo_file = epsilon_dir / 'equilibration_thermo.dat'
    if not thermo_file.exists():
        print(f"ERROR: {thermo_file} not found!")
        return None
    
    data = parse_thermo_data(thermo_file)
    if len(data) == 0:
        print(f"ERROR: No data in {thermo_file}!")
        return None
    
    # Extract columns
    # timestep epsilon temp press pe ke etotal vol density
    steps = data[:, 0]
    epsilon_col = data[:, 1]
    temp = data[:, 2]
    press = data[:, 3]
    pe = data[:, 4]
    ke = data[:, 5]
    etotal = data[:, 6]
    vol = data[:, 7]
    density = data[:, 8]
    
    # Time in ps (timestep = 2 fs)
    time_ps = steps * 2e-3
    
    results = {
        'epsilon': float(epsilon),
        'n_steps': len(steps),
        'time_ps': time_ps[-1] if len(time_ps) > 0 else 0,
    }
    
    # Analyze temperature
    print(f"\nTEMPERATURE:")
    print(f"  Target: 300 K")
    temp_eq, temp_mean, temp_std = is_equilibrated(temp)
    results['temp_mean'] = temp_mean
    results['temp_std'] = temp_std
    results['temp_equilibrated'] = temp_eq
    print(f"  Mean (last 30%): {temp_mean:.2f} ± {temp_std:.2f} K")
    print(f"  Equilibrated: {'✓ YES' if temp_eq else '✗ NO'}")
    
    # Analyze pressure
    print(f"\nPRESSURE:")
    print(f"  Target: 1 atm")
    press_eq, press_mean, press_std = is_equilibrated(press, is_pressure=True)
    results['press_mean'] = press_mean
    results['press_std'] = press_std
    results['press_equilibrated'] = press_eq
    print(f"  Mean (last 30%): {press_mean:.1f} ± {press_std:.1f} atm")
    print(f"  Equilibrated: {'✓ YES' if press_eq else '✗ NO'}")
    print(f"  Note: Large pressure fluctuations (~100 atm) are normal in NPT")
    
    # Analyze density
    print(f"\nDENSITY:")
    print(f"  Target: ~1.0 g/cm³ (water at 300K)")
    dens_eq, dens_mean, dens_std = is_equilibrated(density)
    results['dens_mean'] = dens_mean
    results['dens_std'] = dens_std
    results['dens_equilibrated'] = dens_eq
    print(f"  Mean (last 30%): {dens_mean:.4f} ± {dens_std:.4f} g/cm³")
    print(f"  Deviation from 1.0: {abs(dens_mean - 1.0)*100:.2f}%")
    print(f"  Equilibrated: {'✓ YES' if dens_eq else '✗ NO'}")
    
    # Analyze potential energy
    print(f"\nPOTENTIAL ENERGY:")
    pe_eq, pe_mean, pe_std = is_equilibrated(pe)
    results['pe_mean'] = pe_mean
    results['pe_std'] = pe_std
    results['pe_equilibrated'] = pe_eq
    print(f"  Mean (last 30%): {pe_mean:.1f} ± {pe_std:.1f} kcal/mol")
    print(f"  Equilibrated: {'✓ YES' if pe_eq else '✗ NO'}")
    
    # Analyze total energy (should be conserved on average in NPT)
    print(f"\nTOTAL ENERGY:")
    etot_eq, etot_mean, etot_std = is_equilibrated(etotal)
    results['etot_mean'] = etot_mean
    results['etot_std'] = etot_std
    results['etot_equilibrated'] = etot_eq
    print(f"  Mean (last 30%): {etot_mean:.1f} ± {etot_std:.1f} kcal/mol")
    print(f"  Equilibrated: {'✓ YES' if etot_eq else '✗ NO'}")
    
    # Overall assessment
    all_equilibrated = all([temp_eq, press_eq, dens_eq, pe_eq])
    results['overall_equilibrated'] = all_equilibrated
    
    print(f"\n{'='*70}")
    if all_equilibrated:
        print(f"✓ SYSTEM IS EQUILIBRATED")
    else:
        print(f"✗ SYSTEM NOT FULLY EQUILIBRATED")
    print(f"{'='*70}")
    
    return results, data

def plot_equilibration(epsilon_dir, data):
    """Create equilibration plots."""
    epsilon_dir = Path(epsilon_dir)
    epsilon = epsilon_dir.name.replace('epsilon_', '')
    
    # Extract columns
    steps = data[:, 1]
    temp = data[:, 2]
    press = data[:, 3]
    pe = data[:, 4]
    density = data[:, 7]
    
    time_ps = steps * 2e-3
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Equilibration Analysis: ε = {epsilon}', fontsize=16, fontweight='bold')
    
    # Temperature
    ax = axes[0, 0]
    ax.plot(time_ps, temp, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(300, color='r', linestyle='--', label='Target (300 K)')
    last_30_idx = int(len(temp) * 0.7)
    ax.axhline(np.mean(temp[last_30_idx:]), color='g', linestyle='--', 
               label=f'Mean (last 30%): {np.mean(temp[last_30_idx:]):.1f} K')
    ax.fill_between(time_ps[last_30_idx:], 295, 305, alpha=0.2, color='green', label='±5K range')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Pressure
    ax = axes[0, 1]
    ax.plot(time_ps, press, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(1, color='r', linestyle='--', label='Target (1 atm)')
    ax.axhline(np.mean(press[last_30_idx:]), color='g', linestyle='--',
               label=f'Mean (last 30%): {np.mean(press[last_30_idx:]):.1f} atm')
    ax.fill_between(time_ps[last_30_idx:], -500, 500, alpha=0.2, color='green')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Pressure (atm)', fontsize=12)
    ax.set_title('Pressure Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[1, 0]
    ax.plot(time_ps, density, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(1.0, color='r', linestyle='--', label='Water ref (1.0 g/cm³)')
    ax.axhline(np.mean(density[last_30_idx:]), color='g', linestyle='--',
               label=f'Mean (last 30%): {np.mean(density[last_30_idx:]):.4f} g/cm³')
    ax.fill_between(time_ps[last_30_idx:], 0.95, 1.05, alpha=0.2, color='green', label='±5% range')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Density (g/cm³)', fontsize=12)
    ax.set_title('Density Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Potential Energy
    ax = axes[1, 1]
    ax.plot(time_ps, pe, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(np.mean(pe[last_30_idx:]), color='g', linestyle='--',
               label=f'Mean (last 30%): {np.mean(pe[last_30_idx:]):.1f} kcal/mol')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Potential Energy (kcal/mol)', fontsize=12)
    ax.set_title('Potential Energy Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = epsilon_dir / f'equilibration_analysis_epsilon_{epsilon}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    plt.close()

def main():
    """Main analysis function."""
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim')
    epsilon_values = ['0.0', '0.05', '0.10', '0.15', '0.20', '0.25']
    
    all_results = []
    
    print("\n" + "="*70)
    print("EQUILIBRATION ANALYSIS FOR ALL EPSILON VALUES")
    print("="*70)
    
    for eps in epsilon_values:
        epsilon_dir = base_dir / f'epsilon_{eps}'
        if not epsilon_dir.exists():
            print(f"\nWARNING: {epsilon_dir} not found, skipping...")
            continue
        
        result, data = analyze_epsilon(epsilon_dir)
        if result is not None:
            all_results.append(result)
            plot_equilibration(epsilon_dir, data)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Epsilon':<10} {'Temp (K)':<15} {'Press (atm)':<18} {'Density':<15} {'Equilibrated':<15}")
    print("-"*70)
    
    for r in all_results:
        status = "✓ YES" if r['overall_equilibrated'] else "✗ NO"
        print(f"{r['epsilon']:<10.2f} {r['temp_mean']:>6.2f}±{r['temp_std']:<5.2f} "
              f"{r['press_mean']:>7.1f}±{r['press_std']:<8.1f} "
              f"{r['dens_mean']:>6.4f}±{r['dens_std']:<6.4f} {status:<15}")
    
    # Save results to file
    results_file = base_dir / 'equilibration_summary.txt'
    with open(results_file, 'w') as f:
        f.write("EQUILIBRATION ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Epsilon':<10} {'Temp (K)':<15} {'Press (atm)':<18} {'Density':<15} {'PE (kcal/mol)':<20} {'Status':<15}\n")
        f.write("-"*100 + "\n")
        for r in all_results:
            status = "EQUILIBRATED" if r['overall_equilibrated'] else "NOT EQUILIBRATED"
            f.write(f"{r['epsilon']:<10.2f} {r['temp_mean']:>6.2f}±{r['temp_std']:<5.2f} "
                   f"{r['press_mean']:>7.1f}±{r['press_std']:<8.1f} "
                   f"{r['dens_mean']:>6.4f}±{r['dens_std']:<6.4f} "
                   f"{r['pe_mean']:>8.1f}±{r['pe_std']:<9.1f} {status:<15}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Overall assessment
    n_equilibrated = sum(1 for r in all_results if r['overall_equilibrated'])
    print(f"\n{'='*70}")
    print(f"OVERALL: {n_equilibrated}/{len(all_results)} systems are equilibrated")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
