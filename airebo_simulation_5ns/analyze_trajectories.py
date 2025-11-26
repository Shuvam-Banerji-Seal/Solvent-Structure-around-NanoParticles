#!/usr/bin/env python3
"""
Analyze trajectory files to compute structural properties:
- Radial distribution functions (RDFs)
- Mean squared displacement (MSD)
- Water structure around C60
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_lammpstrj(filepath, max_frames=None):
    """Read LAMMPS trajectory file."""
    frames = []
    current_frame = {'atoms': []}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    frame_count = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == 'ITEM: TIMESTEP':
            if current_frame['atoms']:
                frames.append(current_frame)
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                current_frame = {'atoms': []}
            current_frame['timestep'] = int(lines[i+1].strip())
            i += 2
            continue
        
        if line == 'ITEM: NUMBER OF ATOMS':
            current_frame['natoms'] = int(lines[i+1].strip())
            i += 2
            continue
        
        if line.startswith('ITEM: BOX BOUNDS'):
            box = []
            for j in range(3):
                vals = lines[i+1+j].strip().split()
                box.append([float(vals[0]), float(vals[1])])
            current_frame['box'] = np.array(box)
            i += 4
            continue
        
        if line.startswith('ITEM: ATOMS'):
            headers = line.split()[2:]
            current_frame['headers'] = headers
            for j in range(current_frame['natoms']):
                atom_data = lines[i+1+j].strip().split()
                current_frame['atoms'].append([float(x) if k > 0 else int(x) 
                                              for k, x in enumerate(atom_data)])
            i += current_frame['natoms'] + 1
            continue
        
        i += 1
    
    if current_frame['atoms']:
        frames.append(current_frame)
    
    return frames

def compute_rdf(positions1, positions2, box_length, rmax=15.0, dr=0.05):
    """
    Compute radial distribution function between two sets of atoms.
    """
    nbins = int(rmax / dr)
    g_r = np.zeros(nbins)
    r = np.arange(nbins) * dr + dr/2
    
    n1 = len(positions1)
    n2 = len(positions2)
    
    for i, pos1 in enumerate(positions1):
        for j, pos2 in enumerate(positions2):
            if np.array_equal(pos1, pos2):  # Skip self-interaction
                continue
            
            # Minimum image convention
            delta = pos1 - pos2
            delta = delta - box_length * np.round(delta / box_length)
            dist = np.linalg.norm(delta)
            
            if dist < rmax:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    g_r[bin_idx] += 1
    
    # Normalize by ideal gas
    volume = box_length**3
    for i in range(nbins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        shell_volume = (4.0/3.0) * np.pi * (r_outer**3 - r_inner**3)
        density = n2 / volume
        g_r[i] /= (n1 * density * shell_volume)
    
    return r, g_r

def analyze_trajectory(epsilon_dir, trajectory_name='npt_equilibration.lammpstrj', max_frames=100):
    """Analyze a trajectory file for structural properties."""
    epsilon_dir = Path(epsilon_dir)
    epsilon = epsilon_dir.name.replace('epsilon_', '')
    
    traj_file = epsilon_dir / trajectory_name
    if not traj_file.exists():
        print(f"ERROR: {traj_file} not found!")
        return None
    
    print(f"\nAnalyzing {trajectory_name} for epsilon = {epsilon}")
    print(f"Reading trajectory (max {max_frames} frames)...")
    
    frames = read_lammpstrj(traj_file, max_frames=max_frames)
    print(f"  Loaded {len(frames)} frames")
    
    if len(frames) == 0:
        print("  ERROR: No frames loaded!")
        return None
    
    # Separate atom types (1=C, 2=O, 3=H)
    # Average over last frames for better statistics
    n_avg = min(50, len(frames))
    
    rdf_CO_all = []
    rdf_OO_all = []
    
    for frame in frames[-n_avg:]:
        atoms = np.array(frame['atoms'])
        box = frame['box']
        box_length = box[0, 1] - box[0, 0]  # Assume cubic box
        
        # Get positions by type
        carbon_idx = atoms[:, 1] == 1  # Type 1 = Carbon
        oxygen_idx = atoms[:, 1] == 2  # Type 2 = Oxygen
        
        # Headers should be: id type x y z or id type xu yu zu
        if 'xu' in frame['headers']:
            x_idx = frame['headers'].index('xu')
        else:
            x_idx = frame['headers'].index('x')
        
        carbon_pos = atoms[carbon_idx, x_idx:x_idx+3]
        oxygen_pos = atoms[oxygen_idx, x_idx:x_idx+3]
        
        # Compute RDFs
        r, rdf_CO = compute_rdf(carbon_pos, oxygen_pos, box_length, rmax=15.0, dr=0.05)
        r, rdf_OO = compute_rdf(oxygen_pos, oxygen_pos, box_length, rmax=15.0, dr=0.05)
        
        rdf_CO_all.append(rdf_CO)
        rdf_OO_all.append(rdf_OO)
    
    # Average RDFs
    rdf_CO_avg = np.mean(rdf_CO_all, axis=0)
    rdf_OO_avg = np.mean(rdf_OO_all, axis=0)
    
    # Find first peak in C-O RDF (solvation shell)
    peak_idx = np.argmax(rdf_CO_avg[:100])  # Search up to 5 Å
    first_peak_r = r[peak_idx]
    first_peak_height = rdf_CO_avg[peak_idx]
    
    print(f"  C-O first peak: r = {first_peak_r:.2f} Å, g(r) = {first_peak_height:.2f}")
    
    results = {
        'epsilon': float(epsilon),
        'r': r,
        'rdf_CO': rdf_CO_avg,
        'rdf_OO': rdf_OO_avg,
        'first_peak_r': first_peak_r,
        'first_peak_height': first_peak_height,
    }
    
    return results

def plot_rdfs(all_results, output_dir):
    """Plot RDFs for all epsilon values."""
    output_dir = Path(output_dir)
    
    # C-O RDF comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    
    for i, res in enumerate(all_results):
        eps = res['epsilon']
        ax1.plot(res['r'], res['rdf_CO'], label=f'ε = {eps:.2f}', 
                color=colors[i], linewidth=2)
        ax2.plot(res['r'], res['rdf_OO'], label=f'ε = {eps:.2f}',
                color=colors[i], linewidth=2)
    
    ax1.set_xlabel('r (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.set_title('C60-Water RDF (C-O)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    
    ax2.set_xlabel('r (Å)', fontsize=12)
    ax2.set_ylabel('g(r)', fontsize=12)
    ax2.set_title('Water-Water RDF (O-O)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    
    plt.tight_layout()
    output_file = output_dir / 'rdf_comparison_all_epsilon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nRDF comparison plot saved: {output_file}")
    plt.close()
    
    # First peak analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epsilons = [r['epsilon'] for r in all_results]
    peak_positions = [r['first_peak_r'] for r in all_results]
    peak_heights = [r['first_peak_height'] for r in all_results]
    
    ax.plot(epsilons, peak_positions, 'o-', markersize=10, linewidth=2, 
            color='blue', label='First peak position')
    ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12)
    ax.set_ylabel('First peak position (Å)', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(epsilons, peak_heights, 's-', markersize=10, linewidth=2,
            color='red', label='First peak height')
    ax2.set_ylabel('First peak height g(r)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('C60-Water Solvation Shell vs Epsilon', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'solvation_shell_vs_epsilon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Solvation shell analysis saved: {output_file}")
    plt.close()

def main():
    """Main analysis function."""
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim')
    epsilon_values = ['0.0', '0.05', '0.10', '0.15', '0.20', '0.25']
    
    all_results = []
    
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS: RDFs AND STRUCTURE")
    print("="*70)
    
    for eps in epsilon_values:
        epsilon_dir = base_dir / f'epsilon_{eps}'
        if not epsilon_dir.exists():
            print(f"\nWARNING: {epsilon_dir} not found, skipping...")
            continue
        
        # Analyze NPT equilibration trajectory (has most frames)
        result = analyze_trajectory(epsilon_dir, 'npt_equilibration.lammpstrj', max_frames=100)
        if result is not None:
            all_results.append(result)
            
            # Save individual RDF data
            rdf_file = epsilon_dir / f'rdf_analysis_epsilon_{eps}.dat'
            with open(rdf_file, 'w') as f:
                f.write("# r(A)  g_CO(r)  g_OO(r)\n")
                for i in range(len(result['r'])):
                    f.write(f"{result['r'][i]:.3f}  {result['rdf_CO'][i]:.6f}  {result['rdf_OO'][i]:.6f}\n")
            print(f"  RDF data saved: {rdf_file}")
    
    if len(all_results) > 0:
        plot_rdfs(all_results, base_dir)
        
        # Summary table
        print("\n" + "="*70)
        print("SOLVATION SHELL SUMMARY")
        print("="*70)
        print(f"{'Epsilon':<12} {'1st Peak (Å)':<15} {'Peak Height':<15}")
        print("-"*70)
        
        for r in all_results:
            print(f"{r['epsilon']:<12.2f} {r['first_peak_r']:<15.2f} {r['first_peak_height']:<15.2f}")
        
        print("="*70)
    else:
        print("\nERROR: No results to plot!")

if __name__ == '__main__':
    main()
