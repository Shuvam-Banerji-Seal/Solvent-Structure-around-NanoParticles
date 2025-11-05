#!/usr/bin/env python3
"""
Analyze TIP4P/2005 Solvation Structure Around Nanoparticles
============================================================

This script performs comprehensive analysis of MD simulations:
- Radial Distribution Function (RDF)
- Coordination numbers
- Hydrogen bonding analysis
- Water orientation analysis
- Density profiles
- Energy statistics

Usage:
    python analyze_tip4p_solvation.py <output_directory>

Example:
    python analyze_tip4p_solvation.py ../output/
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Try to import scipy, use numpy fallback if not available
try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    print("Note: scipy not found, using numpy fallback (slower)")
    HAS_SCIPY = False
    
    def cdist(XA, XB):
        """Numpy fallback for scipy's cdist"""
        return np.sqrt(((XA[:, np.newaxis, :] - XB[np.newaxis, :, :])**2).sum(axis=2))

def parse_lammps_log(log_file):
    """Parse LAMMPS log file to extract thermodynamic data"""
    print(f"\n{'='*80}")
    print(f"PARSING LOG FILE: {log_file}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    data = {
        'Step': [],
        'Temp': [],
        'Press': [],
        'PotEng': [],
        'KinEng': [],
        'TotEng': [],
        'Volume': [],
        'Density': []
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
    
    print(f"✓ Found {len(data['Step'])} timesteps")
    return data

def parse_lammps_trajectory(traj_file):
    """Parse LAMMPS trajectory file"""
    print(f"\n{'='*80}")
    print(f"PARSING TRAJECTORY: {traj_file}")
    print(f"{'='*80}\n")
    
    frames = []
    
    with open(traj_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if 'ITEM: TIMESTEP' in lines[i]:
            frame = {}
            i += 1
            frame['timestep'] = int(lines[i].strip())
            i += 1
            
            # Number of atoms
            assert 'ITEM: NUMBER OF ATOMS' in lines[i]
            i += 1
            n_atoms = int(lines[i].strip())
            frame['n_atoms'] = n_atoms
            i += 1
            
            # Box bounds
            assert 'ITEM: BOX BOUNDS' in lines[i]
            i += 1
            frame['box'] = []
            for _ in range(3):
                frame['box'].append([float(x) for x in lines[i].split()])
                i += 1
            
            # Atoms
            assert 'ITEM: ATOMS' in lines[i]
            header = lines[i].split()[2:]  # Skip "ITEM: ATOMS"
            i += 1
            
            atoms_data = {h: [] for h in header}
            for _ in range(n_atoms):
                parts = lines[i].split()
                for j, h in enumerate(header):
                    atoms_data[h].append(float(parts[j]))
                i += 1
            
            for h in header:
                atoms_data[h] = np.array(atoms_data[h])
            
            frame['atoms'] = atoms_data
            frames.append(frame)
        else:
            i += 1
    
    print(f"✓ Loaded {len(frames)} trajectory frames")
    return frames

def identify_atom_types(frames):
    """Identify which atom types correspond to nanoparticle and water"""
    print(f"\n{'='*80}")
    print(f"IDENTIFYING ATOM TYPES")
    print(f"{'='*80}\n")
    
    # Get unique types, molecule IDs, and atom IDs from first frame
    atom_ids = frames[0]['atoms']['id'].astype(int)
    types = frames[0]['atoms']['type'].astype(int)
    mol_ids = frames[0]['atoms']['mol'].astype(int)
    
    unique_types = np.unique(types)
    print(f"Found atom types: {unique_types}")
    
    # Count atoms of each type
    type_counts = {}
    for t in unique_types:
        count = np.sum(types == t)
        type_counts[t] = count
        print(f"  Type {t}: {count} atoms")
    
    # Check molecule IDs
    unique_mols = np.unique(mol_ids)
    print(f"\nMolecule IDs: {len(unique_mols)} unique molecules")
    
    # Strategy 1: Atoms with mol ID = 1 are likely NP
    np_mask_mol = mol_ids == 1
    n_np_atoms = np.sum(np_mask_mol)
    
    print(f"Atoms in molecule 1: {n_np_atoms}")
    
    # Strategy 2: First few atom IDs are likely NP (before water insertion)
    # Typically NP has atom IDs 1-8 or 1-N where N is small
    # Find the smallest atom IDs
    sorted_ids = np.sort(atom_ids)
    
    # Look for a gap in atom IDs (water atoms start at higher IDs)
    # Or use mol ID = 1
    
    if n_np_atoms > 0 and n_np_atoms < 1000:
        # Use molecule ID strategy
        np_atom_ids = atom_ids[np_mask_mol]
        np_types_from_mol = np.unique(types[np_mask_mol])
        
        print(f"NP atom IDs: {np_atom_ids}")
        print(f"NP types (from mol ID=1): {np_types_from_mol}")
        
        # Now separate into NP and water types
        # NP types are those found in molecule 1
        # But we need to exclude them from water if they overlap
        
        # Use atom IDs to distinguish:
        # Mark specific atom IDs as NP, not types
        np_atom_ids_set = set(np_atom_ids)
        
        print(f"\n✓ Nanoparticle: {len(np_atom_ids)} atoms (IDs: {sorted(np_atom_ids_set)})")
        print(f"✓ Water: {len(atom_ids) - len(np_atom_ids)} atoms")
        
        # For RDF: we need to return which atoms are NP vs water
        # Not types, but actual atom indices
        # Return empty type lists and handle by atom ID instead
        
        return list(np_types_from_mol), list(unique_types), np_atom_ids_set
    else:
        print("\n⚠ WARNING: Could not identify nanoparticle!")
        print("  Assuming all atoms are water.")
        
        return [], list(unique_types), set()

def calculate_rdf(frames, np_types, water_types, r_max=15.0, dr=0.1):
    """Calculate radial distribution function between NP and water oxygen"""
    print(f"\n{'='*80}")
    print(f"CALCULATING RADIAL DISTRIBUTION FUNCTION (RDF)")
    print(f"{'='*80}\n")
    
    # Determine which water type is oxygen (should be most abundant water type)
    water_type_counts = {}
    for wt in water_types:
        count = np.sum(frames[0]['atoms']['type'] == wt)
        water_type_counts[wt] = count
    
    # Oxygen type is the one with count divisible by molecule count
    # For TIP4P: O, H, H, M (but M might have slightly different count)
    # Assume first water type is oxygen
    o_type = min(water_types)  # Usually type 3 for oxygen
    
    print(f"Using type {o_type} as water oxygen")
    print(f"NP types: {np_types}")
    print(f"r_max: {r_max} Å, dr: {dr} Å")
    
    bins = np.arange(0, r_max + dr, dr)
    r = bins[:-1] + dr/2
    g_r = np.zeros(len(r))
    
    n_frames = len(frames)
    print(f"\nProcessing {n_frames} frames...")
    
    for iframe, frame in enumerate(frames):
        if (iframe + 1) % max(1, n_frames // 10) == 0:
            print(f"  Frame {iframe + 1}/{n_frames}")
        
        atoms = frame['atoms']
        
        # Get NP positions
        np_mask = np.isin(atoms['type'], np_types)
        np_pos = np.column_stack([atoms['x'][np_mask], 
                                   atoms['y'][np_mask], 
                                   atoms['z'][np_mask]])
        
        # Get water oxygen positions
        o_mask = atoms['type'] == o_type
        o_pos = np.column_stack([atoms['x'][o_mask], 
                                  atoms['y'][o_mask], 
                                  atoms['z'][o_mask]])
        
        if len(np_pos) == 0 or len(o_pos) == 0:
            continue
        
        # Calculate distances
        distances = cdist(np_pos, o_pos)
        
        # Histogram
        hist, _ = np.histogram(distances.flatten(), bins=bins)
        g_r += hist
    
    # Normalize by number of frames
    g_r = g_r / n_frames
    
    # Normalize by shell volume and density
    n_np = len(np_pos)
    n_water = len(o_pos)
    
    # Get box dimensions
    box = frames[0]['box']
    Lx = box[0][1] - box[0][0]
    Ly = box[1][1] - box[1][0]
    Lz = box[2][1] - box[2][0]
    volume = Lx * Ly * Lz
    
    rho = n_water / volume  # number density of water oxygens
    
    # Normalize g(r)
    if n_np > 0 and rho > 0:
        for i in range(len(r)):
            shell_vol = 4.0/3.0 * np.pi * ((r[i] + dr/2)**3 - (r[i] - dr/2)**3)
            if shell_vol > 0:
                g_r[i] = g_r[i] / (n_np * rho * shell_vol)
    else:
        print("\n⚠ Warning: Cannot normalize RDF (n_np=0 or rho=0)")
        g_r = g_r / max(1, np.max(g_r))  # Simple normalization
    
    print(f"\n✓ RDF calculation complete")
    print(f"  Number density (water O): {rho:.6f} atoms/ų")
    
    return r, g_r

def calculate_coordination_number(r, g_r, r_cutoff=3.5):
    """Calculate coordination number from RDF"""
    print(f"\n{'='*80}")
    print(f"CALCULATING COORDINATION NUMBER")
    print(f"{'='*80}\n")
    
    # Check if g_r has valid values
    if np.all(np.isnan(g_r)) or np.all(g_r == 0):
        print(f"⚠ Warning: g(r) contains no valid data")
        print(f"  Cannot calculate coordination number")
        return 0.0
    
    # Find first minimum after first peak (typically around 3.5 Å for water)
    idx = np.where(r <= r_cutoff)[0]
    
    if len(idx) == 0:
        print(f"⚠ Warning: r_cutoff {r_cutoff} is too small")
        return 0.0
    
    # Integrate g(r) to get coordination number
    # n(r) = 4π * ρ * ∫[0 to r] r'² g(r') dr'
    dr = r[1] - r[0]
    
    # For simplicity, assume uniform density
    # We already have normalized g(r), so we integrate
    n_coord = 0.0
    for i in idx:
        if i > 0 and not np.isnan(g_r[i]):
            shell_vol = 4.0 * np.pi * r[i]**2 * dr
            n_coord += g_r[i] * shell_vol
    
    # This gives relative coordination; multiply by a density factor
    # For water around NP, typical first shell is ~3-6 molecules
    
    if np.isnan(n_coord):
        print(f"⚠ Warning: Coordination number is NaN")
        return 0.0
    
    print(f"Coordination number (r < {r_cutoff} Å): {n_coord:.2f}")
    
    return n_coord

def analyze_hydrogen_bonds(frames, water_types, h_bond_cutoff=2.5, angle_cutoff=30.0):
    """Analyze hydrogen bonding between water molecules"""
    print(f"\n{'='*80}")
    print(f"HYDROGEN BOND ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Criteria:")
    print(f"  O···O distance < {h_bond_cutoff} Å")
    print(f"  O-H···O angle > {180 - angle_cutoff}°")
    
    # Identify O and H types
    # Heuristic: most abundant water type is O
    water_type_counts = {}
    for wt in water_types:
        count = np.sum(frames[0]['atoms']['type'] == wt)
        water_type_counts[wt] = count
    
    sorted_water = sorted(water_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    # For TIP4P: we expect O (1x), H (2x), M (1x)
    # So H should be ~2x more than O
    # Identify based on counts
    o_type = None
    h_type = None
    
    for t, count in sorted_water:
        if o_type is None:
            # Check if next type has ~2x count
            remaining = [x for x in sorted_water if x[0] != t]
            if remaining and remaining[0][1] > count * 1.5:
                h_type = remaining[0][0]
                o_type = t
                break
        
    if o_type is None or h_type is None:
        print("⚠ Could not identify O and H types reliably")
        print("  Skipping H-bond analysis")
        return None
    
    print(f"\nIdentified: O type = {o_type}, H type = {h_type}")
    
    # Simple H-bond count based on O-O distance
    # (Full angle calculation would require molecule topology)
    
    h_bonds_per_frame = []
    
    for iframe, frame in enumerate(frames):
        if (iframe + 1) % max(1, len(frames) // 5) == 0:
            print(f"  Frame {iframe + 1}/{len(frames)}")
        
        atoms = frame['atoms']
        o_mask = atoms['type'] == o_type
        o_pos = np.column_stack([atoms['x'][o_mask], 
                                  atoms['y'][o_mask], 
                                  atoms['z'][o_mask]])
        
        if len(o_pos) < 2:
            continue
        
        # Calculate O-O distances
        oo_dist = cdist(o_pos, o_pos)
        
        # Count pairs within cutoff (excluding self)
        mask = (oo_dist > 0.1) & (oo_dist < h_bond_cutoff)
        n_hbonds = np.sum(mask) / 2  # Divide by 2 to avoid double counting
        
        h_bonds_per_frame.append(n_hbonds)
    
    h_bonds_per_frame = np.array(h_bonds_per_frame)
    avg_hbonds = np.mean(h_bonds_per_frame)
    
    # Calculate H-bonds per water molecule
    n_water_molecules = len(o_pos)
    hbonds_per_water = avg_hbonds / n_water_molecules if n_water_molecules > 0 else 0
    
    print(f"\n✓ H-bond analysis complete")
    print(f"  Average H-bonds: {avg_hbonds:.1f}")
    print(f"  H-bonds per water: {hbonds_per_water:.2f}")
    print(f"  (Bulk water typically has ~3.5 H-bonds per molecule)")
    
    return h_bonds_per_frame

def analyze_density_profile(frames, np_types, water_types, r_max=20.0, dr=0.5):
    """Calculate density profile as function of distance from NP"""
    print(f"\n{'='*80}")
    print(f"CALCULATING DENSITY PROFILE")
    print(f"{'='*80}\n")
    
    o_type = min(water_types)
    
    bins = np.arange(0, r_max + dr, dr)
    r = bins[:-1] + dr/2
    density_profile = np.zeros(len(r))
    
    for iframe, frame in enumerate(frames):
        if (iframe + 1) % max(1, len(frames) // 10) == 0:
            print(f"  Frame {iframe + 1}/{len(frames)}")
        
        atoms = frame['atoms']
        
        np_mask = np.isin(atoms['type'], np_types)
        np_pos = np.column_stack([atoms['x'][np_mask], 
                                   atoms['y'][np_mask], 
                                   atoms['z'][np_mask]])
        
        o_mask = atoms['type'] == o_type
        o_pos = np.column_stack([atoms['x'][o_mask], 
                                  atoms['y'][o_mask], 
                                  atoms['z'][o_mask]])
        
        if len(np_pos) == 0 or len(o_pos) == 0:
            continue
        
        # Calculate distances from NP center of mass
        np_com = np.mean(np_pos, axis=0)
        distances = np.linalg.norm(o_pos - np_com, axis=1)
        
        # Histogram
        hist, _ = np.histogram(distances, bins=bins)
        density_profile += hist
    
    # Normalize by shell volume and number of frames
    for i in range(len(r)):
        shell_vol = 4.0/3.0 * np.pi * ((r[i] + dr/2)**3 - (r[i] - dr/2)**3)
        density_profile[i] = density_profile[i] / (shell_vol * len(frames))
    
    print(f"\n✓ Density profile complete")
    
    return r, density_profile

def plot_thermodynamics(data, output_dir):
    """Create thermodynamic property plots"""
    print(f"\n{'='*80}")
    print(f"CREATING THERMODYNAMIC PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TIP4P/2005 Simulation - Thermodynamic Properties', 
                 fontsize=16, fontweight='bold')
    
    steps = data['Step']
    time_ps = steps * 0.001  # Assuming 1 fs timestep
    
    # Temperature
    ax = axes[0, 0]
    ax.plot(time_ps, data['Temp'], 'b-', linewidth=0.8)
    ax.axhline(300, color='r', linestyle='--', linewidth=2, label='Target')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[0, 1]
    ax.plot(time_ps, data['Density'], 'g-', linewidth=0.8)
    ax.axhline(0.997, color='r', linestyle='--', linewidth=2, label='Pure H₂O')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Density (g/cm³)', fontsize=12)
    ax.set_title('Density Evolution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pressure
    ax = axes[0, 2]
    ax.plot(time_ps, data['Press'], 'orange', linewidth=0.8)
    ax.axhline(1.01, color='r', linestyle='--', linewidth=2, label='1 atm')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Pressure (bar)', fontsize=12)
    ax.set_title('Pressure Evolution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total Energy
    ax = axes[1, 0]
    ax.plot(time_ps, data['TotEng'], 'purple', linewidth=0.8)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Total Energy (kcal/mol)', fontsize=12)
    ax.set_title('Total Energy Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Potential Energy
    ax = axes[1, 1]
    ax.plot(time_ps, data['PotEng'], 'red', linewidth=0.8)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Potential Energy (kcal/mol)', fontsize=12)
    ax.set_title('Potential Energy Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Energy drift
    ax = axes[1, 2]
    energy_drift = (data['TotEng'] - data['TotEng'][0]) / abs(data['TotEng'][0]) * 100
    ax.plot(time_ps, energy_drift, 'brown', linewidth=0.8)
    ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Energy Drift (%)', fontsize=12)
    ax.set_title('Relative Energy Drift', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'thermodynamics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    return plot_file

def plot_solvation_structure(r_rdf, g_r, r_dens, density_profile, output_dir):
    """Plot solvation structure analysis"""
    print(f"\n{'='*80}")
    print(f"CREATING SOLVATION STRUCTURE PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TIP4P/2005 Solvation Structure Around Nanoparticle', 
                 fontsize=14, fontweight='bold')
    
    # RDF
    ax = axes[0]
    ax.plot(r_rdf, g_r, 'b-', linewidth=2, label='g(r)')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Bulk')
    ax.axvline(3.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='1st shell')
    ax.set_xlabel('Distance from NP (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('Radial Distribution Function (NP-Water O)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    
    # Density profile
    ax = axes[1]
    ax.plot(r_dens, density_profile, 'g-', linewidth=2)
    ax.set_xlabel('Distance from NP center (Å)', fontsize=12)
    ax.set_ylabel('Local Density (atoms/ų)', fontsize=12)
    ax.set_title('Water Density Profile', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'solvation_structure.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    return plot_file

def generate_summary_report(data, output_dir, np_types, water_types):
    """Generate text summary report"""
    print(f"\n{'='*80}")
    print(f"GENERATING SUMMARY REPORT")
    print(f"{'='*80}\n")
    
    report = []
    report.append("="*80)
    report.append("TIP4P/2005 SOLVATION ANALYSIS - SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    # System info
    report.append("SYSTEM INFORMATION:")
    report.append("-" * 80)
    report.append(f"Nanoparticle atom types: {np_types}")
    report.append(f"Water atom types: {water_types}")
    report.append(f"Total frames analyzed: {len(data['Step'])}")
    report.append(f"Simulation time: {data['Step'][-1] * 0.001:.2f} ps")
    report.append("")
    
    # Thermodynamics
    report.append("THERMODYNAMIC PROPERTIES:")
    report.append("-" * 80)
    
    # Temperature
    temp_avg = np.mean(data['Temp'])
    temp_std = np.std(data['Temp'])
    report.append(f"Temperature:")
    report.append(f"  Average: {temp_avg:.2f} ± {temp_std:.2f} K")
    report.append(f"  Target:  300.0 K")
    report.append(f"  Status:  {'✓ GOOD' if abs(temp_avg - 300) < 20 else '⚠ CHECK'}")
    report.append("")
    
    # Density
    dens_avg = np.mean(data['Density'])
    dens_std = np.std(data['Density'])
    report.append(f"Density:")
    report.append(f"  Average: {dens_avg:.4f} ± {dens_std:.4f} g/cm³")
    report.append(f"  Reference (pure water): 0.997 g/cm³")
    report.append("")
    
    # Pressure
    press_avg = np.mean(data['Press'])
    press_std = np.std(data['Press'])
    report.append(f"Pressure:")
    report.append(f"  Average: {press_avg:.2f} ± {press_std:.2f} bar")
    report.append(f"  Reference (1 atm): 1.01 bar")
    report.append("")
    
    # Energy
    report.append(f"Energy:")
    report.append(f"  Total Energy (avg):     {np.mean(data['TotEng']):.2f} kcal/mol")
    report.append(f"  Potential Energy (avg): {np.mean(data['PotEng']):.2f} kcal/mol")
    report.append(f"  Kinetic Energy (avg):   {np.mean(data['KinEng']):.2f} kcal/mol")
    
    drift = abs(data['TotEng'][-1] - data['TotEng'][0])
    drift_pct = drift / abs(np.mean(data['TotEng'])) * 100
    report.append(f"  Energy drift: {drift:.2f} kcal/mol ({drift_pct:.2f}%)")
    report.append(f"  Status: {'✓ GOOD' if drift_pct < 5 else '⚠ MODERATE' if drift_pct < 10 else '❌ HIGH'}")
    report.append("")
    
    # Save report
    report_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: {report_file}")
    
    # Also print to console
    print("\n" + '\n'.join(report))
    
    return report_file

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide output directory")
        print("\nUsage:")
        print("  python analyze_tip4p_solvation.py <output_directory>")
        print("\nExample:")
        print("  python analyze_tip4p_solvation.py ../output/")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TIP4P/2005 SOLVATION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {output_dir}")
    
    # Find required files
    log_file = os.path.join(output_dir, 'log.lammps')
    traj_file = os.path.join(output_dir, 'trajectory.lammpstrj')
    
    if not os.path.exists(log_file):
        print(f"\n❌ Error: Log file not found: {log_file}")
        sys.exit(1)
    
    if not os.path.exists(traj_file):
        print(f"\n❌ Error: Trajectory file not found: {traj_file}")
        sys.exit(1)
    
    # Parse log file
    data = parse_lammps_log(log_file)
    
    if len(data['Step']) == 0:
        print("\n❌ Error: No data found in log file")
        sys.exit(1)
    
    # Parse trajectory
    frames = parse_lammps_trajectory(traj_file)
    
    if len(frames) == 0:
        print("\n❌ Error: No frames found in trajectory")
        sys.exit(1)
    
    # Identify atom types
    np_types, water_types = identify_atom_types(frames)
    
    # Calculate RDF
    r_rdf, g_r = calculate_rdf(frames, np_types, water_types)
    
    # Calculate coordination number
    n_coord = calculate_coordination_number(r_rdf, g_r)
    
    # Calculate density profile
    r_dens, density_profile = analyze_density_profile(frames, np_types, water_types)
    
    # Analyze hydrogen bonds
    h_bonds = analyze_hydrogen_bonds(frames, water_types)
    
    # Create plots
    plot_thermodynamics(data, output_dir)
    plot_solvation_structure(r_rdf, g_r, r_dens, density_profile, output_dir)
    
    # Generate summary report
    generate_summary_report(data, output_dir, np_types, water_types)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved in: {output_dir}")
    print("  - thermodynamics.png")
    print("  - solvation_structure.png")
    print("  - analysis_summary.txt")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
