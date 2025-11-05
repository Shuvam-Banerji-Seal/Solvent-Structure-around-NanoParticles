#!/usr/bin/env python3
"""
TIP5P Water System Preparation with SHAKE
Creates LAMMPS data file with bonds and angles for SHAKE constraint

Usage:
    python3 prepare_tip5p_shake.py <num_waters> <box_size> [output_mode]
    
Arguments:
    num_waters: Number of water molecules (e.g., 100, 1000, 5000)
    box_size: Box size in Angstroms, or 'auto' for automatic calculation
    output_mode: 'full_box' (default) or 'minimal' for smaller test systems

Example:
    python3 prepare_tip5p_shake.py 500 auto full_box

Reference: Mahoney & Jorgensen, J. Chem. Phys. 112, 8910 (2000)
"""

import numpy as np
import sys
from pathlib import Path

# TIP5P-E Water Model Parameters (Rick 2004)
class TIP5P:
    # Geometry
    O_H_BOND = 0.9572      # Angstroms
    H_O_H_ANGLE = 104.52   # degrees
    O_L_BOND = 0.70        # Angstroms (lone pair distance)
    L_O_L_ANGLE = 109.47   # degrees
    
    # Masses (amu)
    MASS_O = 15.9994
    MASS_H = 1.008
    MASS_L = 0.0001        # Non-zero to avoid LAMMPS error
    
    # Charges (e)
    CHARGE_O = 0.0
    CHARGE_H = +0.241
    CHARGE_L = -0.241
    
    # LJ Parameters (O-O only)
    EPSILON = 0.1780       # kcal/mol
    SIGMA = 3.097          # Angstroms

def read_nanoparticle(filename):
    """Read nanoparticle from LAMMPS data file"""
    atoms = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find Atoms section
    atoms_start = None
    for i, line in enumerate(lines):
        if 'Atoms' in line:
            atoms_start = i + 2  # Skip header and blank line
            break
    
    if atoms_start is None:
        return []
    
    # Read atoms
    for line in lines[atoms_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            break
        
        parts = line.split()
        if len(parts) >= 4:
            # Try full atom style: id mol type q x y z
            if len(parts) >= 7:
                atom_id = int(parts[0])
                mol_id = int(parts[1])
                atom_type = int(parts[2])
                charge = float(parts[3])
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            # Try atomic style: id type x y z
            elif len(parts) >= 5:
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                mol_id = 1
                charge = 0.0
            else:
                continue
            
            atoms.append({
                'id': atom_id,
                'mol': mol_id,
                'type': atom_type,
                'charge': charge,
                'x': x, 'y': y, 'z': z
            })
    
    return atoms

def generate_tip5p_geometry():
    """Generate TIP5P water geometry in local frame"""
    # O at origin
    O = np.array([0.0, 0.0, 0.0])
    
    # H atoms
    angle_rad = np.radians(TIP5P.H_O_H_ANGLE / 2)
    H1 = np.array([
        TIP5P.O_H_BOND * np.sin(angle_rad),
        TIP5P.O_H_BOND * np.cos(angle_rad),
        0.0
    ])
    H2 = np.array([
        -TIP5P.O_H_BOND * np.sin(angle_rad),
        TIP5P.O_H_BOND * np.cos(angle_rad),
        0.0
    ])
    
    # Lone pairs (below the H-O-H plane)
    lp_angle_rad = np.radians(TIP5P.L_O_L_ANGLE / 2)
    L1 = np.array([
        TIP5P.O_L_BOND * np.sin(lp_angle_rad),
        -TIP5P.O_L_BOND * np.cos(lp_angle_rad),
        0.0
    ])
    L2 = np.array([
        -TIP5P.O_L_BOND * np.sin(lp_angle_rad),
        -TIP5P.O_L_BOND * np.cos(lp_angle_rad),
        0.0
    ])
    
    return {'O': O, 'H1': H1, 'H2': H2, 'L1': L1, 'L2': L2}

def random_rotation():
    """Generate random rotation matrix"""
    # Random quaternion
    u = np.random.rand(3)
    q = np.array([
        np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
        np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
        np.sqrt(u[0]) * np.sin(2*np.pi*u[2]),
        np.sqrt(u[0]) * np.cos(2*np.pi*u[2])
    ])
    
    # Convert to rotation matrix
    q0, q1, q2, q3 = q
    R = np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])
    return R

def place_water_molecules(nanoparticle_atoms, num_waters, box_size):
    """Place water molecules around nanoparticle"""
    # Get nanoparticle bounds
    np_coords = np.array([[a['x'], a['y'], a['z']] for a in nanoparticle_atoms])
    np_center = np.mean(np_coords, axis=0)
    np_max_dist = np.max(np.linalg.norm(np_coords - np_center, axis=1))
    
    # Minimum distance from nanoparticle surface
    min_dist = np_max_dist + 3.0  # 3 Angstrom buffer
    max_dist = box_size / 2 - 2.0  # Stay away from box edges
    
    waters = []
    attempts = 0
    max_attempts = num_waters * 100
    
    # Fibonacci sphere for uniform distribution
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(num_waters):
        placed = False
        for attempt in range(100):
            # Fibonacci sphere point
            theta = 2 * np.pi * i / phi
            cos_phi = 1 - (2 * (i + 0.5) / num_waters)
            sin_phi = np.sqrt(1 - cos_phi**2)
            
            # Random radius
            r = min_dist + (max_dist - min_dist) * np.random.rand()
            
            # Position
            pos = np_center + r * np.array([
                np.cos(theta) * sin_phi,
                np.sin(theta) * sin_phi,
                cos_phi
            ])
            
            # Check minimum distance to other waters (3.0 Angstroms)
            too_close = False
            for w in waters:
                if np.linalg.norm(pos - w['O']) < 3.0:
                    too_close = True
                    break
            
            if not too_close:
                # Generate water geometry
                geom = generate_tip5p_geometry()
                R = random_rotation()
                
                water = {
                    'O': R @ geom['O'] + pos,
                    'H1': R @ geom['H1'] + pos,
                    'H2': R @ geom['H2'] + pos,
                    'L1': R @ geom['L1'] + pos,
                    'L2': R @ geom['L2'] + pos
                }
                waters.append(water)
                placed = True
                break
        
        if not placed:
            print(f"Warning: Could only place {len(waters)} waters out of {num_waters}")
            break
    
    return waters

def write_lammps_data(nanoparticle_atoms, waters, box_size, filename):
    """Write LAMMPS data file with bonds and angles for SHAKE"""
    
    # Atom types: 1,2=NP, 3=O, 4=H, 5=L
    num_np_types = max(a['type'] for a in nanoparticle_atoms)
    type_O = num_np_types + 1
    type_H = num_np_types + 2
    type_L = num_np_types + 3
    
    # Counts
    num_np_atoms = len(nanoparticle_atoms)
    num_water_atoms = len(waters) * 5
    total_atoms = num_np_atoms + num_water_atoms
    
    num_bonds = len(waters) * 4  # 2 O-H + 2 O-L per water
    num_angles = len(waters) * 6  # H-O-H + 4 H-O-L + L-O-L
    
    num_bond_types = 2  # O-H, O-L
    num_angle_types = 3  # H-O-H, H-O-L, L-O-L
    
    with open(filename, 'w') as f:
        f.write("TIP5P Water + Nanoparticle System (with SHAKE)\n\n")
        
        # Counts
        f.write(f"{total_atoms} atoms\n")
        f.write(f"{num_bonds} bonds\n")
        f.write(f"{num_angles} angles\n\n")
        
        f.write(f"{type_L} atom types\n")
        f.write(f"{num_bond_types} bond types\n")
        f.write(f"{num_angle_types} angle types\n\n")
        
        # Box
        f.write(f"0.0 {box_size} xlo xhi\n")
        f.write(f"0.0 {box_size} ylo yhi\n")
        f.write(f"0.0 {box_size} zlo zhi\n\n")
        
        # Masses
        f.write("Masses\n\n")
        for i in range(1, num_np_types + 1):
            if i == 1:
                f.write(f"{i} 28.0855  # Si\n")
            else:
                f.write(f"{i} 12.0107  # C\n")
        f.write(f"{type_O} {TIP5P.MASS_O}  # O (water)\n")
        f.write(f"{type_H} {TIP5P.MASS_H}  # H (water)\n")
        f.write(f"{type_L} {TIP5P.MASS_L}  # L (lone pair)\n\n")
        
        # Atoms
        f.write("Atoms  # full\n\n")
        
        atom_id = 1
        
        # Nanoparticle atoms (molecule 1)
        for np_atom in nanoparticle_atoms:
            f.write(f"{atom_id} 1 {np_atom['type']} {np_atom['charge']:.6f} "
                   f"{np_atom['x']:.6f} {np_atom['y']:.6f} {np_atom['z']:.6f}\n")
            atom_id += 1
        
        # Water molecules (molecules 2 to N+1)
        for mol_id, water in enumerate(waters, start=2):
            # O
            f.write(f"{atom_id} {mol_id} {type_O} {TIP5P.CHARGE_O:.6f} "
                   f"{water['O'][0]:.6f} {water['O'][1]:.6f} {water['O'][2]:.6f}\n")
            o_id = atom_id
            atom_id += 1
            
            # H1
            f.write(f"{atom_id} {mol_id} {type_H} {TIP5P.CHARGE_H:.6f} "
                   f"{water['H1'][0]:.6f} {water['H1'][1]:.6f} {water['H1'][2]:.6f}\n")
            h1_id = atom_id
            atom_id += 1
            
            # H2
            f.write(f"{atom_id} {mol_id} {type_H} {TIP5P.CHARGE_H:.6f} "
                   f"{water['H2'][0]:.6f} {water['H2'][1]:.6f} {water['H2'][2]:.6f}\n")
            h2_id = atom_id
            atom_id += 1
            
            # L1
            f.write(f"{atom_id} {mol_id} {type_L} {TIP5P.CHARGE_L:.6f} "
                   f"{water['L1'][0]:.6f} {water['L1'][1]:.6f} {water['L1'][2]:.6f}\n")
            l1_id = atom_id
            atom_id += 1
            
            # L2
            f.write(f"{atom_id} {mol_id} {type_L} {TIP5P.CHARGE_L:.6f} "
                   f"{water['L2'][0]:.6f} {water['L2'][1]:.6f} {water['L2'][2]:.6f}\n")
            l2_id = atom_id
            atom_id += 1
        
        # Bonds
        f.write("\nBonds\n\n")
        bond_id = 1
        atom_id = num_np_atoms + 1
        
        for water in waters:
            o_id = atom_id
            h1_id = atom_id + 1
            h2_id = atom_id + 2
            l1_id = atom_id + 3
            l2_id = atom_id + 4
            
            f.write(f"{bond_id} 1 {o_id} {h1_id}  # O-H1\n")
            bond_id += 1
            f.write(f"{bond_id} 1 {o_id} {h2_id}  # O-H2\n")
            bond_id += 1
            f.write(f"{bond_id} 2 {o_id} {l1_id}  # O-L1\n")
            bond_id += 1
            f.write(f"{bond_id} 2 {o_id} {l2_id}  # O-L2\n")
            bond_id += 1
            
            atom_id += 5
        
        # Angles
        f.write("\nAngles\n\n")
        angle_id = 1
        atom_id = num_np_atoms + 1
        
        for water in waters:
            o_id = atom_id
            h1_id = atom_id + 1
            h2_id = atom_id + 2
            l1_id = atom_id + 3
            l2_id = atom_id + 4
            
            f.write(f"{angle_id} 1 {h1_id} {o_id} {h2_id}  # H-O-H\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h1_id} {o_id} {l1_id}  # H-O-L\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h1_id} {o_id} {l2_id}  # H-O-L\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h2_id} {o_id} {l1_id}  # H-O-L\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h2_id} {o_id} {l2_id}  # H-O-L\n")
            angle_id += 1
            f.write(f"{angle_id} 3 {l1_id} {o_id} {l2_id}  # L-O-L\n")
            angle_id += 1
            
            atom_id += 5

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 prepare_tip5p_shake.py <num_waters> <box_size> [output_mode]")
        print("Example: python3 prepare_tip5p_shake.py 100 auto full_box")
        sys.exit(1)
    
    num_waters = int(sys.argv[1])
    box_size_arg = sys.argv[2]
    output_mode = sys.argv[3] if len(sys.argv) > 3 else "full_box"
    
    # Handle "auto" box size
    if box_size_arg.lower() == "auto":
        # Auto-calculate based on number of waters
        # Assume ~30 Å³ per water molecule
        volume = num_waters * 30.0
        box_size = volume ** (1/3)
    else:
        box_size = float(box_size_arg)
    
    print(f"Creating TIP5P system with {num_waters} waters")
    print(f"Box size: {box_size:.1f} Angstroms")
    
    # Read nanoparticle
    np_file = Path(__file__).parent.parent / "input_files" / "sic_nanoparticle.data"
    nanoparticle_atoms = read_nanoparticle(np_file)
    
    if not nanoparticle_atoms:
        print(f"ERROR: Could not read nanoparticle from {np_file}")
        sys.exit(1)
    
    print(f"Found {len(nanoparticle_atoms)} nanoparticle atoms")
    
    # Place waters
    waters = place_water_molecules(nanoparticle_atoms, num_waters, box_size)
    print(f"Successfully placed {len(waters)} TIP5P water molecules")
    
    # Write output
    output_file = Path(__file__).parent.parent / "input_files" / f"tip5p_system_{len(waters)}waters.data"
    write_lammps_data(nanoparticle_atoms, waters, box_size, output_file)
    
    print(f"\nCreated: {output_file}")
    print(f"Total atoms: {len(nanoparticle_atoms) + len(waters)*5}")
    print(f"Bonds: {len(waters)*4}")
    print(f"Angles: {len(waters)*6}")
    print(f"\nReady for LAMMPS with SHAKE constraint!")

if __name__ == "__main__":
    main()
