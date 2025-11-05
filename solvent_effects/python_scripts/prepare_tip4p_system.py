#!/usr/bin/env python3
"""
TIP4P/2005 Water System Preparation for Nanoparticle Solvation
Uses LAMMPS molecule template and SHAKE constraints

TIP4P/2005 is a 4-site water model with excellent properties:
- Virtual M-site for charge (0.1546 Å from O along H-H bisector)
- Full electrostatics
- Works perfectly with SHAKE and MPI
- Reference: Abascal & Vega, J. Chem. Phys. 123, 234505 (2005)

Usage:
    python3 prepare_tip4p_system.py <nanoparticle_file> <num_waters> <box_size>

Example:
    python3 prepare_tip4p_system.py sic_nanoparticle.data 500 40.0
    python3 prepare_tip4p_system.py GO_nanoparticle.data 1000 auto
"""

import numpy as np
import sys
from pathlib import Path

# TIP4P/2005 Parameters
class TIP4P:
    # Geometry (same as TIP4P in molecule file)
    O_H_BOND = 0.9572      # Angstroms
    H_O_H_ANGLE = 104.52   # degrees
    O_M_DIST = 0.1546      # M-site distance from O (along H-H bisector)
    
    # Masses
    MASS_O = 15.9994       # amu
    MASS_H = 1.008         # amu
    MASS_M = 0.0001        # Virtual site (negligible but non-zero for LAMMPS)
    
    # Charges (TIP4P/2005)
    CHARGE_O = 0.0
    CHARGE_H = +0.5564
    CHARGE_M = -1.1128
    
    # LJ Parameters (O-O only, TIP4P/2005)
    EPSILON = 0.1852       # kcal/mol
    SIGMA = 3.1589         # Angstroms

def read_nanoparticle(filename):
    """Read nanoparticle from LAMMPS data file"""
    atoms = []
    masses = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find sections
    masses_start = None
    atoms_start = None
    
    for i, line in enumerate(lines):
        if 'Masses' in line:
            masses_start = i + 2
        elif 'Atoms' in line:
            atoms_start = i + 2
    
    # Read masses
    if masses_start:
        for line in lines[masses_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if any(x in line.lower() for x in ['bonds', 'velocities', 'atoms']):
                break
            parts = line.split()
            if len(parts) >= 2:
                type_id = int(parts[0])
                mass = float(parts[1])
                masses[type_id] = mass
    
    # Read atoms
    if atoms_start:
        for line in lines[atoms_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if any(x in line.lower() for x in ['bonds', 'velocities']):
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
    
    return atoms, masses

def generate_tip4p_geometry():
    """Generate TIP4P water geometry in local frame"""
    # O at origin
    O = np.array([0.0, 0.0, 0.0])
    
    # H atoms
    angle_rad = np.radians(TIP4P.H_O_H_ANGLE / 2)
    H1 = np.array([
        TIP4P.O_H_BOND * np.sin(angle_rad),
        TIP4P.O_H_BOND * np.cos(angle_rad),
        0.0
    ])
    H2 = np.array([
        -TIP4P.O_H_BOND * np.sin(angle_rad),
        TIP4P.O_H_BOND * np.cos(angle_rad),
        0.0
    ])
    
    # M-site (along H-H bisector, towards H atoms)
    M = np.array([0.0, TIP4P.O_M_DIST, 0.0])
    
    return {'O': O, 'H1': H1, 'H2': H2, 'M': M}

def random_rotation():
    """Generate random rotation matrix using quaternions"""
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
    """Place water molecules around nanoparticle using Fibonacci sphere"""
    # Get nanoparticle bounds
    if len(nanoparticle_atoms) == 0:
        np_center = np.array([0.0, 0.0, 0.0])
        np_max_dist = 0.0
    else:
        np_coords = np.array([[a['x'], a['y'], a['z']] for a in nanoparticle_atoms])
        np_center = np.mean(np_coords, axis=0)
        np_max_dist = np.max(np.linalg.norm(np_coords - np_center, axis=1))
    
    # Placement parameters
    min_dist = np_max_dist + 3.0  # 3 Å buffer from NP surface
    max_dist = box_size / 2 - 2.0  # Stay away from box edges
    
    waters = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(num_waters * 10):  # Try more attempts
        if len(waters) >= num_waters:
            break
        
        # Fibonacci sphere point
        theta = 2 * np.pi * i / phi
        cos_phi = 1 - (2 * (i + 0.5) / (num_waters * 10))
        sin_phi = np.sqrt(max(0, 1 - cos_phi**2))
        
        # Random radius
        r = min_dist + (max_dist - min_dist) * np.random.rand()
        
        # Position
        pos = np_center + r * np.array([
            np.cos(theta) * sin_phi,
            np.sin(theta) * sin_phi,
            cos_phi
        ])
        
        # Check minimum distance to other waters (2.5 Å)
        too_close = False
        for w in waters:
            if np.linalg.norm(pos - w['O']) < 2.5:
                too_close = True
                break
        
        if not too_close:
            # Generate water geometry
            geom = generate_tip4p_geometry()
            R = random_rotation()
            
            water = {
                'O': R @ geom['O'] + pos,
                'H1': R @ geom['H1'] + pos,
                'H2': R @ geom['H2'] + pos,
                'M': R @ geom['M'] + pos
            }
            waters.append(water)
    
    return waters

def write_lammps_data(nanoparticle_atoms, np_masses, waters, box_size, filename):
    """Write LAMMPS data file for TIP4P water system"""
    
    # Determine atom types
    # NP types: 1 to num_np_types
    # Water types: O, H, M (consecutive after NP types)
    if len(nanoparticle_atoms) > 0:
        num_np_types = max(a['type'] for a in nanoparticle_atoms)
    else:
        num_np_types = 0
    
    type_O = num_np_types + 1
    type_H = num_np_types + 2
    type_M = num_np_types + 3
    
    # Counts
    num_np_atoms = len(nanoparticle_atoms)
    num_water_atoms = len(waters) * 4  # O, H, H, M
    total_atoms = num_np_atoms + num_water_atoms
    
    with open(filename, 'w') as f:
        f.write("TIP4P/2005 Water + Nanoparticle System\n")
        f.write("# Generated for LAMMPS with molecule template and SHAKE\n\n")
        
        # Counts
        f.write(f"{total_atoms} atoms\n\n")
        f.write(f"{num_np_types + 3} atom types  # NP types + O,H,M\n\n")
        
        # Box
        half_box = box_size / 2
        f.write(f"{-half_box:.6f} {half_box:.6f} xlo xhi\n")
        f.write(f"{-half_box:.6f} {half_box:.6f} ylo yhi\n")
        f.write(f"{-half_box:.6f} {half_box:.6f} zlo zhi\n\n")
        
        # Masses
        f.write("Masses\n\n")
        # NP masses
        for type_id in sorted(np_masses.keys()):
            f.write(f"{type_id} {np_masses[type_id]:.4f}  # NP\n")
        # Water masses
        f.write(f"{type_O} {TIP4P.MASS_O:.4f}  # O (water)\n")
        f.write(f"{type_H} {TIP4P.MASS_H:.4f}  # H (water)\n")
        f.write(f"{type_M} {TIP4P.MASS_M:.4f}  # M (virtual site)\n\n")
        
        # Atoms
        f.write("Atoms  # full\n\n")
        
        atom_id = 1
        
        # Nanoparticle atoms (molecule 1)
        for np_atom in nanoparticle_atoms:
            f.write(f"{atom_id} 1 {np_atom['type']} {np_atom['charge']:.6f} "
                   f"{np_atom['x']:.6f} {np_atom['y']:.6f} {np_atom['z']:.6f}\n")
            atom_id += 1
        
        # Water molecules (molecules 2 to N+1)
        # Note: Coordinates from molecule file, actual positions set later
        for mol_id, water in enumerate(waters, start=2):
            # O
            f.write(f"{atom_id} {mol_id} {type_O} {TIP4P.CHARGE_O:.6f} "
                   f"{water['O'][0]:.6f} {water['O'][1]:.6f} {water['O'][2]:.6f}\n")
            atom_id += 1
            
            # H1
            f.write(f"{atom_id} {mol_id} {type_H} {TIP4P.CHARGE_H:.6f} "
                   f"{water['H1'][0]:.6f} {water['H1'][1]:.6f} {water['H1'][2]:.6f}\n")
            atom_id += 1
            
            # H2
            f.write(f"{atom_id} {mol_id} {type_H} {TIP4P.CHARGE_H:.6f} "
                   f"{water['H2'][0]:.6f} {water['H2'][1]:.6f} {water['H2'][2]:.6f}\n")
            atom_id += 1
            
            # M
            f.write(f"{atom_id} {mol_id} {type_M} {TIP4P.CHARGE_M:.6f} "
                   f"{water['M'][0]:.6f} {water['M'][1]:.6f} {water['M'][2]:.6f}\n")
            atom_id += 1

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 prepare_tip4p_system.py <nanoparticle_file> <num_waters> <box_size|auto>")
        print("\nExamples:")
        print("  python3 prepare_tip4p_system.py sic_nanoparticle.data 500 40.0")
        print("  python3 prepare_tip4p_system.py GO_nanoparticle.data 1000 auto")
        sys.exit(1)
    
    np_file = sys.argv[1]
    num_waters = int(sys.argv[2])
    box_size_arg = sys.argv[3]
    
    # Handle auto box size
    if box_size_arg.lower() == "auto":
        # Estimate: ~30 Å³ per water molecule
        volume = num_waters * 30.0
        box_size = volume ** (1/3)
        print(f"Auto box size: {box_size:.1f} Å (for {num_waters} waters)")
    else:
        box_size = float(box_size_arg)
        print(f"Box size: {box_size:.1f} Å")
    
    # Read nanoparticle
    input_dir = Path(__file__).parent.parent / "input_files"
    np_path = input_dir / np_file
    
    print(f"\nReading nanoparticle: {np_path}")
    nanoparticle_atoms, np_masses = read_nanoparticle(np_path)
    print(f"Found {len(nanoparticle_atoms)} nanoparticle atoms")
    
    if not np_masses:
        # Default masses for common NP types
        np_masses = {1: 28.0855, 2: 12.0107}
    
    # Place waters
    print(f"\nPlacing {num_waters} TIP4P/2005 water molecules...")
    waters = place_water_molecules(nanoparticle_atoms, num_waters, box_size)
    print(f"Successfully placed {len(waters)} waters")
    
    # Write output
    output_name = f"tip4p_system_{len(waters)}waters.data"
    output_file = input_dir / output_name
    write_lammps_data(nanoparticle_atoms, np_masses, waters, box_size, output_file)
    
    # Calculate density
    volume = box_size ** 3
    mass_water = len(waters) * (TIP4P.MASS_O + 2 * TIP4P.MASS_H)  # grams/mol
    mass_np = sum(np_masses.get(a['type'], 28.0) for a in nanoparticle_atoms)
    total_mass = mass_water + mass_np
    # Convert to g/cm³: (g/mol) / (Å³) * N_A / 10^24
    density = (total_mass / volume) * (6.022e23 / 1e24)
    
    print(f"\n✓ System created successfully!")
    print(f"  Output: {output_file}")
    print(f"  Total atoms: {len(nanoparticle_atoms) + len(waters)*4}")
    print(f"  Water molecules: {len(waters)}")
    print(f"  Box size: {box_size:.1f} Å")
    print(f"  Density: {density:.3f} g/cm³")
    print(f"\nReady for LAMMPS with TIP4P/2005 + SHAKE!")

if __name__ == "__main__":
    main()
