#!/usr/bin/env python3
"""
TIP5P Water Placement Around Nanoparticle

Creates a system with SiC nanoparticle surrounded by TIP5P water molecules.
TIP5P is a 5-site water model with explicit lone pair sites.

Author: Automated System
Date: 2025-11-04
Reference: Rick, S.W., J. Chem. Phys. 120, 6085-6093 (2004)
"""

import numpy as np
import sys
from pathlib import Path

# TIP5P Water Model Parameters
# Reference: https://docs.lammps.org/Howto_tip5p.html#rick

class TIP5PParameters:
    """TIP5P water model parameters"""
    
    # Geometry
    O_H_BOND = 0.9572      # Å
    H_O_H_ANGLE = 104.52   # degrees
    O_L_BOND = 0.70        # Å (lone pair distance from O)
    L_O_L_ANGLE = 109.47   # degrees
    
    # Masses (amu)
    MASS_O = 15.9994
    MASS_H = 1.008
    MASS_L = 0.0  # Lone pairs are massless virtual sites
    
    # Charges (electron charge units)
    CHARGE_O = 0.0
    CHARGE_H = +0.241
    CHARGE_L = -0.241
    
    # LJ Parameters (for oxygen only)
    EPSILON_O = 0.16  # kcal/mol
    SIGMA_O = 3.12    # Å
    
    # Convert to LAMMPS real units
    @classmethod
    def get_lj_params(cls):
        """Return LJ parameters in LAMMPS real units"""
        return cls.EPSILON_O, cls.SIGMA_O

class WaterMolecule:
    """Represents a single TIP5P water molecule"""
    
    def __init__(self, center, orientation=None):
        """
        Create TIP5P water molecule
        
        Args:
            center: Position of oxygen atom (3D array)
            orientation: Optional rotation matrix for molecule orientation
        """
        self.O_pos = np.array(center, dtype=float)
        
        if orientation is None:
            # Random orientation
            orientation = self._random_rotation()
        
        # Generate atom positions in local frame
        local_positions = self._generate_local_geometry()
        
        # Rotate and translate to world frame
        self.H1_pos = orientation @ local_positions['H1'] + self.O_pos
        self.H2_pos = orientation @ local_positions['H2'] + self.O_pos
        self.L1_pos = orientation @ local_positions['L1'] + self.O_pos
        self.L2_pos = orientation @ local_positions['L2'] + self.O_pos
    
    def _generate_local_geometry(self):
        """Generate TIP5P geometry in local coordinate frame"""
        # Place O at origin
        O = np.array([0.0, 0.0, 0.0])
        
        # Place H atoms using H-O-H angle
        half_angle = np.radians(TIP5PParameters.H_O_H_ANGLE / 2.0)
        bond_length = TIP5PParameters.O_H_BOND
        
        H1 = np.array([
            bond_length * np.sin(half_angle),
            0.0,
            bond_length * np.cos(half_angle)
        ])
        
        H2 = np.array([
            -bond_length * np.sin(half_angle),
            0.0,
            bond_length * np.cos(half_angle)
        ])
        
        # Place lone pair sites
        # Lone pairs are in tetrahedral arrangement
        lp_distance = TIP5PParameters.O_L_BOND
        lp_half_angle = np.radians(TIP5PParameters.L_O_L_ANGLE / 2.0)
        
        # Lone pairs point opposite to H atoms (behind O)
        L1 = np.array([
            lp_distance * np.sin(lp_half_angle),
            0.0,
            -lp_distance * np.cos(lp_half_angle)
        ])
        
        L2 = np.array([
            -lp_distance * np.sin(lp_half_angle),
            0.0,
            -lp_distance * np.cos(lp_half_angle)
        ])
        
        return {'H1': H1, 'H2': H2, 'L1': L1, 'L2': L2}
    
    def _random_rotation(self):
        """Generate random rotation matrix using Euler angles"""
        # Random Euler angles
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        psi = np.random.uniform(0, 2*np.pi)
        
        # Rotation matrix
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        
        R = np.array([
            [cos_psi*cos_phi - cos_theta*sin_phi*sin_psi,
             cos_psi*sin_phi + cos_theta*cos_phi*sin_psi,
             sin_psi*sin_theta],
            [-sin_psi*cos_phi - cos_theta*sin_phi*cos_psi,
             -sin_psi*sin_phi + cos_theta*cos_phi*cos_psi,
             cos_psi*sin_theta],
            [sin_theta*sin_phi,
             -sin_theta*cos_phi,
             cos_theta]
        ])
        
        return R
    
    def check_overlap(self, other_positions, min_distance=2.5):
        """
        Check if water overlaps with other atoms
        
        Args:
            other_positions: List of atom positions
            min_distance: Minimum allowed distance (Å)
        
        Returns:
            True if overlap detected, False otherwise
        """
        my_positions = [self.O_pos, self.H1_pos, self.H2_pos]
        
        for my_pos in my_positions:
            for other_pos in other_positions:
                distance = np.linalg.norm(my_pos - other_pos)
                if distance < min_distance:
                    return True
        
        return False

def read_nanoparticle(filename):
    """Read SiC nanoparticle from LAMMPS data file"""
    positions = []
    types = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    in_atoms = False
    for line in lines:
        if 'Atoms' in line:
            in_atoms = True
            continue
        
        if in_atoms and line.strip() and not line.startswith('#'):
            parts = line.split()
            
            # Skip empty lines or section headers
            if len(parts) == 0 or parts[0] in ['Velocities', 'Bonds', 'Angles', 'Masses']:
                break
            
            # Check if this is an atom line
            if parts[0].isdigit():
                # Try to parse based on number of columns
                if len(parts) >= 7:
                    # Full atom style: id mol type q x y z
                    atom_type = int(parts[2])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                elif len(parts) >= 5:
                    # Atomic style: id type x y z
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                else:
                    continue
                
                positions.append([x, y, z])
                types.append(atom_type)
    
    return np.array(positions), np.array(types)

def place_water_grid(np_positions, box_size, n_waters, strategy='full_box'):
    """
    Place water molecules around nanoparticle
    
    Args:
        np_positions: Nanoparticle atom positions
        box_size: Simulation box size (cubic)
        n_waters: Number of water molecules to place
        strategy: 'full_box' or 'shell' (solvation shell only)
    
    Returns:
        List of WaterMolecule objects
    """
    waters = []
    np_center = np.mean(np_positions, axis=0)
    
    if strategy == 'shell':
        # Place in spherical shell around NP
        shell_inner = 3.0  # Å
        shell_outer = min(25.0, box_size / 2 - 5)
    
    attempts = 0
    max_attempts = n_waters * 100
    
    # Use grid-based placement for efficiency
    grid_spacing = 3.1  # Å (approximate water spacing)
    n_per_dim = int(box_size / grid_spacing)
    
    print(f"Placing {n_waters} TIP5P water molecules...")
    print(f"Strategy: {strategy}")
    
    # Generate potential positions on grid
    potential_positions = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                pos = np.array([
                    i * grid_spacing - box_size/2 + box_size/(2*n_per_dim),
                    j * grid_spacing - box_size/2 + box_size/(2*n_per_dim),
                    k * grid_spacing - box_size/2 + box_size/(2*n_per_dim)
                ])
                
                distance_from_center = np.linalg.norm(pos - np_center)
                
                if strategy == 'shell':
                    if shell_inner < distance_from_center < shell_outer:
                        potential_positions.append(pos)
                else:  # full_box
                    # Check if within box bounds
                    if np.all(np.abs(pos) < box_size/2 - 2):
                        potential_positions.append(pos)
    
    # Shuffle to avoid bias
    np.random.shuffle(potential_positions)
    
    # Collect all atom positions (NP + already placed water)
    def get_all_positions():
        all_pos = list(np_positions)
        for water in waters:
            all_pos.extend([water.O_pos, water.H1_pos, water.H2_pos])
        return all_pos
    
    # Place waters
    for pos in potential_positions:
        if len(waters) >= n_waters:
            break
        
        # Try to place water at this position
        water = WaterMolecule(pos)
        
        # Check overlap with NP and other waters
        all_positions = get_all_positions()
        if not water.check_overlap(all_positions, min_distance=2.5):
            waters.append(water)
            
            if len(waters) % 100 == 0:
                print(f"  Placed {len(waters)}/{n_waters} waters...")
        
        attempts += 1
        if attempts > max_attempts:
            print(f"WARNING: Could only place {len(waters)} waters after {attempts} attempts")
            break
    
    print(f"Successfully placed {len(waters)} TIP5P water molecules")
    return waters

def write_lammps_data(np_positions, np_types, waters, box_size, output_file):
    """Write LAMMPS data file with TIP5P water"""
    
    n_np_atoms = len(np_positions)
    n_waters = len(waters)
    n_water_sites = n_waters * 5  # O, H, H, L, L per water
    n_total = n_np_atoms + n_water_sites
    
    # Atom types: 1=Si, 2=C, 3=O(water), 4=H(water), 5=L(water)
    # Bond types: 1=O-H, 2=O-L
    # Angle types: 1=H-O-H, 2=H-O-L, 3=L-O-L
    
    n_bonds = n_waters * 4  # 2 O-H + 2 O-L per water
    n_angles = n_waters * 6  # H-O-H + 4×H-O-L + L-O-L per water
    
    with open(output_file, 'w') as f:
        # Header
        f.write("# TIP5P Water + SiC Nanoparticle System\n")
        f.write("# Generated by prepare_tip5p_system.py\n")
        f.write(f"# System: {n_np_atoms} NP atoms + {n_waters} TIP5P water molecules\n")
        f.write("# Using SHAKE for rigid water geometry\n\n")
        
        f.write(f"{n_total} atoms\n")
        f.write(f"{n_bonds} bonds\n")
        f.write(f"{n_angles} angles\n")
        f.write("0 dihedrals\n")
        f.write("0 impropers\n\n")
        
        f.write("5 atom types  # 1=Si, 2=C, 3=O(water), 4=H(water), 5=L(water)\n")
        f.write("2 bond types  # 1=O-H, 2=O-L\n")
        f.write("3 angle types  # 1=H-O-H, 2=H-O-L, 3=L-O-L\n\n")
        
        # Box
        half_box = box_size / 2
        f.write(f"{-half_box:.6f} {half_box:.6f} xlo xhi\n")
        f.write(f"{-half_box:.6f} {half_box:.6f} ylo yhi\n")
        f.write(f"{-half_box:.6f} {half_box:.6f} zlo zhi\n\n")
        
        # Masses
        # Note: LAMMPS doesn't allow zero mass in data file, so we give lone pairs
        # a tiny mass (0.0001) which will be effectively ignored
        f.write("Masses\n\n")
        f.write("1 28.0855  # Si\n")
        f.write("2 12.0107  # C\n")
        f.write(f"3 {TIP5PParameters.MASS_O}  # O (water)\n")
        f.write(f"4 {TIP5PParameters.MASS_H}  # H (water)\n")
        f.write(f"5 0.0001  # L (lone pair, virtual site with negligible mass)\n\n")
        
        # Atoms
        f.write("Atoms  # full\n\n")
        
        atom_id = 1
        mol_id = 1
        
        # Write nanoparticle atoms
        for i, (pos, atype) in enumerate(zip(np_positions, np_types)):
            charge = 0.0
            f.write(f"{atom_id} {mol_id} {atype} {charge:.4f} ")
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            atom_id += 1
        
        mol_id = 2  # Start water molecules at mol_id 2
        
        # Write water molecules
        for water in waters:
            # Oxygen
            f.write(f"{atom_id} {mol_id} 3 {TIP5PParameters.CHARGE_O:.4f} ")
            f.write(f"{water.O_pos[0]:.6f} {water.O_pos[1]:.6f} {water.O_pos[2]:.6f}\n")
            o_id = atom_id
            atom_id += 1
            
            # Hydrogen 1
            f.write(f"{atom_id} {mol_id} 4 {TIP5PParameters.CHARGE_H:.4f} ")
            f.write(f"{water.H1_pos[0]:.6f} {water.H1_pos[1]:.6f} {water.H1_pos[2]:.6f}\n")
            h1_id = atom_id
            atom_id += 1
            
            # Hydrogen 2
            f.write(f"{atom_id} {mol_id} 4 {TIP5PParameters.CHARGE_H:.4f} ")
            f.write(f"{water.H2_pos[0]:.6f} {water.H2_pos[1]:.6f} {water.H2_pos[2]:.6f}\n")
            h2_id = atom_id
            atom_id += 1
            
            # Lone pair 1
            f.write(f"{atom_id} {mol_id} 5 {TIP5PParameters.CHARGE_L:.4f} ")
            f.write(f"{water.L1_pos[0]:.6f} {water.L1_pos[1]:.6f} {water.L1_pos[2]:.6f}\n")
            l1_id = atom_id
            atom_id += 1
            
            # Lone pair 2
            f.write(f"{atom_id} {mol_id} 5 {TIP5PParameters.CHARGE_L:.4f} ")
            f.write(f"{water.L2_pos[0]:.6f} {water.L2_pos[1]:.6f} {water.L2_pos[2]:.6f}\n")
            l2_id = atom_id
            atom_id += 1
            
            mol_id += 1
        
        # Write bonds
        f.write("\nBonds\n\n")
        bond_id = 1
        atom_base = n_np_atoms + 1  # First water oxygen
        for i in range(n_waters):
            o_id = atom_base + i * 5
            h1_id = o_id + 1
            h2_id = o_id + 2
            l1_id = o_id + 3
            l2_id = o_id + 4
            
            # O-H bonds (type 1)
            f.write(f"{bond_id} 1 {o_id} {h1_id}\n")
            bond_id += 1
            f.write(f"{bond_id} 1 {o_id} {h2_id}\n")
            bond_id += 1
            
            # O-L bonds (type 2)
            f.write(f"{bond_id} 2 {o_id} {l1_id}\n")
            bond_id += 1
            f.write(f"{bond_id} 2 {o_id} {l2_id}\n")
            bond_id += 1
        
        # Write angles
        f.write("\nAngles\n\n")
        angle_id = 1
        atom_base = n_np_atoms + 1
        for i in range(n_waters):
            o_id = atom_base + i * 5
            h1_id = o_id + 1
            h2_id = o_id + 2
            l1_id = o_id + 3
            l2_id = o_id + 4
            
            # H-O-H angle (type 1)
            f.write(f"{angle_id} 1 {h1_id} {o_id} {h2_id}\n")
            angle_id += 1
            
            # H-O-L angles (type 2)
            f.write(f"{angle_id} 2 {h1_id} {o_id} {l1_id}\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h1_id} {o_id} {l2_id}\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h2_id} {o_id} {l1_id}\n")
            angle_id += 1
            f.write(f"{angle_id} 2 {h2_id} {o_id} {l2_id}\n")
            angle_id += 1
            
            # L-O-L angle (type 3)
            f.write(f"{angle_id} 3 {l1_id} {o_id} {l2_id}\n")
            angle_id += 1
    
    print(f"\nWrote LAMMPS data file: {output_file}")
    print(f"  Total atoms: {n_total}")
    print(f"  NP atoms: {n_np_atoms}")
    print(f"  Water molecules: {n_waters} ({n_water_sites} sites)")
    print(f"  Bonds: {n_bonds} (2×O-H + 2×O-L per water)")
    print(f"  Angles: {n_angles} (H-O-H + 4×H-O-L + L-O-L per water)")
    print(f"  Using SHAKE constraints for rigid water geometry")

def main():
    """Main function"""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python prepare_tip5p_system.py <n_waters> [box_size] [strategy]")
        print("  n_waters: Number of water molecules")
        print("  box_size: Box size in Å (default: auto-calculate)")
        print("  strategy: 'full_box' or 'shell' (default: full_box)")
        sys.exit(1)
    
    n_waters = int(sys.argv[1])
    
    # Parse box_size - handle "auto" keyword
    if len(sys.argv) > 2:
        box_size_arg = sys.argv[2]
        if box_size_arg.lower() in ['auto', 'none']:
            box_size = None
        else:
            box_size = float(box_size_arg)
    else:
        box_size = None
    
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'full_box'
    
    # Read nanoparticle
    np_file = Path(__file__).parent.parent / 'input_files' / 'sic_nanoparticle.data'
    print(f"Reading nanoparticle from: {np_file}")
    np_positions, np_types = read_nanoparticle(np_file)
    print(f"  Found {len(np_positions)} nanoparticle atoms")
    
    # Calculate box size if not provided
    if box_size is None:
        # Calculate box size for target density (~1.0 g/cm³)
        # Water: 18.015 g/mol, density 1.0 g/cm³, Avogadro = 6.022e23
        water_volume_per_molecule = 18.015 / (1.0 * 6.022e23) * 1e24  # ų
        total_water_volume = n_waters * water_volume_per_molecule
        box_size = total_water_volume ** (1/3) + 10  # Add 10 Å padding
        print(f"  Auto-calculated box size: {box_size:.1f} Å")
    
    # Place waters
    np.random.seed(42)  # Reproducibility
    waters = place_water_grid(np_positions, box_size, n_waters, strategy)
    
    # Write output
    output_dir = Path(__file__).parent.parent / 'input_files'
    output_file = output_dir / f'tip5p_system_{len(waters)}waters.data'
    write_lammps_data(np_positions, np_types, waters, box_size, output_file)
    
    print(f"\n✓ System preparation complete!")
    print(f"  Output: {output_file}")
    print(f"  Box size: {box_size:.1f} Å")
    print(f"  Density estimate: {(len(waters) * 18.015 / (box_size**3 * 6.022e-4)):.3f} g/cm³")

if __name__ == "__main__":
    main()
