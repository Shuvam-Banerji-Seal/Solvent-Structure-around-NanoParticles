#!/usr/bin/env python3
"""
Simple TIP5P System Generator - No Bonds/Angles (for rigid/small)
"""

import numpy as np

# Read nanoparticle (simplified - hardcoded)
np_atoms = [
    (1, 1, 0.0, 0.0, 0.0, 0.0),  # Si at origin
]

# TIP5P geometry
def create_water(center):
    """Create TIP5P water at center with random orientation"""
    # Random rotation
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.random.uniform(0, np.pi)
    psi = np.random.uniform(0, 2*np.pi)
    
    # Local geometry
    O = np.array([0.0, 0.0, 0.0])
    H1 = np.array([0.7585, 0.5876, 0.0])
    H2 = np.array([-0.7585, 0.5876, 0.0])
    L1 = np.array([0.4311, -0.5522, 0.0])
    L2 = np.array([-0.4311, -0.5522, 0.0])
    
    # Rotation matrix
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R = Rz @ Ry @ Rx
    
    # Rotate and translate
    return {
        'O': R @ O + center,
        'H1': R @ H1 + center,
        'H2': R @ H2 + center,
        'L1': R @ L1 + center,
        'L2': R @ L2 + center
    }

# Place waters
box = 40.0
waters = []
for i in range(100):
    # Random position
    pos = np.random.uniform(-box/4, box/4, 3)
    if np.linalg.norm(pos) > 3.0:  # Stay away from NP
        waters.append(create_water(pos))

print(f"Placed {len(waters)} waters")

# Write data file
with open('../input_files/tip5p_simple_100waters.data', 'w') as f:
    f.write("TIP5P System - No Bonds/Angles (for rigid/small)\n\n")
    f.write(f"{1 + len(waters)*5} atoms\n\n")
    f.write("5 atom types\n\n")
    f.write(f"-{box/2} {box/2} xlo xhi\n")
    f.write(f"-{box/2} {box/2} ylo yhi\n")
    f.write(f"-{box/2} {box/2} zlo zhi\n\n")
    
    f.write("Masses\n\n")
    f.write("1 28.0855\n")
    f.write("2 12.0107\n")
    f.write("3 15.9994\n")
    f.write("4 1.008\n")
    f.write("5 0.0001\n\n")
    
    f.write("Atoms  # full\n\n")
    
    # NP
    f.write("1 1 1 0.0 0.0 0.0 0.0\n")
    
    # Waters
    atom_id = 2
    for mol_id, w in enumerate(waters, start=2):
        f.write(f"{atom_id} {mol_id} 3 0.000 {w['O'][0]:.6f} {w['O'][1]:.6f} {w['O'][2]:.6f}\n")
        atom_id += 1
        f.write(f"{atom_id} {mol_id} 4 0.241 {w['H1'][0]:.6f} {w['H1'][1]:.6f} {w['H1'][2]:.6f}\n")
        atom_id += 1
        f.write(f"{atom_id} {mol_id} 4 0.241 {w['H2'][0]:.6f} {w['H2'][1]:.6f} {w['H2'][2]:.6f}\n")
        atom_id += 1
        f.write(f"{atom_id} {mol_id} 5 -0.241 {w['L1'][0]:.6f} {w['L1'][1]:.6f} {w['L1'][2]:.6f}\n")
        atom_id += 1
        f.write(f"{atom_id} {mol_id} 5 -0.241 {w['L2'][0]:.6f} {w['L2'][1]:.6f} {w['L2'][2]:.6f}\n")
        atom_id += 1

print("Created: ../input_files/tip5p_simple_100waters.data")
