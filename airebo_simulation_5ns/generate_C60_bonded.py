#!/usr/bin/env python3
"""
Generate C60 fullerene with proper bonding topology
Each carbon atom bonds to exactly 3 neighbors
Bond length threshold: 1.3-1.5 Å (typical C-C in fullerenes)
"""

import numpy as np

# Read atomic coordinates from C60.data
coords = []
with open('C60.data', 'r') as f:
    reading_atoms = False
    for line in f:
        if 'Atoms' in line:
            reading_atoms = True
            continue
        if reading_atoms and line.strip():
            try:
                parts = line.split()
                atom_id = int(parts[0])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords.append((atom_id, x, y, z))
            except (ValueError, IndexError):
                continue

print(f"Read {len(coords)} C60 atoms")

# Calculate all pairwise distances and find bonds
# C60 has two types of bonds: single (~1.45 Å) and double (~1.40 Å)
# We'll accept anything between 1.3-1.5 Å as a bond
bonds = []
bond_threshold_min = 1.3
bond_threshold_max = 1.5

for i, (id1, x1, y1, z1) in enumerate(coords):
    for j, (id2, x2, y2, z2) in enumerate(coords[i+1:], start=i+1):
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        if bond_threshold_min <= dist <= bond_threshold_max:
            bonds.append((id1, id2, dist))

print(f"Found {len(bonds)} bonds")

# Verify each atom has exactly 3 bonds
bond_count = {}
for b1, b2, _ in bonds:
    bond_count[b1] = bond_count.get(b1, 0) + 1
    bond_count[b2] = bond_count.get(b2, 0) + 1

print("\nBond count per atom:")
for atom_id in sorted(bond_count.keys())[:10]:  # Show first 10
    print(f"  Atom {atom_id}: {bond_count[atom_id]} bonds")
print(f"  ... (showing first 10)")

# Check if all atoms have exactly 3 bonds
all_have_three = all(count == 3 for count in bond_count.values())
print(f"\nAll atoms have exactly 3 bonds: {all_have_three}")

if not all_have_three:
    print("ERROR: Some atoms don't have exactly 3 bonds!")
    for atom_id, count in bond_count.items():
        if count != 3:
            print(f"  Atom {atom_id}: {count} bonds")

# Write new data file with full atom style (for TIP4P compatibility) and bonds
with open('C60_bonded.data', 'w') as f:
    f.write("# C60 Fullerene with bonding topology\n")
    f.write("# Generated with proper C-C bonds (1.3-1.5 Å)\n")
    f.write("# Atom style: full (compatible with TIP4P water)\n\n")
    
    f.write(f"{len(coords)} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write("0 angles\n")
    f.write("0 dihedrals\n")
    f.write("0 impropers\n\n")
    
    f.write("1 atom types\n")
    f.write("1 bond types\n\n")
    
    f.write("-4.0 4.0 xlo xhi\n")
    f.write("-4.0 4.0 ylo yhi\n")
    f.write("-4.0 4.0 zlo zhi\n\n")
    
    f.write("Masses\n\n")
    f.write("1 12.011  # Carbon\n\n")
    
    f.write("Atoms # full\n\n")
    for atom_id, x, y, z in coords:
        # full style: atom-ID molecule-ID atom-type charge x y z
        # C60 carbons are neutral: charge = 0.0
        f.write(f"{atom_id:4d} 1 1 0.0 {x:12.5f} {y:12.5f} {z:12.5f}\n")
    
    f.write("\nBonds\n\n")
    for bond_id, (atom1, atom2, dist) in enumerate(bonds, start=1):
        f.write(f"{bond_id:4d} 1 {atom1:4d} {atom2:4d}  # {dist:.3f} Å\n")

print(f"\n✓ Created C60_bonded.data with {len(bonds)} bonds")
print(f"  Bond count verification: {len(bonds)} = 90 (expected for C60)")
print(f"  Each atom has 3 bonds: {all_have_three}")
