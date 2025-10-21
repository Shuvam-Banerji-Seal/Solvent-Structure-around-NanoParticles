#!/usr/bin/env python3
"""Convert 3-site water PDB (O,H,H) into 4-site TIP4P-like PDB (adds M sites).

Usage: python3 tools/convert_to_tip4p.py input_3site.pdb output_4site.pdb --d_OM 0.15

This script assumes waters are formatted in the PDB as consecutive O H H records for each water.
It adds a new atom named M at position r_M = r_O + d_OM * unit(bisector(H1,H2)).
"""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', help='input PDB with O H H triplets')
parser.add_argument('output', help='output PDB with added M atoms')
parser.add_argument('--d_OM', type=float, default=0.15, help='distance in Angstrom for M site from O along bisector')
args = parser.parse_args()

lines = Path(args.input).read_text().splitlines()

# parse ATOM/HETATM lines and group in triplets
waters = []
current = []
for ln in lines:
    if ln.startswith(('ATOM', 'HETATM')):
        atomname = ln[12:16].strip()
        x = float(ln[30:38])
        y = float(ln[38:46])
        z = float(ln[46:54])
        current.append((ln, atomname, np.array([x,y,z])))
        if len(current) == 3:
            waters.append(current)
            current = []

out_lines = []
serial = 1
for water_idx, water in enumerate(waters):
    # Expect ordering: O, H1, H2 (if not, we still compute)
    coords = [w[2] for w in water]
    rO = coords[0]
    rH1 = coords[1]
    rH2 = coords[2]
    
    # BUG FIX #4: Validate water geometry (was: no checks)
    # Check O-H distances are reasonable (typical: 0.95-0.98 Å)
    d_OH1 = np.linalg.norm(rH1 - rO)
    d_OH2 = np.linalg.norm(rH2 - rO)
    
    if not (0.85 < d_OH1 < 1.15) or not (0.85 < d_OH2 < 1.15):
        print(f"⚠️  Warning: Water {water_idx+1} has unusual O-H distances:")
        print(f"   d_OH1 = {d_OH1:.3f} Å, d_OH2 = {d_OH2:.3f} Å")
        print(f"   Expected: ~0.95-0.98 Å. Proceeding anyway...")
    
    # Check H-O-H angle is reasonable (typical: 104.5°)
    v1 = rH1 - rO
    v2 = rH2 - rO
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    if not (95 < angle_deg < 115):
        print(f"⚠️  Warning: Water {water_idx+1} has unusual H-O-H angle: {angle_deg:.1f}°")
        print(f"   Expected: ~104.5°. Proceeding anyway...")
    
    # bisector unit vector
    u = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
    u_norm = u / np.linalg.norm(u)
    rM = rO + args.d_OM * u_norm
    # write original three atoms but with new serial numbers
    for atom in water:
        ln = atom[0]
        # replace serial number (columns 6-11) and keep coordinates
        name = atom[1]
        x,y,z = atom[2]
        out_lines.append(f"ATOM  {serial:5d} {name:<4s} MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}")
        serial += 1
    # add M site
    out_lines.append(f"HETATM{serial:5d}  M   MOL     1    {rM[0]:8.3f}{rM[1]:8.3f}{rM[2]:8.3f}  1.00  0.00           M")
    serial += 1

# write PDB
Path(args.output).write_text('\n'.join(out_lines) + '\n')
print('Wrote TIP4P-like PDB with M sites to', args.output)
