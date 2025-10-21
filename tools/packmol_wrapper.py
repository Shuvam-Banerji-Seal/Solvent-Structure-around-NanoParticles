#!/usr/bin/env python3
"""Small PACKMOL wrapper: compute number of waters for target density and write PACKMOL input file.

Usage: python3 tools/packmol_wrapper.py --params configs/params.yaml --solute tools/solute_sphere.pdb --water tools/water_spce.pdb --out data/system.pdb

This will not run packmol automatically unless --run is provided and packmol is on PATH.
"""
import argparse
from pathlib import Path
import yaml
import math

AVOGADRO = 6.02214076e23

parser = argparse.ArgumentParser()
parser.add_argument('--params', required=True)
parser.add_argument('--solute', required=True)
parser.add_argument('--water', required=True)
parser.add_argument('--out', default='data/system.pdb')
parser.add_argument('--run', action='store_true')
args = parser.parse_args()

# BUG FIX #2: Add file validation (was: crashes with FileNotFoundError)
p = Path(args.params)
if not p.exists():
    print(f"❌ ERROR: Config file not found: {args.params}")
    print(f"   Usage: python3 tools/packmol_wrapper.py "
          f"--params configs/params.yaml "
          f"--solute tools/solute_sphere.pdb "
          f"--water tools/water_spce.pdb")
    exit(1)

solute_path = Path(args.solute)
if not solute_path.exists():
    print(f"❌ ERROR: Solute PDB not found: {args.solute}")
    exit(1)

water_path = Path(args.water)
if not water_path.exists():
    print(f"❌ ERROR: Water PDB not found: {args.water}")
    exit(1)

print(f"✓ Config: {args.params}")
print(f"✓ Solute: {args.solute}")
print(f"✓ Water: {args.water}")

with open(p) as f:
    params = yaml.safe_load(f)

box = params.get('box', {}).get('length_A', 60.0)
water_density = params.get('water', {}).get('target_density_g_cm3', 0.997)
water_molar_mass = params.get('water', {}).get('molar_mass_g_mol', 18.015)

# Compute number of water molecules to fill box at target density
# volume (A^3) -> convert to cm^3 by * 1e-24
volume_A3 = box ** 3
volume_cm3 = volume_A3 * 1e-24
mass_g = water_density * volume_cm3
n_moles = mass_g / water_molar_mass
n_molecules = int(round(n_moles * AVOGADRO))

print(f"Box {box} A: volume {volume_A3:.3e} A^3 -> {volume_cm3:.3e} cm^3")
print(f"Target density {water_density} g/cm3 -> {n_molecules} water molecules")

# Safety cap for very large boxes
if n_molecules > 200000:
    print('Warning: computed a very large number of waters; reduce box or use implicit solvent for tests')

in_template = Path('tools/packmol_sphere.inp').read_text()
subs = {
    'SOLUTE_PDB': args.solute,
    'WATER_PDB': args.water,
    'N_WATERS': n_molecules,
    'BOX': box,
    'OUTPUT_PDB': args.out,
    'EXCLUDE_RADIUS': params.get('solute', {}).get('radius_A', 3.0)
}

out_inp = Path('data/packmol_input.inp')
out_inp.write_text(in_template.format(**subs))
print('Wrote PACKMOL input to', out_inp)

if args.run:
    import shutil
    if shutil.which('packmol') is None:
        print('packmol not found on PATH; install packmol to run')
    else:
        import subprocess
        # Run packmol with the generated input by passing the file as stdin
        with open(out_inp, 'r') as inp:
            subprocess.run(['packmol'], stdin=inp, check=True)

