#!/usr/bin/env python3
"""Parse LAMMPS compute rdf output (fix ave/time mode vector) and compute hydration number.

Usage: python3 analysis/compute_rdf.py rdf_solute_O.dat --density 0.997 --rmin 3.5

If --rmin is not provided the script attempts to find first minimum after the first peak.

NOTE (BUG #5 - Known Limitation):
Current implementation assumes single solute atom. For multi-atom solutes (e.g., clusters),
use center-of-mass (COM) calculation instead. The LAMMPS compute rdf command should be:
  compute rdf_com all rdf all all
with appropriate group definitions to get COM-based RDF.
"""
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('rdffile')
parser.add_argument('--density', type=float, default=0.997, help='water density g/cm3')
parser.add_argument('--rmin', type=float, default=None, help='first minimum (Angstrom). If None find automatically')
parser.add_argument('--molar_mass', type=float, default=18.015)
args = parser.parse_args()

lines = [l for l in Path(args.rdffile).read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
# Expect two columns: r g(r) or more, so parse first two numbers per line
data = []
for l in lines:
    parts = l.split()
    if len(parts) >= 2:
        try:
            r = float(parts[0])
            g = float(parts[1])
            data.append((r,g))
        except Exception:
            continue
arr = np.array(data)
if arr.size == 0:
    raise SystemExit('No data parsed from RDF file')
r = arr[:,0]
g = arr[:,1]
# find first peak and subsequent minimum
if args.rmin is None:
    peak_idx = np.argmax(g)
    # search for the first minimum after peak
    min_idx = peak_idx + np.argmin(g[peak_idx:])
    rmin = r[min_idx]
else:
    rmin = args.rmin
# compute bulk number density in molecules / A^3
rho = args.density * 1e-24 / args.molar_mass * 6.02214076e23  # molecules per A^3
# integrate 4*pi*rho*int_0^rmin g(r) r^2 dr
mask = r <= rmin
integral = np.trapz(g[mask] * r[mask]**2, r[mask])
Ncoord = 4 * np.pi * rho * integral
print(f'Found rmin = {rmin:.3f} A')
print(f'Coordination (integrated to rmin): {Ncoord:.3f} molecules')
