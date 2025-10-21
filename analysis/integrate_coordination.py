#!/usr/bin/env python3
"""Helper utility: given RDF (r,g) compute coordination number up to r_min.

This is a tiny wrapper around compute_rdf routines and useful as a library for plotting.
"""
import numpy as np

def coordination_number(r, g, rmin, density_g_cm3=0.997, molar_mass=18.015):
    rho = density_g_cm3 * 1e-24 / molar_mass * 6.02214076e23
    mask = r <= rmin
    integral = np.trapz(g[mask] * r[mask]**2, r[mask])
    return 4 * np.pi * rho * integral

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('rdf_file')
    parser.add_argument('--rmin', type=float, default=None)
    args = parser.parse_args()
    lines = [l for l in Path(args.rdf_file).read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
    data = []
    for l in lines:
        parts = l.split()
        if len(parts) >= 2:
            data.append((float(parts[0]), float(parts[1])))
    r = np.array([d[0] for d in data])
    g = np.array([d[1] for d in data])
    if args.rmin is None:
        peak_idx = np.argmax(g)
        min_idx = peak_idx + np.argmin(g[peak_idx:])
        rmin = r[min_idx]
    else:
        rmin = args.rmin
    N = coordination_number(r, g, rmin)
    print(f'rmin={rmin:.3f} A, Ncoord={N:.3f}')
