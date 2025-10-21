#!/usr/bin/env python3
"""Generate per-epsilon/per-replica LAMMPS inputs from the template.

Usage: python3 scripts/sweep_eps.py configs/params.yaml in/cg_sphere.in.template

This script creates directories under ./experiments/eps_{value}/replica_{i}/
and writes the LAMMPS input file (run.in), and a small run helper (run.sh).
"""
import sys
import os
import math
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

if len(sys.argv) < 3:
    print("Usage: sweep_eps.py <params.yaml> <template.in>")
    sys.exit(1)

params_path = Path(sys.argv[1])
template_path = Path(sys.argv[2])

import re, ast

with open(params_path) as f:
    if yaml is not None:
        params = yaml.safe_load(f)
    else:
        # Minimal fallback parsing for simple YAML files without requiring PyYAML.
        text = f.read()
        # epsilon list like: epsilon_sweep_kcalmol: [0.02, 0.05, 0.1]
        eps_match = re.search(r'epsilon_sweep_kcalmol\s*:\s*(\[[^\]]+\])', text)
        if eps_match:
            epsilon_list = ast.literal_eval(eps_match.group(1))
        else:
            epsilon_list = []
        rep_match = re.search(r'n_replicas\s*:\s*(\d+)', text)
        replicas = int(rep_match.group(1)) if rep_match else 1
        timestep_match = re.search(r'timestep_fs\s*:\s*([0-9.]+)', text)
        timestep_fs = float(timestep_match.group(1)) if timestep_match else 2.0
        equil_match = re.search(r'equilibration_ns\s*:\s*([0-9.]+)', text)
        equil_ns = float(equil_match.group(1)) if equil_match else 1.0
        prod_match = re.search(r'production_ns\s*:\s*([0-9.]+)', text)
        prod_ns = float(prod_match.group(1)) if prod_match else 10.0
        params = {
            'epsilon_sweep_kcalmol': epsilon_list,
            'runtime': {'n_replicas': replicas},
            'simulation': {'timestep_fs': timestep_fs, 'equilibration_ns': equil_ns, 'production_ns': prod_ns},
            'analysis': {'rdf_bins': 200, 'rdf_rmax_A': 30.0}
        }

with open(template_path) as f:
    template = f.read()

out_root = Path('experiments')
out_root.mkdir(exist_ok=True)

# helper to convert ns -> steps
def ns_to_steps(ns, timestep_fs):
    steps = int(round((ns * 1e6) / timestep_fs))
    return max(1, steps)

epsilon_list = params.get('epsilon_sweep_kcalmol', [])
replicas = params.get('runtime', {}).get('n_replicas', 1)

timestep_fs = params.get('simulation', {}).get('timestep_fs', 2.0)
equil_ns = params.get('simulation', {}).get('equilibration_ns', 1.0)
prod_ns = params.get('simulation', {}).get('production_ns', 10.0)

equil_steps = ns_to_steps(equil_ns, timestep_fs)
prod_steps = ns_to_steps(prod_ns, timestep_fs)

# BUG FIX #3: Use absolute path for datafile (was: relative path fails from wrong directory)
# Convert data/system.data to absolute path so LAMMPS can find it from any directory
datafile_abs = str(Path('data/system.data').absolute())

# default substitutions (shared across replicas)
sub_common = {
    'CUTOFF_A': 12.0,
    'KSPACE_TOL': 1e-4,
    'EPS_SOL_SOL': 0.5,
    'SIG_SOL_SOL': params.get('solute', {}).get('sigma_A', 3.4),
    'EPS_OO': 0.0, # placeholder; user should set water LJ in datafile or template
    'SIG_OO': 0.0,
    'EPS_O_H': 0.0,
    'SIG_O_H': 0.0,
    'TIMESTEP_FS': timestep_fs,
    # NOTE: SEED is set per-replica below (Bug Fix #1)
    'DUMP_FREQ': int(round((params.get('simulation', {}).get('write_traj_ps',2.0) * 1000) / timestep_fs)),
    'TRAJ_FILE': 'traj.lammpstrj',
    'RDF_BINS': params.get('analysis', {}).get('rdf_bins', 200),
    'RDF_RMAX': params.get('analysis', {}).get('rdf_rmax_A', 30.0),
    'RDF_FILE': 'rdf_solute_O.dat',
    'EQUIL_STEPS': equil_steps,
    'PROD_STEPS': prod_steps,
    'DATAFILE': datafile_abs,
}

for eps in epsilon_list:
    eps_str = f"{eps:.2f}".replace('.', '_')
    for rep in range(1, replicas + 1):
        run_dir = out_root / f'eps_{eps_str}' / f'replica_{rep}'
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f'Creating run directory: {run_dir}')
        # substitute per-run values
        subs = dict(sub_common)
        subs['EPS_SOL_O'] = eps
        subs['SIG_SOL_O'] = params.get('solute', {}).get('sigma_A', 3.4)
        # BUG FIX #1: Vary random seed per replica (was: 12345 for all)
        # Now: replica 1→12345, replica 2→12346, replica 3→12347, etc.
        subs['SEED'] = 12345 + rep - 1
        # fill placeholders not set
        # Render template with Python's str.format
        content = template.format(**subs)
        infile = run_dir / 'run.in'
        with open(infile, 'w') as f:
            f.write(content)
        print(f'Wrote LAMMPS input to: {infile}')
        # small run helper
        run_sh = run_dir / 'run_cpu.sh'
        with open(run_sh, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('set -e\n')
            f.write('LMP_BIN=${LMP_BIN:-lmp_mpi}\n')
            f.write('export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}\n')
            f.write('$LMP_BIN -in run.in\n')
        os.chmod(run_sh, 0o755)

print('Generated inputs for epsilons:', epsilon_list)
print('Output root:', out_root)
