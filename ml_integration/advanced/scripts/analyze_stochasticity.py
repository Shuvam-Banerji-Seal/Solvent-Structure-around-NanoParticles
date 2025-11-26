#!/usr/bin/env python3
"""
Analyze intrinsic stochasticity in MD trajectories using Numba optimization.
Author: Shuvam Banerji Seal
"""

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

from numba_ops import compute_rmsf_numba, compute_msd_numba

def analyze_fluctuations(epsilon=0.25, generated=False):
    source_type = "Generated" if generated else "Actual"
    print(f"Analyzing stochasticity for epsilon={epsilon} ({source_type})...")
    
    if generated:
        base_dir = Path("/store/shuvam/learning_solvent_effects/ml_integration/advanced/generated") / f"epsilon_{epsilon:.2f}"
    else:
        base_dir = Path("/store/shuvam/learning_solvent_effects/solvent_effects") / f"epsilon_{epsilon:.2f}"
        
    traj_file = base_dir / "production.lammpstrj"
    
    if not traj_file.exists():
        print(f"File not found: {traj_file}")
        return

    # Load trajectory
    try:
        u = mda.Universe(str(traj_file), format='LAMMPSDUMP')
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return
        
    print(f"Loaded {len(u.trajectory)} frames, {len(u.atoms)} atoms.")
    
    # Load all coordinates into memory for Numba
    print("Loading trajectory into memory...")
    t0 = time.time()
    coords = []
    for ts in u.trajectory:
        coords.append(u.atoms.positions.copy())
    coords = np.array(coords, dtype=np.float32)
    print(f"Loaded in {time.time() - t0:.2f}s. Shape: {coords.shape}")
    
    # 1. RMSF (Root Mean Square Fluctuation)
    print("Computing RMSF (Numba optimized)...")
    t0 = time.time()
    rmsf = compute_rmsf_numba(coords)
    print(f"RMSF computed in {time.time() - t0:.4f}s")
    
    print(f"Mean RMSF: {np.mean(rmsf):.4f} Angstrom")
    print(f"Max RMSF: {np.max(rmsf):.4f} Angstrom")
    
    # Plot RMSF histogram
    log_dir = Path("../../ml_integration/advanced/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    prefix = "generated_" if generated else ""
    
    plt.figure(figsize=(10, 6))
    plt.hist(rmsf, bins=50, color='red' if generated else 'blue', alpha=0.7)
    plt.xlabel('RMSF (Angstrom)')
    plt.ylabel('Count')
    plt.title(f'{source_type} Atomic Fluctuations - Epsilon {epsilon}')
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / f'{prefix}rmsf_epsilon_{epsilon}.png')
    plt.close()
    
    # 2. MSD (Mean Squared Displacement)
    print("Computing MSD (Numba optimized)...")
    t0 = time.time()
    msd = compute_msd_numba(coords)
    print(f"MSD computed in {time.time() - t0:.4f}s")
    
    plt.figure(figsize=(10, 6))
    plt.plot(msd, linewidth=2, color='red' if generated else 'blue')
    plt.xlabel('Time (frames)')
    plt.ylabel('MSD (Angstrom^2)')
    plt.title(f'{source_type} Mean Squared Displacement - Epsilon {epsilon}')
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / f'{prefix}msd_epsilon_{epsilon}.png')
    plt.close()
    
    print("✅ Analysis complete. Plots saved to logs/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze intrinsic stochasticity in MD trajectories.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Epsilon value to analyze (default: 0.25)")
    parser.add_argument("--generated", action="store_true", help="Analyze generated trajectory instead of actual")
    args = parser.parse_args()
    
    analyze_fluctuations(args.epsilon, args.generated)
