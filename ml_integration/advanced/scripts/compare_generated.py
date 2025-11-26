#!/usr/bin/env python3
"""
Compare generated files with actual simulation data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from scipy.stats import pearsonr

def load_thermo(file_path):
    return pd.read_csv(file_path, sep=r'\s+', comment='#',
                      names=['TimeStep', 'Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens'])

def load_rdf(file_path):
    """
    Load RDF data from LAMMPS output file.
    Handles multi-frame files by averaging.
    """
    data = {}  # bin_index -> {'r': [], 'g': []}
    
    with open(file_path, 'r') as f:
        current_block = []
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            
            # Check for timestep header (e.g., "610000 150")
            if len(parts) == 2 and parts[1].isdigit():
                continue
                
            if len(parts) >= 3:
                try:
                    # LAMMPS RDF format: Row Dist g(r) Coor(r)
                    # We want Dist (col 1) and g(r) (col 2) - 0-indexed
                    row = int(parts[0])
                    dist = float(parts[1])
                    g_r = float(parts[2])
                    
                    if row not in data:
                        data[row] = {'r': [], 'g': []}
                    
                    data[row]['r'].append(dist)
                    data[row]['g'].append(g_r)
                except ValueError:
                    continue

    # Average over frames
    averaged_data = []
    for row in sorted(data.keys()):
        mean_r = np.mean(data[row]['r'])
        mean_g = np.mean(data[row]['g'])
        averaged_data.append([mean_r, mean_g])
        
    return np.array(averaged_data)

def compare_thermo(actual_file, generated_file):
    print(f"\nComparing Thermodynamics:")
    print(f"  Actual: {actual_file}")
    print(f"  Generated: {generated_file}")
    
    if not actual_file.exists():
        print(f"  ❌ Actual file not found: {actual_file}")
        return
    if not generated_file.exists():
        print(f"  ❌ Generated file not found: {generated_file}")
        return
    
    df_act = load_thermo(actual_file)
    df_gen = load_thermo(generated_file)
    
    # Align lengths (generated is 1000 steps)
    min_len = min(len(df_act), len(df_gen))
    # If actual is much longer, subsample to match generated length roughly
    if len(df_act) > len(df_gen):
        indices = np.linspace(0, len(df_act)-1, len(df_gen), dtype=int)
        df_act = df_act.iloc[indices].reset_index(drop=True)
    
    # Recalculate min_len
    min_len = min(len(df_act), len(df_gen))
    df_act = df_act.iloc[:min_len]
    df_gen = df_gen.iloc[:min_len]
    
    metrics = {}
    for col in ['Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens']:
        act = df_act[col].values
        gen = df_gen[col].values
        
        mse = np.mean((act - gen)**2)
        r2 = 1 - np.sum((act - gen)**2) / np.sum((act - np.mean(act))**2)
        corr, _ = pearsonr(act, gen)
        
        metrics[col] = {'MSE': mse, 'R2': r2, 'Corr': corr}
        print(f"  {col:5s}: MSE={mse:.4f}, R2={r2:.4f}, Corr={corr:.4f}")
        print(f"         Act Mean={np.mean(act):.2f}, Gen Mean={np.mean(gen):.2f}")
    
    return metrics

def compare_rdf(actual_dir, generated_dir):
    print(f"\nComparing RDFs:")
    metrics = {}
    
    for pair in ['CC', 'CO', 'OO']:
        act_file = actual_dir / f"rdf_{pair}.dat"
        gen_file = generated_dir / f"rdf_{pair}.dat"
        
        if not act_file.exists() or not gen_file.exists():
            print(f"  ❌ Missing RDF file for {pair}")
            metrics[pair] = None
            continue
            
        rdf_act = load_rdf(act_file)
        rdf_gen = load_rdf(gen_file)
        
        if len(rdf_act) == 0 or len(rdf_gen) == 0:
            print(f"  ❌ Empty RDF data for {pair}")
            metrics[pair] = None
            continue
        
        # Interpolate actual to match generated bins if needed
        from scipy.interpolate import interp1d
        f_act = interp1d(rdf_act[:, 0], rdf_act[:, 1], kind='linear', fill_value="extrapolate")
        g_r_act_interp = f_act(rdf_gen[:, 0])
        
        mse = np.mean((g_r_act_interp - rdf_gen[:, 1])**2)
        corr, _ = pearsonr(g_r_act_interp, rdf_gen[:, 1])
        
        metrics[pair] = {'MSE': mse, 'Corr': corr}
        print(f"  {pair}: MSE={mse:.4f}, Corr={corr:.4f}")
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare generated files with actual simulation data.")
    parser.add_argument("--epsilon", type=float, default=0.50, help="Epsilon value to compare (default: 0.50)")
    args = parser.parse_args()
    
    base_dir = Path("/store/shuvam/learning_solvent_effects")
    actual_dir = base_dir / f"solvent_effects/epsilon_{args.epsilon:.2f}"
    generated_dir = base_dir / f"ml_integration/advanced/generated/epsilon_{args.epsilon:.2f}"
    
    compare_thermo(actual_dir / "production_detailed_thermo.dat", 
                  generated_dir / "production_detailed_thermo.dat")
    
    compare_rdf(actual_dir, generated_dir)

if __name__ == "__main__":
    main()
