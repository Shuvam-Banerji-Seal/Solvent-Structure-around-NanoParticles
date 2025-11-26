#!/usr/bin/env python3
"""
Batch generate and compare files for all epsilon values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys
import json
# Add current directory to path to import compare_generated
sys.path.append(str(Path(__file__).parent))
from compare_generated import compare_thermo, compare_rdf

def main():
    # Epsilon range: 0.00 to 1.10 with 0.05 step
    # Using linspace to avoid floating point issues, or round carefully
    epsilons = [float(f"{x:.2f}") for x in np.arange(0.0, 1.15, 0.05)]
    
    results = []
    
    base_dir = Path("/store/shuvam/learning_solvent_effects")
    scripts_dir = base_dir / "ml_integration/advanced/scripts"
    checkpoints_dir = base_dir / "ml_integration/advanced/checkpoints"
    generated_base = base_dir / "ml_integration/advanced/generated"
    actual_base = base_dir / "solvent_effects"
    log_dir = base_dir / "ml_integration/advanced/logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Starting batch processing for {len(epsilons)} epsilon values...")
    
    for eps_val in epsilons:
        print(f"\n{'='*60}")
        print(f"Processing Epsilon {eps_val:.2f}")
        print(f"{'='*60}")
        
        # 1. Generate Files
        cmd = [
            "python", "generate_files.py",
            "--model", str(checkpoints_dir / "best_model.pt"),
            "--epsilon", f"{eps_val:.2f}",
            "--output", str(generated_base / f"epsilon_{eps_val:.2f}")
        ]
        
        try:
            # Capture output to avoid cluttering, print only on error
            subprocess.run(cmd, check=True, cwd=scripts_dir, capture_output=True)
            print(f"✅ Generated files for {eps_val}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed for {eps_val}")
            print(e.stderr.decode())
            continue
            
        # 2. Compare
        actual_dir = actual_base / f"epsilon_{eps_val:.2f}"
        generated_dir = generated_base / f"epsilon_{eps_val:.2f}"
    for eps in epsilons:
        print(f"\nAnalyzing epsilon {eps:.2f}...")
        
        gen_dir = generated_base_dir / f"epsilon_{eps:.2f}"
        actual_dir = Path(f"/store/shuvam/learning_solvent_effects/solvent_effects/epsilon_{eps:.2f}")
        
        if not gen_dir.exists():
            print(f"⚠️  Generated directory not found: {gen_dir}")
            continue
            
        if not actual_dir.exists():
            print(f"⚠️  Actual directory not found: {actual_dir}")
            continue
            
        try:
            # Compare thermodynamics
            thermo_metrics = compare_thermo(
                gen_dir / "production_detailed_thermo.dat",
                actual_dir / "production_detailed_thermo.dat"
            )
            
            # Compare RDFs
            rdf_metrics = {}
            for pair in ['CC', 'CO', 'OO']:
                metrics = compare_rdf(
                    gen_dir / f"rdf_{pair}.dat",
                    actual_dir / f"rdf_{pair}.dat",
                    pair
                )
                rdf_metrics[pair] = metrics
            
            # Aggregate results
            res = {
                'epsilon': eps,
                'thermo_mse': {k: v['mse'] for k, v in thermo_metrics.items()},
                'rdf_mse': {k: v['mse'] for k, v in rdf_metrics.items()},
                'rdf_corr': {k: v['correlation'] for k, v in rdf_metrics.items()}
            }
            results.append(res)
            print(f"✅ Epsilon {eps:.2f} analysis complete")
            
        except Exception as e:
            print(f"❌ Error analyzing epsilon {eps:.2f}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save aggregate results
    if results:
        df = pd.DataFrame(results)
        
        # Flatten dictionaries
        thermo_df = pd.json_normalize(df['thermo_mse']).add_prefix('thermo_mse_')
        rdf_mse_df = pd.json_normalize(df['rdf_mse']).add_prefix('rdf_mse_')
        rdf_corr_df = pd.json_normalize(df['rdf_corr']).add_prefix('rdf_corr_')
        
        final_df = pd.concat([df[['epsilon']], thermo_df, rdf_mse_df, rdf_corr_df], axis=1)
        
        csv_path = output_base_dir / "comparison_summary.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"\n💾 Summary saved to {csv_path}")
        
        # Plot summary
        plot_summary(final_df, output_base_dir)

def plot_summary(df, output_dir):
    # RDF MSE
    plt.figure(figsize=(10, 6))
    for pair in ['CC', 'CO', 'OO']:
        if f'rdf_mse_{pair}' in df.columns:
            plt.plot(df['epsilon'], df[f'rdf_mse_{pair}'], marker='o', label=pair)
    plt.xlabel('Epsilon')
    plt.ylabel('RDF MSE')
    plt.title('RDF Prediction Error vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'summary_rdf_mse.png')
    plt.close()
    
    # Thermo MSE (Log scale)
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        if 'thermo_mse' in col:
            name = col.replace('thermo_mse_', '')
            plt.semilogy(df['epsilon'], df[col], marker='o', label=name)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE (Log Scale)')
    plt.title('Thermodynamic Property Error vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'summary_thermo_mse.png')
    plt.close()

if __name__ == "__main__":
    run_batch_comparison()
