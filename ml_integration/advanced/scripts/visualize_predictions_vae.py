#!/usr/bin/env python3
"""
Visualize predictions from the VAE model.
=========================================

Generates:
1. RDF comparison plots (Actual vs Predicted)
2. Thermodynamic property distribution plots
3. Summary error plots

Author: Shuvam Banerji Seal
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from compare_generated import load_rdf, load_thermo

def visualize_vae_predictions():
    # Setup paths
    base_dir = Path("/store/shuvam/learning_solvent_effects")
    generated_dir = base_dir / "ml_integration/advanced/generated_vae"
    actual_base = base_dir / "solvent_effects"
    output_dir = base_dir / "ml_integration/advanced/logs_vae/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Epsilons to visualize (representative set)
    epsilons = [0.55, 0.70, 0.90, 1.10]
    
    print(f"🚀 Generating visualization plots in {output_dir}...")
    
    # 1. RDF Comparisons
    for eps in epsilons:
        print(f"  Plotting RDFs for epsilon {eps:.2f}...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = ['CC', 'CO', 'OO']
        
        gen_path = generated_dir / f"epsilon_{eps:.2f}"
        act_path = actual_base / f"epsilon_{eps:.2f}"
        
        for i, pair in enumerate(pairs):
            ax = axes[i]
            
            # Load data
            gen_file = gen_path / f"rdf_{pair}.dat"
            act_file = act_path / f"rdf_{pair}.dat"
            
            if gen_file.exists() and act_file.exists():
                rdf_gen = load_rdf(gen_file)
                rdf_act = load_rdf(act_file)
                
                if len(rdf_gen) > 0 and len(rdf_act) > 0:
                    ax.plot(rdf_act[:, 0], rdf_act[:, 1], 'k-', linewidth=2, label='Actual', alpha=0.7)
                    ax.plot(rdf_gen[:, 0], rdf_gen[:, 1], 'b--', linewidth=2, label='Predicted (VAE)')
                    
                    ax.set_title(f"{pair} RDF (ε={eps:.2f})")
                    ax.set_xlabel("r (Å)")
                    ax.set_ylabel("g(r)")
                    ax.grid(True, alpha=0.3)
                    if i == 0: ax.legend()
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
                
        plt.tight_layout()
        plt.savefig(output_dir / f"rdf_comparison_eps_{eps:.2f}.png", dpi=150)
        plt.close()

    # 2. Thermodynamic Distributions
    for eps in epsilons:
        print(f"  Plotting Thermodynamics for epsilon {eps:.2f}...")
        
        gen_file = generated_dir / f"epsilon_{eps:.2f}" / "production_detailed_thermo.dat"
        act_file = actual_base / f"epsilon_{eps:.2f}" / "production_detailed_thermo.dat"
        
        if gen_file.exists() and act_file.exists():
            df_gen = load_thermo(gen_file)
            df_act = load_thermo(act_file)
            
            if df_gen is not None and df_act is not None:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                cols = ['Temp', 'Press', 'PE', 'Dens']
                
                for i, col in enumerate(cols):
                    ax = axes[i]
                    if col in df_gen.columns and col in df_act.columns:
                        sns.kdeplot(df_act[col], ax=ax, color='k', fill=True, alpha=0.3, label='Actual')
                        sns.kdeplot(df_gen[col], ax=ax, color='b', fill=True, alpha=0.3, label='Predicted')
                        
                        ax.set_title(f"{col} Distribution (ε={eps:.2f})")
                        ax.grid(True, alpha=0.3)
                        if i == 0: ax.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f"thermo_dist_eps_{eps:.2f}.png", dpi=150)
                plt.close()

    print("✅ Visualization complete!")

if __name__ == "__main__":
    visualize_vae_predictions()
