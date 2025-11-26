#!/usr/bin/env python3
"""
Analyze Latent Space Response Function.
Computes the Area Under the Curve (AUC) of Latent Norm vs Epoch.
Correlates this "Latent Activity" with physical properties.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson

def analyze_latent_response():
    # 1. Load Latent Vectors
    log_dir = Path("../logs_improved")
    csv_path = log_dir / "latent_vectors.csv"
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract z columns
    z_cols = [c for c in df.columns if c.startswith('z_')]
    
    # 2. Compute Norms
    print("Computing latent norms (L1, L2, L-inf)...")
    df['norm_l1'] = np.linalg.norm(df[z_cols].values, ord=1, axis=1)
    df['norm_l2'] = np.linalg.norm(df[z_cols].values, ord=2, axis=1)
    df['norm_linf'] = np.linalg.norm(df[z_cols].values, ord=np.inf, axis=1)
    
    # 3. Analyze per Epsilon
    epsilons = sorted(df['epsilon'].unique())
    results = []
    
    # Plot Norm Evolution for L2 (Standard)
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(epsilons), vmax=max(epsilons))
    
    for eps in epsilons:
        subset = df[df['epsilon'] == eps].sort_values('epoch')
        epochs = subset['epoch'].values
        
        # Calculate AUC for all norms
        auc_l1 = simpson(y=subset['norm_l1'].values, x=epochs)
        auc_l2 = simpson(y=subset['norm_l2'].values, x=epochs)
        auc_linf = simpson(y=subset['norm_linf'].values, x=epochs)
        
        results.append({
            'epsilon': eps,
            'AUC_L1': auc_l1,
            'AUC_L2': auc_l2,
            'AUC_Linf': auc_linf
        })
        
        # Plot L2 evolution
        plt.plot(epochs, subset['norm_l2'].values, color=cmap(norm(eps)), alpha=0.6)
        
    # Add colorbar for epsilon
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Epsilon (Solvent Interaction)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Latent Vector L2 Norm ||z||_2')
    plt.title('Latent Space Dynamics: Norm Evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir / "latent_norm_evolution_l2.png")
    plt.close()
    
    # 4. Analyze Response Functions (AUC vs Epsilon)
    res_df = pd.DataFrame(results)
    
    # Compute Susceptibilities
    res_df['Susc_L1'] = np.gradient(res_df['AUC_L1'], res_df['epsilon'])
    res_df['Susc_L2'] = np.gradient(res_df['AUC_L2'], res_df['epsilon'])
    res_df['Susc_Linf'] = np.gradient(res_df['AUC_Linf'], res_df['epsilon'])
    
    # Plot 1: AUCs for different norms (Normalized to compare shapes)
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['epsilon'].values, (res_df['AUC_L1'] / res_df['AUC_L1'].max()).values, label='L1 Norm (Manhattan)', marker='o')
    plt.plot(res_df['epsilon'].values, (res_df['AUC_L2'] / res_df['AUC_L2'].max()).values, label='L2 Norm (Euclidean)', marker='s')
    plt.plot(res_df['epsilon'].values, (res_df['AUC_Linf'] / res_df['AUC_Linf'].max()).values, label='L-inf Norm (Max)', marker='^')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Normalized Latent Activity (AUC)')
    plt.title('Latent Response Function (Comparison of Norms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / "latent_response_norms_comparison.png")
    plt.close()
    
    # Plot 2: Susceptibility (L2) - The Main Result
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Latent Activity (AUC L2)', color=color, fontweight='bold')
    ax1.plot(res_df['epsilon'].values, res_df['AUC_L2'].values, color=color, marker='o', label='Activity (AUC)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Latent Susceptibility (d(AUC)/dε)', color=color, fontweight='bold')
    ax2.plot(res_df['epsilon'].values, res_df['Susc_L2'].values, color=color, linestyle='--', marker='x', linewidth=2, label='Susceptibility')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Latent Response & Susceptibility (L2 Norm)')
    plt.tight_layout()
    plt.savefig(log_dir / "latent_susceptibility_l2.png")
    plt.close()
    
    # 5. Correlate with Real Data
    batch_metrics_path = log_dir / "batch_metrics.csv"
    if batch_metrics_path.exists():
        print("Correlating with physical properties...")
        phys_df = pd.read_csv(batch_metrics_path)
        merged = pd.merge(res_df, phys_df, on='epsilon')
        
        # Correlation Matrix
        cols = ['AUC_L1', 'AUC_L2', 'AUC_Linf', 'Susc_L2', 'Thermo_PE_MSE', 'Thermo_KE_MSE']
        valid_cols = [c for c in cols if c in merged.columns]
        
        corr_matrix = merged[valid_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        merged.to_csv(log_dir / "latent_physical_correlation_extended.csv", index=False)
        
    print("\n✅ Analysis Complete.")
    print(f"Plots saved to {log_dir}")
    print("  - latent_norm_evolution_l2.png")
    print("  - latent_response_norms_comparison.png")
    print("  - latent_susceptibility_l2.png")

if __name__ == "__main__":
    analyze_latent_response()
