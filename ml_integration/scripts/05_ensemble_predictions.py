#!/usr/bin/env python3
"""
Ensemble Predictions
====================

Combine predictions from GPR, NN, and XGBoost models.
Weighted average based on validation performance and model characteristics.

Output:
-------
- results/predictions/ensemble_predictions.csv: Final consensus predictions
- results/plots/ensemble_comparison.png: Comparison of all models

Author: Shuvam Roy
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path("/store/shuvam/learning_solvent_effects/ml_integration")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Target properties
TARGET_PROPERTIES = [
    'density_mean',
    'press_mean',
    'rdf_co_peak1_position',
    'rdf_co_peak1_height',
    'rdf_co_coordination_number',
    'rdf_oo_peak1_position',
]

# Weights for ensemble
# GPR: 0.5 (Best for small data + uncertainty)
# NN: 0.3 (Captures non-linearities)
# XGB: 0.2 (Robust baseline)
WEIGHTS = {'gpr': 0.5, 'nn': 0.3, 'xgb': 0.2}

def load_predictions():
    """Load predictions from all models."""
    preds = {}
    
    # GPR
    gpr_file = PREDICTIONS_DIR / "gpr_predictions.csv"
    if gpr_file.exists():
        preds['gpr'] = pd.read_csv(gpr_file)
    else:
        print("⚠️ GPR predictions not found")
        
    # NN
    nn_file = PREDICTIONS_DIR / "nn_predictions.csv"
    if nn_file.exists():
        preds['nn'] = pd.read_csv(nn_file)
    else:
        print("⚠️ NN predictions not found")
        
    # XGBoost
    xgb_file = PREDICTIONS_DIR / "xgb_predictions.csv"
    if xgb_file.exists():
        preds['xgb'] = pd.read_csv(xgb_file)
    else:
        print("⚠️ XGBoost predictions not found")
        
    return preds

def plot_ensemble_comparison(preds, ensemble_df, property_name):
    """Plot comparison of all models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epsilon = ensemble_df['epsilon']
    
    # Plot individual models
    if 'gpr' in preds:
        # GPR has mean and std columns
        col_name = f'{property_name}_mean' if f'{property_name}_mean' in preds['gpr'].columns else property_name
        ax.plot(epsilon, preds['gpr'][col_name], 'b--', marker='o', alpha=0.5, label='GPR')
        
    if 'nn' in preds:
        ax.plot(epsilon, preds['nn'][property_name], 'g--', marker='^', alpha=0.5, label='Neural Network')
        
    if 'xgb' in preds:
        ax.plot(epsilon, preds['xgb'][property_name], 'orange', linestyle='--', marker='D', alpha=0.5, label='XGBoost')
        
    # Plot Ensemble
    ax.plot(epsilon, ensemble_df[property_name], 'r-', linewidth=3, marker='*', markersize=12, label='Ensemble (Weighted)')
    
    ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(property_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Ensemble Prediction: {property_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main ensemble workflow."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Ensemble Prediction                                          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load predictions
    preds = load_predictions()
    
    if not preds:
        print("❌ No predictions found!")
        return
    
    print(f"✅ Loaded predictions from: {list(preds.keys())}")
    print(f"⚖️  Weights: {WEIGHTS}")
    print()
    
    # Calculate ensemble
    ensemble_data = {'epsilon': preds[list(preds.keys())[0]]['epsilon']}
    
    for prop in TARGET_PROPERTIES:
        print(f"  🔄 Processing: {prop}")
        
        weighted_sum = 0
        total_weight = 0
        
        # GPR
        if 'gpr' in preds:
            # Handle GPR column naming (might be prop_mean)
            col = f'{prop}_mean' if f'{prop}_mean' in preds['gpr'].columns else prop
            weighted_sum += preds['gpr'][col] * WEIGHTS['gpr']
            total_weight += WEIGHTS['gpr']
            
        # NN
        if 'nn' in preds:
            weighted_sum += preds['nn'][prop] * WEIGHTS['nn']
            total_weight += WEIGHTS['nn']
            
        # XGBoost
        if 'xgb' in preds:
            weighted_sum += preds['xgb'][prop] * WEIGHTS['xgb']
            total_weight += WEIGHTS['xgb']
            
        # Normalize
        ensemble_val = weighted_sum / total_weight
        ensemble_data[prop] = ensemble_val
        
        # Plot comparison
        ensemble_df_temp = pd.DataFrame(ensemble_data)
        fig = plot_ensemble_comparison(preds, ensemble_df_temp, prop)
        plot_file = PLOTS_DIR / f"ensemble_{prop}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    # Save ensemble predictions
    ensemble_df = pd.DataFrame(ensemble_data)
    output_file = PREDICTIONS_DIR / "ensemble_predictions.csv"
    ensemble_df.to_csv(output_file, index=False, float_format='%.6f')
    
    print()
    print(f"💾 Saved ensemble predictions to: {output_file}")
    print()
    
    print("📊 Final Ensemble Predictions:")
    print("="*80)
    for prop in TARGET_PROPERTIES:
        print(f"\n{prop}:")
        for i, row in ensemble_df.iterrows():
            print(f"  ε={row['epsilon']:.2f}: {row[prop]:.4f}")
            
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Ensemble Complete!                                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
