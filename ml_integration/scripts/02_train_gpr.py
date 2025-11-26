#!/usr/bin/env python3
"""
Train Gaussian Process Regression Model
========================================

Train GPR model to predict properties for new epsilon values.
This is the recommended first model due to uncertainty quantification.

Output:
-------
- models/gpr_model.pkl: Trained GPR model
- results/plots/gpr_predictions.png: Prediction plots
- results/predictions/gpr_predictions.csv: Numerical predictions

Author: Shuvam Roy
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json

# Configuration
BASE_DIR = Path("/store/shuvam/learning_solvent_effects/ml_integration")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Target properties to predict
TARGET_PROPERTIES = [
    'density_mean',
    'press_mean',
    'rdf_co_peak1_position',
    'rdf_co_peak1_height',
    'rdf_co_coordination_number',
    'rdf_oo_peak1_position',
]

# New epsilon values to predict
EPSILON_PREDICT = [0.55, 0.60, 0.65, 0.70, 0.75, 0.85]


def load_training_data():
    """Load extracted features."""
    features_file = DATA_DIR / "training_features.csv"
    
    if not features_file.exists():
        raise FileNotFoundError(
            f"Training features not found at {features_file}\n"
            "Please run: python scripts/01_extract_features.py"
        )
    
    df = pd.read_csv(features_file)
    print(f"✅ Loaded training data: {df.shape}")
    return df


def train_gpr_model(X_train, y_train, property_name):
    """Train a single GPR model for one property."""
    # Define kernel
    # RBF kernel for smooth interpolation + White kernel for noise
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e1)) + \
             WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    
    # Create and train GPR
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )
    
    gpr.fit(X_train, y_train)
    
    return gpr


def plot_predictions(df_train, predictions_dict, property_name):
    """Plot training data and predictions with uncertainty."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    ax.scatter(df_train['epsilon'], df_train[property_name], 
              s=100, c='blue', marker='o', label='Training Data', zorder=3)
    
    # Predictions
    eps_pred = predictions_dict['epsilon']
    mean_pred = predictions_dict[f'{property_name}_mean']
    std_pred = predictions_dict[f'{property_name}_std']
    
    ax.scatter(eps_pred, mean_pred, s=100, c='red', marker='s', 
              label='Predictions', zorder=3)
    
    # Uncertainty bands
    ax.fill_between(eps_pred, 
                    mean_pred - 2*std_pred, 
                    mean_pred + 2*std_pred,
                    alpha=0.2, color='red', label='95% Confidence')
    
    # Smooth interpolation line
    eps_fine = np.linspace(0.0, 0.85, 200)
    # This is just for visualization - actual predictions use the GPR model
    
    ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(property_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'GPR Prediction: {property_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main training workflow."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Gaussian Process Regression Training                         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load data
    df_train = load_training_data()
    print()
    
    # Prepare features
    X_train = df_train[['epsilon']].values
    
    print(f"📊 Training GPR models for {len(TARGET_PROPERTIES)} properties")
    print()
    
    # Train models for each property
    models = {}
    predictions_all = {'epsilon': EPSILON_PREDICT}
    
    for prop in TARGET_PROPERTIES:
        if prop not in df_train.columns:
            print(f"  ⚠️  Skipping {prop} (not in training data)")
            continue
        
        print(f"  🔧 Training model for: {prop}")
        
        y_train = df_train[prop].values
        
        # Train GPR
        gpr = train_gpr_model(X_train, y_train, prop)
        models[prop] = gpr
        
        # Make predictions
        X_pred = np.array(EPSILON_PREDICT).reshape(-1, 1)
        y_pred_mean, y_pred_std = gpr.predict(X_pred, return_std=True)
        
        predictions_all[f'{prop}_mean'] = y_pred_mean
        predictions_all[f'{prop}_std'] = y_pred_std
        
        # Calculate training score
        y_train_pred = gpr.predict(X_train)
        r2 = r2_score(y_train, y_train_pred)
        mae = mean_absolute_error(y_train, y_train_pred)
        
        print(f"     ✅ R² = {r2:.4f}, MAE = {mae:.4f}")
        print(f"     📊 Kernel: {gpr.kernel_}")
        
        # Plot
        fig = plot_predictions(df_train, predictions_all, prop)
        plot_file = PLOTS_DIR / f"gpr_{prop}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"     💾 Saved plot: {plot_file.name}")
        print()
    
    # Save models
    models_file = MODELS_DIR / "gpr_models.pkl"
    with open(models_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"💾 Saved models to: {models_file}")
    print()
    
    # Save predictions
    df_predictions = pd.DataFrame(predictions_all)
    pred_file = PREDICTIONS_DIR / "gpr_predictions.csv"
    df_predictions.to_csv(pred_file, index=False, float_format='%.6f')
    print(f"💾 Saved predictions to: {pred_file}")
    print()
    
    # Print predictions
    print("📊 Predictions for new epsilon values:")
    print("="*80)
    for prop in TARGET_PROPERTIES:
        if f'{prop}_mean' in predictions_all:
            print(f"\n{prop}:")
            for i, eps in enumerate(EPSILON_PREDICT):
                mean = predictions_all[f'{prop}_mean'][i]
                std = predictions_all[f'{prop}_std'][i]
                print(f"  ε={eps:.2f}: {mean:.4f} ± {std:.4f}")
    print()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  GPR Training Complete!                                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print("Next steps:")
    print("  1. Review plots in: results/plots/")
    print("  2. Check predictions: results/predictions/gpr_predictions.csv")
    print("  3. Train NN model: python scripts/03_train_nn.py")
    print("  4. Create ensemble: python scripts/05_ensemble_predictions.py")
    print()


if __name__ == "__main__":
    main()
