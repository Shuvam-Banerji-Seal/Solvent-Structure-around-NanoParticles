#!/usr/bin/env python3
"""
Train XGBoost Model
===================

Train XGBoost Gradient Boosting model to predict properties.
This model is robust and handles non-linearities well.

Output:
-------
- models/xgb_models.pkl: Trained XGBoost models
- results/plots/xgb_predictions.png: Prediction plots
- results/predictions/xgb_predictions.csv: Numerical predictions

Author: Shuvam Roy
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
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

def create_features(epsilon_values):
    """Create derived features for XGBoost."""
    eps = np.array(epsilon_values)
    
    features = pd.DataFrame({
        'epsilon': eps,
        'epsilon_sq': eps**2,
        'epsilon_cu': eps**3,
        'sqrt_epsilon': np.sqrt(eps),
        'log_epsilon': np.log(eps + 0.01),  # Avoid log(0)
    })
    
    return features

def plot_predictions(df_train, predictions_dict, property_name):
    """Plot training data and predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    ax.scatter(df_train['epsilon'], df_train[property_name], 
              s=100, c='blue', marker='o', label='Training Data', zorder=3)
    
    # Predictions
    eps_pred = predictions_dict['epsilon']
    val_pred = predictions_dict[property_name]
    
    ax.scatter(eps_pred, val_pred, s=100, c='orange', marker='D', 
              label='XGBoost Predictions', zorder=3)
    
    # Connect predictions with line
    ax.plot(eps_pred, val_pred, color='orange', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(property_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'XGBoost Prediction: {property_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main training workflow."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  XGBoost Training                                             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load data
    df_train = load_training_data()
    
    # Prepare features
    X_train = create_features(df_train['epsilon'])
    
    print(f"📊 Training XGBoost models for {len(TARGET_PROPERTIES)} properties")
    print()
    
    # Train models for each property
    models = {}
    predictions = {'epsilon': EPSILON_PREDICT}
    
    # Prepare prediction features
    X_pred = create_features(EPSILON_PREDICT)
    
    for prop in TARGET_PROPERTIES:
        print(f"  🔧 Training model for: {prop}")
        
        y_train = df_train[prop]
        
        # XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            objective='reg:squarederror',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        models[prop] = model
        
        # Make predictions
        y_pred_new = model.predict(X_pred)
        predictions[prop] = y_pred_new
        
        # Calculate training score
        y_train_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_train_pred)
        mae = mean_absolute_error(y_train, y_train_pred)
        
        print(f"     ✅ R² = {r2:.4f}, MAE = {mae:.4f}")
        
        # Plot
        fig = plot_predictions(df_train, predictions, prop)
        plot_file = PLOTS_DIR / f"xgb_{prop}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"     💾 Saved plot: xgb_{prop}.png")
        print()
        
    # Save models
    models_file = MODELS_DIR / "xgb_models.pkl"
    with open(models_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"💾 Saved models to: {models_file}")
    
    # Save predictions
    df_pred = pd.DataFrame(predictions)
    pred_file = PREDICTIONS_DIR / "xgb_predictions.csv"
    df_pred.to_csv(pred_file, index=False, float_format='%.6f')
    print(f"💾 Saved predictions to: {pred_file}")
    
    print()
    print("📊 Predictions for new epsilon values:")
    print("="*80)
    for prop in TARGET_PROPERTIES:
        print(f"\n{prop}:")
        for i, eps in enumerate(EPSILON_PREDICT):
            val = predictions[prop][i]
            print(f"  ε={eps:.2f}: {val:.4f}")
            
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  XGBoost Training Complete!                                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
