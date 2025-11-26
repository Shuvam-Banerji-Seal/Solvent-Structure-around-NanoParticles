#!/usr/bin/env python3
"""
Train Neural Network Model
==========================

Train a PyTorch Neural Network to predict properties for new epsilon values.
This model captures non-linear relationships and can enforce constraints.

Output:
-------
- models/nn_model.pt: Trained PyTorch model
- models/nn_scaler_X.pkl: Feature scaler
- models/nn_scaler_y.pkl: Target scaler
- results/plots/nn_predictions.png: Prediction plots
- results/predictions/nn_predictions.csv: Numerical predictions

Author: Shuvam Roy
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
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

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EpsilonPropertyNN(nn.Module):
    """Simple MLP for predicting properties from epsilon."""
    def __init__(self, input_dim, output_dim):
        super(EpsilonPropertyNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

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

def train_nn_model(X_train, y_train, epochs=2000, lr=0.001):
    """Train the Neural Network."""
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = EpsilonPropertyNN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    return model, losses

def plot_predictions(df_train, predictions_dict, property_name):
    """Plot training data and predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    ax.scatter(df_train['epsilon'], df_train[property_name], 
              s=100, c='blue', marker='o', label='Training Data', zorder=3)
    
    # Predictions
    eps_pred = predictions_dict['epsilon']
    val_pred = predictions_dict[property_name]
    
    ax.scatter(eps_pred, val_pred, s=100, c='green', marker='^', 
              label='NN Predictions', zorder=3)
    
    # Connect predictions with line
    ax.plot(eps_pred, val_pred, 'g--', alpha=0.5)
    
    ax.set_xlabel('Epsilon (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(property_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Neural Network Prediction: {property_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main training workflow."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Neural Network Training                                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Load data
    df_train = load_training_data()
    
    # Prepare data
    X = df_train[['epsilon']].values
    y = df_train[TARGET_PROPERTIES].values
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"📊 Training NN model for {len(TARGET_PROPERTIES)} properties simultaneously")
    print()
    
    # Train model
    model, losses = train_nn_model(X_scaled, y_scaled)
    
    # Evaluate on training data
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_pred_scaled = model(X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    print()
    print("📊 Training Metrics:")
    for i, prop in enumerate(TARGET_PROPERTIES):
        r2 = r2_score(y[:, i], y_pred[:, i])
        mae = mean_absolute_error(y[:, i], y_pred[:, i])
        print(f"  {prop:<30}: R² = {r2:.4f}, MAE = {mae:.4f}")
        
    # Predict for new epsilon values
    X_new = np.array(EPSILON_PREDICT).reshape(-1, 1)
    X_new_scaled = scaler_X.transform(X_new)
    
    with torch.no_grad():
        X_new_tensor = torch.FloatTensor(X_new_scaled)
        y_new_pred_scaled = model(X_new_tensor).numpy()
        y_new_pred = scaler_y.inverse_transform(y_new_pred_scaled)
        
    # Organize predictions
    predictions = {'epsilon': EPSILON_PREDICT}
    for i, prop in enumerate(TARGET_PROPERTIES):
        predictions[prop] = y_new_pred[:, i]
        
        # Plot
        fig = plot_predictions(df_train, predictions, prop)
        plot_file = PLOTS_DIR / f"nn_{prop}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  💾 Saved plot: nn_{prop}.png")
        
    # Save model and scalers
    torch.save(model.state_dict(), MODELS_DIR / "nn_model.pt")
    with open(MODELS_DIR / "nn_scaler_X.pkl", 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(MODELS_DIR / "nn_scaler_y.pkl", 'wb') as f:
        pickle.dump(scaler_y, f)
        
    print()
    print(f"💾 Saved model to: {MODELS_DIR}/nn_model.pt")
    
    # Save predictions to CSV
    df_pred = pd.DataFrame(predictions)
    pred_file = PREDICTIONS_DIR / "nn_predictions.csv"
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
    print("║  NN Training Complete!                                         ║")
    print("╚════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
