#!/usr/bin/env python3
"""
Latent Space Analysis for Improved Model
========================================

Analyzes:
1. Latent manifold visualization (PCA/t-SNE)
2. AUC correlation (response function capability)
3. Smoothness metrics

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path
import sys
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from model_improved import ImprovedMDGenerativeModel

def analyze_latent_space(model_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = ImprovedMDGenerativeModel(latent_dim=512).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Generate latent vectors for range of epsilons
    epsilons = np.linspace(0.0, 1.2, 121)  # 0.00 to 1.20 step 0.01
    latents = []
    
    with torch.no_grad():
        for eps in epsilons:
            eps_tensor = torch.tensor([[eps]], dtype=torch.float32).to(device)
            # Get latent (ignore mu, logvar)
            output = model.encoder(eps_tensor)
            latent = output[0] if isinstance(output, tuple) else output
            latents.append(latent.cpu().numpy()[0])
            
    latents = np.array(latents)
    
    # 1. PCA Visualization
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=epsilons, cmap='viridis', s=50)
    plt.colorbar(label='Epsilon')
    plt.title('Latent Space PCA Path (Improved Model)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.grid(True, alpha=0.3)
    
    # Mark training range vs extrapolation
    train_mask = epsilons <= 0.50
    plt.scatter(latents_pca[train_mask, 0], latents_pca[train_mask, 1], 
                facecolors='none', edgecolors='k', s=80, alpha=0.2, label='Training Range')
    
    plt.legend()
    plt.savefig(output_dir / 'latent_pca_path.png')
    plt.close()
    
    # 2. AUC / Response Function Analysis
    # Calculate distance from epsilon=0.0 in latent space
    latent_0 = latents[0]
    distances = np.linalg.norm(latents - latent_0, axis=1)
    
    # Fit linear response: Distance ~ k * epsilon
    # Ideally, latent distance should be linear with epsilon if it captures the physics linearly
    r2 = r2_score(epsilons, distances)
    correlation = np.corrcoef(epsilons, distances)[0, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, distances, 'b-', linewidth=2, label='Latent Distance')
    
    # Linear fit
    m, c = np.polyfit(epsilons, distances, 1)
    plt.plot(epsilons, m*epsilons + c, 'r--', label=f'Linear Fit (R²={r2:.3f})')
    
    plt.axvline(x=0.5, color='k', linestyle=':', label='Training Boundary')
    plt.xlabel('Epsilon')
    plt.ylabel('Euclidean Distance in Latent Space')
    plt.title(f'Latent Response Function (Corr={correlation:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'latent_response_function.png')
    plt.close()
    
    # 3. Smoothness Metric
    # Calculate local derivatives (finite difference)
    derivs = np.linalg.norm(latents[1:] - latents[:-1], axis=1) / 0.01
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons[:-1], derivs, 'g-', linewidth=2)
    plt.axvline(x=0.5, color='k', linestyle=':', label='Training Boundary')
    plt.xlabel('Epsilon')
    plt.ylabel('Rate of Change (dLatent/dEpsilon)')
    plt.title('Latent Manifold Smoothness')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'latent_smoothness.png')
    plt.close()
    
    # Save metrics
    metrics = {
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'response_linearity_r2': float(r2),
        'response_correlation': float(correlation),
        'mean_smoothness': float(np.mean(derivs)),
        'std_smoothness': float(np.std(derivs))
    }
    
    with open(output_dir / 'latent_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nLatent Analysis Results:")
    print(f"  Response Correlation: {correlation:.4f}")
    print(f"  Linearity R²: {r2:.4f}")
    print(f"  Mean Smoothness: {np.mean(derivs):.4f}")
    print(f"  Plots saved to: {output_dir}")

if __name__ == "__main__":
    analyze_latent_space(
        model_path="../checkpoints_improved/best_model_improved.pt",
        output_dir="../logs_improved/latent_analysis"
    )
