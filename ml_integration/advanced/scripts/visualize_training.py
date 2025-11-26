#!/usr/bin/env python3
"""
Visualize training progress and latent space.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_training_metrics(log_dir):
    history_file = log_dir / 'training_history_improved.json'
    if not history_file.exists():
        print(f"No history file found at {history_file}")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 1. Loss Curve (Double Descent)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_losses'], label='Val Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training Dynamics: Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / 'loss_curve.png', dpi=300)
    plt.close()
    
    # 2. Perplexity (if available)
    # Estimate perplexity if not in history
    ppl = np.exp(history['val_losses'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ppl, color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (exp(Val Loss))')
    plt.title('Model Uncertainty (Perplexity)')
    plt.grid(True, alpha=0.3)
    plt.savefig(log_dir / 'perplexity.png', dpi=300)
    plt.close()

def plot_latent_space(log_dir):
    csv_file = log_dir / 'latent_vectors.csv'
    if not csv_file.exists():
        print(f"No latent vectors CSV found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # Plot evolution of latent space for a few epochs
    epochs = sorted(df['epoch'].unique())
    selected_epochs = [epochs[0], epochs[len(epochs)//2], epochs[-1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, epoch in enumerate(selected_epochs):
        data = df[df['epoch'] == epoch]
        
        # Use PCA/t-SNE if dim > 2
        # For simplicity here, just plot first 2 dims if available, 
        # but usually we need t-SNE. 
        # Since the CSV has z_0, z_1... we can use them directly if latent_dim=2
        # But latent_dim=512. So we need t-SNE.
        
        from sklearn.manifold import TSNE
        z_cols = [c for c in df.columns if c.startswith('z_')]
        z_data = data[z_cols].values
        
        if len(z_data) > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(z_data)-1))
            z_2d = tsne.fit_transform(z_data)
            
            sc = axes[i].scatter(z_2d[:, 0], z_2d[:, 1], c=data['epsilon'], cmap='viridis', s=100)
            axes[i].set_title(f'Epoch {epoch}')
            if i == 2:
                plt.colorbar(sc, ax=axes[i], label='Epsilon')
        
    plt.tight_layout()
    plt.savefig(log_dir / 'latent_evolution.png', dpi=300)
    plt.close()

def main():
    log_dir = Path('../logs_improved')
    plot_training_metrics(log_dir)
    # plot_latent_space(log_dir) # Latent viz is handled by training script now
    print("✅ Visualizations generated in ../logs_improved/")

if __name__ == "__main__":
    main()
