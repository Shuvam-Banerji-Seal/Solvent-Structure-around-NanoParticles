#!/usr/bin/env python3
"""
IMPROVED Training Script with Full Data, CUDA Optimizations, and Latent Visualization
======================================================================================

Critical Improvements:
1. Use ALL trajectory frames (4001 per epsilon)
2. Mixed precision training (FP16)
3. Gradient accumulation for effective larger batch size
4. Real-time latent space visualization with TensorBoard
5. Better validation and early stopping

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import time
from typing import Dict, Tuple
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model import MDGenerativeModel
from dataset import create_dataloaders


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss functions."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def trajectory_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss for coordinates."""
        return self.mse(pred, target)
    
    def thermodynamics_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        """MSE loss for all thermodynamic properties."""
        total_loss = 0
        count = 0
        for key in ['temperature', 'pressure', 'density', 'potential_energy']:
            if key in pred and key in target:
                total_loss += self.mse(pred[key], target[key])
                count += 1
        return total_loss / max(count, 1)
    
    def rdf_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        """MSE loss for RDF curves."""
        total_loss = 0
        count = 0
        for pair in ['CC', 'CO', 'OO']:
            if pair in pred and pair in target:
                total_loss += self.mse(pred[pair], target[pair]['g_r'])
                count += 1
        return total_loss / max(count, 1)
    
    def smoothness_penalty(self, thermo: Dict) -> torch.Tensor:
        """Penalize non-smooth thermodynamic trajectories."""
        penalty = 0
        count = 0
        for key, values in thermo.items():
            # Compute differences between consecutive timesteps
            diff = values[:, 1:] - values[:, :-1]
            penalty += diff.pow(2).mean()
            count += 1
        return penalty / max(count, 1)
    
    def forward(self, 
                pred: Dict,
                target: Dict,
                alpha_traj: float = 1.0,
                alpha_thermo: float = 2.0,  # Increased weight
                alpha_rdf: float = 3.0,     # Increased weight (more data available)
                alpha_smooth: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss with updated weights."""
        losses = {}
        
        # Trajectory loss
        if pred['trajectory'] is not None and target['trajectory'] is not None:
            losses['trajectory'] = self.trajectory_loss(pred['trajectory'], target['trajectory'])
        else:
            losses['trajectory'] = torch.tensor(0.0, device=list(pred['rdfs'].values())[0].device)
        
        # Thermodynamics loss
        if pred['thermodynamics'] and target['thermodynamics']:
            losses['thermodynamics'] = self.thermodynamics_loss(
                pred['thermodynamics'], 
                target['thermodynamics']
            )
            losses['smoothness'] = self.smoothness_penalty(pred['thermodynamics'])
        else:
            losses['thermodynamics'] = torch.tensor(0.0, device=list(pred['rdfs'].values())[0].device)
            losses['smoothness'] = torch.tensor(0.0, device=list(pred['rdfs'].values())[0].device)
        
        # RDF loss
        if pred['rdfs'] and target['rdfs']:
            losses['rdf'] = self.rdf_loss(pred['rdfs'], target['rdfs'])
        else:
            losses['rdf'] = torch.tensor(0.0, device=list(pred['rdfs'].values())[0].device)
        
        # Total weighted loss
        total_loss = (
            alpha_traj * losses['trajectory'] +
            alpha_thermo * losses['thermodynamics'] +
            alpha_rdf * losses['rdf'] +
            alpha_smooth * losses['smoothness']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


def visualize_latent_space(model: nn.Module,
                           dataloader: DataLoader,
                           device: str,
                           epoch: int,
                           save_dir: Path):
    """
    Visualize latent space using t-SNE (non-blocking).
    """
    model.eval()
    
    latents = []
    epsilons = []
    
    with torch.no_grad():
        for batch in dataloader:
            eps = batch['epsilon'].to(device)
            latent = model.encoder(eps)
            
            latents.append(latent.cpu().numpy())
            epsilons.append(eps.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    epsilons = np.concatenate(epsilons, axis=0).flatten()
    
    # t-SNE (fast for small datasets)
    if len(latents) > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(latents)-1))
        latents_2d = tsne.fit_transform(latents)
    else:
        latents_2d = latents[:, :2]  # Fallback
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=epsilons, cmap='viridis', s=100, alpha=0.7)
    
    # Annotate points
    for i, eps in enumerate(epsilons):
        ax.annotate(f'{eps:.2f}', (latents_2d[i, 0], latents_2d[i, 1]),
                   fontsize=9, alpha=0.8)
    
    plt.colorbar(scatter, label='Epsilon')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'Latent Space Visualization (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'latent_space_epoch_{epoch:03d}.png', dpi=150)
    plt.close()
    
    model.train()


def train_epoch_optimized(model: nn.Module,
                          train_loader: DataLoader,
                          criterion: PhysicsInformedLoss,
                          optimizer: optim.Optimizer,
                          scaler: GradScaler,
                          device: str,
                          epoch: int,
                          accumulation_steps: int = 4) -> Dict[str, float]:
    """
    Optimized training epoch with:
    - Mixed precision (FP16)
    - Gradient accumulation
    """
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'trajectory': 0.0,
        'thermodynamics': 0.0,
        'rdf': 0.0,
        'smoothness': 0.0
    }
    
    n_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        # Move to device
        epsilon = batch['epsilon'].to(device)
        
        # Mixed precision forward pass
        with autocast():
            pred = model(epsilon)
            
            # Prepare targets
            target = {}
            
            if batch['trajectory'][0] is not None:
                valid_trajs = [t for t in batch['trajectory'] if t is not None]
                if valid_trajs:
                    target['trajectory'] = torch.stack(valid_trajs).to(device)
                else:
                    target['trajectory'] = None
            else:
                target['trajectory'] = None
            
            if batch['thermodynamics'][0] is not None:
                target['thermodynamics'] = {
                    k: torch.stack([t[k] for t in batch['thermodynamics']]).to(device)
                    for k in batch['thermodynamics'][0].keys()
                }
            else:
                target['thermodynamics'] = None
            
            if batch['rdfs'][0] is not None:
                target['rdfs'] = {
                    pair: {
                        'g_r': torch.stack([r[pair]['g_r'] for r in batch['rdfs']]).to(device)
                    }
                    for pair in batch['rdfs'][0].keys()
                }
            else:
                target['rdfs'] = None
            
            # Compute loss
            loss, losses = criterion(pred, target)
            loss = loss / accumulation_steps  # Normalize for accumulation
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate losses
        for key in epoch_losses.keys():
            epoch_losses[key] += losses[key].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    # Average losses
    for key in epoch_losses.keys():
        epoch_losses[key] /= max(n_batches, 1)
    
    return epoch_losses


@torch.no_grad()
def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: PhysicsInformedLoss,
            device: str) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    val_losses = {
        'total': 0.0,
        'trajectory': 0.0,
        'thermodynamics': 0.0,
        'rdf': 0.0,
        'smoothness': 0.0
    }
    
    n_batches = 0
    
    for batch in val_loader:
        epsilon = batch['epsilon'].to(device)
        
        pred = model(epsilon)
        
        target = {}
        
        if batch['trajectory'][0] is not None:
            valid_trajs = [t for t in batch['trajectory'] if t is not None]
            if valid_trajs:
                target['trajectory'] = torch.stack(valid_trajs).to(device)
            else:
                target['trajectory'] = None
        else:
            target['trajectory'] = None
        
        if batch['thermodynamics'][0] is not None:
            target['thermodynamics'] = {
                k: torch.stack([t[k] for t in batch['thermodynamics']]).to(device)
                for k in batch['thermodynamics'][0].keys()
            }
        else:
            target['thermodynamics'] = None
        
        if batch['rdfs'][0] is not None:
            target['rdfs'] = {
                pair: {
                    'g_r': torch.stack([r[pair]['g_r'] for r in batch['rdfs']]).to(device)
                }
                for pair in batch['rdfs'][0].keys()
            }
        else:
            target['rdfs'] = None
        
        loss, losses = criterion(pred, target)
        
        for key in val_losses.keys():
            val_losses[key] += losses[key].item()
        n_batches += 1
    
    for key in val_losses.keys():
        val_losses[key] /= max(n_batches, 1)
    
    return val_losses


def train_model_optimized(n_epochs: int = 500,
                          batch_size: int = 2,
                          learning_rate: float = 1e-4,
                          device: str = 'cuda',
                          use_all_frames: bool = True,
                          checkpoint_dir: str = 'ml_integration/advanced/checkpoints',
                          log_dir: str = 'ml_integration/advanced/logs'):
    """
    OPTIMIZED training with:
    - Full data usage
    - Mixed precision
    - Gradient accumulation
    - Latent space visualization
    - TensorBoard logging
    """
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMIZED MD Generative Model Training                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Setup
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    vis_dir = log_dir / 'latent_visualizations'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
    
    # Use ALL available epsilon values
    train_epsilon = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    val_epsilon = [0.50]
    
    print(f"📊 Training Configuration:")
    print(f"  Train epsilon values: {train_epsilon}")
    print(f"  Val epsilon values: {val_epsilon}")
    print(f"  Use all frames: {use_all_frames}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print()
    
    # Dataloaders with optimizations
    print("Creating dataloaders...")
    
    if use_all_frames:
        # Use ALL 4001 frames!
        max_frames = None  # Load all
        stride = 1  # No skipping
        print("  ⚠️  Loading ALL 4001 frames per epsilon (may take time)...")
    else:
        max_frames = 100
        stride = 40
        print(f"  Using {max_frames} frames with stride {stride}")
    
    train_loader, val_loader = create_dataloaders(
        train_epsilon=train_epsilon,
        val_epsilon=val_epsilon,
        batch_size=batch_size,
        traj_stride=stride,
        max_traj_frames=max_frames,
        cache_dir="ml_integration/advanced/data/cache"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Model
    print("Creating model...")
    model = MDGenerativeModel(
        latent_dim=512,
        n_atoms=5541,
        thermo_seq_len=1000,
        rdf_bins=200
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print()
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Optimizer with weight decay for regularization
    criterion = PhysicsInformedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    print(f"🚀 Starting OPTIMIZED training...")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: FP16")
    print(f"  Gradient Accumulation: 4 steps")
    print(f"  Weight Decay: 1e-5")
    print()
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch_optimized(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            accumulation_steps=4
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # TensorBoard logging
        for key in train_losses.keys():
            writer.add_scalar(f'Train/{key}', train_losses[key], epoch)
            writer.add_scalar(f'Val/{key}', val_losses[key], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Console logging
        print(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s) - LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train: {train_losses['total']:.6f} (T:{train_losses['trajectory']:.6f} "
              f"Th:{train_losses['thermodynamics']:.6f} R:{train_losses['rdf']:.6f})")
        print(f"  Val:   {val_losses['total']:.6f}")
        
        # Visualize latent space (every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  📊 Visualizing latent space...")
            visualize_latent_space(model, train_loader, device, epoch, vis_dir)
            visualize_latent_space(model, val_loader, device, epoch, vis_dir)
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✅ Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered (patience={patience})")
            break
        
        # Checkpoint
        if epoch % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        
        print()
    
    writer.close()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Training Complete!                                            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"TensorBoard logs: {log_dir}/tensorboard")
    print(f"Latent visualizations: {vis_dir}")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}/tensorboard")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    print()
    
    train_model_optimized(
        n_epochs=500,
        batch_size=2,
        learning_rate=1e-4,
        device=device,
        use_all_frames=False,  # Start with subset, set True for full training
    )
