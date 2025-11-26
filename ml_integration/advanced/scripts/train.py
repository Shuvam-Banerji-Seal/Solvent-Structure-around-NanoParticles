#!/usr/bin/env python3
"""
Training Script for MD Generative Model
========================================

Train the multi-task generative model with physics-informed losses.

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import time
from typing import Dict, Tuple

from model import MDGenerativeModel
from dataset import create_dataloaders


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss functions to ensure physical validity.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def trajectory_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss for coordinates."""
        return self.mse(pred, target)
    
    def thermodynamics_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        """MSE loss for all thermodynamic properties."""
        total_loss = 0
        for key in ['temperature', 'pressure', 'density', 'potential_energy']:
            if key in pred and key in target:
                total_loss += self.mse(pred[key], target[key])
        return total_loss
    
    def rdf_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        """MSE loss for RDF curves."""
        total_loss = 0
        for pair in ['CC', 'CO', 'OO']:
            if pair in pred and pair in target:
                total_loss += self.mse(pred[pair], target[pair]['g_r'])
        return total_loss
    
    def energy_conservation_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Penalize large variations in total energy (should be roughly constant).
        This is a simplified proxy - full implementation would calculate actual energy.
        """
        # Use center of mass motion as proxy for energy consistency
        com = trajectory.mean(dim=1)  # (batch, 3)
        com_var = com.var(dim=0).mean()
        return com_var
    
    def forward(self, 
                pred: Dict,
                target: Dict,
                alpha_traj: float = 1.0,
                alpha_thermo: float = 1.0,
                alpha_rdf: float = 1.0,
                alpha_physics: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.
        
        Args:
            pred: Predictions from model
            target: Ground truth data
            alpha_*: Loss weights
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual losses for logging
        """
        losses = {}
        
        # Trajectory loss
        if pred['trajectory'] is not None and target['trajectory'] is not None:
            losses['trajectory'] = self.trajectory_loss(pred['trajectory'], target['trajectory'])
        else:
            losses['trajectory'] = torch.tensor(0.0, device=pred['rdfs']['CC'].device)
        
        # Thermodynamics loss
        if pred['thermodynamics'] and target['thermodynamics']:
            losses['thermodynamics'] = self.thermodynamics_loss(
                pred['thermodynamics'], 
                target['thermodynamics']
            )
        else:
            losses['thermodynamics'] = torch.tensor(0.0, device=pred['rdfs']['CC'].device)
        
        # RDF loss
        if pred['rdfs'] and target['rdfs']:
            losses['rdf'] = self.rdf_loss(pred['rdfs'], target['rdfs'])
        else:
            losses['rdf'] = torch.tensor(0.0, device=pred['rdfs']['CC'].device)
        
        # Physics penalty
        if pred['trajectory'] is not None:
            losses['physics'] = self.energy_conservation_penalty(pred['trajectory'])
        else:
            losses['physics'] = torch.tensor(0.0, device=pred['rdfs']['CC'].device)
        
        # Total weighted loss
        total_loss = (
            alpha_traj * losses['trajectory'] +
            alpha_thermo * losses['thermodynamics'] +
            alpha_rdf * losses['rdf'] +
            alpha_physics * losses['physics']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: PhysicsInformedLoss,
                optimizer: optim.Optimizer,
                device: str,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'trajectory': 0.0,
        'thermodynamics': 0.0,
        'rdf': 0.0,
        'physics': 0.0
    }
    
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        epsilon = batch['epsilon'].to(device)
        
        # Forward pass
        pred = model(epsilon)
        
        # Prepare targets
        target = {}
        
        # Trajectory target (handle None values)
        if batch['trajectory'][0] is not None:
            valid_trajs = [t for t in batch['trajectory'] if t is not None]
            if valid_trajs:
                target['trajectory'] = torch.stack(valid_trajs).to(device)
            else:
                target['trajectory'] = None
        else:
            target['trajectory'] = None
        
        # Thermodynamics target
        if batch['thermodynamics'][0] is not None:
            target['thermodynamics'] = {
                k: torch.stack([t[k] for t in batch['thermodynamics']]).to(device)
                for k in batch['thermodynamics'][0].keys()
            }
        else:
            target['thermodynamics'] = None
        
        # RDF target
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses.keys():
            epoch_losses[key] += losses[key].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
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
        'physics': 0.0
    }
    
    n_batches = 0
    
    for batch in val_loader:
        epsilon = batch['epsilon'].to(device)
        
        pred = model(epsilon)
        
        # Prepare targets (same as training)
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


def train_model(train_epsilon: list,
                val_epsilon: list,
                n_epochs: int = 500,
                batch_size: int = 4,
                learning_rate: float = 1e-4,
                device: str = 'cuda',
                checkpoint_dir: str = 'ml_integration/advanced/checkpoints',
                log_dir: str = 'ml_integration/advanced/logs'):
    """
    Main training function.
    """
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Training MD Generative Model                                 ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Create directories
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_epsilon=train_epsilon,
        val_epsilon=val_epsilon,
        batch_size=batch_size,
        traj_stride=40,
        max_traj_frames=100,
        cache_dir="ml_integration/advanced/data/cache"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Create model
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
    
    # Create loss and optimizer
    criterion = PhysicsInformedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_history = []
    
    print(f"Starting training for {n_epochs} epochs...")
    print(f"Device: {device}")
    print()
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"    - Trajectory: {train_losses['trajectory']:.6f}")
        print(f"    - Thermodynamics: {train_losses['thermodynamics']:.6f}")
        print(f"    - RDF: {train_losses['rdf']:.6f}")
        print(f"    - Physics: {train_losses['physics']:.6f}")
        print(f"  Val Loss: {val_losses['total']:.6f}")
        
        # Update scheduler
        scheduler.step(val_losses['total'])
        
        # Save history
        train_history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'time': epoch_time
        })
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  ✅ Saved best model (val_loss={best_val_loss:.6f})")
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        
        print()
    
    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pt')
    
    with open(log_dir / 'train_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Training Complete!                                            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Training history saved to: {log_dir}/train_history.json")


if __name__ == "__main__":
    # Configure training
    train_epsilon = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    val_epsilon = [0.50]
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Train
    train_model(
        train_epsilon=train_epsilon,
        val_epsilon=val_epsilon,
        n_epochs=100,  # Start with 100 epochs for testing
        batch_size=2,   # Small batch size due to large model
        learning_rate=1e-4,
        device=device
    )
