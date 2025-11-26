#!/usr/bin/env python3
"""
PRODUCTION-READY Training Script
=================================

Features:
- Comprehensive tqdm progress tracking
- Advanced checkpoint management
- Overfitting detection
- Double descent detection 
- TensorBoard logging
- Latent space visualization
- Full CUDA optimization

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
from typing import Dict, Tuple, List
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from model import MDGenerativeModel
from dataset import create_dataloaders


class TrainingMonitor:
    """
    Monitor training for overfitting and double descent.
    """
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Double descent tracking
        self.val_loss_trend = []
        self.double_descent_detected = False
        
    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Update with new epoch results."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            return True  # Improved
        else:
            self.patience_counter += 1
            return False  # No improvement
    
    def detect_overfitting(self) -> Dict:
        """Detect overfitting based on train/val loss gap."""
        if len(self.train_losses) < 10:
            return {'overfitting': False, 'severity': 0.0}
        
        recent_train = np.mean(self.train_losses[-10:])
        recent_val = np.mean(self.val_losses[-10:])
        
        gap = recent_val - recent_train
        gap_ratio = gap / (recent_train + 1e-8)
        
        overfitting = gap_ratio > 0.2  # 20% gap threshold
        severity = min(gap_ratio, 1.0)
        
        return {
            'overfitting': overfitting,
            'severity': severity,
            'gap': gap,
            'gap_ratio': gap_ratio
        }
    
    def detect_double_descent(self, window: int = 20) -> Dict:
        """
        Detect double descent phenomenon.
        
        Double descent: val loss goes down → up (overfitting) → down again
        """
        if len(self.val_losses) < window * 2:
            return {'detected': False, 'phase': 'early'}
        
        # Smooth validation loss
        val_smooth = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
        
        if len(val_smooth) < 3:
            return {'detected': False, 'phase': 'early'}
        
        # Find local minima and maxima
        diffs = np.diff(val_smooth)
        
        # Check for U-shape followed by another descent
        # This is simplified - true double descent is complex
        recent_trend = diffs[-window:] if len(diffs) >= window else diffs
        
        if len(recent_trend) > 5:
            # Check if we're improving after previous worsening
            earlier_worse = np.mean(recent_trend[:len(recent_trend)//2]) > 0
            recent_better = np.mean(recent_trend[len(recent_trend)//2:]) < 0
            
            if earlier_worse and recent_better:
                self.double_descent_detected = True
                return {'detected': True, 'phase': 'second_descent'}
        
        return {'detected': self.double_descent_detected, 'phase': 'first_descent'}
    
    def should_stop(self) -> bool:
        """Check if should stop training."""
        return self.patience_counter >= self.patience
    
    def get_status(self) -> Dict:
        """Get current training status."""
        overfit = self.detect_overfitting()
        double_desc = self.detect_double_descent()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'overfitting': overfit,
            'double_descent': double_desc
        }


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss functions."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def trajectory_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)
    
    def thermodynamics_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        total_loss = 0
        count = 0
        for key in ['temperature', 'pressure', 'density', 'potential_energy']:
            if key in pred and key in target:
                total_loss += self.mse(pred[key], target[key])
                count += 1
        return total_loss / max(count, 1)
    
    def rdf_loss(self, pred: Dict, target: Dict) -> torch.Tensor:
        total_loss = 0
        count = 0
        for pair in ['CC', 'CO', 'OO']:
            if pair in pred and pair in target:
                total_loss += self.mse(pred[pair], target[pair]['g_r'])
                count += 1
        return total_loss / max(count, 1)
    
    def smoothness_penalty(self, thermo: Dict) -> torch.Tensor:
        penalty = 0
        count = 0
        for key, values in thermo.items():
            diff = values[:, 1:] - values[:, :-1]
            penalty += diff.pow(2).mean()
            count += 1
        return penalty / max(count, 1)
    
    def forward(self, pred: Dict, target: Dict,
                alpha_traj: float = 1.0,
                alpha_thermo: float = 2.0,
                alpha_rdf: float = 3.0,
                alpha_smooth: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        
        losses = {}
        device = list(pred['rdfs'].values())[0].device
        
        # Trajectory
        if pred['trajectory'] is not None and target['trajectory'] is not None:
            losses['trajectory'] = self.trajectory_loss(pred['trajectory'], target['trajectory'])
        else:
            losses['trajectory'] = torch.tensor(0.0, device=device)
        
        # Thermodynamics
        if pred['thermodynamics'] and target['thermodynamics']:
            losses['thermodynamics'] = self.thermodynamics_loss(pred['thermodynamics'], target['thermodynamics'])
            losses['smoothness'] = self.smoothness_penalty(pred['thermodynamics'])
        else:
            losses['thermodynamics'] = torch.tensor(0.0, device=device)
            losses['smoothness'] = torch.tensor(0.0, device=device)
        
        # RDF
        if pred['rdfs'] and target['rdfs']:
            losses['rdf'] = self.rdf_loss(pred['rdfs'], target['rdfs'])
        else:
            losses['rdf'] = torch.tensor(0.0, device=device)
        
        total_loss = (
            alpha_traj * losses['trajectory'] +
            alpha_thermo * losses['thermodynamics'] +
            alpha_rdf * losses['rdf'] +
            alpha_smooth * losses['smoothness']
        )
        
        losses['total'] = total_loss
        return total_loss, losses


def visualize_latent_space(model: nn.Module, dataloader: DataLoader,
                           device: str, epoch: int, save_dir: Path):
    """Visualize latent space with t-SNE."""
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
    
    if len(latents) > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(latents)-1))
        latents_2d = tsne.fit_transform(latents)
    else:
        latents_2d = latents[:, :2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                        c=epsilons, cmap='viridis', s=100, alpha=0.7)
    
    for i, eps in enumerate(epsilons):
        ax.annotate(f'{eps:.2f}', (latents_2d[i, 0], latents_2d[i, 1]),
                   fontsize=9, alpha=0.8)
    
    plt.colorbar(scatter, label='Epsilon')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'Latent Space (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'latent_epoch_{epoch:04d}.png', dpi=150)
    plt.close()
    
    # Save to CSV for high-quality rendering
    csv_file = save_dir.parent / 'latent_vectors.csv'
    mode = 'a' if csv_file.exists() else 'w'
    
    with open(csv_file, mode) as f:
        if mode == 'w':
            header = "epoch,epsilon," + ",".join([f"z_{i}" for i in range(latents.shape[1])])
            f.write(header + "\n")
        
        for i in range(len(epsilons)):
            row = [f"{epoch}", f"{epsilons[i]:.4f}"] + [f"{z:.6f}" for z in latents[i]]
            f.write(",".join(row) + "\n")

    model.train()


def train_production(n_epochs: int = 500,
                     batch_size: int = 2,
                     learning_rate: float = 1e-4,
                     device: str = 'cuda',
                     use_all_frames: bool = False,
                     checkpoint_dir: str = '../checkpoints',
                     log_dir: str = '../logs',
                     patience: int = 50):
    """
    PRODUCTION training with all optimizations and monitoring.
    """
    
    # Setup
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    vis_dir = log_dir / 'latent_viz'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
    
    # Training monitor
    monitor = TrainingMonitor(patience=patience)
    
    # Print header
    print("\n" + "="*70)
    print("  PRODUCTION MD GENERATIVE MODEL TRAINING".center(70))
    print("="*70 + "\n")
    
    # Configuration
    # Train on 0.0-0.50 (80% frames), Validate on 0.0-0.50 (20% frames)
    # Test on 0.55-1.10 (Extrapolation) done after training
    train_epsilon = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    val_epsilon = train_epsilon  # Use same epsilons, dataset handles splitting
    
    print("📋 Configuration:")
    print(f"  Train epsilon: {train_epsilon}")
    print(f"  Val epsilon: {val_epsilon}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Use all frames: {use_all_frames}")
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Dataloaders
    print("📊 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_epsilon=train_epsilon,
        val_epsilon=val_epsilon,
        batch_size=batch_size,
        traj_stride=1 if use_all_frames else 40,
        max_traj_frames=None if use_all_frames else 100,
        cache_dir="../data/cache"
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Save normalization stats
    print("💾 Saving normalization stats...")
    stats = {
        'trajectory': {
            'mean': train_loader.dataset.coord_mean.tolist(),
            'std': train_loader.dataset.coord_std.tolist()
        },
        'thermodynamics': {
            k: {
                'mean': float(v['mean']),
                'std': float(v['std'])
            } for k, v in train_loader.dataset.thermo_stats.items()
        }
    }
    with open(log_dir / 'normalization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print()
    
    # Model
    print("🧠 Creating model...")
    model = MDGenerativeModel(
        latent_dim=512,
        n_atoms=5541,
        thermo_seq_len=1000,
        rdf_bins=200
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print()
    
    # Training components
    criterion = PhysicsInformedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training loop
    print("🚀 Starting training...\n")
    
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # TRAIN
        model.train()
        train_losses = {k: 0.0 for k in ['total', 'trajectory', 'thermodynamics', 'rdf', 'smoothness']}
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", position=1, leave=False)
        for i, batch in enumerate(batch_pbar):
            epsilon = batch['epsilon'].to(device)
            
            # Targets - need to define before model call for seq_len inference
            target = {
                'trajectory': batch['trajectory'].to(device) if batch['trajectory'] is not None else None,
                'thermodynamics': {k: v.to(device) for k, v in batch['thermodynamics'].items()} if batch['thermodynamics'] is not None else None,
                'rdfs': {pair: {'g_r': val['g_r'].to(device)} for pair, val in batch['rdfs'].items()} if batch['rdfs'] is not None else None
            }
            
            with autocast():
                pred = model(epsilon, target=target)
                
                loss, losses = criterion(pred, target)
                loss = loss / 4  # Gradient accumulation
            
            scaler.scale(loss).backward()
            
            if (i + 1) % 4 == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            for key in train_losses.keys():
                train_losses[key] += losses[key].item()
            
            batch_pbar.set_postfix({'loss': f"{loss.item() * 4:.4f}"})
        
        for key in train_losses.keys():
            train_losses[key] /= len(train_loader)
        
        #VALIDATE
        model.eval()
        val_losses = {k: 0.0 for k in ['total', 'trajectory', 'thermodynamics', 'rdf', 'smoothness']}
        
        with torch.no_grad():
            for batch in val_loader:
                epsilon = batch['epsilon'].to(device)
                
                # Targets - need to define before model call for seq_len inference
                target = {
                    'trajectory': batch['trajectory'].to(device) if batch['trajectory'] is not None else None,
                    'thermodynamics': {k: v.to(device) for k, v in batch['thermodynamics'].items()} if batch['thermodynamics'] is not None else None,
                    'rdfs': {pair: {'g_r': val['g_r'].to(device)} for pair, val in batch['rdfs'].items()} if batch['rdfs'] is not None else None
                }
                
                pred = model(epsilon, target=target)
                
                loss, losses = criterion(pred, target)
                
                for key in val_losses.keys():
                    val_losses[key] += losses[key].item()
        
        for key in val_losses.keys():
            val_losses[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update monitor
        improved = monitor.update(epoch, train_losses['total'], val_losses['total'], current_lr)
        status = monitor.get_status()
        
        # Calculate metrics
        train_loss = train_losses['total']
        val_loss = val_losses['total']
        perplexity = np.exp(val_loss)
        
        # TensorBoard logging
        for key in train_losses.keys():
            writer.add_scalar(f'Train/{key}', train_losses[key], epoch)
            writer.add_scalar(f'Val/{key}', val_losses[key], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Overfitting/Severity', status['overfitting']['severity'], epoch)
        writer.add_scalar('Metrics/Perplexity', perplexity, epoch)
        
        # Update progress bar
        epoch_time = time.time() - epoch_start
        epoch_pbar.set_postfix({
            'train': f"{train_loss:.4f}",
            'val': f"{val_loss:.4f}",
            'ppl': f"{perplexity:.2f}",
            'best': f"{status['best_val_loss']:.4f}",
            'patience': f"{status['patience_counter']}/{monitor.patience}"
        })
        
        # Detailed logging
        tqdm.write(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
        tqdm.write(f"  Train: {train_loss:.6f} | Val: {val_loss:.6f} | Ppl: {perplexity:.2f}")
        tqdm.write(f"  Best: {status['best_val_loss']:.6f} @ epoch {status['best_epoch']}")
        
        if status['overfitting']['overfitting']:
            tqdm.write(f"  ⚠️  Overfitting detected (severity: {status['overfitting']['severity']:.2%})")
        
        if status['double_descent']['detected']:
            tqdm.write(f"  📈 Double descent: {status['double_descent']['phase']}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'monitor_state': status
        }
        
        if improved:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            tqdm.write(f"  ✅ Best model saved")
        
        if epoch % 25 == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt')
            tqdm.write(f"  💾 Checkpoint saved")
        
        # Latent visualization
        if epoch % 10 == 0:
            tqdm.write(f"  📊 Generating latent space visualization...")
            visualize_latent_space(model, train_loader, device, epoch, vis_dir)
        
        # Early stopping
        if monitor.should_stop():
            tqdm.write(f"\n⚠️  Early stopping at epoch {epoch}")
            break
        
        tqdm.write("")  # Newline
    
    # Final save
    torch.save(checkpoint, checkpoint_dir / 'final_model.pt')
    
    # Save training history
    history = {
        'train_losses': monitor.train_losses,
        'val_losses': monitor.val_losses,
        'learning_rates': monitor.learning_rates,
        'best_epoch': monitor.best_epoch,
        'best_val_loss': monitor.best_val_loss
    }
    
    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE".center(70))
    print("="*70)
    print(f"\nBest Val Loss: {monitor.best_val_loss:.6f} @ epoch {monitor.best_epoch}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print(f"\nView TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}/tensorboard\n")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_production(
        n_epochs=1000,  # Increased epochs for more exploration
        batch_size=8,  # Increased from 2 to better utilize A100 GPU
        learning_rate=1e-4,
        device=device,
        use_all_frames=False,
        patience=1000  # Effectively disable early stopping
    )
