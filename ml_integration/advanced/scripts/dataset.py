#!/usr/bin/env python3
"""
PyTorch Dataset for MD Simulations
===================================

Creates PyTorch Dataset with preprocessing and augmentation.

Author: Shuvam Banerji Seal
Date: November 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm

from data_loader import MDDataLoader


class MDSimulationDataset(Dataset):
    """
    PyTorch Dataset for MD simulations with comprehensive preprocessing.
    """
    
    def __init__(self, 
                 epsilon_values: List[float],
                 base_dir: str = "/store/shuvam/learning_solvent_effects",
                 traj_stride: int = 40,
                 max_traj_frames: int = 100,
                 cache_dir: Optional[str] = None,
                 augment: bool = True,
                 split: str = 'all',  # 'train', 'val', or 'all'
                 split_ratio: float = 0.8):
        """
        Args:
            epsilon_values: List of epsilon values to load
            base_dir: Base directory for data
            traj_stride: Stride for loading trajectory frames
            max_traj_frames: Maximum trajectory frames to load
            cache_dir: Directory to cache processed data
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'all')
            split_ratio: Ratio of data to use for training
        """
        self.epsilon_values = epsilon_values
        self.base_dir = Path(base_dir)
        self.traj_stride = traj_stride
        self.max_traj_frames = max_traj_frames
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment
        self.split = split
        self.split_ratio = split_ratio
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.data = self._load_all_data()
        
        # Compute normalization statistics
        self.compute_normalization_stats()
        
    def _load_all_data(self) -> List[Dict]:
        """Load data for all epsilon values."""
        loader = MDDataLoader(self.base_dir)
        all_data = []
        
        for eps in tqdm(self.epsilon_values, desc="Loading datasets"):
            # Check cache first
            if self.cache_dir:
                cache_file = self.cache_dir / f"epsilon_{eps:.2f}.pkl"
                if cache_file.exists():
                    print(f"  Loading from cache: {cache_file}")
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                else:
                    # Load fresh data
                    data = loader.load_all_data(
                        eps, 
                        traj_stride=self.traj_stride,
                        max_traj_frames=self.max_traj_frames
                    )
                    # Save to cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"  Saved to cache: {cache_file}")
            else:
                # Load fresh data without caching
                data = loader.load_all_data(
                    eps, 
                    traj_stride=self.traj_stride,
                    max_traj_frames=self.max_traj_frames
                )
            
            # Apply split (to both cached and fresh data)
            if self.split != 'all' and data['trajectory'] and data['trajectory']['coordinates'] is not None:
                n_frames = len(data['trajectory']['coordinates'])
                split_idx = int(n_frames * self.split_ratio)
                
                if self.split == 'train':
                    # Slice trajectory
                    data['trajectory']['coordinates'] = data['trajectory']['coordinates'][:split_idx]
                    
                    # Slice thermodynamics (only if arrays, not scalars)
                    if data['thermodynamics']:
                        for key in data['thermodynamics']:
                            if hasattr(data['thermodynamics'][key], '__len__'):  # Check if iterable
                                data['thermodynamics'][key] = data['thermodynamics'][key][:split_idx]
                            
                elif self.split == 'val':
                    # Slice trajectory
                    data['trajectory']['coordinates'] = data['trajectory']['coordinates'][split_idx:]
                    
                    # Slice thermodynamics (only if arrays, not scalars)
                    if data['thermodynamics']:
                        for key in data['thermodynamics']:
                            if hasattr(data['thermodynamics'][key], '__len__'):  # Check if iterable
                                data['thermodynamics'][key] = data['thermodynamics'][key][split_idx:]
            
            all_data.append(data)
        
        return all_data
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        print("\nComputing normalization statistics...")
        
        # Trajectory statistics
        all_coords = []
        for data in self.data:
            if data['trajectory'] and data['trajectory']['coordinates'] is not None:
                coords = data['trajectory']['coordinates']
                all_coords.append(coords.reshape(-1, 3))
        
        if all_coords:
            all_coords = np.concatenate(all_coords, axis=0)
            self.coord_mean = all_coords.mean(axis=0, keepdims=True).astype(np.float32)
            self.coord_std = (all_coords.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
        else:
            self.coord_mean = np.zeros((1, 3), dtype=np.float32)
            self.coord_std = np.ones((1, 3), dtype=np.float32)
        
        # Thermodynamics statistics
        thermo_data = {
            'temperature': [],
            'pressure': [],
            'density': [],
            'potential_energy': []
        }
        
        for data in self.data:
            if data['thermodynamics']:
                for key in thermo_data.keys():
                    thermo_data[key].append(data['thermodynamics'][key])
        
        self.thermo_stats = {}
        for key, values in thermo_data.items():
            if values:
                all_values = np.concatenate(values)
                self.thermo_stats[key] = {
                    'mean': all_values.mean().astype(np.float32),
                    'std': (all_values.std() + 1e-8).astype(np.float32)
                }
            else:
                self.thermo_stats[key] = {'mean': 0.0, 'std': 1.0}
        
        # RDF statistics (keep g(r) in [0, ~5] range, no normalization needed)
        print("  ✅ Normalization statistics computed")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'epsilon': scalar
                - 'trajectory': (n_frames, n_atoms, 3)
                - 'thermodynamics': dict of time series
                - 'rdfs': dict of RDF curves
        """
        data = self.data[idx]
        sample = {}
        
        # Epsilon
        sample['epsilon'] = torch.tensor([data['epsilon']], dtype=torch.float32)
        
        # Trajectory
        if data['trajectory'] and data['trajectory']['coordinates'] is not None:
            coords = data['trajectory']['coordinates'].astype(np.float32)
            
            # Normalize
            coords = (coords - self.coord_mean) / self.coord_std
            
            # Apply augmentation
            if self.augment:
                coords = self._augment_trajectory(coords)
            
            sample['trajectory'] = torch.from_numpy(coords).float()
        else:
            sample['trajectory'] = None
        
        # Thermodynamics
        if data['thermodynamics']:
            thermo = {}
            for key in ['temperature', 'pressure', 'density', 'potential_energy']:
                values = data['thermodynamics'][key].astype(np.float32)
                
                # Normalize
                mean = self.thermo_stats[key]['mean']
                std = self.thermo_stats[key]['std']
                values = (values - mean) / std
                
                thermo[key] = torch.from_numpy(values).float()
            
            sample['thermodynamics'] = thermo
        else:
            sample['thermodynamics'] = None
        
        # RDFs
        if data['rdfs']:
            rdfs = {}
            for pair in ['CC', 'CO', 'OO']:
                if pair in data['rdfs']:
                    rdf_data = data['rdfs'][pair]
                    rdfs[pair] = {
                        'r': torch.from_numpy(rdf_data['r'].astype(np.float32)).float(),
                        'g_r': torch.from_numpy(rdf_data['g_r'].astype(np.float32)).float()
                    }
            sample['rdfs'] = rdfs
        else:
            sample['rdfs'] = None
        
        return sample
    
    def _augment_trajectory(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to trajectory.
        
        Args:
            coords: (n_frames, n_atoms, 3)
            
        Returns:
            Augmented coordinates
        """
        # Random rotation (around random axis)
        if np.random.rand() < 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            axis = np.random.choice(['x', 'y', 'z'])
            coords = self._rotate_coords(coords, theta, axis)
        
        # Random translation (small, within normalized space)
        if np.random.rand() < 0.5:
            translation = np.random.normal(0, 0.1, size=(1, 1, 3))
            coords = coords + translation
        
        # Add thermal noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, size=coords.shape)
            coords = coords + noise
        
        return coords
    
    def _rotate_coords(self, coords: np.ndarray, theta: float, axis: str) -> np.ndarray:
        """Rotate coordinates around axis."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, cos_t, -sin_t],
                [0, sin_t, cos_t]
            ])
        elif axis == 'y':
            R = np.array([
                [cos_t, 0, sin_t],
                [0, 1, 0],
                [-sin_t, 0, cos_t]
            ])
        else:  # z
            R = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ])
        
        # Apply rotation
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        coords_rotated = coords_flat @ R.T
        return coords_rotated.reshape(original_shape)


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle None values properly.
    
    PyTorch default collate stacks tensors, but we have Nones and dicts.
    This function properly batches our data structure.
    """
    # Batch epsilon values
    epsilons = torch.stack([item['epsilon'] for item in batch])
    
    # Batch trajectories (average across frames first, then stack)
    trajectories = []
    for item in batch:
        if item['trajectory'] is not None:
            # item['trajectory'] is (frames, atoms, 3)
            # Average across frames to get equilibrium configuration (atoms, 3)
            traj_mean = item['trajectory'].mean(dim=0)
            trajectories.append(traj_mean)
        else:
            trajectories.append(None)
            
    # Filter Nones for stacking check
    valid_trajs = [t for t in trajectories if t is not None]
    
    if valid_trajs:
        # Check if all shapes match (should be true now as (atoms, 3))
        shapes = [t.shape for t in valid_trajs]
        if len(set(shapes)) == 1:
            trajectories = torch.stack(valid_trajs)  # (batch, atoms, 3)
        # else: keep as list (should not happen if atoms count is constant)
    else:
        trajectories = None
    
    # Batch thermodynamics - stack each property across batch
    thermo_list = [item['thermodynamics'] for item in batch if item['thermodynamics'] is not None]
    if thermo_list and len(thermo_list) == len(batch):
        # All items have thermodynamics - stack them
        thermodynamics = {}
        for key in thermo_list[0].keys():
            thermodynamics[key] = torch.stack([t[key] for t in thermo_list])
    else:
        thermodynamics = None
    
    # Batch RDFs - stack g_r values across batch
    rdfs_list = [item['rdfs'] for item in batch if item['rdfs'] is not None]
    if rdfs_list and len(rdfs_list) == len(batch):
        # All items have RDFs - stack them
        rdfs = {}
        for pair in rdfs_list[0].keys():
            rdfs[pair] = {
                'r': rdfs_list[0][pair]['r'],  # r is same for all
                'g_r': torch.stack([r[pair]['g_r'] for r in rdfs_list])
            }
    else:
        rdfs = None
    
    return {
        'epsilon': epsilons,
        'trajectory': trajectories,  # Stacked tensor or None
        'thermodynamics': thermodynamics,  # Dict of stacked tensors or None
        'rdfs': rdfs  # Dict with stacked g_r or None
    }


def create_dataloaders(train_epsilon: List[float],
                       val_epsilon: List[float],
                       batch_size: int = 4,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_epsilon: List of epsilon values for training
        val_epsilon: List of epsilon values for validation
        batch_size: Batch size
        **dataset_kwargs: Additional arguments for MDSimulationDataset
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = MDSimulationDataset(
        epsilon_values=train_epsilon,
        augment=True,
        split='train',
        split_ratio=0.8,
        **dataset_kwargs
    )
    
    val_dataset = MDSimulationDataset(
        epsilon_values=val_epsilon,  # Can be same as train_epsilon now
        augment=False,
        split='val',
        split_ratio=0.8,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for production
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use custom collate function
        drop_last=True  # Drop incomplete batches to avoid size mismatches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use custom collate function
        drop_last=False  # Keep all validation samples
    )
    
    return train_loader, val_loader


def test_dataset():
    """Test the dataset."""
    print("Testing MDSimulationDataset...")
    
    # Create dataset with a few epsilon values
    epsilon_values = [0.0, 0.05, 0.10]
    
    dataset = MDSimulationDataset(
        epsilon_values=epsilon_values,
        traj_stride=40,
        max_traj_frames=10,
        cache_dir="ml_integration/advanced/data/cache",
        augment=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample structure:")
    print(f"  Epsilon: {sample['epsilon']}")
    if sample['trajectory'] is not None:
        print(f"  Trajectory shape: {sample['trajectory'].shape}")
    if sample['thermodynamics']:
        print(f"  Thermodynamics keys: {list(sample['thermodynamics'].keys())}")
        for key, val in sample['thermodynamics'].items():
            print(f"    {key}: shape={val.shape}, mean={val.mean():.4f}, std={val.std():.4f}")
    if sample['rdfs']:
        print(f"  RDF pairs: {list(sample['rdfs'].keys())}")
    
    # Test dataloader
    print(f"\nTesting DataLoader...")
    train_loader, val_loader = create_dataloaders(
        train_epsilon=[0.0, 0.05],
        val_epsilon=[0.10],
        batch_size=2,
        traj_stride=40,
        max_traj_frames=10,
        cache_dir="ml_integration/advanced/data/cache"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch epsilon: {batch['epsilon'].shape}")
    if batch['trajectory'][0] is not None:
        # Filter out None trajectories
        valid_trajs = [t for t in batch['trajectory'] if t is not None]
        if valid_trajs:
            print(f"Batch trajectory: {valid_trajs[0].shape}")


if __name__ == "__main__":
    test_dataset()
