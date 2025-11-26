import torch
import json
import numpy as np
from pathlib import Path
from dataset import MDSimulationDataset

def save_stats():
    print("Loading dataset to compute normalization stats...")
    # Use same parameters as in train_production.py
    train_epsilon = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    dataset = MDSimulationDataset(
        epsilon_values=train_epsilon,
        base_dir="/store/shuvam/learning_solvent_effects",
        traj_stride=40,
        max_traj_frames=100,
        cache_dir="../data/cache",
        split='train',
        split_ratio=0.8
    )
    
    stats = {
        'trajectory': {
            'mean': dataset.coord_mean.tolist(),
            'std': dataset.coord_std.tolist()
        },
        'thermodynamics': {
            k: {
                'mean': float(v['mean']),
                'std': float(v['std'])
            } for k, v in dataset.thermo_stats.items()
        }
    }
    
    output_path = Path("../logs/normalization_stats.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"✅ Normalization stats saved to {output_path}")

if __name__ == "__main__":
    save_stats()
