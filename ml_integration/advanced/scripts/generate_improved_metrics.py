#!/usr/bin/env python3
"""
Generate Improved Model Metrics
===============================

Calculates physical error metrics (MSE) for the improved model
across the full range of epsilons.
Saves to: ../logs_improved/batch_metrics.csv
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from model_improved import ImprovedMDGenerativeModel
from dataset import create_dataloaders

def generate_metrics():
    # Setup paths
    checkpoint_path = "../checkpoints_improved/best_model_improved.pt"
    output_dir = Path("../logs_improved")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = ImprovedMDGenerativeModel(latent_dim=512).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Load Data
    print("Loading dataset...")
    # We use the validation loader to get all data without shuffling if possible, 
    # but create_dataloaders splits it. We'll use the full dataset logic if needed,
    # or just iterate through val_loader and train_loader to get all epsilons.
    # Actually, let's just use the validation loader for now as it covers the range 
    # if the split was random, OR better: create a custom loader for all files.
    
    # Define epsilons (0.00 to 1.10)
    all_epsilons = [f"{i/100:.2f}" for i in range(0, 111)]
    
    # Pass all to validation so we can iterate once
    train_loader, val_loader = create_dataloaders(
        base_dir="../../data",
        train_epsilon=[], # Empty
        val_epsilon=all_epsilons,
        batch_size=1,
        num_workers=4
    )
    
    metrics_list = []
    seen_epsilons = set()
    
    def process_loader(loader, name):
        print(f"Processing {name} data...")
        with torch.no_grad():
            for batch in tqdm(loader):
                epsilon = batch['epsilon'].to(device)
                eps_val = float(epsilon.item())
                
                # Skip if we already processed this epsilon (to avoid duplicates if any)
                # Although in MD datasets, multiple frames exist for same epsilon.
                # We want the AVERAGE error for that epsilon.
                
                target_traj = batch['trajectory'].to(device)
                target_thermo = batch['thermodynamics'].to(device)
                target_rdf = batch['rdf'].to(device)
                
                # Forward pass
                outputs = model(epsilon)
                pred_traj = outputs['trajectory']
                pred_thermo = outputs['thermodynamics']
                pred_rdf = outputs['rdfs']
                
                # Calculate MSEs
                traj_mse = torch.mean((pred_traj - target_traj) ** 2).item()
                thermo_mse = torch.mean((pred_thermo - target_thermo) ** 2, dim=(1, 2)) # Mean over seq and props
                # Actually we want specific property MSEs
                
                # Thermo shape: [B, Seq, 4] -> T, P, Rho, PE
                thermo_diff = (pred_thermo - target_thermo) ** 2
                pe_mse = torch.mean(thermo_diff[:, :, 3]).item() # PE is index 3
                ke_mse = torch.mean(thermo_diff[:, :, 0]).item() # T is index 0 (proxy for KE)
                
                rdf_mse = torch.mean((pred_rdf - target_rdf) ** 2).item()
                
                metrics_list.append({
                    'epsilon': eps_val,
                    'Trajectory_MSE': traj_mse,
                    'Thermo_PE_MSE': pe_mse,
                    'Thermo_KE_MSE': ke_mse,
                    'RDF_MSE': rdf_mse
                })

    process_loader(train_loader, "Train")
    process_loader(val_loader, "Validation")
    
    # Aggregate by epsilon
    print("Aggregating metrics per epsilon...")
    df = pd.DataFrame(metrics_list)
    
    # Group by epsilon and take mean
    final_df = df.groupby('epsilon').mean().reset_index()
    final_df = final_df.sort_values('epsilon')
    
    # Save
    output_path = output_dir / "batch_metrics.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")
    print(final_df.head())

if __name__ == "__main__":
    generate_metrics()
