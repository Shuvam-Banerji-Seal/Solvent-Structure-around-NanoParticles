#!/usr/bin/env python3
"""
Generate MD files using the VAE trained model.
==============================================

Modified to use ImprovedMDGenerativeModel with VAE enabled.

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from model_improved import ImprovedMDGenerativeModel

class MDFileGeneratorVAE:
    def __init__(self, model_path: str, stats_path: str = "../logs_vae/normalization_stats.json", 
                 device: str = 'cuda'):
        self.device = device
        self.stats_path = stats_path
        
        # Load model
        print(f"Loading VAE model from {model_path}...")
        self.model = ImprovedMDGenerativeModel(
            latent_dim=512,
            n_atoms=5541,
            thermo_seq_len=1000,
            rdf_bins=200,
            use_vae=True # ENABLE VAE
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("✅ Model loaded successfully")
        
        # Load normalization stats
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
            
        # Convert trajectory stats to numpy
        self.traj_mean = np.array(self.stats['trajectory']['mean'])
        self.traj_std = np.array(self.stats['trajectory']['std'])
        
        # System constants
        self.n_atoms = 5541
        self.n_c60 = 180
        self.box_size = 38.2  # Approximate
        
    def generate_trajectory_file(self, epsilon: float, output_dir: Path,
                                 n_frames: int = 1):
        """
        Generate production.lammpstrj file.
        """
        output_path = output_dir / "production.lammpstrj"
        
        # Prepare input
        eps_tensor = torch.tensor([[epsilon]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(eps_tensor)
            coords_norm = outputs['trajectory'].cpu().numpy()  # (1, n_atoms, 3)
            
        # Denormalize
        coords = coords_norm * self.traj_std + self.traj_mean
        
        # Write LAMMPS trajectory format
        with open(output_path, 'w') as f:
            for frame in range(n_frames):
                # Header
                f.write(f"ITEM: TIMESTEP\n{frame*1000}\n")
                f.write("ITEM: NUMBER OF ATOMS\n5541\n")
                f.write(f"ITEM: BOX BOUNDS pp pp pp\n0.0 {self.box_size}\n0.0 {self.box_size}\n0.0 {self.box_size}\n")
                f.write("ITEM: ATOMS id type x y z\n")
                
                # Atoms
                current_coords = coords[0] # Single frame generation for now
                for i in range(self.n_atoms):
                    f.write(f"{i+1} 1 {current_coords[i,0]:.3f} {current_coords[i,1]:.3f} {current_coords[i,2]:.3f}\n")
                    
        print(f"    ✅ Saved: {output_path}")

    def generate_thermo_file(self, epsilon: float, output_dir: Path):
        """
        Generate production_detailed_thermo.dat file.
        """
        output_path = output_dir / "production_detailed_thermo.dat"
        
        eps_tensor = torch.tensor([[epsilon]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(eps_tensor)
            thermo_pred = outputs['thermodynamics']
            
        # Denormalize and extract
        data = {}
        for key in ['temperature', 'pressure', 'density', 'potential_energy']:
            val_norm = thermo_pred[key].cpu().numpy()[0] # (seq_len,)
            mean = self.stats['thermodynamics'][key]['mean']
            std = self.stats['thermodynamics'][key]['std']
            data[key] = val_norm * std + mean
            
        seq_len = len(data['temperature'])
        
        with open(output_path, 'w') as f:
            f.write("# TimeStep Temp Press PotEng KinEng Volume Density\n")
            for i in range(seq_len):
                step = i * 100
                T = data['temperature'][i]
                P = data['pressure'][i]
                PE = data['potential_energy'][i]
                Dens = data['density'][i]
                
                # Dummy/Derived values for missing cols
                KE = 3350.0 # Approx mean
                Vol = 54000.0 # Approx mean
                
                f.write(f"{step} {T:.2f} {P:.2f} {PE:.2f} {KE:.2f} {Vol:.2f} {Dens:.4f}\n")
                
        print(f"    ✅ Saved: {output_path}")

    def generate_rdf_files(self, epsilon: float, output_dir: Path):
        """
        Generate rdf_*.dat files.
        """
        eps_tensor = torch.tensor([[epsilon]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(eps_tensor)
            rdfs = outputs['rdfs']
            
        # RDF bins (0 to 10 Angstroms approx)
        bins = np.linspace(0, 10, 200)
        
        for pair in ['CC', 'CO', 'OO']:
            if pair in rdfs:
                g_r = rdfs[pair].cpu().numpy()[0]
                
                output_path = output_dir / f"rdf_{pair}.dat"
                with open(output_path, 'w') as f:
                    f.write(f"# Radial distribution function: {pair}\n")
                    f.write("# Row Dist g(r)\n")
                    for i in range(len(bins)):
                        f.write(f"{i+1} {bins[i]:.6f} {g_r[i]:.6f}\n")
                print(f"    ✅ Saved: rdf_{pair}.dat")

def main():
    parser = argparse.ArgumentParser(description="Generate MD files from VAE model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n============================================================")
    print(f"Generating files for epsilon = {args.epsilon:.2f} (VAE MODEL)")
    print(f"Output directory: {output_dir}")
    print(f"============================================================\n")
    
    try:
        generator = MDFileGeneratorVAE(args.model, device=args.device)
        
        print("  Generating trajectory file: production.lammpstrj")
        generator.generate_trajectory_file(args.epsilon, output_dir)
        
        print("  Generating thermodynamics file: production_detailed_thermo.dat")
        generator.generate_thermo_file(args.epsilon, output_dir)
        
        print("  Generating RDF files...")
        generator.generate_rdf_files(args.epsilon, output_dir)
        
        print("\n============================================================")
        print("✅ All files generated successfully!")
        print("============================================================\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
