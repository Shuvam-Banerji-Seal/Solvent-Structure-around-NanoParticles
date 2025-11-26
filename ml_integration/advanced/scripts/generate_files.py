#!/usr/bin/env python3
"""
File Generator for MD Simulation Outputs
=========================================

Generate actual LAMMPS files (.lammpstrj, .dat) from trained model predictions.

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
from model import MDGenerativeModel


import json

class MDFileGenerator:
    """
    Generate LAMMPS-format files from model predictions.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', stats_path: str = '../logs/normalization_stats.json'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
        """
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = MDGenerativeModel(
            latent_dim=512,
            n_atoms=5541,
            thermo_seq_len=1000,
            rdf_bins=200
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
        
        Args:
            epsilon: Epsilon value
            output_dir: Output directory
            n_frames: Number of frames to generate (currently 1)
        """
        output_file = output_dir / "production.lammpstrj"
        
        print(f"  Generating trajectory file: {output_file.name}")
        
        # Generate coordinates
        generated = self.model.generate(epsilon, device=self.device)
        coords = generated['trajectory']  # (n_atoms, 3)
        
        # Denormalize coordinates
        # coords is (atoms, 3)
        coords = coords * self.traj_std.reshape(1, 3) + self.traj_mean.reshape(1, 3)
        
        # Write LAMMPS trajectory format
        with open(output_file, 'w') as f:
            for frame_idx in range(n_frames):
                timestep = 601000 + frame_idx * 1000
                
                # Header
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{timestep}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{self.n_atoms}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                half_box = self.box_size / 2
                f.write(f"{-half_box} {half_box}\n")
                f.write(f"{-half_box} {half_box}\n")
                f.write(f"{-half_box} {half_box}\n")
                f.write("ITEM: ATOMS id type xu yu zu\n")
                
                # Atom data
                for atom_id in range(1, self.n_atoms + 1):
                    idx = atom_id - 1
                    
                    # Determine atom type
                    if atom_id <= self.n_c60:
                        atom_type = 1  # Carbon
                    elif (atom_id - self.n_c60) % 3 == 1:
                        atom_type = 2  # Oxygen
                    else:
                        atom_type = 3  # Hydrogen
                    
                    x, y, z = coords[idx]
                    
                    f.write(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")
        
        print(f"    ✅ Saved: {output_file}")
    
    def generate_thermodynamics_file(self, epsilon: float, output_dir: Path):
        """
        Generate production_detailed_thermo.dat file.
        """
        output_file = output_dir / "production_detailed_thermo.dat"
        
        print(f"  Generating thermodynamics file: {output_file.name}")
        
        # Generate thermodynamics
        generated = self.model.generate(epsilon, device=self.device)
        thermo = generated['thermodynamics']
        
        # Write data
        with open(output_file, 'w') as f:
            # Match LAMMPS header format exactly
            f.write("# Time-averaged data for fix thermo_detailed\n")
            f.write("# TimeStep v_temp v_press v_pe v_ke v_vol v_dens\n")
            
            n_steps = len(thermo['temperature'])
            for i in range(n_steps):
                timestep = 600100 + i * 100  # Start at 600100, every 0.1 ps (100 steps)
                
                # Denormalize thermodynamics
                temp = thermo['temperature'][i] * self.stats['thermodynamics']['temperature']['std'] + self.stats['thermodynamics']['temperature']['mean']
                press = thermo['pressure'][i] * self.stats['thermodynamics']['pressure']['std'] + self.stats['thermodynamics']['pressure']['mean']
                pe = thermo['potential_energy'][i] * self.stats['thermodynamics']['potential_energy']['std'] + self.stats['thermodynamics']['potential_energy']['mean']
                dens = thermo['density'][i] * self.stats['thermodynamics']['density']['std'] + self.stats['thermodynamics']['density']['mean']
                
                # KE from temperature: KE = (3/2) * N * kB * T
                # For N atoms: KE ≈ 1.5 * kB * T * N_atoms
                # kB in kcal/mol/K ≈ 0.001987, but LAMMPS uses 3*N*kB*T/2
                ke = 0.002019 * temp * self.n_atoms  # Empirical factor from data (approx N*kB*T)
                
                # Volume from density: V = mass / density
                # Total mass = n_water * 18.01528 amu
                n_water = (self.n_atoms - self.n_c60) / 3  # 3 atoms per water
                mass_amu = n_water * 18.01528 + self.n_c60 * 12.011  # Water + C60
                vol = mass_amu / (dens * 0.6022140857)  # Convert to Angstrom^3
                
                f.write(f"{timestep} {temp:.6f} {press:.6f} {pe:.6f} {ke:.6f} {vol:.6f} {dens:.6f}\n")
        
        print(f"    ✅ Saved: {output_file}")
    
    def generate_rdf_files(self, epsilon: float, output_dir: Path):
        """
        Generate rdf_CC.dat, rdf_CO.dat, rdf_OO.dat files.
        """
        print(f"  Generating RDF files...")
        
        # Generate RDFs
        generated = self.model.generate(epsilon, device=self.device)
        rdfs = generated['rdfs']
        
        # Generate r values (0.5 to 20.0 Å)
        r_values = np.linspace(0.5, 20.0, 200)
        
        for pair in ['CC', 'CO', 'OO']:
            output_file = output_dir / f"rdf_{pair}.dat"
            
            g_r = rdfs[pair]
            
            with open(output_file, 'w') as f:
                f.write(f"# Radial distribution function: {pair}\n")
                f.write("# Row Dist g(r)\n")
                
                for i, (r, g) in enumerate(zip(r_values, g_r)):
                    f.write(f"{i+1} {r:.6f} {g:.6f}\n")
            
            print(f"    ✅ Saved: {output_file.name}")
    
    def generate_all_files(self, epsilon: float, output_dir: str):
        """
        Generate all simulation files for a given epsilon.
        
        Args:
            epsilon: Epsilon value to generate
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating files for epsilon = {epsilon:.2f}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        # Generate each file type
        self.generate_trajectory_file(epsilon, output_dir)
        self.generate_thermodynamics_file(epsilon, output_dir)
        self.generate_rdf_files(epsilon, output_dir)
        
        print(f"\n{'='*60}")
        print(f"✅ All files generated successfully!")
        print(f"{'='*60}\n")
        
        # Print file list
        print("Generated files:")
        for file in sorted(output_dir.glob("*")):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  - {file.name} ({size_mb:.2f} MB)")


def main():
    """Test file generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MD simulation files from trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--epsilon', type=float, required=True,
                       help='Epsilon value to generate')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MDFileGenerator(args.model, device=args.device)
    
    # Generate files
    generator.generate_all_files(args.epsilon, args.output)


if __name__ == "__main__":
    main()
