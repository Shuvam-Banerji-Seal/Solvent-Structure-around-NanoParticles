#!/usr/bin/env python3
"""
Advanced Generative Neural Network Architecture
================================================

Multi-task neural network that generates complete MD simulation outputs
from a single epsilon value.

Components:
1. Epsilon Encoder: epsilon → latent representation
2. Trajectory Decoder: latent → atomic coordinates
3. Thermodynamics Decoder: latent → T, P, ρ, E time series
4. RDF Decoder: latent → radial distribution functions

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EpsilonEncoder(nn.Module):
    """
    Encodes epsilon value into rich latent representation.
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, latent_dim),
        )
        
    def forward(self, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            epsilon: (batch_size, 1)
            
        Returns:
            latent: (batch_size, latent_dim)
        """
        return self.network(epsilon)


class TrajectoryDecoder(nn.Module):
    """
    Generates trajectory from latent representation.
    
    Strategy:
    - Generate mean coordinates for all atoms
    - Add learned per-atom variations
    - Output: (batch, n_atoms, 3)
    
    Note: For simplicity, we generate a single representative frame
    rather than full time series. Full temporal generation would require
    RNN/Transformer which is more complex.
    """
    
    def __init__(self, latent_dim: int = 512, n_atoms: int = 5541):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        
        # Generate mean atomic positions
        self.coord_generator = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(4096, n_atoms * 3),
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch_size, latent_dim)
            
        Returns:
            coordinates: (batch_size, n_atoms, 3)
        """
        batch_size = latent.shape[0]
        
        # Generate coordinates
        coords_flat = self.coord_generator(latent)
        coords = coords_flat.view(batch_size, self.n_atoms, 3)
        
        return coords


class ThermodynamicsDecoder(nn.Module):
    """
    Generates thermodynamic time series from latent representation.
    
    Uses LSTM to generate realistic time-correlated dynamics.
    """
    
    def __init__(self, latent_dim: int = 512, seq_len: int = 1000, n_properties: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_properties = n_properties  # T, P, ρ, E
        
        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, 256)
        
        # LSTM for temporal generation
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.output_layer = nn.Linear(256, n_properties)
        
    def forward(self, latent: torch.Tensor, seq_len: int = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (batch_size, latent_dim)
            seq_len: Optional sequence length (uses self.seq_len if None)
            
        Returns:
            dict with keys:
                - 'temperature': (batch_size, seq_len)
                - 'pressure': (batch_size, seq_len)
                - 'density': (batch_size, seq_len)
                - 'potential_energy': (batch_size, seq_len)
        """
        batch_size = latent.shape[0]
        if seq_len is None:
            seq_len = self.seq_len
        
        # Repeat latent for each time step
        latent_seq = latent.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(latent).unsqueeze(0).repeat(3, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Generate sequence
        lstm_out, _ = self.lstm(latent_seq, (h0, c0))
        
        # Project to properties
        properties = self.output_layer(lstm_out)  # (batch, seq_len, n_properties)
        
        return {
            'temperature': properties[:, :, 0],
            'pressure': properties[:, :, 1],
            'density': properties[:, :, 2],
            'potential_energy': properties[:, :, 3]
        }


class RDFDecoder(nn.Module):
    """
    Generates RDF curves from latent representation.
    """
    
    def __init__(self, latent_dim: int = 512, n_bins: int = 200):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Separate heads for each pair type
        self.cc_head = nn.Linear(256, n_bins)
        self.co_head = nn.Linear(256, n_bins)
        self.oo_head = nn.Linear(256, n_bins)
        
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (batch_size, latent_dim)
            
        Returns:
            dict with keys 'CC', 'CO', 'OO', each: (batch_size, n_bins)
        """
        features = self.shared(latent)
        
        return {
            'CC': F.softplus(self.cc_head(features)),  # Softplus: smooth, always positive, gradients flow
            'CO': F.softplus(self.co_head(features)),
            'OO': F.softplus(self.oo_head(features))
        }


class MDGenerativeModel(nn.Module):
    """
    Complete multi-task generative model.
    
    Input: epsilon value
    Output: Complete MD simulation data (trajectory, thermodynamics, RDFs)
    """
    
    def __init__(self,
                 latent_dim: int = 512,
                 n_atoms: int = 5541,
                 thermo_seq_len: int = 1000,
                 rdf_bins: int = 200):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = EpsilonEncoder(latent_dim=latent_dim)
        
        # Decoders
        self.trajectory_decoder = TrajectoryDecoder(
            latent_dim=latent_dim,
            n_atoms=n_atoms
        )
        
        self.thermo_decoder = ThermodynamicsDecoder(
            latent_dim=latent_dim,
            seq_len=thermo_seq_len,
            n_properties=4
        )
        
        self.rdf_decoder = RDFDecoder(
            latent_dim=latent_dim,
            n_bins=rdf_bins
        )
        
    def forward(self, epsilon: torch.Tensor, target: Dict = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            epsilon: (batch_size, 1)
            target: Optional dict with target data (used to infer sequence lengths)
            
        Returns:
            dict with keys:
                - 'trajectory': (batch_size, n_atoms, 3)
                - 'thermodynamics': dict of time series
                - 'rdfs': dict of RDF curves
        """
        # Encode epsilon
        latent = self.encoder(epsilon)
        
        # Infer thermodynamics sequence length from target if available
        thermo_seq_len = None
        if target and target.get('thermodynamics'):
            # Get sequence length from any thermodynamics key
            first_key = next(iter(target['thermodynamics']))
            thermo_seq_len = target['thermodynamics'][first_key].shape[1]
        
        # Decode to all outputs
        outputs = {
            'trajectory': self.trajectory_decoder(latent),
            'thermodynamics': self.thermo_decoder(latent, seq_len=thermo_seq_len),
            'rdfs': self.rdf_decoder(latent),
            'latent': latent  # For analysis
        }
        
        return outputs
    
    def generate(self, epsilon: float, device: str = 'cuda') -> Dict:
        """
        Generate simulation data for a single epsilon value.
        
        Args:
            epsilon: Epsilon value (scalar)
            device: Device to run on
            
        Returns:
            Dictionary with all generated outputs
        """
        self.eval()
        
        with torch.no_grad():
            eps_tensor = torch.tensor([[epsilon]], dtype=torch.float32, device=device)
            outputs = self.forward(eps_tensor)
            
            # Convert to numpy and remove batch dimension
            generated = {
                'epsilon': epsilon,
                'trajectory': outputs['trajectory'][0].cpu().numpy(),
                'thermodynamics': {
                    k: v[0].cpu().numpy() 
                    for k, v in outputs['thermodynamics'].items()
                },
                'rdfs': {
                    k: v[0].cpu().numpy()
                    for k, v in outputs['rdfs'].items()
                }
            }
            
        return generated


def test_model():
    """Test the model architecture."""
    print("Testing MDGenerativeModel...")
    
    # Create model
    model = MDGenerativeModel(
        latent_dim=512,
        n_atoms=5541,
        thermo_seq_len=1000,
        rdf_bins=200
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    epsilon = torch.randn(batch_size, 1)
    
    print(f"\nInput epsilon shape: {epsilon.shape}")
    
    outputs = model(epsilon)
    
    print(f"\nOutput shapes:")
    print(f"  Trajectory: {outputs['trajectory'].shape}")
    print(f"  Thermodynamics:")
    for key, val in outputs['thermodynamics'].items():
        print(f"    {key}: {val.shape}")
    print(f"  RDFs:")
    for key, val in outputs['rdfs'].items():
        print(f"    {key}: {val.shape}")
    
    # Test generation
    print(f"\nTesting generation for epsilon=0.55...")
    generated = model.generate(epsilon=0.55, device='cpu')
    
    print(f"Generated data:")
    print(f"  Trajectory: {generated['trajectory'].shape}")
    print(f"  Thermodynamics keys: {list(generated['thermodynamics'].keys())}")
    print(f"  RDFs keys: {list(generated['rdfs'].keys())}")
    
    print(f"\n✅ Model architecture test passed!")


if __name__ == "__main__":
    test_model()
