#!/usr/bin/env python3
"""
IMPROVED Generative Neural Network Architecture
=================================================

Enhanced with:
- Dropout for regularization
- Spectral normalization for Lipschitz constraint
- Deeper epsilon conditioning
- VAE-style latent space (optional)
- Better initialization

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict, Tuple, Optional


class ImprovedEpsilonEncoder(nn.Module):
    """
    Enhanced encoder with spectral normalization and optional VAE.
    """
    
    def __init__(self, latent_dim: int = 512, use_vae: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        
        # Spectral normalization prevents unbounded gradients
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(1, 128)),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            
            spectral_norm(nn.Linear(128, 256)),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            
            spectral_norm(nn.Linear(256, 512)),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
        )
        
        if use_vae:
            # Separate heads for mean and log-variance
            self.fc_mu = spectral_norm(nn.Linear(512, latent_dim))
            self.fc_logvar = spectral_norm(nn.Linear(512, latent_dim))
        else:
            self.fc_out = spectral_norm(nn.Linear(512, latent_dim))
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, epsilon: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            epsilon: (batch_size, 1)
            
        Returns:
            latent: (batch_size, latent_dim)
            mu: (batch_size, latent_dim) if VAE, else None
            logvar: (batch_size, latent_dim) if VAE, else None
        """
        features = self.network(epsilon)
        
        if self.use_vae:
            mu = self.fc_mu(features)
            logvar = self.fc_logvar(features)
            latent = self.reparameterize(mu, logvar)
            return latent, mu, logvar
        else:
            latent = self.fc_out(features)
            return latent, None, None


class ImprovedTrajectoryDecoder(nn.Module):
    """
    Enhanced with dropout and deeper epsilon conditioning.
    """
    
    def __init__(self, latent_dim: int = 512, n_atoms: int = 5541, epsilon_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.epsilon_dim = epsilon_dim
        
        # Epsilon embedding for injection
        self.epsilon_embed = nn.Linear(1, epsilon_dim)
        
        # Deeper conditioning: inject epsilon at multiple layers
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim + epsilon_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(2048 + epsilon_dim, 4096),  # Re-inject epsilon
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        
        self.output = nn.Linear(4096, n_atoms * 3)
        
    def forward(self, latent: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch_size, latent_dim)
            epsilon: (batch_size, 1)
            
        Returns:
            coordinates: (batch_size, n_atoms, 3)
        """
        batch_size = latent.shape[0]
        
        # Embed epsilon
        eps_emb = torch.relu(self.epsilon_embed(epsilon))
        
        # Layer 1 with epsilon
        x = torch.cat([latent, eps_emb], dim=-1)
        x = self.layer1(x)
        
        # Layer 2 with re-injected epsilon
        x = torch.cat([x, eps_emb], dim=-1)
        x = self.layer2(x)
        
        # Output
        coords_flat = self.output(x)
        coords = coords_flat.view(batch_size, self.n_atoms, 3)
        
        return coords


class ImprovedThermodynamicsDecoder(nn.Module):
    """
    Enhanced LSTM with dropout and epsilon conditioning.
    """
    
    def __init__(self, latent_dim: int = 512, seq_len: int = 1000, n_properties: int = 4, epsilon_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_properties = n_properties
        self.epsilon_dim = epsilon_dim
        
        # Epsilon embedding
        self.epsilon_embed = nn.Linear(1, epsilon_dim)
        
        # Project latent + epsilon to hidden state
        self.latent_to_hidden = nn.Linear(latent_dim + epsilon_dim, 256)
        
        # LSTM with higher dropout
        self.lstm = nn.LSTM(
            input_size=latent_dim + epsilon_dim,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.15  # Increased from 0.2
        )
        
        # Output layer with dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(256, n_properties)
        )
        
    def forward(self, latent: torch.Tensor, epsilon: torch.Tensor, seq_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (batch_size, latent_dim)
            epsilon: (batch_size, 1)
            seq_len: Optional sequence length override
            
        Returns:
            dict with thermodynamic properties
        """
        batch_size = latent.shape[0]
        if seq_len is None:
            seq_len = self.seq_len
        
        # Embed epsilon
        eps_emb = torch.relu(self.epsilon_embed(epsilon))
        
        # Concatenate latent + epsilon
        latent_eps = torch.cat([latent, eps_emb], dim=-1)
        
        # Repeat for sequence
        latent_seq = latent_eps.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Initialize hidden state
        h0 = self.latent_to_hidden(latent_eps).unsqueeze(0).repeat(3, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Generate sequence
        lstm_out, _ = self.lstm(latent_seq, (h0, c0))
        
        # Project to properties
        properties = self.output_layer(lstm_out)
        
        return {
            'temperature': properties[:, :, 0],
            'pressure': properties[:, :, 1],
            'density': properties[:, :, 2],
            'potential_energy': properties[:, :, 3]
        }


class ImprovedRDFDecoder(nn.Module):
    """
    Enhanced with dropout and epsilon conditioning.
    """
    
    def __init__(self, latent_dim: int = 512, n_bins: int = 200, epsilon_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        self.epsilon_dim = epsilon_dim
        
        # Epsilon embedding
        self.epsilon_embed = nn.Linear(1, epsilon_dim)
        
        # Shared trunk with epsilon conditioning
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + epsilon_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Separate heads for each pair type
        self.cc_head = nn.Linear(256, n_bins)
        self.co_head = nn.Linear(256, n_bins)
        self.oo_head = nn.Linear(256, n_bins)
        
    def forward(self, latent: torch.Tensor, epsilon: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (batch_size, latent_dim)
            epsilon: (batch_size, 1)
            
        Returns:
            dict with RDF curves
        """
        # Embed epsilon
        eps_emb = torch.relu(self.epsilon_embed(epsilon))
        
        # Concatenate latent + epsilon
        latent_eps = torch.cat([latent, eps_emb], dim=-1)
        
        # Shared features
        features = self.shared(latent_eps)
        
        return {
            'CC': F.softplus(self.cc_head(features)),
            'CO': F.softplus(self.co_head(features)),
            'OO': F.softplus(self.oo_head(features))
        }


class ImprovedMDGenerativeModel(nn.Module):
    """
    Complete improved generative model.
    """
    
    def __init__(self,
                 latent_dim: int = 512,
                 n_atoms: int = 5541,
                 thermo_seq_len: int = 1000,
                 rdf_bins: int = 200,
                 use_vae: bool = False):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.use_vae = use_vae
        
        # Encoder
        self.encoder = ImprovedEpsilonEncoder(latent_dim, use_vae=use_vae)
        
        # Decoders
        self.trajectory_decoder = ImprovedTrajectoryDecoder(latent_dim, n_atoms)
        self.thermo_decoder = ImprovedThermodynamicsDecoder(latent_dim, thermo_seq_len)
        self.rdf_decoder = ImprovedRDFDecoder(latent_dim, rdf_bins)
        
    def forward(self, epsilon: torch.Tensor, target: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            epsilon: (batch_size, 1)
            target: Optional target data for seq_len inference
            
        Returns:
            dict with all outputs + latent + VAE params if applicable
        """
        # Encode
        latent, mu, logvar = self.encoder(epsilon)
        
        # Infer sequence length from target if provided
        thermo_seq_len = None
        if target is not None and target.get('thermodynamics') is not None:
            # Get sequence length from first thermodynamic property
            first_key = list(target['thermodynamics'].keys())[0]
            thermo_seq_len = target['thermodynamics'][first_key].shape[1]
        
        # Decode with epsilon conditioning
        outputs = {
            'trajectory': self.trajectory_decoder(latent, epsilon),
            'thermodynamics': self.thermo_decoder(latent, epsilon, thermo_seq_len),
            'rdfs': self.rdf_decoder(latent, epsilon),
            'latent': latent
        }
        
        if self.use_vae:
            outputs['mu'] = mu
            outputs['logvar'] = logvar
        
        return outputs
