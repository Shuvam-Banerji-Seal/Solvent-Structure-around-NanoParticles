#!/usr/bin/env python3
"""
Physics-Informed Loss Functions
================================

Implements physical constraints to guide the model toward
physically plausible predictions.

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    Combines multiple physics-based constraints.
    """
    
    def __init__(self):
        super().__init__()
        
    def energy_conservation_loss(self, thermo: dict) -> torch.Tensor:
        """
        Penalize drift in total energy (PE + KE should be ~constant).
        
        Args:
            thermo: Dictionary with 'potential_energy' and keys
            
        Returns:
            Energy drift penalty
        """
        if 'potential_energy' not in thermo:
            return torch.tensor(0.0, device=list(thermo.values())[0].device)
            
        pe = thermo['potential_energy']
        
        # Total energy should be relatively constant over time
        # Penalize standard deviation
        energy_std = torch.std(pe, dim=1).mean()
        
        return energy_std / 1000.0  # Scale down
    
    def rdf_normalization_loss(self, rdfs: dict, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Enforce RDF integral constraints.
        
        For C-C pair, the first shell coordination number should be
        physically reasonable (typically 2-6 for C60 cage).
        
        Args:
            rdfs: Dictionary of RDF curves
            epsilon: Epsilon values
            
        Returns:
            RDF constraint penalty
        """
        if 'CC' not in rdfs:
            return torch.tensor(0.0, device=epsilon.device)
            
        # Get C-C RDF
        g_r = rdfs['CC']  # (batch, n_bins)
        
        # Simple check: RDF should be positive and bounded
        # g(r) typically ranges from 0 to ~20 for first peak
        max_g = torch.max(g_r, dim=1)[0]  # (batch,)
        
        # Penalize if peak is too high (>30) or too low (<1)
        high_penalty = torch.relu(max_g - 30.0).mean()
        low_penalty = torch.relu(1.0 - max_g).mean()
        
        return (high_penalty + low_penalty) * 0.1
    
    def thermodynamic_bounds_loss(self, thermo: dict) -> torch.Tensor:
        """
        Ensure thermodynamic properties stay within physical bounds.
        
        Args:
            thermo: Dictionary of thermodynamic properties
            
        Returns:
            Bounds penalty
        """
        device = list(thermo.values())[0].device
        penalty = torch.tensor(0.0, device=device)
        
        # Temperature should be ~300K (thermostatted)
        if 'temperature' in thermo:
            temp = thermo['temperature']
            # Penalize if temperature drifts too far from 300K
            temp_penalty = torch.relu(torch.abs(temp.mean(dim=1) - 300.0) - 10.0).mean()
            penalty += temp_penalty * 0.01
        
        # Density should be positive and reasonable (~0.5-2.0 g/cm³)
        if 'density' in thermo:
            dens = thermo['density']
            # Penalize negative or extreme densities
            neg_penalty = torch.relu(-dens).mean()
            high_penalty = torch.relu(dens - 3.0).mean()
            penalty += (neg_penalty + high_penalty) * 0.1
            
        return penalty
    
    def forward(self, pred: dict, epsilon: torch.Tensor,
                alpha_energy: float = 0.1,
                alpha_rdf: float = 0.2,
                alpha_bounds: float = 0.1) -> torch.Tensor:
        """
        Compute total physics-informed loss.
        
        Args:
            pred: Model predictions
            epsilon: Epsilon values
            alpha_*: Loss weights
            
        Returns:
            Total physics loss
        """
        losses = {}
        
        # Energy conservation
        if pred.get('thermodynamics'):
            losses['energy'] = self.energy_conservation_loss(pred['thermodynamics'])
            losses['bounds'] = self.thermodynamic_bounds_loss(pred['thermodynamics'])
        else:
            device = epsilon.device
            losses['energy'] = torch.tensor(0.0, device=device)
            losses['bounds'] = torch.tensor(0.0, device=device)
        
        # RDF constraints
        if pred.get('rdfs'):
            losses['rdf_norm'] = self.rdf_normalization_loss(pred['rdfs'], epsilon)
        else:
            losses['rdf_norm'] = torch.tensor(0.0, device=epsilon.device)
        
        total = (
            alpha_energy * losses['energy'] +
            alpha_rdf * losses['rdf_norm'] +
            alpha_bounds * losses['bounds']
        )
        
        return total, losses
