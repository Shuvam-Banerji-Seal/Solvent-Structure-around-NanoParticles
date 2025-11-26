#!/usr/bin/env python3
"""
Extract Features from MD Simulations
=====================================

This script extracts thermodynamic and structural properties from
epsilon 0.0-0.50 simulations to create training data for ML models.

Output:
-------
- data/training_features.csv: Features for ML training
- data/property_descriptions.json: Metadata about each property

Author: Shuvam Roy
Date: November 2025
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.signal import find_peaks
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
BASE_DIR = Path("/store/shuvam/learning_solvent_effects")
SOLVENT_EFFECTS_DIR = BASE_DIR / "solvent_effects"
OUTPUT_DIR = BASE_DIR / "ml_integration" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Epsilon values to process (training data)
EPSILON_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_thermodynamic_data(epsilon):
    """Load thermodynamic data for a given epsilon value."""
    if epsilon == 0.0:
        eps_dir = SOLVENT_EFFECTS_DIR / "epsilon_0.0"
    else:
        eps_dir = SOLVENT_EFFECTS_DIR / f"epsilon_{epsilon:.2f}"
    
    thermo_file = eps_dir / "production_detailed_thermo.dat"
    
    if not thermo_file.exists():
        print(f"  ⚠️  Warning: {thermo_file} not found")
        return None
    
    # Read thermodynamic data
    df = pd.read_csv(thermo_file, sep=r'\s+', comment='#',
                     names=['TimeStep', 'Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens'])
    
    return df


def load_rdf_data(epsilon, rdf_type='CO'):
    """Load RDF data for a given epsilon value."""
    if epsilon == 0.0:
        eps_dir = SOLVENT_EFFECTS_DIR / "epsilon_0.0"
    else:
        eps_dir = SOLVENT_EFFECTS_DIR / f"epsilon_{epsilon:.2f}"
    
    rdf_file = eps_dir / f"rdf_{rdf_type}.dat"
    
    if not rdf_file.exists():
        print(f"  ⚠️  Warning: {rdf_file} not found")
        return None
    
    # Read RDF data (skip header lines)
    data = []
    with open(rdf_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    r = float(parts[1])  # Distance
                    g_r = float(parts[2])  # g(r)
                    data.append([r, g_r])
                except ValueError:
                    continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=['r', 'g_r'])
    return df


def extract_rdf_features(rdf_df):
    """Extract features from RDF curve."""
    if rdf_df is None:
        return {}
    
    r = rdf_df['r'].values
    g_r = rdf_df['g_r'].values
    
    # Find peaks
    peaks, properties = find_peaks(g_r, prominence=0.1, distance=5)
    
    features = {}
    
    if len(peaks) > 0:
        # First peak
        features['peak1_position'] = r[peaks[0]]
        features['peak1_height'] = g_r[peaks[0]]
        
        # First minimum (coordination shell boundary)
        if peaks[0] > 0:
            first_min_idx = np.argmin(g_r[:peaks[0]]) if peaks[0] > 10 else 0
            if len(peaks) > 1:
                first_min_idx = peaks[0] + np.argmin(g_r[peaks[0]:peaks[1]])
            features['first_minimum'] = r[first_min_idx] if first_min_idx < len(r) else r[-1]
        else:
            features['first_minimum'] = 0.0
    else:
        features['peak1_position'] = 0.0
        features['peak1_height'] = 0.0
        features['first_minimum'] = 0.0
    
    if len(peaks) > 1:
        # Second peak
        features['peak2_position'] = r[peaks[1]]
        features['peak2_height'] = g_r[peaks[1]]
    else:
        features['peak2_position'] = 0.0
        features['peak2_height'] = 0.0
    
    # Coordination number (integrate up to first minimum)
    # n(r) = 4πρ ∫ g(r) r² dr
    # Approximate density of water oxygens
    rho_water = 0.033  # atoms/Å³ (approximate)
    
    if features['first_minimum'] > 0:
        mask = r <= features['first_minimum']
        if np.sum(mask) > 1:
            coord_num = 4 * np.pi * rho_water * np.trapz(g_r[mask] * r[mask]**2, r[mask])
            features['coordination_number'] = coord_num
        else:
            features['coordination_number'] = 0.0
    else:
        features['coordination_number'] = 0.0
    
    return features


def extract_features_for_epsilon(epsilon):
    """Extract all features for a single epsilon value."""
    print(f"  Processing epsilon = {epsilon:.2f}")
    
    features = {'epsilon': epsilon}
    
    # Thermodynamic properties
    thermo_df = load_thermodynamic_data(epsilon)
    if thermo_df is not None:
        features['temp_mean'] = thermo_df['Temp'].mean()
        features['temp_std'] = thermo_df['Temp'].std()
        features['press_mean'] = thermo_df['Press'].mean()
        features['press_std'] = thermo_df['Press'].std()
        features['density_mean'] = thermo_df['Dens'].mean()
        features['density_std'] = thermo_df['Dens'].std()
        features['pe_mean'] = thermo_df['PE'].mean()
        features['pe_std'] = thermo_df['PE'].std()
        features['vol_mean'] = thermo_df['Vol'].mean()
        features['vol_std'] = thermo_df['Vol'].std()
    else:
        print(f"    ❌ No thermodynamic data found")
        return None
    
    # RDF features (C-O)
    rdf_co = load_rdf_data(epsilon, 'CO')
    if rdf_co is not None:
        rdf_features = extract_rdf_features(rdf_co)
        for key, value in rdf_features.items():
            features[f'rdf_co_{key}'] = value
    else:
        print(f"    ⚠️  No C-O RDF data found")
    
    # RDF features (O-O) - water structure
    rdf_oo = load_rdf_data(epsilon, 'OO')
    if rdf_oo is not None:
        rdf_features = extract_rdf_features(rdf_oo)
        for key, value in rdf_features.items():
            features[f'rdf_oo_{key}'] = value
    else:
        print(f"    ⚠️  No O-O RDF data found")
    
    print(f"    ✅ Extracted {len(features)} features")
    return features


def main():
    """Main extraction workflow."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Feature Extraction from MD Simulations                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"📊 Epsilon values: {EPSILON_VALUES}")
    print()
    
    # Extract features for all epsilon values
    all_features = []
    for epsilon in EPSILON_VALUES:
        features = extract_features_for_epsilon(epsilon)
        if features is not None:
            all_features.append(features)
    
    print()
    print(f"✅ Successfully extracted features for {len(all_features)} epsilon values")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_file = OUTPUT_DIR / "training_features.csv"
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"💾 Saved features to: {output_file}")
    print(f"   Shape: {df.shape}")
    print()
    
    # Print summary
    print("📊 Feature Summary:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    print()
    
    # Save metadata
    metadata = {
        'description': 'Training features extracted from MD simulations',
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # Exclude epsilon
        'epsilon_range': [float(df['epsilon'].min()), float(df['epsilon'].max())],
        'features': {
            'thermodynamic': ['temp_mean', 'temp_std', 'press_mean', 'press_std', 
                             'density_mean', 'density_std', 'pe_mean', 'pe_std',
                             'vol_mean', 'vol_std'],
            'rdf_co': ['rdf_co_peak1_position', 'rdf_co_peak1_height', 
                      'rdf_co_peak2_position', 'rdf_co_peak2_height',
                      'rdf_co_first_minimum', 'rdf_co_coordination_number'],
            'rdf_oo': ['rdf_oo_peak1_position', 'rdf_oo_peak1_height',
                      'rdf_oo_peak2_position', 'rdf_oo_peak2_height',
                      'rdf_oo_first_minimum', 'rdf_oo_coordination_number']
        }
    }
    
    metadata_file = OUTPUT_DIR / "property_descriptions.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata to: {metadata_file}")
    print()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Feature Extraction Complete!                                 ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print("Next steps:")
    print("  1. Train GPR model: python scripts/02_train_gpr.py")
    print("  2. Train NN model: python scripts/03_train_nn.py")
    print("  3. Make predictions: python scripts/04_predict_new_epsilon.py")
    print()


if __name__ == "__main__":
    main()
