#!/usr/bin/env python3
"""
Comprehensive Data Loader for MD Simulations
=============================================

Loads ALL data from MD simulations:
- Trajectories (.lammpstrj): 4001 frames × 5541 atoms × 3 coords
- Thermodynamics (.dat): Temperature, Pressure, Density, Energy time series
- RDFs (.dat): Radial distribution functions (C-C, C-O, O-O)
- MSD (.dat): Mean squared displacement

Author: Shuvam Banerji Seal
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import MDAnalysis as mda
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MDDataLoader:
    """
    Comprehensive loader for all MD simulation data.
    """
    
    def __init__(self, base_dir: str = "/store/shuvam/learning_solvent_effects"):
        self.base_dir = Path(base_dir)
        self.solvent_effects_dir = self.base_dir / "solvent_effects"
        
        # System constants
        self.n_atoms = 5541
        self.n_c60_atoms = 180  # 3 C60 molecules × 60 atoms
        self.n_water_atoms = 5361  # ~1787 water molecules × 3 atoms
        self.box_size = 38.2  # Approximate box size in Angstroms
        
    def get_epsilon_dir(self, epsilon: float) -> Path:
        """Get directory path for a given epsilon value."""
        if epsilon == 0.0:
            return self.solvent_effects_dir / "epsilon_0.0"
        else:
            return self.solvent_effects_dir / f"epsilon_{epsilon:.2f}"
    
    def load_trajectory(self, epsilon: float, max_frames: Optional[int] = None,
                       stride: int = 1) -> Dict[str, np.ndarray]:
        """
        Load trajectory data from .lammpstrj file.
        
        Args:
            epsilon: Epsilon value
            max_frames: Maximum number of frames to load (None = all)
            stride: Load every nth frame
            
        Returns:
            dict with keys:
                - 'coordinates': (n_frames, n_atoms, 3) array
                - 'atom_types': (n_atoms,) array
                - 'atom_ids': (n_atoms,) array
                - 'timesteps': (n_frames,) array
                - 'box_bounds': (n_frames, 3, 2) array
        """
        eps_dir = self.get_epsilon_dir(epsilon)
        traj_file = eps_dir / "production.lammpstrj"
        
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_file}")
        
        print(f"  Loading trajectory for epsilon={epsilon:.2f}...")
        
        # Use MDAnalysis for efficient loading
        try:
            u = mda.Universe(str(traj_file), format='LAMMPSDUMP')
            
            n_frames = len(u.trajectory)
            if max_frames is not None:
                n_frames = min(n_frames, max_frames)
            
            # Pre-allocate arrays
            frames_to_load = range(0, n_frames, stride)
            actual_frames = len(frames_to_load)
            
            coordinates = np.zeros((actual_frames, self.n_atoms, 3), dtype=np.float32)
            timesteps = np.zeros(actual_frames, dtype=np.int64)
            box_bounds = np.zeros((actual_frames, 3, 2), dtype=np.float32)
            
            # Load frames
            for i, frame_idx in enumerate(tqdm(frames_to_load, desc="Loading frames")):
                ts = u.trajectory[frame_idx]
                coordinates[i] = ts.positions
                timesteps[i] = ts.frame
                box_bounds[i] = ts.dimensions[:3].reshape(-1, 1) * np.array([[-0.5, 0.5]])
            
            # Get atom info from first frame
            atom_types = u.atoms.types
            atom_ids = u.atoms.ids
            
            return {
                'coordinates': coordinates,
                'atom_types': atom_types,
                'atom_ids': atom_ids,
                'timesteps': timesteps,
                'box_bounds': box_bounds,
                'n_frames': actual_frames,
                'epsilon': epsilon
            }
            
        except Exception as e:
            print(f"    Error with MDAnalysis, falling back to manual parsing: {e}")
            return self._load_trajectory_manual(traj_file, max_frames, stride)
    
    def _load_trajectory_manual(self, traj_file: Path, max_frames: Optional[int],
                                stride: int) -> Dict[str, np.ndarray]:
        """Manual parsing of LAMMPS trajectory (fallback)."""
        coordinates_list = []
        timesteps_list = []
        box_bounds_list = []
        atom_types = None
        atom_ids = None
        
        with open(traj_file, 'r') as f:
            frame_count = 0
            while True:
                # Read ITEM: TIMESTEP
                line = f.readline()
                if not line:
                    break
                if 'ITEM: TIMESTEP' not in line:
                    continue
                
                timestep = int(f.readline().strip())
                
                # Skip frames based on stride
                if frame_count % stride != 0:
                    # Skip this frame
                    for _ in range(5 + self.n_atoms):
                        f.readline()
                    frame_count += 1
                    continue
                
                # ITEM: NUMBER OF ATOMS
                f.readline()
                n_atoms = int(f.readline().strip())
                
                # ITEM: BOX BOUNDS
                f.readline()
                box = []
                for _ in range(3):
                    bounds = list(map(float, f.readline().split()[:2]))
                    box.append(bounds)
                box_bounds_list.append(box)
                
                # ITEM: ATOMS
                f.readline()
                coords = np.zeros((n_atoms, 3), dtype=np.float32)
                types = np.zeros(n_atoms, dtype=np.int32)
                ids = np.zeros(n_atoms, dtype=np.int32)
                
                for i in range(n_atoms):
                    parts = f.readline().split()
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    
                    # Store in order of atom_id
                    idx = atom_id - 1
                    coords[idx] = [x, y, z]
                    types[idx] = atom_type
                    ids[idx] = atom_id
                
                coordinates_list.append(coords)
                timesteps_list.append(timestep)
                
                if atom_types is None:
                    atom_types = types
                    atom_ids = ids
                
                frame_count += 1
                if max_frames is not None and len(coordinates_list) >= max_frames:
                    break
        
        return {
            'coordinates': np.array(coordinates_list, dtype=np.float32),
            'atom_types': atom_types,
            'atom_ids': atom_ids,
            'timesteps': np.array(timesteps_list),
            'box_bounds': np.array(box_bounds_list, dtype=np.float32),
            'n_frames': len(coordinates_list),
            'epsilon': None
        }
    
    def load_thermodynamics(self, epsilon: float, max_steps: int = 1000) -> Dict[str, np.ndarray]:
        """
        Load thermodynamic data.
        
        Args:
            max_steps: Maximum number of steps to return (subsampled if needed)
        
        Returns:
            dict with keys:
                - 'timestep': (max_steps,)
                - 'temperature': (max_steps,)
                - 'pressure': (max_steps,)
                - 'density': (max_steps,)
                - 'potential_energy': (max_steps,)
                - 'kinetic_energy': (max_steps,)
                - 'volume': (max_steps,)
        """
        eps_dir = self.get_epsilon_dir(epsilon)
        thermo_file = eps_dir / "production_detailed_thermo.dat"
        
        if not thermo_file.exists():
            print(f"    Warning: {thermo_file.name} not found, trying alternative...")
            thermo_file = eps_dir / "production_thermo.dat"
        
        if not thermo_file.exists():
            raise FileNotFoundError(f"Thermodynamic data not found for epsilon={epsilon}")
        
        print(f"  Loading thermodynamics for epsilon={epsilon:.2f}...")
        
        # Read thermodynamic data
        df = pd.read_csv(thermo_file, sep=r'\s+', comment='#',
                        names=['TimeStep', 'Temp', 'Press', 'PE', 'KE', 'Vol', 'Dens'])
        
        # Subsample if needed
        if len(df) > max_steps:
            indices = np.linspace(0, len(df)-1, max_steps, dtype=int)
            df = df.iloc[indices]
        
        return {
            'timestep': df['TimeStep'].values.astype(np.float32),
            'temperature': df['Temp'].values.astype(np.float32),
            'pressure': df['Press'].values.astype(np.float32),
            'density': df['Dens'].values.astype(np.float32),
            'potential_energy': df['PE'].values.astype(np.float32),
            'kinetic_energy': df['KE'].values.astype(np.float32),
            'volume': df['Vol'].values.astype(np.float32),
            'epsilon': epsilon
        }
    
    def load_rdf(self, epsilon: float, max_bins: int = 200) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load RDF data for all pairs.
        
        Args:
            max_bins: Maximum number of bins to return (subsampled if needed)
        
        Returns:
            dict with keys 'CC', 'CO', 'OO', each containing:
                - 'r': (max_bins,) distances
                - 'g_r': (max_bins,) g(r) values
        """
        eps_dir = self.get_epsilon_dir(epsilon)
        
        print(f"  Loading RDFs for epsilon={epsilon:.2f}...")
        
        rdfs = {}
        
        for pair in ['CC', 'CO', 'OO']:
            rdf_file = eps_dir / f"rdf_{pair}.dat"
            
            if not rdf_file.exists():
                print(f"    Warning: {rdf_file.name} not found")
                continue
            
            # Parse RDF file
            data = []
            with open(rdf_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            r = float(parts[1])
                            g_r = float(parts[2])
                            data.append([r, g_r])
                        except ValueError:
                            continue
            
            if data:
                data = np.array(data, dtype=np.float32)
                
                # Subsample if needed
                if len(data) > max_bins:
                    indices = np.linspace(0, len(data)-1, max_bins, dtype=int)
                    data = data[indices]
                
                rdfs[pair] = {
                    'r': data[:, 0],
                    'g_r': data[:, 1]
                }
        
        return rdfs
    
    def load_all_data(self, epsilon: float, traj_stride: int = 40,
                     max_traj_frames: Optional[int] = 100) -> Dict:
        """
        Load ALL data for a given epsilon value.
        
        Args:
            epsilon: Epsilon value to load
            traj_stride: Load every nth trajectory frame
            max_traj_frames: Maximum trajectory frames to load
            
        Returns:
            Comprehensive dictionary with all simulation data
        """
        print(f"\n{'='*60}")
        print(f"Loading all data for epsilon = {epsilon:.2f}")
        print(f"{'='*60}")
        
        data = {'epsilon': epsilon}
        
        # Load trajectory
        try:
            data['trajectory'] = self.load_trajectory(
                epsilon, 
                max_frames=max_traj_frames,
                stride=traj_stride
            )
        except Exception as e:
            print(f"  ❌ Error loading trajectory: {e}")
            data['trajectory'] = None
        
        # Load thermodynamics
        try:
            data['thermodynamics'] = self.load_thermodynamics(epsilon)
        except Exception as e:
            print(f"  ❌ Error loading thermodynamics: {e}")
            data['thermodynamics'] = None
        
        # Load RDFs
        try:
            data['rdfs'] = self.load_rdf(epsilon)
        except Exception as e:
            print(f"  ❌ Error loading RDFs: {e}")
            data['rdfs'] = None
        
        print(f"{'='*60}\n")
        
        return data


def test_data_loader():
    """Test the data loader."""
    loader = MDDataLoader()
    
    # Test with epsilon = 0.05
    data = loader.load_all_data(epsilon=0.05, max_traj_frames=10)
    
    print("\n📊 Data Summary:")
    print(f"Epsilon: {data['epsilon']}")
    
    if data['trajectory']:
        traj = data['trajectory']
        print(f"\nTrajectory:")
        print(f"  Frames: {traj['n_frames']}")
        print(f"  Atoms: {traj['coordinates'].shape[1]}")
        print(f"  Coordinates shape: {traj['coordinates'].shape}")
        print(f"  Memory: {traj['coordinates'].nbytes / 1024**2:.2f} MB")
    
    if data['thermodynamics']:
        thermo = data['thermodynamics']
        print(f"\nThermodynamics:")
        print(f"  Time steps: {len(thermo['timestep'])}")
        print(f"  Temperature: {thermo['temperature'].mean():.2f} ± {thermo['temperature'].std():.2f} K")
        print(f"  Pressure: {thermo['pressure'].mean():.2f} ± {thermo['pressure'].std():.2f} atm")
        print(f"  Density: {thermo['density'].mean():.4f} ± {thermo['density'].std():.4f} g/cm³")
    
    if data['rdfs']:
        print(f"\nRDFs:")
        for pair, rdf_data in data['rdfs'].items():
            print(f"  {pair}: {len(rdf_data['r'])} bins")


if __name__ == "__main__":
    test_data_loader()
