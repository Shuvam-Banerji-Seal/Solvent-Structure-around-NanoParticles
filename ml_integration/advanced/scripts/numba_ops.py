"""
Numba-optimized operations for data augmentation.
Author: Shuvam Banerji Seal
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def rotate_coords_numba(coords, theta, axis_idx):
    """
    Rotate coordinates around axis using Numba.
    
    Args:
        coords: (n_frames, n_atoms, 3) array
        theta: rotation angle
        axis_idx: 0 for x, 1 for y, 2 for z
        
    Returns:
        Rotated coordinates
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Create rotation matrix
    R = np.zeros((3, 3), dtype=np.float32)
    
    if axis_idx == 0:  # x
        R[0, 0] = 1
        R[1, 1] = cos_t
        R[1, 2] = -sin_t
        R[2, 1] = sin_t
        R[2, 2] = cos_t
    elif axis_idx == 1:  # y
        R[0, 0] = cos_t
        R[0, 2] = sin_t
        R[1, 1] = 1
        R[2, 0] = -sin_t
        R[2, 2] = cos_t
    else:  # z
        R[0, 0] = cos_t
        R[0, 1] = -sin_t
        R[1, 0] = sin_t
        R[1, 1] = cos_t
        R[2, 2] = 1
        
    # Apply rotation: coords @ R.T
    # Manual loop for maximum speed (though dot is also supported)
    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]
    
    out = np.zeros_like(coords)
    
    for f in range(n_frames):
        for i in range(n_atoms):
            x = coords[f, i, 0]
            y = coords[f, i, 1]
            z = coords[f, i, 2]
            
            out[f, i, 0] = x * R[0, 0] + y * R[0, 1] + z * R[0, 2]
            out[f, i, 1] = x * R[1, 0] + y * R[1, 1] + z * R[1, 2]
            out[f, i, 2] = x * R[2, 0] + y * R[2, 1] + z * R[2, 2]
            
    return out

@jit(nopython=True)
def add_noise_numba(coords, scale):
    """Add Gaussian noise."""
    noise = np.random.normal(0, scale, coords.shape)
    return coords + noise

@jit(nopython=True, parallel=True)
def compute_rmsf_numba(coords):
    """
    Compute RMSF using Numba.
    Args:
        coords: (n_frames, n_atoms, 3)
    Returns:
        rmsf: (n_atoms,)
    """
    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]
    
    # Compute mean position
    mean_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        for f in range(n_frames):
            mean_pos[i, 0] += coords[f, i, 0]
            mean_pos[i, 1] += coords[f, i, 1]
            mean_pos[i, 2] += coords[f, i, 2]
        mean_pos[i, 0] /= n_frames
        mean_pos[i, 1] /= n_frames
        mean_pos[i, 2] /= n_frames
        
    # Compute fluctuations
    sq_diff_sum = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        for f in range(n_frames):
            dx = coords[f, i, 0] - mean_pos[i, 0]
            dy = coords[f, i, 1] - mean_pos[i, 1]
            dz = coords[f, i, 2] - mean_pos[i, 2]
            sq_diff_sum[i] += dx*dx + dy*dy + dz*dz
            
    rmsf = np.sqrt(sq_diff_sum / n_frames)
    return rmsf

@jit(nopython=True)
def compute_msd_numba(coords):
    """
    Compute MSD (simplified) using Numba.
    Args:
        coords: (n_frames, n_atoms, 3)
    Returns:
        msd_per_frame: (n_frames,)
    """
    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]
    
    start_pos = coords[0]  # (n_atoms, 3)
    msd = np.zeros(n_frames, dtype=np.float32)
    
    for f in range(n_frames):
        sq_disp_sum = 0.0
        for i in range(n_atoms):
            dx = coords[f, i, 0] - start_pos[i, 0]
            dy = coords[f, i, 1] - start_pos[i, 1]
            dz = coords[f, i, 2] - start_pos[i, 2]
            sq_disp_sum += dx*dx + dy*dy + dz*dz
        msd[f] = sq_disp_sum / n_atoms
        
    return msd
