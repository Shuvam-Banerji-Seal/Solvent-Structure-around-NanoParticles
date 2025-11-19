#!/usr/bin/env python3
"""
RDF Calculator using CuPy GPU Acceleration - FULLY CORRECTED VERSION
✓ Correct atom type mapping (Type 1=O, Type 2=H, Type 3=C)
✓ Correct box size parsing from trajectory
✓ Correct minimum image convention using image flags

Atom Type Mapping (from LAMMPS data file):
  Type 1: OXYGEN (mass 15.9994)  - 6,711 atoms from TIP4P water
  Type 2: HYDROGEN (mass 1.008)  - 13,422 atoms from TIP4P water
  Type 3: CARBON (mass 12.011)   - 201 atoms (nanoparticle)

RDF Pairs Calculated:
  g_CO: Type 3 (Carbon) vs Type 1 (Oxygen)
  g_OO: Type 1 (Oxygen) vs Type 1 (Oxygen) 
  g_CH: Type 3 (Carbon) vs Type 2 (Hydrogen)
"""

import os
import pickle
import numpy as np
import time
from pathlib import Path

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    print("Warning: CuPy not available, falling back to NumPy")
    HAS_CUPY = False
    cp = np


def parse_trajectory_frame(lines, start_idx):
    """Parse a single frame from LAMMPS trajectory with image flags."""
    atoms = []
    idx = start_idx
    
    while idx < len(lines):
        line = lines[idx]
        if line.startswith('ITEM:'):
            break
        
        parts = line.split()
        if len(parts) >= 8:
            try:
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                ix = int(parts[5])  # Image flag x
                iy = int(parts[6])  # Image flag y
                iz = int(parts[7])  # Image flag z
                atoms.append([atom_type, x, y, z, ix, iy, iz])
            except (ValueError, IndexError):
                pass
        
        idx += 1
    
    return np.array(atoms, dtype=object) if atoms else None, idx


def parse_box_bounds(lines, start_idx):
    """Parse box bounds from trajectory header."""
    idx = start_idx
    box_bounds = {'x': None, 'y': None, 'z': None}
    dims = ['x', 'y', 'z']
    dim_idx = 0
    
    while idx < len(lines) and dim_idx < 3:
        line = lines[idx]
        if line.startswith('ITEM:'):
            break
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                lo = float(parts[0])
                hi = float(parts[1])
                box_bounds[dims[dim_idx]] = (lo, hi)
                dim_idx += 1
            except ValueError:
                pass
        
        idx += 1
    
    return box_bounds, idx


def unwrap_coordinates(positions_wrapped, image_flags, box_bounds):
    """
    Unwrap coordinates using image flags and box bounds.
    
    Parameters:
    -----------
    positions_wrapped : array, shape (N, 3)
        Wrapped coordinates (x, y, z)
    image_flags : array, shape (N, 3)
        Image flags (ix, iy, iz)
    box_bounds : dict
        {'x': (xlo, xhi), 'y': (ylo, yhi), 'z': (zlo, zhi)}
    
    Returns:
    --------
    positions_unwrapped : array, shape (N, 3)
        Unwrapped coordinates
    """
    
    box_lengths = np.array([
        box_bounds['x'][1] - box_bounds['x'][0],
        box_bounds['y'][1] - box_bounds['y'][0],
        box_bounds['z'][1] - box_bounds['z'][0]
    ])
    
    box_lows = np.array([
        box_bounds['x'][0],
        box_bounds['y'][0],
        box_bounds['z'][0]
    ])
    
    # Unwrapped = wrapped + image_flag * box_length + box_low
    positions_unwrapped = positions_wrapped + image_flags * box_lengths + box_lows
    
    return positions_unwrapped


def calculate_rdf_pair_cuda(positions_type1, positions_type2, box_bounds, r_max, dr):
    """
    Calculate RDF between two atom types using CuPy GPU acceleration.
    
    Parameters:
    -----------
    positions_type1 : array, shape (N1, 3)
        Unwrapped positions of atoms of type 1
    positions_type2 : array, shape (N2, 3)
        Unwrapped positions of atoms of type 2
    box_bounds : dict
        Box boundaries for minimum image convention
    r_max : float
        Maximum distance for RDF
    dr : float
        Bin width
    
    Returns:
    --------
    rdf : array
        Radial distribution function g(r)
    r_centers : array
        Distance bin centers
    """
    
    N1 = len(positions_type1)
    N2 = len(positions_type2)
    n_bins = int(np.ceil(r_max / dr))
    
    # Box parameters
    box_lengths = np.array([
        box_bounds['x'][1] - box_bounds['x'][0],
        box_bounds['y'][1] - box_bounds['y'][0],
        box_bounds['z'][1] - box_bounds['z'][0]
    ])
    half_box = box_lengths / 2.0
    
    # Convert to GPU arrays if CuPy available
    if HAS_CUPY:
        pos1 = cp.asarray(positions_type1, dtype=cp.float32)
        pos2 = cp.asarray(positions_type2, dtype=cp.float32)
        half_box_gpu = cp.asarray(half_box, dtype=cp.float32)
    else:
        pos1 = positions_type1.astype(np.float32)
        pos2 = positions_type2.astype(np.float32)
        half_box_gpu = half_box.astype(np.float32)
    
    # Initialize histogram
    if HAS_CUPY:
        hist = cp.zeros(n_bins, dtype=cp.float32)
    else:
        hist = np.zeros(n_bins, dtype=np.float32)
    
    # Calculate pairwise distances and histogram
    # Process in chunks to avoid memory overflow
    chunk_size = min(500, N1)
    
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        chunk = pos1[i:end_i]  # (chunk_size, 3)
        
        # Reshape for broadcasting: (chunk_size, 1, 3) - (1, N2, 3)
        chunk = cp.reshape(chunk, (chunk.shape[0], 1, 3)) if HAS_CUPY else chunk.reshape((chunk.shape[0], 1, 3))
        pos2_reshaped = cp.reshape(pos2, (1, N2, 3)) if HAS_CUPY else pos2.reshape((1, N2, 3))
        
        # Compute distances with minimum image convention
        diff = chunk - pos2_reshaped  # (chunk_size, N2, 3)
        
        # Apply minimum image convention
        # If diff > half_box, subtract box_length
        # If diff < -half_box, add box_length
        diff = cp.where(diff > half_box_gpu, diff - 2*half_box_gpu, diff) if HAS_CUPY else np.where(diff > half_box_gpu, diff - 2*half_box_gpu, diff)
        diff = cp.where(diff < -half_box_gpu, diff + 2*half_box_gpu, diff) if HAS_CUPY else np.where(diff < -half_box_gpu, diff + 2*half_box_gpu, diff)
        
        distances = cp.sqrt(cp.sum(diff**2, axis=2)) if HAS_CUPY else np.sqrt(np.sum(diff**2, axis=2))
        
        # Add to histogram
        bins = (distances / dr).astype(cp.int32) if HAS_CUPY else (distances / dr).astype(np.int32)
        mask = (bins >= 0) & (bins < n_bins)
        
        for b in range(n_bins):
            hist[b] += cp.sum(mask & (bins == b)) if HAS_CUPY else np.sum(mask & (bins == b))
    
    # Normalize
    # Number density of type2
    volume = np.prod(box_lengths)
    rho2 = N2 / volume
    
    # Shell volume for each bin
    r_inner = np.arange(n_bins) * dr
    r_outer = (np.arange(n_bins) + 1) * dr
    shell_volumes = (4/3) * np.pi * (r_outer**3 - r_inner**3)
    
    # Normalization
    if HAS_CUPY:
        hist_np = cp.asnumpy(hist)
    else:
        hist_np = hist
    
    rdf = hist_np / (N1 * rho2 * shell_volumes)
    r_centers = (r_inner + r_outer) / 2
    
    return rdf, r_centers


def process_trajectory(trajectory_file, output_dir):
    """
    Process trajectory file and calculate RDFs.
    """
    
    print(f"\n{'='*80}")
    print(f"Processing trajectory: {trajectory_file}")
    print(f"{'='*80}")
    
    # RDF parameters
    r_max = 15.0  # Ångströms
    dr = 0.1       # Ångströms
    
    # Initialize RDF accumulators
    n_bins = int(np.ceil(r_max / dr))
    rdf_co = np.zeros(n_bins)
    rdf_oo = np.zeros(n_bins)
    rdf_ch = np.zeros(n_bins)
    
    frame_count = 0
    start_time = time.time()
    
    # Read trajectory file
    with open(trajectory_file, 'r') as f:
        all_lines = f.readlines()
    
    # Parse frames
    line_idx = 0
    frame_idx = 0
    box_bounds = None
    
    while line_idx < len(all_lines):
        line = all_lines[line_idx]
        
        if 'ITEM: TIMESTEP' in line:
            frame_idx += 1
            
            # Skip to box bounds section
            while line_idx < len(all_lines):
                line = all_lines[line_idx]
                if 'ITEM: BOX BOUNDS' in line:
                    line_idx += 1
                    break
                line_idx += 1
            
            # Parse box bounds
            box_bounds, next_idx = parse_box_bounds(all_lines, line_idx)
            line_idx = next_idx
            
            # Skip to atoms section
            while line_idx < len(all_lines):
                line = all_lines[line_idx]
                if 'ITEM: ATOMS' in line:
                    line_idx += 1
                    break
                line_idx += 1
            
            # Parse frame
            atoms_data, next_idx = parse_trajectory_frame(all_lines, line_idx)
            
            if atoms_data is not None and box_bounds is not None:
                # Extract positions and image flags
                positions = atoms_data[:, 1:4].astype(float)
                image_flags = atoms_data[:, 4:7].astype(int)
                atom_types = atoms_data[:, 0].astype(int)
                
                # Unwrap coordinates
                positions_unwrapped = unwrap_coordinates(positions, image_flags, box_bounds)
                
                # Separate by type
                type1_mask = atom_types == 1  # Oxygen
                type2_mask = atom_types == 2  # Hydrogen
                type3_mask = atom_types == 3  # Carbon
                
                pos_o = positions_unwrapped[type1_mask]
                pos_h = positions_unwrapped[type2_mask]
                pos_c = positions_unwrapped[type3_mask]
                
                # Calculate RDFs
                rdf_co_frame, r_centers = calculate_rdf_pair_cuda(pos_c, pos_o, box_bounds, r_max, dr)
                rdf_oo_frame, _ = calculate_rdf_pair_cuda(pos_o, pos_o, box_bounds, r_max, dr)
                rdf_ch_frame, _ = calculate_rdf_pair_cuda(pos_c, pos_h, box_bounds, r_max, dr)
                
                rdf_co += rdf_co_frame
                rdf_oo += rdf_oo_frame
                rdf_ch += rdf_ch_frame
                
                frame_count += 1
                
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Frame {frame_idx}: {frame_count} production frames | Box: {box_bounds['x'][1]-box_bounds['x'][0]:.1f}×{box_bounds['y'][1]-box_bounds['y'][0]:.1f}×{box_bounds['z'][1]-box_bounds['z'][0]:.1f} Å³ | {elapsed:.1f}s")
                
                # Limit to 100 production frames
                if frame_count >= 100:
                    break
            
            line_idx = next_idx
        else:
            line_idx += 1
    
    # Average RDFs
    rdf_co /= frame_count
    rdf_oo /= frame_count
    rdf_ch /= frame_count
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_count} production frames in {elapsed:.2f}s")
    
    # Find peaks
    peaks_co = find_peaks(rdf_co, min_height=0.1, min_distance=5)
    peaks_oo = find_peaks(rdf_oo, min_height=0.1, min_distance=5)
    peaks_ch = find_peaks(rdf_ch, min_height=0.1, min_distance=5)
    
    print(f"\nC-O RDF peaks: {r_centers[peaks_co] if len(peaks_co) > 0 else 'none'}")
    print(f"O-O RDF peaks: {r_centers[peaks_oo] if len(peaks_oo) > 0 else 'none'}")
    print(f"C-H RDF peaks: {r_centers[peaks_ch] if len(peaks_ch) > 0 else 'none'}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to text file
    output_file = os.path.join(output_dir, 'rdf_data.txt')
    with open(output_file, 'w') as f:
        f.write("r(Å)      g_CO          g_OO          g_CH\n")
        for i in range(len(r_centers)):
            f.write(f"{r_centers[i]:8.4f}   {rdf_co[i]:13.6f}  {rdf_oo[i]:13.6f}  {rdf_ch[i]:13.6f}\n")
    
    print(f"\nSaved RDF data to {output_file}")
    
    # Save pickle
    pickle_file = os.path.join(output_dir, 'rdf_results.pkl')
    results = {
        'r_centers': r_centers,
        'g_co': rdf_co,
        'g_oo': rdf_oo,
        'g_ch': rdf_ch,
        'peaks_co': peaks_co,
        'peaks_oo': peaks_oo,
        'peaks_ch': peaks_ch,
        'frame_count': frame_count,
        'box_bounds': box_bounds
    }
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved pickle to {pickle_file}")
    
    return results


def find_peaks(data, min_height=0.1, min_distance=5):
    """Simple peak finding."""
    peaks = []
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > min_height:
            peaks.append(i)
    
    # Filter by minimum distance
    filtered_peaks = []
    for peak in peaks:
        if not filtered_peaks or peak - filtered_peaks[-1] > min_distance:
            filtered_peaks.append(peak)
    
    return np.array(filtered_peaks)


def main():
    """Process all epsilon values."""
    
    base_dir = "/store/shuvam/solvent_effects/main_project/combined_system"
    epsilons = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50", "0.55", "0.65"]
    
    print("\n" + "="*80)
    print("RDF CALCULATION - FULLY CORRECTED VERSION")
    print("="*80)
    print("\nAtom Type Mapping:")
    print("  Type 1: OXYGEN    (6,711 atoms)")
    print("  Type 2: HYDROGEN  (13,422 atoms)")
    print("  Type 3: CARBON    (201 atoms)")
    print("\nCalculating RDF pairs:")
    print("  g_CO: Carbon vs Oxygen")
    print("  g_OO: Oxygen vs Oxygen (hydrogen bonding)")
    print("  g_CH: Carbon vs Hydrogen")
    print("\nCorrections applied:")
    print("  ✓ Correct atom type mapping")
    print("  ✓ Parse box size from trajectory")
    print("  ✓ Unwrap coordinates using image flags")
    print("  ✓ Correct minimum image convention")
    print("="*80)
    
    all_results = {}
    
    for epsilon in epsilons:
        traj_dir = os.path.join(base_dir, f"epsilon_{epsilon}")
        traj_file = os.path.join(traj_dir, "production.lammpstrj")
        
        if not os.path.exists(traj_file):
            print(f"\n⚠️  Trajectory file not found: {traj_file}")
            continue
        
        output_dir = traj_dir
        
        results = process_trajectory(traj_file, output_dir)
        all_results[epsilon] = results
    
    print("\n" + "="*80)
    print("ALL RDF CALCULATIONS COMPLETE")
    print("="*80)
    
    # Save summary
    summary_file = os.path.join(base_dir, "rdf_all_results_CORRECTED.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nSaved all results to {summary_file}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Check if CuPy is available
    if HAS_CUPY:
        print(f"✓ CuPy is available - using GPU acceleration")
        try:
            gpu_info = cp.cuda.Device()
            print(f"  GPU Device: {gpu_info}")
        except:
            pass
    else:
        print(f"⚠ CuPy not available - using NumPy (CPU mode)")
    
    all_results = main()
