# TIP5P Water Solvation Study - Setup Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide helps you set up and run molecular dynamics (MD) simulations of a SiC nanoparticle surrounded by TIP5P water molecules. The TIP5P model is a 5-site water model with explicit lone pair sites, providing accurate representation of water structure, hydrogen bonding, and dielectric properties.

**Key Features:**
- Full electrostatic interactions (Ewald summation via PPPM)
- Rigid water constraints (SHAKE algorithm)
- Accurate hydrogen bonding capabilities
- Low energy drift (<5% target)
- Professional-grade simulation protocol

---

## Prerequisites

### Required Software

1. **Python 3.7+** with NumPy
   ```bash
   python3 --version  # Should be 3.7 or higher
   python3 -c "import numpy; print(numpy.__version__)"
   ```

2. **LAMMPS** (with TIP5P support)
   - Must be compiled with KSPACE package (for PPPM)
   - Must support `pair_style lj/cut/tip5p/long`
   - Recommended: MPI version for parallel execution
   
   ```bash
   # Check LAMMPS installation
   lmp -help
   # or
   lmp_mpi -help
   ```

3. **MPI** (optional, for parallel runs)
   ```bash
   mpirun --version
   ```

4. **VMD** (optional, for visualization)
   ```bash
   vmd -h
   ```

### Installing LAMMPS with TIP5P Support

If you need to compile LAMMPS:

```bash
# Download LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps

# Build with required packages
mkdir build && cd build
cmake ../cmake -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_RIGID=yes
make -j 4
sudo make install
```

---

## Installation

### 1. Navigate to Project Directory

```bash
cd /path/to/solvent_effects
```

### 2. Verify File Structure

Your directory should contain:

```
solvent_effects/
├── input_files/
│   ├── sic_nanoparticle.data     # SiC nanoparticle structure
│   └── solvation_tip5p.in        # LAMMPS input script
├── python_scripts/
│   └── prepare_tip5p_system.py   # Water placement script
├── shell_scripts/
│   └── run_tip5p_simulation.sh   # Main run script
├── docs/
│   └── SETUP_GUIDE.md            # This file
├── output/                        # (Created automatically)
└── analysis/                      # (Created automatically)
```

### 3. Test Dependencies

```bash
cd shell_scripts
./run_tip5p_simulation.sh --help
```

This will show help information and verify that dependencies are accessible.

---

## Quick Start

### Run a Small Test Simulation

Start with a small system to verify everything works:

```bash
cd shell_scripts

# Run 100-water test (very fast, ~2 minutes)
./run_tip5p_simulation.sh -n 100 -t 10 -c 4
```

**Parameters:**
- `-n 100`: 100 water molecules
- `-t 10`: 10 ps production time
- `-c 4`: Use 4 CPU cores

**Expected output:**
```
========================================
TIP5P WATER SOLVATION SIMULATION
========================================

✓ Python 3 found: version 3.x.x
✓ NumPy found: version 1.x.x
✓ LAMMPS found: /usr/local/bin/lmp
...
✓ System preparation complete
...
✓ Simulation completed successfully!
```

### Run a Production Simulation

For actual research, use more water molecules and longer simulation time:

```bash
# 5000 waters, 500 ps production, 10 cores
./run_tip5p_simulation.sh -n 5000 -t 500 -c 10
```

**Runtime estimate:** 
- 100 waters, 10 ps: ~2 minutes
- 1000 waters, 100 ps: ~20 minutes
- 5000 waters, 500 ps: ~4-6 hours (with 10 cores)
- 10000 waters, 1000 ps: ~12-24 hours (with 10 cores)

---

## Detailed Usage

### Command Line Options

```bash
./run_tip5p_simulation.sh [options]

Options:
  -n, --nwaters N      Number of water molecules (default: 3000)
  -b, --boxsize SIZE   Box size in Å (default: auto-calculated)
  -s, --strategy TYPE  Placement: 'full_box' or 'shell' (default: full_box)
  -t, --time TIME      Production time in ps (default: 500)
  -c, --cores N        Number of MPI cores (default: 10)
  -h, --help           Show help message
```

### Placement Strategies

#### 1. Full Box (Default)

Places water throughout the entire simulation box, aiming for bulk water density (~1.0 g/cm³).

```bash
./run_tip5p_simulation.sh -n 5000 -s full_box
```

**Use when:**
- Studying bulk water properties
- Want to compare with experimental water density
- Need full solvation environment

#### 2. Solvation Shell

Places water only in a shell around the nanoparticle (3-25 Å from NP center).

```bash
./run_tip5p_simulation.sh -n 2000 -s shell
```

**Use when:**
- Focusing on interface properties
- Limited computational resources
- Only interested in first/second solvation shells

### Box Size Selection

**Option 1: Auto-calculate (recommended)**
```bash
./run_tip5p_simulation.sh -n 5000
```
The script calculates box size to achieve ~1.0 g/cm³ water density.

**Option 2: Manual specification**
```bash
./run_tip5p_simulation.sh -n 5000 -b 50.0
```
Useful when you need a specific box size for comparison studies.

### Production Time

**Short runs (testing):**
```bash
./run_tip5p_simulation.sh -n 1000 -t 50  # 50 ps
```

**Standard runs:**
```bash
./run_tip5p_simulation.sh -n 5000 -t 500  # 500 ps (default)
```

**Long runs (better statistics):**
```bash
./run_tip5p_simulation.sh -n 5000 -t 1000  # 1 ns
```

**Note:** All runs include 50 ps equilibration automatically.

---

## Output Files

### Output Directory Structure

Each simulation creates a timestamped directory:

```
output/production_tip5p_5000waters_500ps_20251104_143052/
├── production.lammpstrj          # Main trajectory (all atoms)
├── production_custom.dump        # Detailed trajectory (positions, velocities, forces)
├── temperature.dat               # Temperature vs time
├── pressure.dat                  # Pressure vs time
├── energy.dat                    # Energy components vs time
├── final_configuration.data      # Final atomic positions
├── restart.*.lmp                 # Restart files (every 100 ps)
├── log.lammps                    # Complete LAMMPS log
└── simulation_info.txt           # Simulation parameters summary
```

### File Descriptions

#### 1. production.lammpstrj

Main trajectory file in LAMMPS dump format.

**Format:**
```
ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
25008
ITEM: BOX BOUNDS pp pp pp
-25.0 25.0
-25.0 25.0
-25.0 25.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
...
```

**Use for:**
- VMD visualization
- RDF analysis
- Coordination number calculations
- Structure analysis

#### 2. production_custom.dump

Detailed trajectory with velocities and forces.

**Columns:** `id mol type q x y z vx vy vz fx fy fz`

**Use for:**
- Detailed dynamics analysis
- Force analysis
- Velocity distributions
- Advanced research questions

#### 3. temperature.dat, pressure.dat, energy.dat

Time series data for thermodynamic properties.

**Format (3 columns):**
```
# Timestep   Value   Value
1000        299.85   ...
2000        300.12   ...
...
```

**Use for:**
- Checking equilibration
- Verifying stability
- Calculating drift
- Quality control

#### 4. log.lammps

Complete LAMMPS output log.

**Contains:**
- All LAMMPS commands executed
- Thermo output (every 1000 steps)
- Timing information
- Any warnings or errors

**Critical for:**
- Debugging
- Performance analysis
- Verification

#### 5. simulation_info.txt

Human-readable summary of simulation parameters.

**Contains:**
```
TIP5P Water Solvation Simulation
Generated: Mon Nov  4 14:30:52 2025

System Parameters:
  Water molecules: 5000
  Box size: 50.0 Å
  
Simulation Parameters:
  Equilibration: 50 ps
  Production: 500 ps
  Temperature: 300 K
  
Water Model:
  Type: TIP5P (5-site)
  Charges: q_H = +0.241, q_L = -0.241
  ...
```

---

## Troubleshooting

### Problem: "LAMMPS not found"

**Solution:**
```bash
# Check if LAMMPS is in PATH
which lmp
which lmp_mpi

# If not found, add to PATH
export PATH=/path/to/lammps/bin:$PATH

# Or use full path in script
LAMMPS_CMD="/full/path/to/lmp"
```

### Problem: "Python import error: numpy"

**Solution:**
```bash
pip3 install numpy

# Or with conda
conda install numpy
```

### Problem: "Could only place N waters (less than requested)"

**Cause:** Box too small or waters too close together.

**Solutions:**
```bash
# Option 1: Increase box size
./run_tip5p_simulation.sh -n 5000 -b 60.0  # Larger box

# Option 2: Reduce water count
./run_tip5p_simulation.sh -n 4000

# Option 3: Use shell strategy (more space efficient)
./run_tip5p_simulation.sh -n 5000 -s shell
```

### Problem: High Energy Drift (>5%)

**Possible causes:**
1. System not equilibrated long enough
2. Timestep too large
3. Numerical instability

**Solutions:**

**A. Longer equilibration:**
Edit `solvation_tip5p.in`, increase equilibration steps:
```lammps
run 200000  # 100 ps equilibration instead of 50 ps
```

**B. Smaller timestep:**
Edit `solvation_tip5p.in`:
```lammps
timestep 0.2  # Reduce from 0.5 fs to 0.2 fs
```
Note: This increases runtime by 2.5x.

**C. Check for bad contacts:**
```bash
# Visualize initial structure
vmd ../input_files/tip5p_system_5000waters.data
```
Look for overlapping atoms.

### Problem: Simulation Very Slow

**Optimization strategies:**

**1. Use more CPU cores:**
```bash
./run_tip5p_simulation.sh -n 5000 -c 20  # Use 20 cores
```

**2. Reduce output frequency:**
Edit `solvation_tip5p.in`:
```lammps
dump prod_traj all atom 2000  # Every 2000 steps instead of 500
```

**3. Smaller system:**
```bash
./run_tip5p_simulation.sh -n 2000 -s shell  # Shell only
```

**4. Check LAMMPS build:**
```bash
lmp -help | grep "PPPM"  # Should show PPPM support
```

### Problem: "PPPM initialization failed"

**Cause:** Box too small for PPPM accuracy.

**Solution:**
```bash
# Use larger box
./run_tip5p_simulation.sh -n 5000 -b 60.0
```

Or edit `solvation_tip5p.in`:
```lammps
kspace_style pppm/tip5p 1.0e-3  # Relax accuracy from 1e-4 to 1e-3
```

### Problem: Temperature Not Stable at 300 K

**Expected behavior:** Temperature should fluctuate around 300 K with ±10 K variation.

**If temperature drifts significantly:**

1. Check thermostat parameters:
```lammps
fix nvt_prod real_atoms nvt temp 300.0 300.0 100.0
#                                 ^start ^end   ^Tdamp
```

2. Verify equilibration completed:
```bash
grep "^[0-9]" output/.../log.lammps | awk '{print $1, $2}' | head -200
# Should see temperature stabilizing
```

3. Check for energy drift (see above)

---

## Next Steps

After successful simulation:

1. **Visualize:**
   ```bash
   vmd output/production_tip5p_5000waters_500ps_*/production.lammpstrj
   ```

2. **Analyze:**
   - See [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) for analysis scripts
   - RDF, coordination numbers, H-bonding, orientation

3. **Compare:**
   - Run multiple simulations with different parameters
   - Compare with previous SPC/E results
   - Validate against experimental data

4. **Extend:**
   - Longer production runs
   - Different temperatures
   - Pressure coupling (NPT ensemble)
   - Different nanoparticle materials

---

## Additional Resources

- **LAMMPS Documentation:** https://docs.lammps.org/
- **TIP5P Reference:** Rick, S.W., J. Chem. Phys. 120, 6085-6093 (2004)
- **Project README:** [../README.md](../README.md)
- **Analysis Guide:** [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)
- **Parameter Details:** [TIP5P_PARAMETERS.md](TIP5P_PARAMETERS.md)

---

## Support

If you encounter issues not covered here:

1. Check LAMMPS log file: `output/.../log.lammps`
2. Verify all dependencies are correctly installed
3. Test with minimal system (100 waters, 10 ps)
4. Review simulation_info.txt for parameter verification

---

*Last updated: November 4, 2025*
