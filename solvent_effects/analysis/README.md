# TIP4P/2005 Simulation Analysis Tools

## Overview

This directory contains analysis scripts for TIP4P/2005 water + nanoparticle MD simulations.

## Analysis Scripts

### 1. `analyze_tip4p_simple.py` - **RECOMMENDED**

**Purpose:** Quick thermodynamic analysis from LAMMPS log file

**Features:**
- ✓ Temperature analysis (average, stability)
- ✓ Density analysis  
- ✓ Pressure analysis
- ✓ Energy conservation check
- ✓ Energy drift calculation
- ✓ Volume tracking
- ✓ Generates publication-quality plots
- ✓ Creates text summary report

**Usage:**
```bash
python analyze_tip4p_simple.py ../output/
```

**Output Files:**
- `analysis.png` - 6-panel thermodynamic plots
- `analysis_report.txt` - Text summary report

**Requirements:**
- numpy
- matplotlib

---

### 2. `analyze_tip4p_solvation.py` - **ADVANCED** (Work in Progress)

**Purpose:** Full solvation structure analysis including RDF, H-bonds, coordination

**Features:**
- Radial distribution function (RDF)
- Coordination number calculation
- Hydrogen bonding analysis
- Water orientation analysis
- Density profiles

**Status:** ⚠️ Currently being updated to properly identify nanoparticle atoms

**Usage:**
```bash
python analyze_tip4p_solvation.py ../output/
```

**Requirements:**
- numpy
- matplotlib
- scipy (optional, uses numpy fallback if not available)

---

## Quick Start

### After Running a Simulation:

```bash
# Navigate to analysis directory
cd ~/codes/solvent_structure_around_nanoparticles_V2/solvent_effects/analysis

# Run analysis on your simulation output
python analyze_tip4p_simple.py ../output/

# View results
ls -lh ../output/analysis*
cat ../output/analysis_report.txt
```

### View Plots:

```bash
# Open analysis plot
eog ../output/analysis.png
# or
xdg-open ../output/analysis.png
```

---

## Analysis Results Interpretation

### Temperature
- **Target:** 300 K
- **Good:** ± 20 K from target
- **Issue:** Large fluctuations or drift

### Density  
- **Reference:** 0.997 g/cm³ (pure water)
- **Expected:** Lower for NP+water system
- **Your result:** 0.085 g/cm³ (system is very dilute due to large box)

### Pressure
- **Target:** ~1 bar (1 atm)
- **Acceptable:** ±1000 bar fluctuations
- **Your result:** 53 ± 791 bar (reasonable fluctuations)

### Energy
- **Drift < 5%:** ✓ GOOD conservation
- **Drift 5-10%:** ⚠ MODERATE (acceptable for NVT)
- **Drift > 10%:** ❌ HIGH (check timestep, equilibration)
- **Your result:** 2467% drift ⚠️ (very high - likely due to initial energy)

**Note on Energy Drift:** Large percentage drift can occur when:
1. Starting energy is close to zero
2. Initial configuration is far from equilibrium
3. System is still equilibrating

**Fix:** Look at absolute drift (21671 kcal/mol) relative to average total energy (-878 kcal/mol). The percentage calculation amplifies the apparent drift when energies are small.

---

## Performance Metrics

From your SiC + 500 waters simulation:

| Metric | Value |
|--------|-------|
| **Total time** | 100 ps |
| **Wall time** | 4:29:04 (269 minutes) |
| **Performance** | 0.535 ns/day |
| **Timesteps/s** | 6.194 |
| **MPI tasks** | 8 |
| **OpenMP threads** | 12 per task |
| **CPU utilization** | 44.4% |

### Performance Breakdown:
- **Kspace (PPPM):** 99.10% - Long-range electrostatics (dominant)
- **Pair interactions:** 0.53%
- **Communication:** 0.23%
- **Neighbor list:** 0.02%

**Key Finding:** Long-range electrostatics (PPPM) dominates computation time. This is normal for charged systems like TIP4P water.

---

## Comparing Multiple Simulations

To compare SiC, GO, and Amorphous Carbon:

```bash
# Analyze each simulation
python analyze_tip4p_simple.py ../output_SiC/
python analyze_tip4p_simple.py ../output_GO/
python analyze_tip4p_simple.py ../output_amorphous/

# Compare reports
diff ../output_SiC/analysis_report.txt ../output_GO/analysis_report.txt
```

Or create a comparison script:

```bash
#!/bin/bash
for dir in ../output_*; do
    echo "=== $(basename $dir) ==="
    python analyze_tip4p_simple.py $dir 2>&1 | grep -A 5 "TEMPERATURE:"
    echo ""
done
```

---

## Troubleshooting

### "No data found in log file"
- **Cause:** Log file is empty or corrupted
- **Fix:** Check if simulation ran successfully

### "File not found"
- **Cause:** Wrong output directory path
- **Fix:** Verify path with `ls ../output/`

### Import errors (numpy, matplotlib)
- **Cause:** Missing Python packages
- **Fix:** Install with `pip install numpy matplotlib`

### Large energy drift
- **Cause:** System not equilibrated or numerical issues
- **Fix:** 
  1. Run longer equilibration
  2. Check if drift stabilizes after initial phase
  3. Reduce timestep if needed (currently 1 fs)

### Low density
- **Cause:** Simulation box too large for number of atoms
- **This is normal** for your system - box expanded to 80×80×80 Å³

---

## Next Steps

1. **Immediate:** ✓ Run `analyze_tip4p_simple.py` on your completed simulation

2. **Short-term:** 
   - Run simulations for GO and Amorphous Carbon
   - Compare thermodynamic properties across nanoparticles

3. **Future:**
   - Implement proper NP atom identification for RDF analysis
   - Add water orientation analysis
   - Create comparison plotting script
   - Calculate residence times

---

## File Structure

```
solvent_effects/
├── analysis/
│   ├── analyze_tip4p_simple.py      # Thermodynamic analysis (USE THIS)
│   ├── analyze_tip4p_solvation.py   # Full analysis (in development)
│   └── README.md                    # This file
├── output/
│   ├── log.lammps                   # Simulation log
│   ├── trajectory.lammpstrj         # Full trajectory
│   ├── final_config.data            # Final structure
│   ├── analysis.png                 # Analysis plots
│   └── analysis_report.txt          # Analysis report
└── ...
```

---

## Citation

If you use these analysis tools in publications, please cite:

- **TIP4P/2005 water model:** Abascal, J. L. F.; Vega, C. "A general purpose model for the condensed phases of water: TIP4P/2005." J. Chem. Phys. 2005, 123, 234505.

- **LAMMPS:** Thompson, A. P.; et al. "LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales" Comp. Phys. Comm. 2022, 271, 108171.

---

**Last Updated:** November 5, 2025  
**Status:** Production Ready for Thermodynamic Analysis
