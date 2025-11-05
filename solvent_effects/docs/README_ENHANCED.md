# Solvation Structure Analysis - Enhanced System

This enhanced version provides comprehensive tools for studying solvent structure around nanoparticles at multiple scales (ps to ns).

## 🎯 Key Features

### 1. **Advanced Simulation Runner** (`run_solvation_study.sh`)
- ✅ **Configurable parameters** via command-line flags
- ✅ **Nanosecond scale** support (`--ns` flag)
- ✅ **Flexible timesteps** and dump frequencies
- ✅ **Restart capability** for continuing simulations
- ✅ **Auto box sizing** or manual control
- ✅ **Equilibration control** (separate from production)
- ✅ **Color-coded output** with progress tracking
- ✅ **Timestamped output directories**

### 2. **Advanced Analysis** (`analyze_solvation_advanced.py`)
- ✅ **RDF calculations** between NP and water
- ✅ **Coordination number** analysis
- ✅ **Hydrogen bonding** statistics
- ✅ **Configurable cutoffs** and parameters
- ✅ **Frame selection** (skip/stride)
- ✅ **Automatic plotting** and data export
- ✅ **Comprehensive reports**

### 3. **Batch Runner** (`batch_run_solvation.sh`)
- ✅ **Multiple nanoparticles** in one command
- ✅ **Test and production modes**
- ✅ **Success/failure tracking**
- ✅ **Time estimates**

## 🚀 Quick Start

### Run a Quick Test (100 ps)
```bash
cd shell_scripts
./run_solvation_study.sh SiC 500 100 8
```

### Production Run (5 nanoseconds)
```bash
./run_solvation_study.sh GO 1000 5 16 --ns --dump-freq 5000
```

### Batch Run All Nanoparticles (Test)
```bash
./batch_run_solvation.sh --all-nps --test --cores 16
```

### Batch Production Runs
```bash
./batch_run_solvation.sh --all-nps --production --cores 16
```

### Analyze Results
```bash
cd ../analysis
python analyze_solvation_advanced.py ../output/SiC_500w_5ns_*/ --all
```

## 📊 Complete Workflow Examples

### Example 1: Systematic Study (All NPs, 1 ns)
```bash
# Run simulations
cd shell_scripts
./batch_run_solvation.sh --all-nps --waters 500 --time 1 --ns --cores 16

# Analyze each
cd ../analysis
for dir in ../output/SiC_500w_1ns_*/; do
    python analyze_solvation_advanced.py "$dir" --all --verbose
done

# Compare results
python compare_nanoparticles.py ../output/*_500w_1ns_*/
```

### Example 2: Long Production Run (Single NP, 10 ns)
```bash
# High dump frequency for reduced storage
./run_solvation_study.sh SiC 1000 10 16 \
    --ns \
    --dump-freq 10000 \
    --timestep 2.0 \
    --equilibration 50000 \
    --label "production_10ns"

# Analyze with selective frames
python analyze_solvation_advanced.py ../output/SiC_1000w_10ns_production_10ns/ \
    --all \
    --skip 50 \
    --stride 2 \
    --r-max 20.0
```

### Example 3: Continue from Restart
```bash
# Initial run (5 ns)
./run_solvation_study.sh GO 500 5 16 --ns --label "run1"

# Continue for another 5 ns
./run_solvation_study.sh GO 500 5 16 --ns \
    --restart ../output/GO_500w_5ns_run1/final.restart \
    --label "run2_continued"
```

## 🔧 Advanced Options

### Simulation Runner Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--timestep` | 1.0 fs | Integration timestep |
| `--temp` | 300 K | Temperature |
| `--dump-freq` | 1000 | Trajectory dump frequency |
| `--thermo-freq` | 500 | Thermo output frequency |
| `--box-size` | auto | Box size (Å) |
| `--equilibration` | 10000 | Equilibration steps |
| `--production` | - | Skip equilibration |
| `--ns` | - | Use nanosecond units |
| `--restart` | - | Restart file path |
| `--output-dir` | auto | Custom output directory |
| `--label` | timestamp | Custom label |

### Analysis Script Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rdf` | - | Calculate RDF |
| `--coordination` | - | Calculate coordination |
| `--hbonds` | - | Analyze H-bonds |
| `--all` | - | Run all analyses |
| `--r-max` | 15.0 Å | Max RDF radius |
| `--r-bin` | 0.1 Å | RDF bin width |
| `--skip` | 0 | Skip first N frames |
| `--stride` | 1 | Use every Nth frame |
| `--cutoff` | 3.5 Å | Coordination cutoff |
| `--hb-distance` | 3.5 Å | H-bond distance cutoff |
| `--no-plots` | - | Skip plotting |
| `--output` | solvation_analysis | Output prefix |
| `--verbose` | - | Verbose output |

## 📁 Output Structure

### Simulation Output
```
output/
├── SiC_500w_5ns_20240101_120000/
│   ├── simulation.in              # LAMMPS input
│   ├── trajectory.lammpstrj       # Full trajectory
│   ├── final_config.data          # Final structure
│   ├── final.restart              # Restart file
│   ├── rdf_np_water.dat          # RDF from LAMMPS
│   ├── np_com.dat                 # NP center of mass
│   ├── equilibrated.data          # After equilibration
│   ├── lammps_output.log          # Full LAMMPS log
│   └── simulation_info.txt        # Metadata
```

### Analysis Output
```
output/SiC_500w_5ns_20240101_120000/
├── solvation_analysis_plots.png       # Combined plots
├── solvation_analysis_rdf.dat         # RDF data
├── solvation_analysis_coordination.dat # Coordination vs time
├── solvation_analysis_hbonds.dat      # H-bonds vs time
└── solvation_analysis_report.txt      # Summary report
```

## 🎓 Understanding the Output

### RDF Plot (g(r))
- **Peak position**: Solvation shell distance
- **Peak height**: Structural ordering (>1 = enhanced, <1 = depleted)
- **g(r) = 1**: Bulk water behavior
- **First minimum**: Defines first solvation shell

### Coordination Number
- **Mean value**: Average waters in first shell
- **Fluctuations**: Shell stability
- **Range**: Dynamic exchange with bulk

### Hydrogen Bonds
- **Count**: Number of water-NP H-bonds
- **Stability**: Consistent = strong interaction
- **Variations**: Water exchange events

## 🧪 Performance Guidelines

### Timestep Selection
- **1.0 fs**: Standard, stable, recommended
- **2.0 fs**: Faster, requires careful equilibration
- **0.5 fs**: Very stable, slower, for difficult systems

### Time Scale Recommendations
| Purpose | Time | Cores | Wall Time (est.) |
|---------|------|-------|------------------|
| Quick test | 100 ps | 8 | ~4 hours |
| Structure test | 1 ns | 16 | ~20 hours |
| Production | 5-10 ns | 16-32 | 2-5 days |
| Long dynamics | 50-100 ns | 32+ | 1-2 weeks |

### Dump Frequency Optimization
```bash
# Testing (100 ps): Every 1 ps
--dump-freq 1000    # 100 frames, ~5 MB

# Standard (1-5 ns): Every 10 ps
--dump-freq 10000   # 100-500 frames, ~50-250 MB

# Production (10+ ns): Every 50 ps
--dump-freq 50000   # ~200 frames, ~100 MB
```

## 🔬 Available Nanoparticles

### 1. Silicon Carbide (SiC)
```bash
NP_NAME="SiC"
# Small, well-defined cubic structure
# 8 atoms
# Good for testing
```

### 2. Graphene Oxide (GO)
```bash
NP_NAME="GO"
# Large planar structure with functional groups
# ~90 KB data file
# Interesting surface chemistry
```

### 3. Amorphous Carbon
```bash
NP_NAME="amorphous"
# Large disordered structure
# ~664 KB data file
# Complex surface topology
```

## 🐛 Troubleshooting

### Simulation Crashes
1. **Check equilibration**: Increase `--equilibration 50000`
2. **Reduce timestep**: Use `--timestep 0.5`
3. **Check box size**: Manually set `--box-size 50`

### Analysis Fails
1. **Check trajectory**: Verify `trajectory.lammpstrj` exists
2. **Frame count**: Use `--skip` and `--stride` to reduce
3. **Memory**: Process fewer frames at once

### Slow Performance
1. **PPPM dominates**: 99% of time is long-range electrostatics (normal)
2. **Reduce dump frequency**: Use `--dump-freq 5000`
3. **Increase cores**: Scale to 16-32 cores
4. **Check LAMMPS**: Ensure MPI version installed

## 📈 Analysis Best Practices

### Frame Selection
```bash
# Skip equilibration (first 50 frames)
--skip 50

# Reduce to every 2nd frame (faster)
--stride 2

# Combine for production analysis
--skip 50 --stride 2
```

### RDF Resolution
```bash
# Fine resolution (publication quality)
--r-max 20.0 --r-bin 0.05

# Standard (good balance)
--r-max 15.0 --r-bin 0.1

# Coarse (quick check)
--r-max 10.0 --r-bin 0.2
```

### Coordination Cutoff
```bash
# First minimum from RDF (typical: 3.5-4.0 Å)
--cutoff 3.5    # Standard

# Sensitive to surface effects
--cutoff 4.0    # Include second shell partially
```

## 🎯 Next Steps

### Immediate (Done)
- ✅ Advanced simulation runner with flags
- ✅ Enhanced analysis with configurable parameters
- ✅ Batch processing capability
- ✅ ns-scale support

### Short-term (To-do)
- [ ] Fix atom identification in original `analyze_tip4p_solvation.py`
- [ ] Run comparative study (all NPs, 1 ns each)
- [ ] Create comparative analysis script
- [ ] Water orientation analysis implementation
- [ ] Residence time calculations

### Long-term
- [ ] 10-50 ns production runs
- [ ] Temperature variation studies
- [ ] Different water models comparison
- [ ] Surface charge effects
- [ ] Dynamic property analysis (diffusion, etc.)

## 📚 Additional Resources

### Visualization
```bash
# VMD visualization
vmd trajectory.lammpstrj

# Plot RDF quickly
gnuplot -e "plot 'rdf_np_water.dat' with lines; pause -1"
```

### Data Processing
```bash
# Extract specific columns
awk '{print $1, $2}' solvation_analysis_rdf.dat > rdf_simple.dat

# Calculate statistics
python -c "import numpy as np; print(np.loadtxt('solvation_analysis_coordination.dat', usecols=1).mean())"
```

## 🤝 Contributing

For issues or improvements:
1. Test changes with quick runs (100 ps)
2. Document new features
3. Update this README
4. Provide example commands

---

**Last Updated**: $(date)
**Version**: 2.0 (Enhanced with configurable flags and ns-scale support)
**Status**: Production ready ✅
