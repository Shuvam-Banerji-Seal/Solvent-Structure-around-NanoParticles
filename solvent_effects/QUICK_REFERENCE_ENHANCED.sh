#!/bin/bash
# QUICK REFERENCE - Enhanced Solvation System
# ============================================

# Navigate to scripts
cd solvent_effects/shell_scripts

# ═══════════════════════════════════════════════════════════
# SIMULATION COMMANDS
# ═══════════════════════════════════════════════════════════

# Quick test (100 ps)
./run_solvation_study.sh SiC 500 100 8

# Production run (5 ns)
./run_solvation_study.sh GO 1000 5 16 --ns --dump-freq 5000

# High temperature study
./run_solvation_study.sh SiC 500 1 16 --ns --temp 350 --label "high_temp"

# Long equilibration
./run_solvation_study.sh amorphous 500 1 16 --ns --equilibration 50000

# Continue from restart
./run_solvation_study.sh SiC 500 5 16 --ns --restart ../output/SiC_*/final.restart

# Custom timestep (faster but less stable)
./run_solvation_study.sh GO 500 1 16 --ns --timestep 2.0

# ═══════════════════════════════════════════════════════════
# BATCH COMMANDS
# ═══════════════════════════════════════════════════════════

# Test all nanoparticles (100 ps each)
./batch_run_solvation.sh --all-nps --test --cores 16

# Production all NPs (5 ns each)  
./batch_run_solvation.sh --all-nps --production --cores 32

# Custom: SiC and GO, 1 ns
./batch_run_solvation.sh --nps SiC,GO --waters 500 --time 1 --ns --cores 16

# With custom args
./batch_run_solvation.sh --all-nps --production --cores 32 --custom-args "--timestep 2.0 --temp 350"

# ═══════════════════════════════════════════════════════════
# ANALYSIS COMMANDS
# ═══════════════════════════════════════════════════════════

cd ../analysis

# Full analysis
python analyze_solvation_advanced.py ../output/SiC_500w_100ps_*/ --all --verbose

# Just RDF
python analyze_solvation_advanced.py ../output/GO_*/ --rdf --r-max 20 --r-bin 0.05

# Just coordination and H-bonds
python analyze_solvation_advanced.py ../output/amorphous_*/ --coordination --hbonds

# Skip equilibration (first 50 frames)
python analyze_solvation_advanced.py ../output/SiC_*/ --all --skip 50

# Reduce frames (every 2nd)
python analyze_solvation_advanced.py ../output/GO_*/ --all --stride 2

# Combined: skip + stride + custom cutoffs
python analyze_solvation_advanced.py ../output/SiC_*/ \
    --all \
    --skip 50 \
    --stride 2 \
    --r-max 20 \
    --cutoff 4.0 \
    --hb-distance 3.0

# Save data only (no plots)
python analyze_solvation_advanced.py ../output/GO_*/ --all --no-plots

# Custom output prefix
python analyze_solvation_advanced.py ../output/SiC_*/ --all --output my_analysis

# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

# VMD trajectory
vmd ../output/SiC_*/trajectory.lammpstrj

# Plot RDF quickly
gnuplot -e "plot '../output/SiC_*/rdf_np_water.dat' with lines; pause -1"

# View analysis plots
eog ../output/SiC_*/solvation_analysis_plots.png

# ═══════════════════════════════════════════════════════════
# FILE OPERATIONS
# ═══════════════════════════════════════════════════════════

# Check simulation status
tail -f ../output/SiC_*/lammps_output.log

# View simulation info
cat ../output/SiC_*/simulation_info.txt

# Check analysis report
cat ../output/SiC_*/solvation_analysis_report.txt

# List all output directories
ls -lhd ../output/*/

# Archive old simulations
tar -czf SiC_test.tar.gz ../output/SiC_500w_100ps_*/

# ═══════════════════════════════════════════════════════════
# COMMON WORKFLOWS
# ═══════════════════════════════════════════════════════════

# WORKFLOW 1: Quick test all NPs
cd shell_scripts
./batch_run_solvation.sh --all-nps --test --cores 16
cd ../analysis
for dir in ../output/*/; do
    python analyze_solvation_advanced.py "$dir" --all
done

# WORKFLOW 2: Production study
cd shell_scripts
./run_solvation_study.sh SiC 1000 10 32 --ns --dump-freq 10000 --label "production"
# Wait for completion...
cd ../analysis
python analyze_solvation_advanced.py ../output/SiC_*_production/ \
    --all --skip 100 --stride 2 --verbose

# WORKFLOW 3: Temperature series
for T in 300 325 350 375; do
    ./run_solvation_study.sh SiC 500 1 16 --ns --temp $T --label "temp${T}K"
done

# WORKFLOW 4: Convergence check
for TIME in 1 2 5 10; do
    ./run_solvation_study.sh GO 500 $TIME 16 --ns --label "${TIME}ns"
done

# ═══════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════

# Check for errors
grep -i error ../output/*/lammps_output.log

# Check performance
grep "Performance" ../output/*/lammps_output.log

# Verify trajectory
head -100 ../output/*/trajectory.lammpstrj

# Check completion
grep "SIMULATION COMPLETE" ../output/*/lammps_output.log

# Re-analyze with different settings
python analyze_solvation_advanced.py ../output/SiC_*/ \
    --rdf --r-max 15 --r-bin 0.1 --verbose

# ═══════════════════════════════════════════════════════════
# PARAMETERS REFERENCE
# ═══════════════════════════════════════════════════════════

# Simulation flags:
#   --timestep 1.0      # fs (default)
#   --temp 300          # K (default)
#   --dump-freq 1000    # steps (default)
#   --thermo-freq 500   # steps (default)  
#   --box-size 40       # Å (default: auto)
#   --equilibration 10000  # steps (default)
#   --production        # Skip equilibration
#   --ns                # Use nanosecond units
#   --restart FILE      # Continue from restart
#   --output-dir DIR    # Custom directory
#   --label LABEL       # Custom label

# Analysis flags:
#   --rdf               # Calculate RDF
#   --coordination      # Calculate coordination
#   --hbonds            # Analyze H-bonds
#   --all               # All analyses (default)
#   --r-max 15.0        # Max RDF radius (Å)
#   --r-bin 0.1         # RDF bin width (Å)
#   --skip 0            # Skip first N frames
#   --stride 1          # Use every Nth frame
#   --cutoff 3.5        # Coordination cutoff (Å)
#   --hb-distance 3.5   # H-bond distance (Å)
#   --hb-angle 30       # H-bond angle (degrees)
#   --no-plots          # Skip plotting
#   --output PREFIX     # Output file prefix
#   --verbose           # Verbose logging

# ═══════════════════════════════════════════════════════════
# PERFORMANCE ESTIMATES (500 waters, 8-16 cores)
# ═══════════════════════════════════════════════════════════
#
# 100 ps:   4-5 hours
# 1 ns:     10-12 hours  
# 5 ns:     2-3 days
# 10 ns:    5-7 days
#
# PPPM (electrostatics) = 99% of compute time
# Scaling good up to ~32 cores
# ═══════════════════════════════════════════════════════════
