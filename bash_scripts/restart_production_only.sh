#!/bin/bash
# Restart Production MD Only (from NPT checkpoint)
# Use this if production MD crashes but NPT equilibration completed successfully
# This saves ~1200 ps of computation time

echo "════════════════════════════════════════════════════════════════════════════"
echo "RESTARTING PRODUCTION MD ONLY (from NPT checkpoint)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This script will:"
echo "  • Resume from npt_equilibration_complete.restart"
echo "  • Skip NPT equilibration (already complete)"
echo "  • Run only production MD (4000 ps)"
echo ""
echo "Starting all 6 epsilon values in parallel..."
echo ""

cd /store/shuvam/solvent_effects/6ns_sim

# Check if NPT checkpoints exist
missing_checkpoints=0
for eps in 0.0 0.05 0.10 0.15 0.20 0.25; do
    if [ ! -f "epsilon_${eps}/npt_equilibration_complete.restart" ]; then
        echo "⚠️  WARNING: epsilon_${eps}/npt_equilibration_complete.restart not found!"
        missing_checkpoints=1
    fi
done

if [ $missing_checkpoints -eq 1 ]; then
    echo ""
    echo "❌ ERROR: Some NPT checkpoints are missing!"
    echo "   Use restart_all_simulations.sh instead to restart from pressure_ramp"
    exit 1
fi

echo "✓ All NPT checkpoints found"
echo ""

# Launch all 6 epsilon values in parallel
for eps in 0.0 0.05 0.10 0.15 0.20 0.25; do
    echo "Launching epsilon = ${eps}..."
    cd epsilon_${eps}
    
    # Run with GPU acceleration in background
    /opt/lammps/bin/lmp_mpi \
        -pk gpu 1 neigh yes newton off binsize 2.8 split 1.0 \
        -sf gpu \
        -var EPSILON_CO ${eps} \
        -var RANDOM_SEED 42 \
        -in ../4_restart_production_only.lmp \
        -log production_restart.log &
    
    cd ..
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "All simulations launched!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress:"
echo "  tail -f epsilon_*/production_restart.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep lmp_mpi | grep -v grep"
echo ""
echo "Expected completion time: ~14-16 hours (4000 ps at ~40 ts/s)"
echo ""
