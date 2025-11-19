#!/bin/bash
# Restart all 6 epsilon simulations from pressure ramp checkpoint
# With CPU core pinning (taskset) to match run_parallel_equilibration_v2.sh

# Array of epsilon values
epsilons=(0.0 0.05 0.10 0.15 0.20 0.25)

# Random seed (same for all for reproducibility)
RANDOM_SEED=42

# CPU core allocation per simulation
OMP_THREADS_PER_JOB=4  # 4 cores per epsilon

echo "════════════════════════════════════════════════════════════════════"
echo "RESTARTING SIMULATIONS FROM PRESSURE RAMP CHECKPOINT"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "All 6 epsilon simulations will restart from step 100,010"
echo "This will complete:"
echo "  - NPT Equilibration Phase 1 (400 ps)"
echo "  - NPT Equilibration Phase 2 (600 ps)"
echo "  - Production MD (4000 ps)"
echo ""
echo "CPU Core Allocation (with taskset pinning):"
echo "  Epsilon 0.0   → Cores c0-c3"
echo "  Epsilon 0.05  → Cores c4-c7"
echo "  Epsilon 0.10  → Cores c8-c11"
echo "  Epsilon 0.15  → Cores c12-c15"
echo "  Epsilon 0.20  → Cores c16-c19"
echo "  Epsilon 0.25  → Cores c20-c23"
echo ""
echo "GPU Configuration: 1 GPU (A100) shared by all 6 simulations"
echo "  GPU suffix:       -sf gpu"
echo "  GPU package:      -pk gpu 1"
echo "  GPU binding:      Serial access (each epsilon serialized)"
echo ""
echo "With FULL GPU acceleration (pppm/gpu, npt/gpu):"
echo "  Expected time: ~3-4 hours for full completion (5× faster than CPU)"
echo "  GPU Utilization: ~80-90%"
echo ""
echo "════════════════════════════════════════════════════════════════════"

echo ""
echo "Starting all 6 simulations in parallel..."
echo ""

# Start each simulation in background with core pinning
job_count=0
for eps in "${epsilons[@]}"; do
    # Calculate CPU core range for this job
    start_core=$((job_count * OMP_THREADS_PER_JOB))
    end_core=$((start_core + OMP_THREADS_PER_JOB - 1))
    
    echo "  → Starting epsilon=${eps} (Cores c${start_core}-c${end_core})..."
    
    cd epsilon_${eps}
    
    # Set environment for this job
    export OMP_NUM_THREADS=$OMP_THREADS_PER_JOB
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads
    
    # Run LAMMPS with GPU acceleration and CPU core pinning
    # Using taskset to pin each epsilon to specific 4 cores
    taskset -c $start_core-$end_core /opt/lammps/bin/lmp_mpi \
        -pk gpu 1 neigh yes newton off binsize 2.8 split 1.0 \
        -sf gpu \
        -var EPSILON_CO ${eps} \
        -var RANDOM_SEED ${RANDOM_SEED} \
        -in ../3_restart_from_pressure_ramp.lmp \
        -log restart_equilibration.log \
        > restart_run.out 2>&1 &
    
    cd ..
    ((job_count++))
    
    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✓ All 6 simulations launched!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress with:"
echo "  tail -f epsilon_*/restart_equilibration.log"
echo ""
echo "Check active simulations:"
echo "  ps aux | grep lmp_mpi | grep -v grep"
echo ""
echo "Monitor GPU utilization (should see 80-90%):"
echo "  nvidia-smi"
echo ""
echo "Total runtime should be ~3-4 hours per epsilon with full GPU acceleration"
echo "════════════════════════════════════════════════════════════════════"
