#!/bin/bash
# Restart all 12 epsilon simulations from pressure ramp checkpoint
# With CPU core pinning (taskset) to match run_parallel_equilibration_v2.sh

# Array of epsilon values - extended range for better hydrophilicity sampling
epsilons=(0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0 1.05 1.10)

# Random seed (same for all for reproducibility)
RANDOM_SEED=42

# CPU core allocation per simulation
OMP_THREADS_PER_JOB=5  # 5 cores per epsilon (12 jobs x 5 cores = 60 cores)

echo "════════════════════════════════════════════════════════════════════"
echo "RESTARTING SIMULATIONS FROM PRESSURE RAMP CHECKPOINT"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "All 12 epsilon simulations will restart from step 100,010"
echo "This will complete:"
echo "  - NPT Equilibration Phase 1 (400 ps)"
echo "  - NPT Equilibration Phase 2 (600 ps)"
echo "  - Production MD (4000 ps)"
echo ""
echo "Epsilon Range: 0.55 - 1.10 kcal/mol (hydrophilic C60-water interactions)"
echo "This samples the hydrophilic regime where C60-water attraction increases"
echo ""
echo "CPU Core Allocation (with taskset pinning):"
echo "  Epsilon 0.55  → Cores c0-c4   (5 cores)"
echo "  Epsilon 0.60  → Cores c5-c9   (5 cores)"
echo "  Epsilon 0.65  → Cores c10-c14 (5 cores)"
echo "  Epsilon 0.70  → Cores c15-c19 (5 cores)"
echo "  Epsilon 0.75  → Cores c20-c24 (5 cores)"
echo "  Epsilon 0.80  → Cores c25-c29 (5 cores)"
echo "  Epsilon 0.85  → Cores c30-c34 (5 cores)"
echo "  Epsilon 0.90  → Cores c35-c39 (5 cores)"
echo "  Epsilon 0.95  → Cores c40-c44 (5 cores)"
echo "  Epsilon 1.00  → Cores c45-c49 (5 cores)"
echo "  Epsilon 1.05  → Cores c50-c54 (5 cores)"
echo "  Epsilon 1.10  → Cores c55-c59 (5 cores)"
echo "Total: 60 cores (of 64 available)"
echo ""
echo "GPU Configuration: 1 GPU (RTX 6000) shared by all 12 simulations"
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
echo "✓ All 12 simulations launched!"
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
