#!/bin/bash
# Batch Simulation Runner for Solvation Studies
# 
# Run multiple simulations in sequence with different parameters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_solvation_study.sh"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Usage
if [ $# -lt 1 ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all-nps           Run all nanoparticles (SiC, GO, amorphous)"
    echo "  --nps LIST          Comma-separated list (e.g., SiC,GO)"
    echo "  --waters N          Number of water molecules (default: 500)"
    echo "  --time T            Simulation time (default: 100)"
    echo "  --ns                Use nanosecond units"
    echo "  --cores N           CPU cores (default: 8)"
    echo "  --test              Quick test mode (100 ps)"
    echo "  --production        Production mode (5 ns)"
    echo "  --custom-args ARGS  Additional args to pass to runner"
    echo ""
    echo "Examples:"
    echo "  # Test all nanoparticles"
    echo "  $0 --all-nps --test"
    echo ""
    echo "  # Production runs"
    echo "  $0 --all-nps --production --cores 16"
    echo ""
    echo "  # Custom: SiC and GO with 1000 waters, 1 ns"
    echo "  $0 --nps SiC,GO --waters 1000 --time 1 --ns --cores 16"
    exit 0
fi

# Defaults
NANOPARTICLES=()
NUM_WATERS=500
RUN_TIME=100
USE_NS=""
NUM_CORES=8
CUSTOM_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-nps)
            NANOPARTICLES=(SiC GO amorphous)
            shift
            ;;
        --nps)
            IFS=',' read -ra NANOPARTICLES <<< "$2"
            shift 2
            ;;
        --waters)
            NUM_WATERS="$2"
            shift 2
            ;;
        --time)
            RUN_TIME="$2"
            shift 2
            ;;
        --ns)
            USE_NS="--ns"
            shift
            ;;
        --cores)
            NUM_CORES="$2"
            shift 2
            ;;
        --test)
            RUN_TIME=100
            USE_NS=""
            shift
            ;;
        --production)
            RUN_TIME=5
            USE_NS="--ns"
            shift
            ;;
        --custom-args)
            CUSTOM_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ ${#NANOPARTICLES[@]} -eq 0 ]; then
    echo "Error: No nanoparticles specified. Use --all-nps or --nps LIST"
    exit 1
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Batch Solvation Study Runner                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Nanoparticles: ${NANOPARTICLES[*]}"
echo "  Waters: ${NUM_WATERS}"
if [ -n "$USE_NS" ]; then
    echo "  Time: ${RUN_TIME} ns"
else
    echo "  Time: ${RUN_TIME} ps"
fi
echo "  Cores: ${NUM_CORES}"
if [ -n "$CUSTOM_ARGS" ]; then
    echo "  Custom args: ${CUSTOM_ARGS}"
fi
echo ""
echo "Total simulations: ${#NANOPARTICLES[@]}"
echo ""

# Confirm
read -p "Proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Run simulations
TOTAL_START=$(date +%s)
FAILED=()
SUCCEEDED=()

for i in "${!NANOPARTICLES[@]}"; do
    NP="${NANOPARTICLES[$i]}"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    info "Starting simulation $((i+1))/${#NANOPARTICLES[@]}: ${NP}"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    SIM_START=$(date +%s)
    
    # Build command
    CMD="$RUNNER $NP $NUM_WATERS $RUN_TIME $NUM_CORES $USE_NS $CUSTOM_ARGS"
    
    info "Command: $CMD"
    echo ""
    
    # Run simulation
    if $CMD; then
        SIM_END=$(date +%s)
        SIM_TIME=$((SIM_END - SIM_START))
        
        HOURS=$((SIM_TIME / 3600))
        MINUTES=$(( (SIM_TIME % 3600) / 60 ))
        SECONDS=$((SIM_TIME % 60))
        
        success "Simulation complete: ${NP} (${HOURS}h ${MINUTES}m ${SECONDS}s)"
        SUCCEEDED+=("$NP")
    else
        echo "ERROR: Simulation failed: ${NP}"
        FAILED+=("$NP")
    fi
    
    echo ""
done

# Summary
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(( (TOTAL_TIME % 3600) / 60 ))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                BATCH RUN COMPLETE                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Succeeded: ${#SUCCEEDED[@]}/${#NANOPARTICLES[@]}"

if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    echo "  ✓ ${SUCCEEDED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED[@]}"
    echo "  ✗ ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Analyze each simulation:"
echo "     cd ../analysis"
echo "     python analyze_solvation_advanced.py ../output/NP_NAME_*"
echo ""
echo "  2. Compare results across NPs"
echo ""

exit 0
