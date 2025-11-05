#!/bin/bash
#
# Run TIP5P Water Solvation Simulation
#
# This script prepares and runs a molecular dynamics simulation of a SiC nanoparticle
# in TIP5P water with full electrostatics and rigid water constraints.
#
# Usage:
#   ./run_tip5p_simulation.sh [options]
#
# Options:
#   -n, --nwaters N      Number of water molecules (default: 3000)
#   -b, --boxsize SIZE   Box size in Å (default: auto-calculated)
#   -s, --strategy TYPE  Placement strategy: 'full_box' or 'shell' (default: full_box)
#   -t, --time TIME      Production time in ps (default: 500)
#   -c, --cores N        Number of MPI cores (default: 10)
#   -h, --help           Show this help message
#
# Author: Automated System
# Date: 2025-11-04

set -e  # Exit on error

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

N_WATERS=3000
BOX_SIZE=""
STRATEGY="full_box"
PROD_TIME_PS=500
N_CORES=10

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_DIR/input_files"
PYTHON_DIR="$PROJECT_DIR/python_scripts"
OUTPUT_BASE="$PROJECT_DIR/output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo -e "  $1"
}

show_help() {
    cat << EOF
Usage: ./run_tip5p_simulation.sh [options]

Run TIP5P water solvation simulation around SiC nanoparticle.

Options:
  -n, --nwaters N      Number of water molecules (default: 3000)
  -b, --boxsize SIZE   Box size in Å (default: auto-calculated from density)
  -s, --strategy TYPE  Placement: 'full_box' or 'shell' (default: full_box)
  -t, --time TIME      Production time in ps (default: 500)
  -c, --cores N        Number of MPI cores (default: 10)
  -h, --help           Show this help message

Examples:
  # Run with 5000 waters, auto box size
  ./run_tip5p_simulation.sh -n 5000

  # Run with 3000 waters in 50 Å box, using 20 cores
  ./run_tip5p_simulation.sh -n 3000 -b 50 -c 20

  # Run solvation shell only (not full box)
  ./run_tip5p_simulation.sh -n 2000 -s shell

  # Long production run (1000 ps)
  ./run_tip5p_simulation.sh -n 5000 -t 1000

Output:
  All output files are saved to: $OUTPUT_BASE/
  
  Directory structure:
    production_tip5p_<N>waters_<TIME>ps_<TIMESTAMP>/
    ├── production.lammpstrj          # Main trajectory
    ├── production_custom.dump        # Detailed trajectory
    ├── temperature.dat               # Temperature vs time
    ├── pressure.dat                  # Pressure vs time
    ├── energy.dat                    # Energy components
    ├── final_configuration.data      # Final structure
    ├── log.lammps                    # LAMMPS log file
    └── simulation_info.txt           # Simulation parameters

EOF
    exit 0
}

check_dependencies() {
    print_header "CHECKING DEPENDENCIES"
    
    local all_ok=true
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python 3 found: version $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
        all_ok=false
    fi
    
    # Check NumPy
    if python3 -c "import numpy" 2>/dev/null; then
        NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
        print_success "NumPy found: version $NUMPY_VERSION"
    else
        print_error "NumPy not found (pip install numpy)"
        all_ok=false
    fi
    
    # Check LAMMPS
    if command -v lmp &> /dev/null; then
        print_success "LAMMPS found: $(which lmp)"
    elif command -v lmp_mpi &> /dev/null; then
        print_success "LAMMPS MPI found: $(which lmp_mpi)"
    else
        print_error "LAMMPS not found (need 'lmp' or 'lmp_mpi' in PATH)"
        all_ok=false
    fi
    
    # Check MPI (if using multiple cores)
    if [ $N_CORES -gt 1 ]; then
        if command -v mpirun &> /dev/null; then
            print_success "MPI found: $(which mpirun)"
        else
            print_warning "MPI not found, will run in serial mode"
            N_CORES=1
        fi
    fi
    
    # Check input files
    if [ -f "$INPUT_DIR/sic_nanoparticle.data" ]; then
        print_success "Nanoparticle data file found"
    else
        print_error "Nanoparticle data file not found: $INPUT_DIR/sic_nanoparticle.data"
        all_ok=false
    fi
    
    if [ -f "$INPUT_DIR/solvation_tip5p.in" ]; then
        print_success "LAMMPS input script found"
    else
        print_error "LAMMPS input script not found: $INPUT_DIR/solvation_tip5p.in"
        all_ok=false
    fi
    
    if [ -f "$PYTHON_DIR/prepare_tip5p_system.py" ]; then
        print_success "Water placement script found"
    else
        print_error "Water placement script not found: $PYTHON_DIR/prepare_tip5p_system.py"
        all_ok=false
    fi
    
    if [ "$all_ok" = false ]; then
        print_error "Missing dependencies. Please install required software."
        exit 1
    fi
    
    echo ""
}

prepare_system() {
    print_header "PREPARING TIP5P WATER SYSTEM"
    
    print_info "Parameters:"
    print_info "  Water molecules: $N_WATERS"
    print_info "  Strategy: $STRATEGY"
    if [ -n "$BOX_SIZE" ]; then
        print_info "  Box size: $BOX_SIZE Å"
    else
        print_info "  Box size: auto-calculated"
    fi
    echo ""
    
    # Build command
    CMD="python3 $PYTHON_DIR/prepare_tip5p_system.py $N_WATERS"
    if [ -n "$BOX_SIZE" ]; then
        CMD="$CMD $BOX_SIZE"
    else
        CMD="$CMD auto"
    fi
    CMD="$CMD $STRATEGY"
    
    print_info "Running: $CMD"
    echo ""
    
    # Run water placement
    if $CMD; then
        print_success "System preparation complete"
    else
        print_error "System preparation failed"
        exit 1
    fi
    
    # Find generated data file
    DATA_FILE=$(ls -t "$INPUT_DIR"/tip5p_system_*waters.data 2>/dev/null | head -1)
    
    if [ -f "$DATA_FILE" ]; then
        print_success "Data file created: $(basename $DATA_FILE)"
        
        # Get actual number of waters placed
        ACTUAL_WATERS=$(grep "# System:" "$DATA_FILE" | grep -oP '\d+(?= TIP5P)')
        print_info "  Actually placed: $ACTUAL_WATERS water molecules"
    else
        print_error "Data file not created"
        exit 1
    fi
    
    echo ""
}

run_simulation() {
    print_header "RUNNING LAMMPS SIMULATION"
    
    # Create output directory with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$OUTPUT_BASE/production_tip5p_${ACTUAL_WATERS}waters_${PROD_TIME_PS}ps_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    
    print_success "Output directory: $OUTPUT_DIR"
    echo ""
    
    # Adjust timesteps based on production time
    # timestep = 0.5 fs, so steps = time_ps * 2000
    PROD_STEPS=$((PROD_TIME_PS * 2000))
    EQ_STEPS=100000  # 50 ps equilibration
    
    print_info "Simulation parameters:"
    print_info "  Equilibration: 50 ps ($EQ_STEPS steps)"
    print_info "  Production: $PROD_TIME_PS ps ($PROD_STEPS steps)"
    print_info "  Timestep: 0.5 fs"
    print_info "  MPI cores: $N_CORES"
    echo ""
    
    # Write simulation info
    cat > "$OUTPUT_DIR/simulation_info.txt" << EOF
TIP5P Water Solvation Simulation
Generated: $(date)

System Parameters:
  Water molecules: $ACTUAL_WATERS
  Strategy: $STRATEGY
  Box size: $(grep "xlo xhi" "$DATA_FILE" | awk '{print $2 - $1}') Å
  
Simulation Parameters:
  Equilibration: 50 ps ($EQ_STEPS steps)
  Production: $PROD_TIME_PS ps ($PROD_STEPS steps)
  Timestep: 0.5 fs
  Thermostat: Nosé-Hoover NVT
  Temperature: 300 K
  
Water Model:
  Type: TIP5P (5-site model)
  Sites: O, H, H, L, L (2 lone pairs)
  Charges: q_H = +0.241 e, q_L = -0.241 e, q_O = 0
  Electrostatics: PPPM Ewald (accuracy 1e-4)
  Constraints: SHAKE (O-H bonds and H-O-H angle)
  
Force Field:
  Pair style: lj/cut/tip5p/long
  LJ cutoff: 10.0 Å
  Coulomb cutoff: 10.0 Å
  Mixing rule: arithmetic
  
Computational:
  MPI cores: $N_CORES
  LAMMPS version: $(lmp -help 2>&1 | head -1 || echo "unknown")
  
Data File: $(basename $DATA_FILE)
Input Script: solvation_tip5p.in
EOF
    
    # Modify LAMMPS input to set correct production time
    # Create temporary input file with correct run steps
    TMP_INPUT="$OUTPUT_DIR/solvation_tip5p_tmp.in"
    sed "s/run             1000000/run             $PROD_STEPS/" "$INPUT_DIR/solvation_tip5p.in" > "$TMP_INPUT"
    
    # Determine LAMMPS command
    if command -v lmp_mpi &> /dev/null && [ $N_CORES -gt 1 ]; then
        LAMMPS_CMD="mpirun -np $N_CORES lmp_mpi"
    elif command -v lmp &> /dev/null; then
        LAMMPS_CMD="lmp"
    else
        print_error "LAMMPS executable not found"
        exit 1
    fi
    
    # Build LAMMPS command with variables
    LAMMPS_FULL_CMD="$LAMMPS_CMD -in $TMP_INPUT -var datafile $DATA_FILE -var outputdir $OUTPUT_DIR -log $OUTPUT_DIR/log.lammps"
    
    print_info "Starting LAMMPS..."
    print_info "Command: $LAMMPS_FULL_CMD"
    echo ""
    echo -e "${YELLOW}==================== LAMMPS OUTPUT ====================${NC}"
    echo ""
    
    # Run LAMMPS
    START_TIME=$(date +%s)
    
    if $LAMMPS_FULL_CMD; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        
        echo ""
        echo -e "${YELLOW}======================================================${NC}"
        echo ""
        print_success "Simulation completed successfully!"
        print_info "  Elapsed time: $ELAPSED seconds ($(($ELAPSED / 60)) minutes)"
    else
        print_error "Simulation failed!"
        print_info "Check log file: $OUTPUT_DIR/log.lammps"
        exit 1
    fi
    
    # Clean up temporary file
    rm -f "$TMP_INPUT"
    
    echo ""
}

analyze_results() {
    print_header "ANALYZING RESULTS"
    
    # Check if trajectory file exists and get statistics
    if [ -f "$OUTPUT_DIR/production.lammpstrj" ]; then
        TRAJ_SIZE=$(du -h "$OUTPUT_DIR/production.lammpstrj" | cut -f1)
        N_FRAMES=$(grep -c "ITEM: TIMESTEP" "$OUTPUT_DIR/production.lammpstrj" || echo "0")
        print_success "Trajectory file: $TRAJ_SIZE ($N_FRAMES frames)"
    else
        print_warning "Trajectory file not found"
    fi
    
    # Quick energy analysis
    if [ -f "$OUTPUT_DIR/log.lammps" ]; then
        print_info "Energy statistics:"
        
        # Extract energy from last 100 steps
        FINAL_TEMP=$(grep "^[0-9]" "$OUTPUT_DIR/log.lammps" | tail -100 | awk '{sum+=$2; count++} END {printf "%.2f", sum/count}')
        FINAL_PE=$(grep "^[0-9]" "$OUTPUT_DIR/log.lammps" | tail -100 | awk '{sum+=$6; count++} END {printf "%.2f", sum/count}')
        FINAL_ETOTAL=$(grep "^[0-9]" "$OUTPUT_DIR/log.lammps" | tail -100 | awk '{sum+=$4; count++} END {printf "%.2f", sum/count}')
        
        print_info "  Temperature: $FINAL_TEMP K (last 100 steps avg)"
        print_info "  Potential energy: $FINAL_PE kcal/mol"
        print_info "  Total energy: $FINAL_ETOTAL kcal/mol"
        
        # Calculate energy drift
        INITIAL_E=$(grep "^[0-9]" "$OUTPUT_DIR/log.lammps" | head -100 | awk '{sum+=$4; count++} END {print sum/count}')
        FINAL_E=$(grep "^[0-9]" "$OUTPUT_DIR/log.lammps" | tail -100 | awk '{sum+=$4; count++} END {print sum/count}')
        
        if [ -n "$INITIAL_E" ] && [ -n "$FINAL_E" ]; then
            DRIFT=$(python3 -c "print(abs(($FINAL_E - $INITIAL_E) / $INITIAL_E * 100))")
            print_info "  Energy drift: ${DRIFT}%"
            
            if (( $(echo "$DRIFT < 5.0" | bc -l) )); then
                print_success "  ✓ Energy drift within acceptable range (<5%)"
            else
                print_warning "  Energy drift higher than recommended"
            fi
        fi
    fi
    
    echo ""
    print_info "Output files in: $OUTPUT_DIR/"
    print_info "  production.lammpstrj       - Trajectory"
    print_info "  production_custom.dump     - Detailed trajectory"
    print_info "  temperature.dat            - Temperature time series"
    print_info "  pressure.dat               - Pressure time series"
    print_info "  energy.dat                 - Energy components"
    print_info "  final_configuration.data   - Final structure"
    print_info "  log.lammps                 - Full LAMMPS log"
    print_info "  simulation_info.txt        - Simulation parameters"
    
    echo ""
}

print_next_steps() {
    print_header "NEXT STEPS"
    
    cat << EOF
The simulation completed successfully! Here's what to do next:

1. Visualize trajectory with VMD:
   vmd $OUTPUT_DIR/production.lammpstrj

2. Analyze radial distribution functions (RDF):
   python3 $PYTHON_DIR/analyze_rdf.py $OUTPUT_DIR/production.lammpstrj

3. Analyze hydrogen bonding:
   python3 $PYTHON_DIR/analyze_hbonds.py $OUTPUT_DIR/production.lammpstrj

4. Analyze water orientation:
   python3 $PYTHON_DIR/analyze_orientation.py $OUTPUT_DIR/production.lammpstrj

5. Calculate coordination numbers:
   python3 $PYTHON_DIR/analyze_coordination.py $OUTPUT_DIR/production.lammpstrj

Note: Analysis scripts will be created next if they don't exist yet.

Documentation:
  - Setup guide: $PROJECT_DIR/docs/SETUP_GUIDE.md
  - Analysis guide: $PROJECT_DIR/docs/ANALYSIS_GUIDE.md
  - TIP5P parameters: $PROJECT_DIR/docs/TIP5P_PARAMETERS.md

EOF
}

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nwaters)
            N_WATERS="$2"
            shift 2
            ;;
        -b|--boxsize)
            BOX_SIZE="$2"
            shift 2
            ;;
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        -t|--time)
            PROD_TIME_PS="$2"
            shift 2
            ;;
        -c|--cores)
            N_CORES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# MAIN EXECUTION
# ============================================================================

clear
print_header "TIP5P WATER SOLVATION SIMULATION"

echo "Project directory: $PROJECT_DIR"
echo "Python scripts: $PYTHON_DIR"
echo "Input files: $INPUT_DIR"
echo "Output base: $OUTPUT_BASE"
echo ""

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Run workflow
check_dependencies
prepare_system
run_simulation
analyze_results
print_next_steps

print_success "All done! 🎉"
echo ""
