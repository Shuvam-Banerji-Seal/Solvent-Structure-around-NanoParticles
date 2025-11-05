#!/bin/bash
# Advanced TIP4P/2005 Simulation Runner with Full Control
# 
# Usage: ./run_solvation_study.sh [NP] [waters] [time] [cores] [options]
#
# Options:
#   --timestep FLOAT    Timestep in fs (default: 1.0)
#   --temp FLOAT        Temperature in K (default: 300.0)
#   --dump-freq INT     Dump frequency in steps (default: 1000)
#   --thermo-freq INT   Thermo output frequency (default: 500)
#   --box-size FLOAT    Box size in Angstroms (default: auto = 40 per axis)
#   --restart FILE      Restart from previous simulation
#   --equilibration INT Equilibration steps before production (default: 10000)
#   --production        Production run only (skip equilibration)
#   --ns                Use nanosecond time units instead of ps
#   --output-dir DIR    Custom output directory
#   --label LABEL       Custom label for output
#   --help              Show this help message

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
show_help() {
    head -18 "$0" | tail -17
    echo ""
    echo "Examples:"
    echo "  # Short test (100 ps)"
    echo "  $0 SiC 500 100 8"
    echo ""
    echo "  # Production run (5 nanoseconds)"
    echo "  $0 GO 1000 5 16 --ns --dump-freq 5000"
    echo ""
    echo "  # Continue from restart"
    echo "  $0 SiC 500 100 8 --restart output/final.restart"
    echo ""
    echo "  # Custom equilibration"
    echo "  $0 amorphous 500 1 16 --ns --equilibration 50000"
    exit 0
}

error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check required arguments
if [ $# -lt 4 ]; then
    show_help
fi

# Default parameters
NP_NAME="$1"
NUM_WATERS="$2"
RUN_TIME="$3"
NUM_CORES="$4"
TIMESTEP=1.0            # fs
TEMPERATURE=300.0       # K
DUMP_FREQ=1000          # steps
THERMO_FREQ=500         # steps
BOX_SIZE="auto"         # or specific value in Angstroms
RESTART_FILE=""
EQUILIBRATION_STEPS=10000
PRODUCTION_MODE=false
USE_NS=false
OUTPUT_DIR=""
CUSTOM_LABEL=""

# Parse optional flags
shift 4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timestep)
            TIMESTEP="$2"
            shift 2
            ;;
        --temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --dump-freq)
            DUMP_FREQ="$2"
            shift 2
            ;;
        --thermo-freq)
            THERMO_FREQ="$2"
            shift 2
            ;;
        --box-size)
            BOX_SIZE="$2"
            shift 2
            ;;
        --restart)
            RESTART_FILE="$2"
            shift 2
            ;;
        --equilibration)
            EQUILIBRATION_STEPS="$2"
            shift 2
            ;;
        --production)
            PRODUCTION_MODE=true
            shift
            ;;
        --ns)
            USE_NS=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --label)
            CUSTOM_LABEL="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Convert time to ps if ns specified
ORIGINAL_TIME=$RUN_TIME
if [ "$USE_NS" = true ]; then
    RUN_TIME=$((RUN_TIME * 1000))
    TIME_UNIT="ns"
    info "Converting ${ORIGINAL_TIME} ns to ${RUN_TIME} ps"
else
    TIME_UNIT="ps"
fi

# Setup directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$OUTPUT_DIR" ]; then
    if [ -n "$CUSTOM_LABEL" ]; then
        OUTPUT_DIR="${SCRIPT_DIR}/../output/${NP_NAME}_${NUM_WATERS}w_${ORIGINAL_TIME}${TIME_UNIT}_${CUSTOM_LABEL}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="${SCRIPT_DIR}/../output/${NP_NAME}_${NUM_WATERS}w_${ORIGINAL_TIME}${TIME_UNIT}_${TIMESTAMP}"
    fi
fi
INPUT_DIR="${SCRIPT_DIR}/../input_files"

# Validate CPU cores
AVAILABLE_CORES=$(nproc)
if [ "$NUM_CORES" -gt "$AVAILABLE_CORES" ]; then
    error_exit "Requested $NUM_CORES cores but only $AVAILABLE_CORES available. Use fewer cores."
fi

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Header
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     TIP4P/2005 Solvation Study - Advanced Runner          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Select and configure nanoparticle
info "Configuring nanoparticle..."

case "${NP_NAME,,}" in
    sic|si|silicon)
        NP_FILE="sic_nanoparticle.data"
        NP_LABEL="Silicon Carbide (SiC)"
        TYPE_O=3
        TYPE_H=4
        TYPE_M=5
        NP_TYPES="1 2"
        NP_DESCRIPTION="Small cubic SiC (8 atoms)"
        ;;
    go|graphene)
        NP_FILE="GO_nanoparticle.data"
        NP_LABEL="Graphene Oxide (GO)"
        TYPE_O=1
        TYPE_H=2
        TYPE_M=5
        NP_TYPES="3 4 5"
        NP_DESCRIPTION="Functionalized graphene sheet"
        ;;
    amorphous|carbon|am)
        NP_FILE="amorphous_carbon.data"
        NP_LABEL="Amorphous Carbon"
        TYPE_O=2
        TYPE_H=3
        TYPE_M=5
        NP_TYPES="1"
        NP_DESCRIPTION="Large amorphous carbon structure"
        ;;
    *)
        error_exit "Unknown nanoparticle: $NP_NAME"
        ;;
esac

# Auto-calculate box size if needed
if [ "$BOX_SIZE" = "auto" ]; then
    # Estimate: ~30 ų per water molecule
    VOLUME_NEEDED=$(echo "$NUM_WATERS * 30" | bc)
    BOX_SIZE=$(echo "scale=2; ($VOLUME_NEEDED)^(1/3)" | bc)
    
    # Ensure minimum size of 40 Å
    if (( $(echo "$BOX_SIZE < 40" | bc -l) )); then
        BOX_SIZE=40
    fi
    
    info "Auto-calculated box size: ${BOX_SIZE} Å (cube)"
fi

# Calculate total steps
TOTAL_STEPS=$((RUN_TIME * 1000 / ${TIMESTEP%.*}))
if [ "$PRODUCTION_MODE" = false ]; then
    PRODUCTION_STEPS=$((TOTAL_STEPS - EQUILIBRATION_STEPS))
else
    PRODUCTION_STEPS=$TOTAL_STEPS
    EQUILIBRATION_STEPS=0
fi

# Display configuration
echo ""
echo "Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Nanoparticle:    $NP_LABEL"
echo "  Description:     $NP_DESCRIPTION"
echo "  Water molecules: $NUM_WATERS (TIP4P/2005)"
echo "  Simulation time: $ORIGINAL_TIME $TIME_UNIT (${RUN_TIME} ps)"
echo "  Total steps:     $TOTAL_STEPS"
if [ "$PRODUCTION_MODE" = false ]; then
    echo "  Equilibration:   $EQUILIBRATION_STEPS steps ($(echo "scale=1; $EQUILIBRATION_STEPS * $TIMESTEP / 1000" | bc) ps)"
    echo "  Production:      $PRODUCTION_STEPS steps ($(echo "scale=1; $PRODUCTION_STEPS * $TIMESTEP / 1000" | bc) ps)"
fi
echo "  Timestep:        $TIMESTEP fs"
echo "  Temperature:     $TEMPERATURE K"
echo "  Box size:        $BOX_SIZE × $BOX_SIZE × $BOX_SIZE Å³"
echo "  CPU cores:       $NUM_CORES"
echo "  Output dir:      $(basename $OUTPUT_DIR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if NP file exists
if [ ! -f "${INPUT_DIR}/${NP_FILE}" ]; then
    error_exit "Nanoparticle file not found: ${INPUT_DIR}/${NP_FILE}"
fi

# Create LAMMPS input file
info "Generating LAMMPS input file..."

cat > simulation.in << EOF
# TIP4P/2005 Water Solvation Study
# Nanoparticle: $NP_LABEL
# Generated: $(date)
# Configuration: ${NUM_WATERS} waters, ${ORIGINAL_TIME} ${TIME_UNIT}

# ═══════════════════════════════════════════════════════════════
# SYSTEM SETUP
# ═══════════════════════════════════════════════════════════════

units           real
atom_style      full
bond_style      harmonic
angle_style     harmonic
boundary        p p p

# TIP4P/2005 pair style with long-range electrostatics
pair_style      lj/cut/tip4p/long $TYPE_O $TYPE_H 1 1 0.1546 12.0
kspace_style    pppm/tip4p  1.0e-5
pair_modify     mix arithmetic tail yes

# ═══════════════════════════════════════════════════════════════
# READ NANOPARTICLE STRUCTURE
# ═══════════════════════════════════════════════════════════════

read_data       $NP_FILE &
                extra/bond/per/atom 2 &
                extra/angle/per/atom 1 &
                extra/special/per/atom 3

# ═══════════════════════════════════════════════════════════════
# EXPAND BOX FOR WATER
# ═══════════════════════════════════════════════════════════════

variable box_min equal -${BOX_SIZE}/2
variable box_max equal ${BOX_SIZE}/2

change_box      all x final \${box_min} \${box_max} &
                    y final \${box_min} \${box_max} &
                    z final \${box_min} \${box_max} &
                    remap units box

# ═══════════════════════════════════════════════════════════════
# ADD WATER MOLECULES
# ═══════════════════════════════════════════════════════════════

molecule        h2o_mol H2O_TIP4P.txt

# Bond and angle coefficients for water
bond_coeff      1  1000.0  0.9572
angle_coeff     1  1000.0  104.52

# Insert water molecules randomly
create_atoms    0 random $NUM_WATERS 654321 NULL mol h2o_mol 545474 &
                overlap 2.0 maxtry 5000

# ═══════════════════════════════════════════════════════════════
# FORCE FIELD PARAMETERS
# ═══════════════════════════════════════════════════════════════

# TIP4P/2005 water parameters
pair_coeff      $TYPE_O $TYPE_O  0.1852  3.1589   # O-O
pair_coeff      $TYPE_H $TYPE_H  0.0     0.0      # H-H  
pair_coeff      $TYPE_M $TYPE_M  0.0     0.0      # M-M (virtual site)

# Nanoparticle parameters (generic LJ)
pair_coeff      1 1  0.40    3.826    # Type 1 (Si/C/etc)
pair_coeff      2 2  0.095   3.851    # Type 2

# ═══════════════════════════════════════════════════════════════
# ATOM GROUPS
# ═══════════════════════════════════════════════════════════════

group           nanoparticle type $NP_TYPES
group           water type $TYPE_O $TYPE_H $TYPE_M
group           water_O type $TYPE_O

# ═══════════════════════════════════════════════════════════════
# COMPUTE PROPERTIES FOR ANALYSIS
# ═══════════════════════════════════════════════════════════════

# RDF between NP and water oxygen
compute         rdf_np_water all rdf 200 1 $TYPE_O 2 $TYPE_O

# Temperature of each group
compute         temp_np nanoparticle temp
compute         temp_water water temp

# Center of mass of nanoparticle
compute         com_np nanoparticle com

# Coordination number (water O within 3.5 Å of NP)
compute         coord_water water_O coord/atom cutoff 3.5 group nanoparticle

# ═══════════════════════════════════════════════════════════════
# OUTPUT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Trajectory output
dump            traj all custom $DUMP_FREQ trajectory.lammpstrj &
                id mol type x y z vx vy vz

# Thermodynamic output
thermo          $THERMO_FREQ
thermo_style    custom step time temp press pe ke etotal vol density &
                c_temp_np c_temp_water

# RDF output (every N steps)
fix             rdf_output all ave/time 1000 1 1000 &
                c_rdf_np_water[*] file rdf_np_water.dat mode vector

# COM trajectory of nanoparticle
fix             com_output nanoparticle ave/time 100 1 100 &
                c_com_np[1] c_com_np[2] c_com_np[3] &
                file np_com.dat

# ═══════════════════════════════════════════════════════════════
# DYNAMICS SETUP
# ═══════════════════════════════════════════════════════════════

# SHAKE constraints for rigid water
fix             shake_water water shake 1.0e-4 200 0 b 1 a 1 mol h2o_mol

# NVT thermostat (Nose-Hoover)
fix             nvt_thermo all nvt temp $TEMPERATURE $TEMPERATURE 100.0

# Remove center-of-mass drift
fix             momentum_fix all momentum 500 linear 1 1 1

# Timestep
timestep        $TIMESTEP

# ═══════════════════════════════════════════════════════════════
# EQUILIBRATION PHASE
# ═══════════════════════════════════════════════════════════════

EOF

if [ "$PRODUCTION_MODE" = false ] && [ $EQUILIBRATION_STEPS -gt 0 ]; then
    cat >> simulation.in << EOF
print           ""
print           "════════════════════════════════════════"
print           " EQUILIBRATION PHASE"
print           "════════════════════════════════════════"
print           ""

run             $EQUILIBRATION_STEPS

# Save equilibration state
write_data      equilibrated.data
write_restart   equilibrated.restart

print           ""
print           "✓ Equilibration complete"
print           ""

# ═══════════════════════════════════════════════════════════════
# PRODUCTION PHASE
# ═══════════════════════════════════════════════════════════════

print           "════════════════════════════════════════"
print           " PRODUCTION PHASE"
print           "════════════════════════════════════════"
print           ""

EOF
fi

cat >> simulation.in << EOF
run             $PRODUCTION_STEPS

# ═══════════════════════════════════════════════════════════════
# SAVE FINAL STATE
# ═══════════════════════════════════════════════════════════════

write_data      final_config.data
write_restart   final.restart

print           ""
print           "════════════════════════════════════════"
print           " SIMULATION COMPLETE!"
print           "════════════════════════════════════════"
print           ""
print           "Output files:"
print           "  - trajectory.lammpstrj       Full trajectory"
print           "  - rdf_np_water.dat          Radial distribution function"
print           "  - np_com.dat                 NP center of mass trajectory"
print           "  - final_config.data          Final structure"
print           "  - final.restart              Restart file"
EOF

if [ "$PRODUCTION_MODE" = false ] && [ $EQUILIBRATION_STEPS -gt 0 ]; then
    cat >> simulation.in << EOF
print           "  - equilibrated.data          Equilibrated structure"
print           "  - equilibrated.restart       Equilibration restart"
EOF
fi

cat >> simulation.in << EOF
print           ""
EOF

success "LAMMPS input file created: simulation.in"

# Link required files
info "Linking input files..."
ln -sf "${INPUT_DIR}/H2O_TIP4P.txt" H2O_TIP4P.txt
ln -sf "${INPUT_DIR}/${NP_FILE}" ${NP_FILE}
success "Input files linked"

# Save simulation metadata
cat > simulation_info.txt << EOF
Simulation Information
======================
Date: $(date)
Nanoparticle: $NP_LABEL ($NP_FILE)
Waters: $NUM_WATERS
Simulation time: $ORIGINAL_TIME $TIME_UNIT ($RUN_TIME ps)
Total steps: $TOTAL_STEPS
Equilibration: $EQUILIBRATION_STEPS steps
Production: $PRODUCTION_STEPS steps
Timestep: $TIMESTEP fs
Temperature: $TEMPERATURE K
Box size: $BOX_SIZE Å
CPU cores: $NUM_CORES
Restart: ${RESTART_FILE:-none}
Output directory: $OUTPUT_DIR
EOF

success "Simulation metadata saved"

# Run LAMMPS
echo ""
info "Starting LAMMPS simulation..."
echo ""

START_TIME=$(date +%s)

if mpirun -np $NUM_CORES lmp_mpi -in simulation.in > lammps_output.log 2>&1; then
    END_TIME=$(date +%s)
    WALL_TIME=$((END_TIME - START_TIME))
    
    # Calculate performance
    HOURS=$((WALL_TIME / 3600))
    MINUTES=$(( (WALL_TIME % 3600) / 60 ))
    SECONDS=$((WALL_TIME % 60))
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              SIMULATION COMPLETED SUCCESSFULLY!            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    success "Simulation finished"
    echo ""
    echo "Performance:"
    echo "  Wall time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Simulated: $ORIGINAL_TIME $TIME_UNIT"
    echo "  Speed: $(echo "scale=3; $RUN_TIME * 1000 / $WALL_TIME" | bc) ps/s"
    echo ""
    echo "Output files:"
    ls -lh trajectory.lammpstrj final_config.data final.restart rdf_np_water.dat 2>/dev/null | awk '{print "  "$9": "$5}'
    echo ""
    echo "Next steps:"
    echo "  1. Visualize: vmd trajectory.lammpstrj"
    echo "  2. Analyze: cd ../analysis && python analyze_tip4p_simple.py $OUTPUT_DIR"
    echo "  3. RDF plot: plot rdf_np_water.dat"
    echo ""
else
    END_TIME=$(date +%s)
    WALL_TIME=$((END_TIME - START_TIME))
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  SIMULATION FAILED!                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    error_exit "LAMMPS simulation failed. Check lammps_output.log for details"
fi
