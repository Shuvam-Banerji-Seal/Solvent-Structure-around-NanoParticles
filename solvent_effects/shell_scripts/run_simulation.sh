#!/bin/bash
# Quick Start: Run TIP4P/2005 Simulation
# This script automates the complete workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/../output"
INPUT_DIR="${SCRIPT_DIR}/../input_files"

# Create output directory
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  TIP4P/2005 Solvation Study - Quick Start  ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Configuration
NP_NAME="${1:-SiC}"  # SiC, GO, or amorphous_carbon
NUM_WATERS="${2:-500}"
RUN_TIME="${3:-100}"  # ps
NUM_CORES="${4:-8}"

echo "Configuration:"
echo "  Nanoparticle: $NP_NAME"
echo "  Waters: $NUM_WATERS"
echo "  Run time: $RUN_TIME ps"
echo "  CPU cores: $NUM_CORES"
echo ""

# Select nanoparticle
case "$NP_NAME" in
    SiC|sic)
        NP_FILE="sic_nanoparticle.data"
        NP_LABEL="Silicon Carbide"
        # For SiC: types 1=Si, 2=C, 3=O, 4=H, 5=M
        TYPE_O=3
        TYPE_H=4
        NP_TYPES="1 2"
        ;;
    GO|go)
        NP_FILE="GO_nanoparticle.data"
        NP_LABEL="Graphene Oxide"
        # For GO: different type mapping - will detect from data file
        TYPE_O=1
        TYPE_H=2
        NP_TYPES="3 4 5"
        ;;
    amorphous|carbon|am)
        NP_FILE="amorphous_carbon.data"
        NP_LABEL="Amorphous Carbon"
        # For AC: different type mapping
        TYPE_O=2
        TYPE_H=3
        NP_TYPES="1"
        ;;
    *)
        echo "Usage: $0 [SiC|GO|amorphous] [num_waters] [time_ps] [cores]"
        echo "Examples:"
        echo "  $0 SiC 500 100 8      # SiC + 500 waters, 100 ps, 8 cores"
        echo "  $0 GO 1000 500 16     # GO + 1000 waters, 500 ps, 16 cores"
        exit 1
        ;;
esac

echo "Nanoparticle: $NP_LABEL"
echo ""

# Create LAMMPS input file
echo "Step 1: Creating LAMMPS input file..."

cat > simulation.in << EOF
# TIP4P/2005 Water + $NP_LABEL Solvation
# Auto-generated for $NUM_WATERS waters, $RUN_TIME ps

units           real
atom_style      full
bond_style      harmonic
angle_style     harmonic
boundary        p p p

pair_style      lj/cut/tip4p/long $TYPE_O $TYPE_H 1 1 0.1546 12.0
kspace_style    pppm/tip4p  1.0e-5
pair_modify     mix arithmetic tail yes

# Read nanoparticle with extra space for water bonds/angles
read_data       $NP_FILE &
                extra/bond/per/atom 2 &
                extra/angle/per/atom 1 &
                extra/special/per/atom 3

# Expand box to fit water molecules
# Estimate needed box size: ~30 Å³ per water molecule
change_box      all x final -40 40 y final -40 40 z final -40 40 remap units box

# Molecule template for water
molecule        h2o_mol H2O_TIP4P.txt

# Bond/angle coefficients
bond_coeff      1  1000.0  0.9572
angle_coeff     1  1000.0  104.52

# Pair coefficients - TIP4P water (O-O only)
pair_coeff      $TYPE_O $TYPE_O  0.1852  3.1589   # O-O TIP4P/2005
pair_coeff      $((TYPE_H)) $((TYPE_H))  0.0     0.0      # H-H
pair_coeff      5 5  0.0     0.0      # M-M

# NP parameters (SiC example - adjust for your NP)
pair_coeff      1 1  0.40    3.826    # Type 1 (Si/C/etc)
pair_coeff      2 2  0.095   3.851    # Type 2

# Water insertion
create_atoms    0 random $NUM_WATERS 654312 NULL mol h2o_mol 545474 overlap 2.0 maxtry 2000

# Groups
group           nanoparticle type $NP_TYPES
group           water type $TYPE_O $TYPE_H 5

# Output
dump            1 all custom 1000 trajectory.lammpstrj id mol type x y z
thermo          500
thermo_style    custom step temp pe ke etotal press vol density

# SHAKE for rigid water
fix             myshk water shake 1.0e-4 200 0 b 1 a 1 mol h2o_mol
fix             mynvt all nvt temp 300.0 300.0 100.0
fix             mymom all momentum 500 linear 1 1 1

timestep        1.0

# Run simulation
run             $(($RUN_TIME * 1000))

# Save final state
write_data      final_config.data
write_restart   final.restart

print           ""
print           "Simulation complete!"
print           "Output files:"
print           "  - trajectory.lammpstrj (full trajectory)"
print           "  - final_config.data (final structure)"
print           "  - final.restart (for continuation)"
print           ""
EOF

echo "  Created: simulation.in"
echo ""

# Copy input files
echo "Step 2: Linking input files..."
ln -sf "${INPUT_DIR}/H2O_TIP4P.txt" H2O_TIP4P.txt
ln -sf "${INPUT_DIR}/${NP_FILE}" ${NP_FILE}
echo "  Linked: H2O_TIP4P.txt, $NP_FILE"
echo ""

# Run LAMMPS
echo "Step 3: Running LAMMPS simulation..."
echo "  Command: mpirun -np $NUM_CORES lmp_mpi -in simulation.in"
echo ""

START_TIME=$(date +%s)

mpirun -np $NUM_CORES lmp_mpi -in simulation.in 2>&1 | tee simulation.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  Simulation Complete!                      ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  Total time: $(($ELAPSED / 60)) min $(($ELAPSED % 60)) sec"
echo "  Performance: ~$((($ELAPSED) / $RUN_TIME)) sec per ps"
echo ""
echo "Output files in: $WORK_DIR"
ls -lh trajectory.lammpstrj final_config.data final.restart 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Visualize:  vmd trajectory.lammpstrj"
echo "  2. Analyze RDF and hydration structure (see analysis scripts)"
echo "  3. Run other nanoparticles for comparison"
echo ""
