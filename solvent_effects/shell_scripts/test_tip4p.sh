#!/bin/bash
# Quick Test Script for TIP4P/2005 Water + Nanoparticle
# Tests the complete workflow with a small system

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/../input_files"
PYTHON_DIR="${SCRIPT_DIR}/../python_scripts"

echo "=========================================="
echo "  TIP4P/2005 Quick Test"
echo "=========================================="
echo ""

# Default parameters
NP_FILE="sic_nanoparticle.data"
NUM_WATERS=100
BOX_SIZE="auto"
NUM_CORES=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nanoparticle)
            NP_FILE="$2"
            shift 2
            ;;
        -w|--waters)
            NUM_WATERS="$2"
            shift 2
            ;;
        -b|--box)
            BOX_SIZE="$2"
            shift 2
            ;;
        -c|--cores)
            NUM_CORES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-n nanoparticle.data] [-w num_waters] [-b box_size] [-c cores]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Nanoparticle: ${NP_FILE}"
echo "  Waters: ${NUM_WATERS}"
echo "  Box size: ${BOX_SIZE}"
echo "  CPU cores: ${NUM_CORES}"
echo ""

# Step 1: Generate system
echo "Step 1: Generating TIP4P system..."
cd "${PYTHON_DIR}"
python3 prepare_tip4p_system.py "${NP_FILE}" "${NUM_WATERS}" "${BOX_SIZE}"

if [ $? -ne 0 ]; then
    echo "ERROR: System generation failed!"
    exit 1
fi

# Find the generated file
SYSTEM_FILE=$(ls -t "${INPUT_DIR}"/tip4p_system_*waters.data 2>/dev/null | head -1)
if [ -z "${SYSTEM_FILE}" ]; then
    echo "ERROR: Could not find generated system file!"
    exit 1
fi

echo "  Created: ${SYSTEM_FILE}"
echo ""

# Step 2: Create short test input file
echo "Step 2: Creating test input file..."
cd "${INPUT_DIR}"

cat > tip4p_test_quick.in << 'EOF'
# Quick TIP4P/2005 Test (10 ps)
units           real
atom_style      full
boundary        p p p

read_data       tip4p_system.data
molecule        h2o_mol H2O_TIP4P.txt

# Adjust type numbers based on nanoparticle!
# For SiC: 1=Si, 2=C, 3=O(water), 4=H(water), 5=M(water)
pair_style      lj/cut/tip4p/long 3 4 1 1 0.1546 12.0
pair_modify     tail yes

pair_coeff      3 3  0.1852  3.1589   # TIP4P/2005 O-O
pair_coeff      4 4  0.0     0.0      # H-H
pair_coeff      5 5  0.0     0.0      # M-M
pair_coeff      1 1  0.40    3.826    # Si-Si
pair_coeff      2 2  0.095   3.851    # C-C

kspace_style    pppm/tip4p  1.0e-4

bond_style      harmonic
bond_coeff      1  1000.0  0.9572

angle_style     harmonic
angle_coeff     1  1000.0  104.52

group           water type 3 4 5
fix             myshk water shake 1.0e-4 200 0 b 1 a 1 mol h2o_mol

thermo          50
thermo_style    custom step temp pe ke etotal press vol density

timestep        1.0

# Quick minimization
minimize        1.0e-4 1.0e-6 100 1000

# Short NVT run (10 ps)
reset_timestep  0
velocity        all create 300.0 12345
fix             mynvt all nvt temp 300.0 300.0 100.0
dump            1 all custom 500 tip4p_test.lammpstrj id mol type x y z
run             10000

write_data      tip4p_test_final.data
print           "Test complete!"
EOF

# Link the system file
ln -sf "$(basename ${SYSTEM_FILE})" tip4p_system.data

echo "  Input file: tip4p_test_quick.in"
echo ""

# Step 3: Run LAMMPS
echo "Step 3: Running LAMMPS test..."
echo "  Command: mpirun -np ${NUM_CORES} lmp_mpi -in tip4p_test_quick.in"
echo ""

timeout 300 mpirun -np "${NUM_CORES}" lmp_mpi -in tip4p_test_quick.in 2>&1 | tail -50

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ TEST SUCCESSFUL!"
    echo "=========================================="
    echo ""
    echo "Output files created:"
    ls -lh tip4p_test*.* 2>/dev/null | head -10
    echo ""
    echo "Next steps:"
    echo "  1. Run full simulation: cd ../shell_scripts && ./run_tip4p_simulation.sh"
    echo "  2. Visualize: vmd tip4p_test.lammpstrj"
    echo "  3. Try other nanoparticles (GO, amorphous carbon)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  ✗ TEST FAILED"
    echo "=========================================="
    echo ""
    echo "Check the output above for errors."
    echo "Common issues:"
    echo "  - Adjust pair_coeff type numbers for your nanoparticle"
    echo "  - Check if H2O_TIP4P.txt exists in input_files/"
    echo "  - Verify lmp_mpi is available"
    echo ""
    exit 1
fi
