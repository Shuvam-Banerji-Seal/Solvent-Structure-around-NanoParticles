#!/bin/bash
# TIP4P/2005 Simulation using create_atoms approach (like water-co2 example)
# This avoids data file bond/angle issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/../input_files"

cd "${INPUT_DIR}"

echo "Creating TIP4P/2005 system using LAMMPS create_atoms..."

# Create LAMMPS input that builds the system
cat > tip4p_create_system.in << 'EOF'
# TIP4P/2005 Water + Nanoparticle using create_atoms
# Based on water-co2 example

units           real
atom_style      full
boundary        p p p

# Create simulation box (40 Angstroms cubic)
region          box block -20 20 -20 20 -20 20
create_box      5 box &
                bond/types 1 &
                angle/types 1 &
                extra/bond/per/atom 2 &
                extra/angle/per/atom 1 &
                extra/special/per/atom 3

# Force field
pair_style      lj/cut/tip4p/long 3 4 1 1 0.1546 12.0
pair_modify     tail yes mix arithmetic

# TIP4P/2005 parameters
pair_coeff      3 3  0.1852  3.1589   # O-O
pair_coeff      4 4  0.0     0.0      # H-H
pair_coeff      5 5  0.0     0.0      # M-M

# Nanoparticle parameters (SiC)
pair_coeff      1 1  0.40    3.826    # Si-Si
pair_coeff      2 2  0.095   3.851    # C-C

kspace_style    pppm/tip4p  1.0e-5

bond_style      harmonic
bond_coeff      1  1000.0  0.9572

angle_style     harmonic
angle_coeff     1  1000.0  104.52

# Masses
mass            1  28.0855  # Si
mass            2  12.0107  # C
mass            3  15.9994  # O (water)
mass            4  1.008    # H (water)
mass            5  0.0001   # M (virtual site)

# Add nanoparticle atoms manually (SiC, 8 atoms at origin)
create_atoms    1 single 0.0 0.0 0.0 units box
create_atoms    2 single 1.0 0.0 0.0 units box
create_atoms    1 single 0.0 1.0 0.0 units box
create_atoms    2 single 1.0 1.0 0.0 units box
create_atoms    1 single 0.0 0.0 1.0 units box
create_atoms    2 single 1.0 0.0 1.0 units box
create_atoms    1 single 0.0 1.0 1.0 units box
create_atoms    2 single 1.0 1.0 1.0 units box

# Insert water molecules
molecule        h2o_mol H2O_TIP4P.txt
create_atoms    0 random 100 12345 NULL mol h2o_mol 54321 overlap 2.0 maxtry 2000

# Groups
group           nanoparticle type 1 2
group           water type 3 4 5

# SHAKE
fix             myshk water shake 1.0e-4 200 0 b 1 a 1 mol h2o_mol

# Output
thermo          50
thermo_style    custom step temp pe ke etotal press vol density

timestep        1.0

# Minimization
minimize        1.0e-4 1.0e-6 1000 10000

write_data      tip4p_created_system.data
print           "System created and minimized!"

# Short NVT test (10 ps)
reset_timestep  0
velocity        all create 300.0 12345
fix             mynvt all nvt temp 300.0 300.0 100.0
dump            1 all custom 500 tip4p_create_test.lammpstrj id mol type x y z
run             10000

write_data      tip4p_create_final.data
print           "Test complete!"
EOF

echo "Running LAMMPS..."
mpirun -np 4 lmp_mpi -in tip4p_create_system.in 2>&1 | tail -60

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ SUCCESS!"
    echo "=========================================="
    ls -lh tip4p_create*.* 2>/dev/null
else
    echo ""
    echo "Failed - see output above"
    exit 1
fi
EOF

chmod +x "${SCRIPT_DIR}/test_tip4p_create.sh"
bash "${SCRIPT_DIR}/test_tip4p_create.sh"
