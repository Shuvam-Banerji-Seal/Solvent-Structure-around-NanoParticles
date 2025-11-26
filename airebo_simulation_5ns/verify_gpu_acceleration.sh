#!/bin/bash
#
# Verify GPU acceleration capabilities in LAMMPS
# This script checks if pppm/gpu and npt/gpu are available
#

LAMMPS_BIN="/opt/lammps/bin/lmp_mpi"

echo "════════════════════════════════════════════════════════════════════════════"
echo "LAMMPS GPU Acceleration Verification"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if LAMMPS binary exists
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "❌ ERROR: LAMMPS binary not found: $LAMMPS_BIN"
    exit 1
fi

echo "LAMMPS Binary: $LAMMPS_BIN"
echo ""

# Get LAMMPS version
echo "Version Info:"
$LAMMPS_BIN -h 2>/dev/null | head -1
echo ""

# Check for GPU pair styles
echo "GPU Pair Styles Available:"
$LAMMPS_BIN -h 2>/dev/null | grep "lj/cut/tip4p/long/gpu"
if [ $? -eq 0 ]; then
    echo "  ✅ lj/cut/tip4p/long/gpu found"
else
    echo "  ❌ lj/cut/tip4p/long/gpu NOT found"
fi
echo ""

# Check for GPU Coulomb solver
echo "GPU Coulomb Solvers Available:"
$LAMMPS_BIN -h 2>/dev/null | grep "pppm/gpu" | head -3
if [ $? -eq 0 ]; then
    echo "  ✅ pppm/gpu found - FULL GPU Coulomb acceleration!"
else
    echo "  ❌ pppm/gpu NOT found - Coulomb still on CPU"
    echo "  Note: pppm/tip4p available but runs on CPU (bottleneck)"
fi
echo ""

# Check for GPU integrators
echo "GPU Integrators Available:"
$LAMMPS_BIN -h 2>/dev/null | grep "npt/gpu\|nvt/gpu\|nve/gpu"
if $LAMMPS_BIN -h 2>/dev/null | grep -q "npt/gpu"; then
    echo "  ✅ npt/gpu found - Full GPU NPT integration!"
else
    echo "  ❌ npt/gpu NOT found - NPT still on CPU (bottleneck)"
fi
if $LAMMPS_BIN -h 2>/dev/null | grep -q "nvt/gpu"; then
    echo "  ✅ nvt/gpu found"
else
    echo "  ❌ nvt/gpu NOT found"
fi
if $LAMMPS_BIN -h 2>/dev/null | grep -q "nve/gpu"; then
    echo "  ✅ nve/gpu found"
else
    echo "  ❌ nve/gpu NOT found"
fi
echo ""

# Check for GPU package
echo "GPU Package Configuration:"
$LAMMPS_BIN -h 2>/dev/null | grep -i "gpu" | grep -i "suffix" | head -1
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════════════════"
echo "GPU Acceleration Capability Summary"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

has_pair_gpu=0
has_pppm_gpu=0
has_npt_gpu=0

$LAMMPS_BIN -h 2>/dev/null | grep -q "lj/cut/tip4p/long/gpu" && has_pair_gpu=1
$LAMMPS_BIN -h 2>/dev/null | grep -q "pppm/gpu" && has_pppm_gpu=1
$LAMMPS_BIN -h 2>/dev/null | grep -q "npt/gpu" && has_npt_gpu=1

if [ $has_pair_gpu -eq 1 ] && [ $has_pppm_gpu -eq 1 ] && [ $has_npt_gpu -eq 1 ]; then
    echo "✅✅✅ FULL GPU ACCELERATION AVAILABLE!"
    echo ""
    echo "All required components for complete GPU acceleration are available:"
    echo "  • lj/cut/tip4p/long/gpu  (LJ pair forces)"
    echo "  • pppm/gpu                (Coulomb forces) ← CRITICAL"
    echo "  • npt/gpu                 (Integration)    ← CRITICAL"
    echo ""
    echo "Expected GPU Utilization: 80-90%"
    echo "Expected Speedup: 5-6× vs CPU-only"
    echo "Estimated Runtime for 5 ns: ~3-4 hours (vs ~18-20 hours)"
    echo ""
    exit 0
else
    echo "⚠️  PARTIAL GPU ACCELERATION"
    echo ""
    echo "Status:"
    [ $has_pair_gpu -eq 1 ] && echo "  ✅ LJ pair forces (GPU)" || echo "  ❌ LJ pair forces (CPU)"
    [ $has_pppm_gpu -eq 1 ] && echo "  ✅ Coulomb forces (GPU)" || echo "  ❌ Coulomb forces (CPU) - Bottleneck!"
    [ $has_npt_gpu -eq 1 ] && echo "  ✅ Integration (GPU)" || echo "  ❌ Integration (CPU) - Bottleneck!"
    echo ""
    if [ $has_pppm_gpu -eq 0 ] || [ $has_npt_gpu -eq 0 ]; then
        echo "Recommendation: Rebuild LAMMPS with GPU package for full acceleration"
        echo ""
        echo "Build command:"
        echo "  cd lammps/src"
        echo "  cmake -DPKG_GPU=ON -DGPU_API=cuda ../cmake"
        echo "  make -j 4"
    fi
    echo ""
    exit 1
fi
