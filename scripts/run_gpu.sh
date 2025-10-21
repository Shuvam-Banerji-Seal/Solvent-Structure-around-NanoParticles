#!/usr/bin/env bash
# GPU runner template. Configure LMP_BIN and KOKKOS/GPU args as appropriate for your LAMMPS build.
set -e
LMP_BIN=${LMP_BIN:-lmp_kokkos}
LMP_ARGS=${LMP_ARGS:-"-k on g 1"}
# Example for GPU-enabled lmp_kokkos: lmp_kokkos -k on g 1 -in run.in
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

${LMP_BIN} ${LMP_ARGS} -in "$1"
