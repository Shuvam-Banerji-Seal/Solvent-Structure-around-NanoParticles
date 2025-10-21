#!/usr/bin/env bash
# CPU runner template. Set LMP_BIN, MPI_PROCS and OMP_NUM_THREADS as needed.
set -e
LMP_BIN=${LMP_BIN:-lmp_mpi}
MPI_PROCS=${MPI_PROCS:-1}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_NUM_THREADS

if command -v mpirun >/dev/null 2>&1; then
    mpirun -np ${MPI_PROCS} ${LMP_BIN} -in "$1"
else
    ${LMP_BIN} -in "$1"
fi
