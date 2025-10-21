#!/usr/bin/env bash
# Run the small smoke-test using CPU wrapper
set -e
pushd $(dirname "$0")/.. >/dev/null
LMP_BIN=${LMP_BIN:-lmp_mpi}
$LMP_BIN -in in/test_small.in
popd >/dev/null
