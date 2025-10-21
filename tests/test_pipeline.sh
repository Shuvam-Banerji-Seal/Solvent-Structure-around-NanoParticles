#!/usr/bin/env bash
# Very small smoke test: run sweep driver in dry-run mode (no packmol invocation)
set -e
python3 scripts/sweep_eps.py configs/params.yaml in/cg_sphere.in.template
echo "Sweep generation ok"
