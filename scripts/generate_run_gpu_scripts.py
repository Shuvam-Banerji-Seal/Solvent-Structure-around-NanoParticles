#!/usr/bin/env python3
"""Generate a `run_gpu.sh` in every experiment replica directory.

Usage: python3 scripts/generate_run_gpu_scripts.py [experiments_root]
Default experiments_root: ./experiments
"""
import sys
from pathlib import Path
import os

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('experiments')
if not root.exists():
    print('No experiments directory found at', root)
    sys.exit(1)

count = 0
for run_in in root.rglob('run.in'):
    run_dir = run_in.parent
    gpu_script = run_dir / 'run_gpu.sh'
    content = """#!/usr/bin/env bash
set -e
# GPU runner for this experiment. Edit LMP_BIN and LMP_ARGS as needed.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
LMP_BIN=${LMP_BIN:-lmp_kokkos}
LMP_ARGS=${LMP_ARGS:-"-k on g 1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

$LMP_BIN $LMP_ARGS -in run.in
"""
    gpu_script.write_text(content)
    gpu_script.chmod(0o755)
    count += 1

print(f'Wrote {count} run_gpu.sh scripts under {root}')
