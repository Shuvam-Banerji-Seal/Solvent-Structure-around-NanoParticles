#!/usr/bin/env bash
# Create Python environment and install required packages.
# Prefer 'uv' package manager if present, otherwise fallback to venv + pip.
set -e
if command -v uv >/dev/null 2>&1; then
    echo "Found uv; attempting to create environment and install packages with uv."
    # The exact uv commands may vary by uv version; adapt as needed.
    uv new mdenv -y || true
    uv install -y pyyaml numpy mdanalysis mdtraj parmed matplotlib
    echo "Activate environment with: uv shell mdenv" 
else
    echo "uv not found; creating a Python venv at .venv and installing packages with pip."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install pyyaml numpy MDAnalysis mdtraj parmed matplotlib
    echo "Activate environment with: source .venv/bin/activate"
fi
