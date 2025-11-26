#!/bin/bash
# Run Advanced Generative Model Training
# This script activates the environment and starts training

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Advanced MD Generative Model Training                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: cd .. && ./setup_env.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../.venv/bin/activate

echo "✅ Environment activated"
echo ""

# Run training
echo "🚀 Starting training..."
echo ""

cd scripts
python train.py

echo ""
echo "✅ Training complete!"
echo ""
