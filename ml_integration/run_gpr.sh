#!/bin/bash
# Run Gaussian Process Regression model
# This script activates the environment and runs the GPR training

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Running GPR Model Training                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./setup_env.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "✅ Environment activated"
echo "   Python: $(which python)"
echo ""

# Run GPR training
echo "🚀 Starting GPR training..."
echo ""

python scripts/02_train_gpr.py

echo ""
echo "✅ GPR training complete!"
echo ""
echo "Results saved to:"
echo "  - models/gpr_models.pkl"
echo "  - results/plots/gpr_*.png"
echo "  - results/predictions/gpr_predictions.csv"
echo ""
