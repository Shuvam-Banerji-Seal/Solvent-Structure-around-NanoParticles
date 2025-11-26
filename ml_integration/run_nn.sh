#!/bin/bash
# Run Neural Network model training
# This script activates the environment and runs the NN training

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Running Neural Network Model Training                         ║"
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

# Run NN training
echo "🚀 Starting NN training..."
echo ""

python scripts/03_train_nn.py

echo ""
echo "✅ NN training complete!"
echo ""
echo "Results saved to:"
echo "  - models/nn_model.pt"
echo "  - results/plots/nn_*.png"
echo "  - results/predictions/nn_predictions.csv"
echo ""
