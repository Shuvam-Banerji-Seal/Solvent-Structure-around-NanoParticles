#!/bin/bash
# Run Full ML Pipeline
# This script runs XGBoost training and then the Ensemble prediction

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Running Full ML Pipeline (XGBoost + Ensemble)                 ║"
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
echo ""

# Run XGBoost training
echo "🚀 Starting XGBoost training..."
python scripts/04_train_xgboost.py
echo "✅ XGBoost training complete"
echo ""

# Run Ensemble
echo "🚀 Starting Ensemble prediction..."
python scripts/05_ensemble_predictions.py
echo "✅ Ensemble prediction complete"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Pipeline Complete!                                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Final results available in:"
echo "  - results/predictions/ensemble_predictions.csv"
echo "  - results/plots/ensemble_*.png"
echo ""
