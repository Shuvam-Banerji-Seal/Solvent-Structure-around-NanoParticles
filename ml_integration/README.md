"""
ML Integration for Nanoparticle Solvation Study
================================================

This package provides machine learning models to predict properties
for untested epsilon values (0.55+) while simulations are running.

Project Structure:
------------------
ml_integration/
├── models/              # Trained model files (.pkl, .pt)
├── data/                # Extracted features and datasets
├── scripts/             # Python scripts for training and prediction
├── notebooks/           # Jupyter notebooks for exploration
├── results/             # Predictions and plots
│   ├── plots/
│   └── predictions/
├── logs/                # Training logs
├── requirements.txt     # Python dependencies
├── setup_env.sh         # Environment setup script
├── run_gpr.sh          # Run Gaussian Process model
├── run_nn.sh           # Run Neural Network model
├── run_ensemble.sh     # Run ensemble prediction
└── README.md           # This file

Quick Start:
-----------
1. Set up environment:
   ./setup_env.sh

2. Extract data from simulations:
   source .venv/bin/activate
   python scripts/01_extract_features.py

3. Train Gaussian Process model:
   ./run_gpr.sh

4. Make predictions for epsilon 0.55+:
   python scripts/04_predict_new_epsilon.py

Models:
-------
1. Gaussian Process Regression (GPR) - Primary model
2. Neural Network with Physics Constraints
3. XGBoost Gradient Boosting
4. Ensemble (weighted combination of all 3)

Author: Shuvam Roy
Date: November 2025
"""

__version__ = "0.1.0"
