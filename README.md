# Solvation of C60 Nanoparticles in Water

This repository contains a computational science project investigating the solvation of C60 nanoparticles in water. The project follows a clear, three-stage workflow: 1) Simulation, 2) Feature Extraction, and 3) Machine Learning.

## Workflow

1.  **Simulation:** The core of the project is a series of molecular dynamics simulations run using LAMMPS. An initial system is built using scripts in `lammps_scripts/`. The main, production-level simulations are executed using a highly optimized and corrected workflow located in the `airebo_simulation_5ns/` directory. The entire campaign is managed by `bash_scripts/run_parallel_equilibration_v2.sh`, which runs simulations for a range of 'epsilon' values (controlling C60-water interaction strength) in parallel.

2.  **Feature Extraction:** After the simulations complete, the `ml_integration/scripts/01_extract_features.py` script is used to parse the raw output files (thermodynamic data and radial distribution functions) from each simulation run. It calculates key features like mean temperature, pressure, RDF peak heights, and coordination numbers.

3.  **Machine Learning:** The extracted features are used to train a suite of machine learning models (GPR, NN, XGBoost) in the `ml_integration/` directory. The ultimate goal is to use these trained models to predict the properties of the system at new epsilon values without the need for running more costly simulations.

## Key Directories

-   `bash_scripts/`: Contains the master script to run the entire simulation campaign.
-   `airebo_simulation_5ns/`: Contains the primary, optimized LAMMPS scripts for production runs.
-   `ml_integration/`: Contains the complete Python-based machine learning pipeline, from feature extraction to prediction.
-   `lammps_scripts/`: Contains scripts for the initial setup of the simulation system.
-   `data_files/`: A repository for input data like the C60 structure and water model.
-   `python_scripts_for_c60_generation/`: Utility script to generate the C60 nanoparticle data file.

## How to Run

1.  **Generate C60 data file:**
    ```bash
    python python_scripts_for_c60_generation/generate_C60_bonded.py
    ```
2.  **Build the initial system:**
    ```bash
    lmp -in lammps_scripts/1_build_large_C60_system.lmp
    ```
3.  **Run the simulation campaign:**
    ```bash
    bash bash_scripts/run_parallel_equilibration_v2.sh
    ```
4.  **Extract features:**
    ```bash
    python ml_integration/scripts/01_extract_features.py
    ```
5.  **Train ML models:**
    ```bash
    python ml_integration/scripts/02_train_gpr.py
    python ml_integration/scripts/03_train_nn.py
    python ml_integration/scripts/04_train_xgboost.py
    ```
6.  **Make predictions:**
    ```bash
    python ml_integration/scripts/05_ensemble_predictions.py
    ```
