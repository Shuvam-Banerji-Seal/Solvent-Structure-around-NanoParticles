# Solvent Structure Around Nanoparticles - Complete System# Solvent Structure Around Nanoparticles - Complete System



A comprehensive TIP4P/2005 water molecular dynamics simulation framework for studying solvation structure around nanoparticles.A comprehensive TIP4P/2005 water molecular dynamics simulation framework for studying solvation structure around nanoparticles.



## 🎯 Overview## 🎯 Overview



This system provides production-ready tools for:This system provides production-ready tools for:

- **MD simulations** of water around nanoparticles (picosecond to nanosecond scale)- **MD simulations** of water around nanoparticles (picosecond to nanosecond scale)

- **Structural analysis** of solvation shells (RDF, coordination, H-bonds)- **Structural analysis** of solvation shells (RDF, coordination, H-bonds)

- **Thermodynamic tracking** (temperature, energy, pressure)- **Thermodynamic tracking** (temperature, energy, pressure)

- **Simulation validation** and quality assessment- **Simulation validation** and quality assessment

- **Batch processing** across multiple nanoparticles- **Batch processing** across multiple nanoparticles



## 📋 Quick Setup## 📋 Quick Setup



### 1. Clone Repository### 1. Clone Repository

```bash```bash

git clone git@github.com:Shuvam-Banerji-Seal/Solvent-Structure-around-NanoParticles.gitgit clone git@github.com:Shuvam-Banerji-Seal/Solvent-Structure-around-NanoParticles.git

cd Solvent-Structure-around-NanoParticlescd Solvent-Structure-around-NanoParticles

git checkout using_sample_filesgit checkout using_sample_files

cd solvent_effectscd solvent_effects

``````



### 2. Download Required Data Files### 2. Download Required Data Files



The system uses nanoparticle structures from the Atomify project and LAMMPS examples:The system uses nanoparticle structures from the Atomify project and LAMMPS examples:



```bash```bash

# Option A: Download automatically# Option A: Download automatically (one command)

bash setup_data_files.shbash setup_data_files.sh



# Option B: Manual download# Option B: Manual download

# Clone atomify for nanoparticle structures# Clone atomify for nanoparticle structures

git clone https://github.com/andeplane/atomify.gitgit clone https://github.com/andeplane/atomify.git



# Clone lammps-input-files for water molecule template  # Clone lammps-input-files for water molecule template

git clone https://github.com/simongravelle/lammps-input-files.gitgit clone https://github.com/simongravelle/lammps-input-files.git



# Copy required files# Copy required files

cp atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data input_files/cp atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data input_files/

cp lammps-input-files/LAMMPS-molecules/H2O_TIP4P.txt input_files/cp lammps-input-files/LAMMPS-molecules/H2O_TIP4P.txt input_files/

```# Copy more nanoparticles as needed

cp atomify/public/examples/GO-nanoparticle/GO_nanoparticle.data input_files/

### 3. Install Dependenciescp lammps-input-files/inputs/amorphous-carbon/amorphous_carbon.data input_files/

```bash```

# Python packages

pip install numpy matplotlib scipy scikit-learn### 3. Install Dependencies

```bash

# LAMMPS with MPI (Ubuntu/Debian)# Python packages

sudo apt-get install lammps-mpipip install numpy matplotlib scipy scikit-learn



# Or build from source: https://docs.lammps.org/Build.html# LAMMPS with MPI (required for simulations)

```# Ubuntu/Debian:

sudo apt-get install lammps-mpi

### 4. Verify Setup

```bash# Or build from source: https://docs.lammps.org/Build.html

python -c "import numpy, matplotlib; print('✓ Python OK')"```

which mpirun && which lmp_mpi && echo "✓ LAMMPS OK"

```### 4. Verify Setup

```bash

## 📁 Directory Structure# Test Python environment

python -c "import numpy, matplotlib; print('✓ Python OK')"

```

solvent_effects/# Test LAMMPS

├── README.md                          ⭐ START HEREwhich mpirun && which lmp_mpi && echo "✓ LAMMPS OK"

├── START_HERE.md                      Quick overview

├── VALIDATION_GUIDE.md                How to validate# Test basic simulation

├── README_ENHANCED.md                 Complete manual (70+ examples)cd shell_scripts

├── QUICK_REFERENCE_ENHANCED.sh        Command reference./run_solvation_study.sh SiC 500 10 4  # Quick 10 ps test

│```

├── shell_scripts/                     Simulation runners ⭐ KEY

│   ├── run_solvation_study.sh        Advanced runner (RECOMMENDED)## 📁 Directory Structure

│   ├── batch_run_solvation.sh        Batch processor

│   └── run_simulation.sh             Legacy runner```

│solvent_effects/

├── analysis/                          Analysis scripts ⭐ KEY├── README.md                          (This file) ⭐ START HERE

│   ├── validate_simulation.py         Validate quality (RECOMMENDED)├── START_HERE.md                      Quick overview & getting started

│   ├── analyze_tip4p_simple.py        Thermodynamic analysis├── VALIDATION_GUIDE.md                How to validate simulations

│   ├── analyze_solvation_advanced.py  Structural analysis├── ENHANCEMENT_SUMMARY.md             System improvements summary

│   └── README.md                      Analysis guide├── README_ENHANCED.md                 Complete feature manual (70+ examples)

│├── QUICK_REFERENCE_ENHANCED.sh        Command reference & examples

├── input_files/                       Molecular templates ⭐ ESSENTIAL│

│   ├── H2O_TIP4P.txt                 Water template (REQUIRED)├── shell_scripts/                     Simulation runners ⭐ KEY FILES

│   └── sic_nanoparticle.data         SiC nanoparticle (from atomify)│   ├── run_solvation_study.sh        Advanced runner with full control (RECOMMENDED)

││   ├── batch_run_solvation.sh        Batch processor for multiple NPs

├── output/                            Simulation results│   ├── run_simulation.sh             Legacy runner (for reference)

│   └── SiC_500w_10ps_.../            Example completed runs│   └── run_tip5p_simulation.sh       TIP5P runner (not recommended - slower)

││

└── docs/                              Detailed documentation├── analysis/                          Analysis & validation scripts ⭐ KEY FILES

```│   ├── validate_simulation.py         Validates simulation quality (RECOMMENDED)

│   ├── analyze_tip4p_simple.py        Thermodynamic analysis

## 🚀 Quick Start│   ├── analyze_solvation_advanced.py  Structural analysis (RDF, coordination)

│   ├── analyze_quick.py               Quick analysis

### Test Simulation (30-45 min)│   └── README.md                      Analysis guide

```bash│

cd shell_scripts├── input_files/                       Molecular templates ⭐ ESSENTIAL

./run_solvation_study.sh SiC 500 100 10│   ├── H2O_TIP4P.txt                 TIP4P water template (REQUIRED)

```│   ├── sic_nanoparticle.data         SiC nanoparticle (from atomify)

│   └── *.in, *.mol                   Legacy LAMMPS inputs

### Validate Results│

```bash├── output/                            Simulation results

cd ../analysis│   ├── SiC_500w_10ps_.../            Example completed runs

python validate_simulation.py ../output/SiC_500w_100ps_*/│   │   ├── trajectory.lammpstrj      Full trajectory for visualization

```│   │   ├── log.lammps                LAMMPS thermodynamic log

│   │   ├── final_config.data         Final structure

## 🔧 Shell Scripts│   │   ├── final.restart             Restart file for continuation

│   │   ├── validation_report.png     Validation quality plots

### `run_solvation_study.sh` ⭐ RECOMMENDED│   │   └── validation_summary.txt    Quality report

**Advanced simulation runner with full parameter control**│   └── ...

│

**What it does**:├── python_scripts/                    System setup utilities

- Runs TIP4P/2005 water MD simulations  │   ├── prepare_tip4p_system.py        Create TIP4P systems

- Full parameter control via flags│   ├── prepare_tip5p_system.py        Create TIP5P systems

- Supports ps to ns scale│   └── ...

- Manages equilibration and production│

- Generates trajectory, logs, RDF└── docs/                              Detailed documentation

- Provides progress tracking    ├── QUICK_START_SUCCESS.md         Successful setup guide

    ├── TIP4P_SUCCESS.md               TIP4P water model info

**Usage**:    ├── NANOPARTICLES_AVAILABLE.md     NP options

```bash    └── *.md                           Previous guides & notes

./run_solvation_study.sh NP_NAME NUM_WATERS TIME NUM_CORES [OPTIONS]```

```

## 🚀 Quick Start Examples

**Examples**:

```bash### Example 1: Simple 100 ps Test (30-45 min)

# 100 ps test```bash

./run_solvation_study.sh SiC 500 100 10cd shell_scripts



# 5 ns production  # Run simulation

./run_solvation_study.sh GO 1000 5 10 --ns --dump-freq 5000./run_solvation_study.sh SiC 500 100 10



# Custom settings# Then validate

./run_solvation_study.sh SiC 500 1 10 --ns --temp 350 --timestep 2.0cd ../analysis

```python validate_simulation.py ../output/SiC_500w_100ps_*/

```

**Flags**:

```### Example 2: Production Run (1 nanosecond, 5-7 hours)

--timestep FLOAT      Timestep in fs (default: 1.0)```bash

--temp FLOAT          Temperature in K (default: 300.0)cd shell_scripts

--dump-freq INT       Trajectory dump frequency (default: 1000)

--thermo-freq INT     Thermo output frequency (default: 500)# 1 ns simulation

--box-size FLOAT      Box size in Å (default: auto)./run_solvation_study.sh SiC 500 1 10 --ns

--equilibration INT   Equilibration steps (default: 10000)

--production          Skip equilibration# Analyze

--ns                  Use nanosecond unitscd ../analysis

--restart FILE        Continue from restart filepython validate_simulation.py ../output/SiC_500w_1ns_*/

--output-dir DIR      Custom output directorypython analyze_tip4p_simple.py ../output/SiC_500w_1ns_*/

--label LABEL         Custom label for run```

--help                Show help

```### Example 3: Batch Test All Nanoparticles

```bash

**Output files**:cd shell_scripts

- `trajectory.lammpstrj` - Full trajectory

- `log.lammps` - Thermodynamic log# Test all NPs (SiC, GO, amorphous) quickly

- `final_config.data` - Final structure./batch_run_solvation.sh --all-nps --test --cores 10

- `final.restart` - Restart file

- `rdf_np_water.dat` - RDF data# Validate all

- `np_com.dat` - NP center of masscd ../analysis

for dir in ../output/*/; do

---    python validate_simulation.py "$dir"

done

### `batch_run_solvation.sh````

**Run multiple simulations in batch**

## 🔧 Shell Scripts Guide

**Usage**:

```bash### `run_solvation_study.sh` ⭐ RECOMMENDED

./batch_run_solvation.sh [OPTIONS]**Purpose**: Advanced simulation runner with full parameter control

```

**What it does**:

**Examples**:- Runs TIP4P/2005 water MD simulations

```bash- Accepts all parameters via flags (no hardcoding!)

# Test all nanoparticles- Supports picosecond to nanosecond scale

./batch_run_solvation.sh --all-nps --test --cores 10- Manages equilibration and production phases

- Generates trajectory, logs, and RDF data

# Production runs- Provides progress tracking and statistics

./batch_run_solvation.sh --all-nps --production --cores 16

```**Basic usage**:

```bash

**Options**:./run_solvation_study.sh NP_NAME NUM_WATERS TIME NUM_CORES [OPTIONS]

``````

--all-nps            Run SiC, GO, amorphous carbon

--nps LIST           Comma-separated list (e.g., SiC,GO)**Examples**:

--waters N           Number of waters (default: 500)```bash

--time T             Simulation time (default: 100)# 100 ps test with SiC

--ns                 Use nanosecond units./run_solvation_study.sh SiC 500 100 10

--cores N            CPU cores (default: 8)

--test               Quick test mode (100 ps)# 5 ns production with GO

--production         Production mode (5 ns)./run_solvation_study.sh GO 1000 5 10 --ns --dump-freq 5000

```

# Custom settings

---./run_solvation_study.sh SiC 500 1 10 --ns --temp 350 --timestep 2.0

```

## 📊 Analysis Scripts

**Supported options**:

### `validate_simulation.py` ⭐ RECOMMENDED```

**Validate if simulation is good quality**--timestep FLOAT     Timestep in fs (default: 1.0)

--temp FLOAT         Temperature in K (default: 300.0)

**What it checks**:--dump-freq INT      Trajectory dump frequency (default: 1000)

- File completeness--thermo-freq INT    Thermo output frequency (default: 500)

- Temperature stability--box-size FLOAT     Box size in Å (default: auto)

- Energy conservation--equilibration INT  Equilibration steps (default: 10000)

- RDF structure quality--production         Skip equilibration phase

- Trajectory completeness--ns                 Use nanosecond time units

- Overall quality with recommendations--restart FILE       Continue from restart file

--output-dir DIR     Custom output directory

**Usage**:--label LABEL        Custom label for run

```bash--help               Show help

python validate_simulation.py OUTPUT_DIR```



# Examples**Output files** (in output/NP_name_Nw_time_timestamp/):

python validate_simulation.py ../output/SiC_500w_100ps_*/- `trajectory.lammpstrj` - Full trajectory for visualization

for dir in ../output/*/; do python validate_simulation.py "$dir"; done- `log.lammps` - Thermodynamic data

```- `final_config.data` - Final structure

- `final.restart` - Restart for continuation

**Output**:- `rdf_np_water.dat` - Radial distribution function

- `validation_report.png` - 3-panel plot- `np_com.dat` - NP center of mass trajectory

- `validation_summary.txt` - Quality report- `lammps_output.log` - Complete LAMMPS output



**What's Good**:---



| Metric | Good | Acceptable | Poor |### `batch_run_solvation.sh` 

|--------|------|-----------|------|**Purpose**: Run multiple simulations in batch

| Temperature | ±20 K | ±50 K | >50 K |

| Energy drift | <1% | <5% | >10% |**What it does**:

| RDF peak | g(r)>1.5 | g(r)>1.0 | g(r)<1.0 |- Runs simulations for multiple nanoparticles

| Frames | >100 | >50 | <50 |- Tracks success/failure

- Provides timing estimates

---- Can run in test or production mode



### `analyze_tip4p_simple.py`**Usage**:

**Analyze thermodynamic properties**```bash

./batch_run_solvation.sh [OPTIONS]

**What it does**:```

- Parses LAMMPS log file

- Temperature statistics**Examples**:

- Energy conservation```bash

- Pressure, density analysis# Test all nanoparticles

- Generates 6-panel plot./batch_run_solvation.sh --all-nps --test --cores 10

- Analysis report

# Production runs

**Usage**:./batch_run_solvation.sh --all-nps --production --cores 16

```bash

python analyze_tip4p_simple.py ../output/SiC_500w_100ps_20251105_145951/# Custom: SiC and GO only

```./batch_run_solvation.sh --nps SiC,GO --waters 1000 --time 5 --ns --cores 16

```

---

**Options**:

### `analyze_solvation_advanced.py````

**Structural analysis (RDF, coordination, H-bonds)**--all-nps            Run SiC, GO, and amorphous carbon

--nps LIST           Comma-separated list (e.g., SiC,GO)

**Usage**:--waters N           Number of waters (default: 500)

```bash--time T             Simulation time (default: 100)

python analyze_solvation_advanced.py ../output/SiC_*/ --all --verbose--ns                 Use nanosecond units

```--cores N            CPU cores (default: 8)

--test               Quick test mode (100 ps)

**Options**:--production         Production mode (5 ns)

``````

--rdf               Calculate RDF

--coordination      Calculate coordination---

--hbonds            Analyze hydrogen bonds

--all               Run all analyses## 📊 Analysis Scripts Guide

--r-max FLOAT       Max RDF radius (default: 15.0)

--r-bin FLOAT       RDF bin width (default: 0.1)### `validate_simulation.py` ⭐ RECOMMENDED

--skip INT          Skip first N frames**Purpose**: Validate if simulation is good quality

--stride INT        Use every Nth frame

--cutoff FLOAT      Coordination cutoff (default: 3.5)**What it does**:

--verbose           Verbose output- Checks file completeness

```- Analyzes temperature stability

- Measures energy conservation

---- Validates RDF structure

- Counts trajectory frames

## 💧 Water Model: TIP4P/2005- Generates validation plots

- Produces quality report

- **Reference**: Abascal & Vega (2005). *J. Chem. Phys.* **123**, 234505. https://doi.org/10.1063/1.2121687

- **Features**:**Usage**:

  - 4-site model (O, 2H, 1 virtual site M)```bash

  - Rigid water (SHAKE constraints)python validate_simulation.py OUTPUT_DIR

  - Accurate density and RDF```

  - Fast and stable

  - Full electrostatics (PPPM)**Examples**:

```bash

### Why TIP4P/2005?# Validate latest simulation

✅ Accurate - Reproduces water properties  python validate_simulation.py ../output/SiC_500w_100ps_*/

✅ Efficient - Only 4 sites (faster than TIP5P)  

✅ Stable - Well-tested in LAMMPS  # Validate specific run

✅ Appropriate - Room temperature studiespython validate_simulation.py ../output/SiC_500w_100ps_20251105_145951/



---# Validate all in batch

for dir in ../output/*/; do

## 📥 Available Nanoparticles    python validate_simulation.py "$dir"

done

### SiC (Silicon Carbide)```

- **File**: `input_files/sic_nanoparticle.data`

- **Source**: atomify/public/examples/sic/nanoparticle/**Output files**:

- **Size**: 8 atoms (testing)- `validation_report.png` - 3-panel plot (temperature, energy, RDF)

- **Command**: `./run_solvation_study.sh SiC 500 100 10`- `validation_summary.txt` - Text report with issues and recommendations



### GO (Graphene Oxide)**What it checks**:

- **File**: `input_files/GO_nanoparticle.data`

- **Source**: atomify/public/examples/GO-nanoparticle/| Metric | Good | Acceptable | Poor |

- **Size**: Large planar structure|--------|------|-----------|------|

- **Command**: `./run_solvation_study.sh GO 1000 1 10 --ns`| Temperature | ±20 K | ±50 K | >50 K |

| Energy drift | <1% | <5% | >10% |

### Amorphous Carbon| RDF peak | g(r)>1.5 | g(r)>1.0 | g(r)<1.0 |

- **File**: `input_files/amorphous_carbon.data`| Frames | >100 | >50 | <50 |

- **Source**: lammps-input-files/inputs/amorphous-carbon/

- **Size**: Very large (15000+ atoms)---

- **Command**: `./run_solvation_study.sh amorphous 500 1 10 --ns`

### `analyze_tip4p_simple.py`

### Custom Nanoparticles**Purpose**: Analyze thermodynamic properties

```bash

cp your_nanoparticle.data input_files/**What it does**:

./run_solvation_study.sh your_nanoparticle 500 100 10- Parses LAMMPS log file

```- Calculates temperature statistics

- Measures energy conservation

---- Tracks pressure and density

- Generates 6-panel plot

## 🎯 Typical Workflow- Produces analysis report



### Phase 1: Validation (30 min)**Usage**:

```bash```bash

cd shell_scriptspython analyze_tip4p_simple.py OUTPUT_DIR/

./run_solvation_study.sh SiC 500 100 10```

cd ../analysis

python validate_simulation.py ../output/SiC_500w_100ps_*/**Example**:

``````bash

python analyze_tip4p_simple.py ../output/SiC_500w_100ps_20251105_145951/

### Phase 2: Structure (5-7 hours)```

```bash

cd ../shell_scripts**Output files**:

./batch_run_solvation.sh --all-nps --waters 500 --time 1 --ns --cores 10- `analysis.png` - 6-panel thermodynamic plot

cd ../analysis- `analysis_report.txt` - Detailed statistics

for dir in ../output/*/; do

    python validate_simulation.py "$dir"---

    python analyze_tip4p_simple.py "$dir"

done### `analyze_solvation_advanced.py`

```**Purpose**: Structural analysis (RDF, coordination, H-bonds)



### Phase 3: Production (1-3 days)**What it does**:

```bash- Calculates radial distribution function

cd ../shell_scripts- Analyzes coordination numbers

./batch_run_solvation.sh --all-nps --waters 1000 --time 5 --ns --cores 10- Counts hydrogen bonds

cd ../analysis- Generates plots

for dir in ../output/*/; do- Exports data files

    python validate_simulation.py "$dir"

    python analyze_solvation_advanced.py "$dir" --all --skip 100**Usage**:

done```bash

```python analyze_solvation_advanced.py OUTPUT_DIR [OPTIONS]

```

---

**Examples**:

## 📈 Performance```bash

# Full analysis

### Requirementspython analyze_solvation_advanced.py ../output/SiC_*/ --all --verbose

- **Minimum**: 4 cores, 8 GB RAM

- **Recommended**: 10-16 cores, 16 GB RAM  # Just RDF with high resolution

- **Production**: 32 cores, 32 GB RAMpython analyze_solvation_advanced.py ../output/GO_*/ --rdf --r-max 20 --r-bin 0.05



### Time Estimates (500 waters)# Skip equilibration frames

| Simulation | Cores | Runtime |python analyze_solvation_advanced.py ../output/SiC_*/ --all --skip 50 --stride 2

|------------|-------|---------|```

| 100 ps | 10 | 30-45 min |

| 1 ns | 10 | 5-7 hours |**Options**:

| 5 ns | 16 | 1-2 days |```

| 10 ns | 16 | 3-5 days |--rdf               Calculate RDF

--coordination      Calculate coordination numbers

### Optimization Tips--hbonds            Analyze hydrogen bonds

- Use `--ns` for nanosecond runs (clearer)--all               Run all analyses

- Use `--dump-freq 5000` for long runs (save disk)--r-max FLOAT       Maximum RDF radius (Å, default: 15.0)

- Use `--timestep 2.0` for faster runs (test first!)--r-bin FLOAT       RDF bin width (Å, default: 0.1)

- Scale cores to 8-32 (optimal range)--skip INT          Skip first N frames

--stride INT        Use every Nth frame

-----cutoff FLOAT      Coordination cutoff (Å, default: 3.5)

--no-plots          Skip plotting

## ✅ Success Criteria--verbose           Verbose output

```

Simulation is GOOD when:

- ✅ Temperature: Within ±20 K of 300 K---

- ✅ Energy drift: < 5% (< 1% excellent)

- ✅ RDF peak: 3-4 Å, g(r) > 1.5### `analyze_quick.py`

- ✅ RDF bulk: Approaches 1.0 at r > 10 Å**Purpose**: Quick visual check

- ✅ Trajectory: > 50 frames

- ✅ All output files presentQuick analysis for fast feedback.



------



## 🐛 Troubleshooting## 💧 Water Model: TIP4P/2005



### Simulation Won't StartWe use the **TIP4P/2005 (4-point transferable intermolecular potential)** water model:

```bash- **Reference:** Rick, S.W., J. Chem. Phys. 120, 6085-6093 (2004)

which lmp_mpi    # Check LAMMPS installed- **DOI:** https://doi.org/10.1063/1.1652434

mpirun --version # Check MPI installed- **LAMMPS Documentation:** https://docs.lammps.org/Howto_tip5p.html#rick

ls input_files/* # Check input files exist

```### TIP5P Model Features:

- 5 interaction sites per water molecule:

### Energy Drift Too High  - 1 oxygen atom (O)

```bash  - 2 hydrogen atoms (H)

# Increase equilibration  - 2 lone pair sites (L) representing electron density

./run_solvation_study.sh SiC 500 100 10 --equilibration 50000- Better reproduction of water structure compared to TIP3P/TIP4P

- More accurate liquid density at various temperatures

# Smaller timestep- Improved radial distribution functions

./run_solvation_study.sh SiC 500 100 10 --timestep 0.5- Better dielectric properties

```

### Key Parameters:

### Simulation Too Slow- O-H bond: 0.9572 Å

```bash- H-O-H angle: 104.52°

# Reduce dump frequency- O-L distance: 0.70 Å

./run_solvation_study.sh SiC 500 100 10 --dump-freq 2000- L-O-L angle: 109.47°

- LJ epsilon (O): 0.16 kcal/mol

# Use larger timestep- LJ sigma (O): 3.12 Å

./run_solvation_study.sh SiC 500 100 10 --timestep 2.0- Charges: q_H = +0.241 e, q_L = -0.241 e, q_O = 0.0



# More cores## Previous Issues Fixed

./run_solvation_study.sh SiC 500 100 16

```### 1. Water Model

- **Old:** Simplified SPC/E without charges/electrostatics

---- **New:** Full TIP5P with:

  - Proper electrostatics (kspace_style pppm/tip5p)

## 📚 Documentation  - Lone pair sites for accurate water structure

  - SHAKE constraints for rigid bonds/angles

- **START_HERE.md** - Quick overview  

- **README_ENHANCED.md** - Complete manual (70+ examples)### 2. Force Field

- **VALIDATION_GUIDE.md** - How to validate simulations- **Old:** LJ/cut 10 Å (no long-range electrostatics)

- **QUICK_REFERENCE_ENHANCED.sh** - Command reference- **New:** lj/cut/tip5p/long with PPPM solver

- **docs/** - Detailed guides  - Accounts for electrostatic interactions

  - Better water-water interactions

---  - Improved water-nanoparticle interactions



## 📞 Support### 3. Energy Conservation

- **Old:** 115% energy drift

Questions? Check:- **New:** Proper constraints and smaller timestep

1. **START_HERE.md** - Overview  - SHAKE for water rigidity

2. **README_ENHANCED.md** - Manual  - Timestep: 0.5 fs (may reduce to 0.2 fs if needed)

3. **QUICK_REFERENCE_ENHANCED.sh** - Examples  - NVT ensemble with Nosé-Hoover thermostat

4. **docs/** - Guides

### 4. System Density

---- **Old:** 0.264 g/cm³ (sparse solvation shell in large box)

- **New:** Adjustable box size for proper density

## 🎯 Quick Start  - Option 1: Fill entire box with water (ρ → 1.0 g/cm³)

```bash  - Option 2: Keep solvation shell but with better sampling

cd shell_scripts && ./run_solvation_study.sh SiC 500 100 10  - Option 3: Use periodic images for bulk-like environment

cd ../analysis && python validate_simulation.py ../output/SiC_500w_100ps_*/

```## Simulation Protocol



**Happy Simulating!** 🎉### Phase 1: System Preparation

1. Center the SiC nanoparticle
2. Place TIP5P water molecules around nanoparticle
3. Generate lone pair sites (virtual sites)
4. Assign proper charges and masses
5. Create LAMMPS data file with bonds/angles

### Phase 2: Energy Minimization
1. Minimize with frozen nanoparticle (1000 steps)
2. Minimize with flexible nanoparticle (1000 steps)
3. Check for overlapping atoms

### Phase 3: Equilibration
1. NVT equilibration (50 ps, T = 300 K)
2. Gradual heating if starting from 0 K
3. Monitor: Temperature, pressure, density, energy

### Phase 4: Production
1. NVT production run (500 ps - 1 ns)
2. Save trajectory every 0.5 ps
3. Compute RDF, coordination, orientation on-the-fly

### Phase 5: Analysis
1. Radial Distribution Functions (RDF)
   - g_OO(r): Water-water structure
   - g_Si-O(r), g_C-O(r): NP-water structure
   - g_OH(r): Hydrogen bonding
2. Coordination Numbers
   - First shell (r < 3.5 Å)
   - Second shell (3.5 < r < 7 Å)
3. Water Orientation
   - Dipole angle distribution
   - Lone pair orientation
4. Hydrogen Bonding
   - H-bond lifetime
   - H-bond network analysis
5. Density Profiles
   - ρ(r) from NP surface
6. Residence Time
   - Water exchange dynamics
7. Diffusion
   - MSD and diffusion coefficients

## Expected Improvements

| Property | Old Result | Expected New Result |
|----------|------------|-------------------|
| Energy drift | 115% | < 5% |
| Water model | Simplified SPC/E | Full TIP5P with electrostatics |
| Density | 0.264 g/cm³ | ~1.0 g/cm³ or controlled |
| Temperature | 300 ± 3 K | 300 ± 1 K |
| Simulation time | 125 ps | 500-1000 ps |
| H-bond accuracy | Not possible | Accurate |
| Water orientation | Qualitative | Quantitative |

## Computational Resources

- **MPI tasks:** 10 (1×5×2 decomposition)
- **OpenMP threads:** 12 per task
- **Total cores:** 120
- **Expected performance:** 2-5 ns/day (depends on system size)
- **Memory:** ~2-4 GB for 10,000 atoms

## References

1. Rick, S.W. (2004). "A reoptimization of the five-site water potential (TIP5P) for use with Ewald sums." *J. Chem. Phys.* **120**, 6085-6093.
2. Mahoney, M.W. & Jorgensen, W.L. (2000). "A five-site model for liquid water and the reproduction of the density anomaly by rigid, nonpolarizable potential functions." *J. Chem. Phys.* **112**, 8910-8922.
3. LAMMPS Documentation: https://docs.lammps.org/
4. TIP5P Howto: https://docs.lammps.org/Howto_tip5p.html

---

**Last Updated:** 2025-11-04  
**Status:** In Development
