# Core Algorithms and Mathematical Formulas
## Solvent Effects Analysis Suite

This document details the mathematical foundations and algorithms used in the analysis modules for the C60 Solvation Study.

---

### 1. Thermodynamic Analysis
**Modules:** `01`, `02`, `12`, `15`

#### 1.1 Statistical Mechanics Averages
For any thermodynamic property $X$ (Temperature, Pressure, Energy, Density):
- **Ensemble Average:** $\langle X \rangle = \frac{1}{N} \sum_{i=1}^{N} X_i$
- **Fluctuations (Standard Deviation):** $\sigma_X = \sqrt{\langle X^2 \rangle - \langle X \rangle^2}$

#### 1.2 Equilibration Detection
- **Drift Analysis:** Linear regression of property $X(t)$ over time window $t \in [t_{start}, t_{end}]$:
  $$ X(t) = \alpha t + \beta $$
  Equilibrium is assumed when slope $\alpha \approx 0$ (within statistical noise).
- **Block Averaging:** The trajectory is divided into $M$ blocks of size $B$. The variance of block averages $\sigma_B^2$ decays as $1/B$ for uncorrelated data. Deviation from this scaling indicates correlation/non-equilibrium.

#### 1.3 Autocorrelation Function (ACF)
Used to determine the correlation time $\tau$ and effective sample size.
$$ C_X(\tau) = \frac{\langle (X(t) - \langle X \rangle)(X(t+\tau) - \langle X \rangle) \rangle}{\sigma_X^2} $$
Computed efficiently using Fast Fourier Transform (FFT) (Wiener-Khinchin theorem).

---

### 2. Structural Analysis
**Modules:** `03`, `10`, `14`

#### 2.1 Radial Distribution Function (RDF)
The probability of finding a particle at distance $r$ relative to an ideal gas.
$$ g(r) = \frac{1}{4\pi r^2 \rho N} \sum_{i=1}^{N} \sum_{j \neq i} \delta(r - r_{ij}) $$
Where $\rho$ is the number density.
**Implementation:** Histogram of pairwise distances normalized by shell volume $dV = 4\pi r^2 dr$.

#### 2.2 Coordination Number
The average number of neighbors within a cutoff distance $r_{cut}$ (typically the first minimum of $g(r)$).
$$ N(r_{cut}) = 4\pi \rho \int_0^{r_{cut}} r^2 g(r) dr $$

---

### 3. Dynamic Analysis
**Modules:** `06`, `07`

#### 3.1 Mean Squared Displacement (MSD)
Measure of particle mobility over time.
$$ \text{MSD}(t) = \langle | \mathbf{r}(t) - \mathbf{r}(0) |^2 \rangle $$
**Implementation:** Averaged over all water molecules (or C60 COMs) and multiple time origins $t_0$.

#### 3.2 Diffusion Coefficient ($D$)
Derived from MSD using the Einstein relation for 3D diffusion in the long-time limit:
$$ D = \lim_{t \to \infty} \frac{\text{MSD}(t)}{6t} $$
**Implementation:** Slope of the linear fit to MSD vs $t$ in the diffusive regime (typically 1-5 ns).

---

### 4. Advanced Water Structure (CUDA Accelerated)
**Modules:** `04`, `16`

#### 4.1 Tetrahedral Order Parameter ($q$)
Quantifies the local tetrahedrality of the water network.
$$ q = 1 - \frac{3}{8} \sum_{j=1}^{3} \sum_{k=j+1}^{4} \left( \cos \psi_{jk} + \frac{1}{3} \right)^2 $$
Where $\psi_{jk}$ is the angle formed by the central oxygen atom and its nearest neighbors $j$ and $k$.
- $q=1$: Perfect tetrahedral geometry (Ice Ih).
- $q=0$: Random ideal gas arrangement.
- $q \approx 0.5-0.8$: Liquid water.

#### 4.2 Hydrogen Bonds
Geometric definition based on distance and angle:
1.  $r_{OO} < 3.5 \, \text{Å}$ (Distance between oxygens)
2.  $\angle H-O\dots O < 30^\circ$ (Angle between O-H bond and O-O vector)

#### 4.3 Steinhardt Order Parameters ($Q_l$)
Rotational invariants used to distinguish crystal structures.
**Proxy Implementation (Module 04):**
Due to computational cost of full spherical harmonics, a proxy based on neighbor distance variance is used:
$$ Q_{proxy} = \exp(-\text{Var}(r_{neighbors})) $$
Higher values indicate more regular (crystal-like) shells.

#### 4.4 Shape Parameters (Asphericity & Acylindricity)
Derived from the eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ of the Gyration Tensor (or Moment of Inertia Tensor) of the water cluster.
- **Asphericity ($b$):** Deviation from spherical symmetry.
  $$ b = \lambda_1 - \frac{1}{2}(\lambda_2 + \lambda_3) $$
- **Acylindricity ($c$):** Deviation from cylindrical symmetry.
  $$ c = \lambda_2 - \lambda_3 $$

#### 4.5 3D Density Maps
Spatial probability distribution of water oxygen atoms relative to the nearest C60 molecule.
$$ \rho(\mathbf{r}) = \langle \sum_i \delta(\mathbf{r} - (\mathbf{r}_i - \mathbf{r}_{C60})) \rangle $$
**Implementation:** 3D histogram accumulation on a grid centered on C60.

#### 4.6 Orientation Analysis
Correlation between water dipole orientation and distance from the surface.
$$ \cos \theta = \hat{\mu}_{H2O} \cdot \hat{r}_{C60 \to O} $$
- $\cos \theta = 1$: Dipole pointing away from surface.
- $\cos \theta = -1$: Dipole pointing towards surface.

---

### 5. Thermodynamic Response Functions
**Module:** `07`

Calculated from fluctuations in the NPT ensemble.

#### 5.1 Isochoric Heat Capacity ($C_V$)
$$ C_V = \frac{\langle \delta E^2 \rangle}{k_B T^2} $$
Where $E$ is the total energy.

#### 5.2 Isothermal Compressibility ($\kappa_T$)
$$ \kappa_T = \frac{\langle \delta V^2 \rangle}{k_B T \langle V \rangle} $$
Where $V$ is the system volume.

#### 5.3 Thermal Expansion Coefficient ($\alpha_P$)
$$ \alpha_P = \frac{\langle \delta V \delta H \rangle}{k_B T^2 \langle V \rangle} $$
Where $H$ is the enthalpy ($H = E + PV$).

---

### 6. Image Analysis
**Module:** `08`

#### 6.1 Brightness & Contrast
Analysis of PPM snapshots.
- **Brightness:** Mean pixel intensity $\mu_I$.
- **Contrast:** Standard deviation of pixel intensity $\sigma_I$.
Used as a proxy for visual clustering or phase separation (qualitative).

---

### 7. Equilibration Pathway Analysis
**Module:** `09`

Tracks the system evolution through four distinct equilibration stages:
1.  **NVT Thermalization:** Heating to 300K.
2.  **Pre-equilibration:** Initial relaxation.
3.  **Pressure Ramp:** Transition from 0 to 1 atm.
4.  **NPT Equilibration:** Density stabilization.

**Metrics:**
- **Transition Smoothness:** Continuity of $T, P, \rho$ across stage boundaries.
- **Convergence Rate:** Time derivative of properties $\frac{dX}{dt}$.

---

### 8. Structural Data Analysis
**Module:** `10`

Parses LAMMPS data files (`.data`) to analyze static topology.
- **Bond Length Distribution:** Histogram of C-C bonds in C60.
- **Hydration Shell Composition:** Identification of water molecules with $r_{C-O} < r_{shell}$.
- **Displacement Maps:** Vector field $\Delta \mathbf{r} = \mathbf{r}_{final} - \mathbf{r}_{initial}$ to visualize drift.

---

### 9. Trajectory Visualization
**Module:** `11`

Generates DCD trajectory movies.
- **Rendering:** 3D scatter/line plots of atomic coordinates over time.
- **Frame Interpolation:** Linear interpolation for smooth playback (if enabled).

---

### 10. Equilibration Convergence
**Module:** `12`

Quantitative assessment of convergence.
- **Running Average:** $\bar{X}(t) = \frac{1}{t} \int_0^t X(t') dt'$
- **Standard Error of the Mean (SEM):** $\sigma_{\bar{X}} = \frac{\sigma_X}{\sqrt{N_{eff}}}$
- **Convergence Criterion:** $|\text{Slope}(X)| < \epsilon_{tol}$ and $\sigma_X < \sigma_{tol}$.

---

### 11. Performance Analysis
**Module:** `13`

Parses log files for computational efficiency metrics.
- **Speed:** Timesteps per second (TPS).
- **Throughput:** Nanoseconds per day ($\text{ns/day} = \frac{\text{TPS} \times \Delta t \times 86400}{10^6}$).
- **Parallel Efficiency:** CPU vs GPU time ratio.

---

### 12. System Validation
**Module:** `14`

Validates force field parameters against standard values.
- **TIP4P/2005 Geometry:**
    - $r_{OH} = 0.9572 \, \text{Å}$
    - $\theta_{HOH} = 104.52^\circ$
- **C60 Geometry:**
    - $r_{CC} \approx 1.42 \, \text{Å}$
    - Diameter $\approx 7.1 \, \text{Å}$
- **Lennard-Jones Parameters:** Checks $\epsilon$ and $\sigma$ values in data file.

---

### 13. Thermal Trajectory Analysis
**Module:** `15`

Analyzes energy conservation and thermalization.
- **Total Energy:** $E_{tot} = E_{kin} + E_{pot}$
- **Energy Drift:** $\frac{1}{E_{tot}} \frac{dE_{tot}}{dt}$ (Should be $< 10^{-4} / \text{ns}$ for NVE, stable for NPT).
- **Equipartition Theorem Check:** $E_{kin} = \frac{3}{2} N k_B T$.


---

## Module Data Specifications
This section details the input data requirements and output files for each analysis module.

### Module 01: Thermodynamic Analysis
*   **Input Files:**
    *   `production.log`: Main thermodynamic data source.
    *   `equilibration.log`: Used for comparison.
*   **Analysis Method:** Parses LAMMPS log files to extract T, P, Density, Energy time series. Computes statistical moments (mean, std dev).
*   **Output Files:**
    *   `01_temperature_evolution.png`: Temperature vs Time.
    *   `02_pressure_analysis.png`: Pressure vs Time.
    *   `03_density_analysis.png`: Density vs Time.
    *   `04_energy_analysis.png`: Total/Potential/Kinetic Energy vs Time.
    *   `05_comparison_matrix.png`: Statistical summary table.
    *   `thermodynamic_statistics.csv`: Raw statistical data.

### Module 02: Equilibration Stability
*   **Input Files:**
    *   `npt_equilibration.log`: Final equilibration stage.
    *   `production.log`: Production run.
*   **Analysis Method:** Computes autocorrelation functions, block averages, and running averages to verify statistical convergence.
*   **Output Files:**
    *   `06_autocorrelation_analysis.png`: ACF decay plots.
    *   `07_block_averaging.png`: Error estimate vs block size.
    *   `08_running_averages.png`: Cumulative average convergence.
    *   `09_stability_metrics.png`: Drift and noise metrics.
    *   `equilibration_stability_metrics.csv`: Stability quantification.

### Module 03: RDF Structural Analysis
*   **Input Files:**
    *   `rdf_data.txt` (or similar): Pre-computed RDF data from LAMMPS (fix ave/time).
*   **Analysis Method:** visualizes Radial Distribution Functions (g(r)) for C-C, C-O, and O-O pairs.
*   **Output Files:**
    *   `10_rdf_comparison.png`: Comparative RDFs across epsilon values.
    *   `11_co_rdf_detailed.png`: Detailed C60-Water structure.
    *   `12_coordination_numbers.png`: Integrated coordination numbers.
    *   `rdf_peak_analysis.csv`: Peak locations and heights.

### Module 04: Comprehensive Water Structure (CUDA)
*   **Input Files:**
    *   `production.dcd` (or `.lammpstrj`): Full trajectory.
    *   `equilibrated_system.data`: Topology.
*   **Analysis Method:** GPU-accelerated calculation of tetrahedral order (q), Steinhardt parameters (Q4, Q6), and shape tensors.
*   **Output Files:**
    *   `comprehensive_water_structure_eps*.json`: Intermediate data file containing time-series of structural metrics (consumed by Module 05).

### Module 05: Plot Water Structure
*   **Input Files:**
    *   `comprehensive_water_structure_eps*.json`: Output from Module 04.
*   **Analysis Method:** Visualizes the structural metrics computed by Module 04.
*   **Output Files:**
    *   `13_tetrahedral_order_analysis.png`: q-parameter distributions.
    *   `14_steinhardt_order_parameters.png`: Q4/Q6 order parameters.
    *   `15_shape_parameters_oblate_prolate.png`: Asphericity and Acylindricity.
    *   `16_coordination_hbond_analysis.png`: H-bond network statistics.
    *   `17_msd_diffusion_analysis.png`: Water diffusion coefficients.
    *   `tetrahedral_order_summary.csv`, `steinhardt_order_summary.csv`, `shape_parameters_summary.csv`, `coordination_hbond_summary.csv`, `diffusion_coefficients.csv`.

### Module 06: MSD Validation
*   **Input Files:**
    *   `msd_data.txt`: MSD data computed by LAMMPS.
*   **Analysis Method:** Fits MSD vs Time to Einstein relation to extract diffusion coefficients.
*   **Output Files:**
    *   `18_msd_evolution.png`: MSD plots.
    *   `msd_evolution_data.csv`: Computed diffusion coefficients.

### Module 07: High Priority Additional Analysis
*   **Input Files:**
    *   `production.lammpstrj`: Trajectory file.
    *   `production.log`: Thermodynamic data.
*   **Analysis Method:** Computes C60-C60 distances, aggregation metrics, and thermodynamic response functions (Cv, Kappa).
*   **Output Files:**
    *   `19_c60_distance_timeseries.png`: Inter-fullerene distances.
    *   `20_c60_aggregation_analysis.png`: Contact probability.
    *   `21_c60_diffusion.png`: C60 diffusion analysis.
    *   `22_thermodynamic_response_functions.png`: Heat capacity and compressibility.
    *   `c60_distances_summary.csv`, `c60_diffusion_coefficients.csv`, `thermodynamic_response_functions.csv`.

### Module 08: PPM Snapshot Analysis
*   **Input Files:**
    *   `production_*.ppm`: Image snapshots from LAMMPS dump.
*   **Analysis Method:** Image processing to quantify phase separation via brightness/contrast analysis.
*   **Output Files:**
    *   `23_ppm_snapshot_montage.png`: Visual montage of system states.
    *   `24_ppm_metrics.png`: Quantitative image metrics.
    *   `ppm_snapshot_statistics.csv`.

### Module 09: Equilibration Pathway Analysis
*   **Input Files:**
    *   `nvt_thermalization.dcd`, `pre_equilibration.dcd`, `pressure_ramp.dcd`, `npt_equilibration.dcd`: Trajectories from all stages.
    *   Corresponding log files.
*   **Analysis Method:** Visualizes the continuous evolution of the system through all preparation stages.
*   **Output Files:**
    *   `25_equilibration_pathway.png`: Multi-panel evolution plot.
    *   `equilibration_stages_summary.csv`.

### Module 10: Structural Data Analysis
*   **Input Files:**
    *   `equilibrated_system.data`: Final system topology.
*   **Analysis Method:** Static analysis of bond lengths and hydration shell composition from the data file.
*   **Output Files:**
    *   `26_bond_lengths.png`: C60 bond length distribution.
    *   `27_hydration_shell.png`: Water count in first solvation shell.
    *   `28_radial_distributions.png`: Static RDFs.
    *   `c60_structural_integrity.csv`, `hydration_shell_composition.csv`.

### Module 11: DCD Trajectory Movies
*   **Input Files:**
    *   `production.lammpstrj` (or `.dcd`): Trajectory.
*   **Analysis Method:** Renders 3D animations of the simulation trajectory.
*   **Output Files:**
    *   `trajectory_movie.mp4` (or similar video format).

### Module 12: Equilibration Convergence Analysis
*   **Input Files:**
    *   Log files from all equilibration stages.
*   **Analysis Method:** Detailed convergence checks for T, P, Density, Energy during equilibration.
*   **Output Files:**
    *   `35_convergence_comparison_eps*.png`: Convergence plots per epsilon.
    *   `convergence_metrics_eps*.csv`.

### Module 13: Log File Performance Analysis
*   **Input Files:**
    *   Log files from all stages.
*   **Analysis Method:** Extracts performance metrics (ns/day, hours/ns) from LAMMPS logs.
*   **Output Files:**
    *   `production_performance.csv`: Performance metrics.
    *   `equilibration_performance_by_stage.csv`.

### Module 14: System Validation
*   **Input Files:**
    *   `equilibrated_system.data`: Topology.
*   **Analysis Method:** Validates force field parameters (bond lengths, angles, LJ params) against reference values.
*   **Output Files:**
    *   `system_validation_metrics.csv`.

### Module 15: Thermal Trajectory Analysis
*   **Input Files:**
    *   Log files.
*   **Analysis Method:** Checks energy conservation and thermalization efficiency.
*   **Output Files:**
    *   `thermal_analysis_metrics_eps*.csv`.

### Module 16: Advanced CUDA Trajectory Analysis
*   **Input Files:**
    *   `production.lammpstrj`: Trajectory.
*   **Analysis Method:** Advanced GPU-accelerated structural analysis including 3D density maps and orientation maps.
*   **Output Files:**
    *   `52_advanced_rdf.png`: High-resolution RDF.
    *   `53_tetrahedral_order.png`: Tetrahedral order parameter.
    *   `54_orientation_heatmaps_all.png`: Water orientation vs distance.
    *   `55_coordination_number.png`: Coordination analysis.
    *   `56_tetrahedral_vs_dist_all.png`: Order parameter spatial distribution.
    *   `57_dynamics_thermodynamics.png`: Correlation plots.
    *   `comprehensive_analysis_results.json`, `comprehensive_metrics.csv`.
