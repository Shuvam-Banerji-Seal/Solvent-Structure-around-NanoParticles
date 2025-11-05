# Nanoparticle Examples - Available for Solvent Study

## Overview

This directory now contains multiple nanoparticle structures ready for solvation studies. Each nanoparticle represents a different material and structure type, enabling comparative analysis of solvent behavior.

## Available Nanoparticles

### 1. SiC Nanoparticle (Silicon Carbide)
**File:** `sic_nanoparticle.data`  
**Size:** 772 bytes  
**Atoms:** 8 atoms  
**Composition:** Silicon and Carbon  
**Source:** Original project nanoparticle  

**Properties:**
- Simple cubic-like structure
- Mixed Si-C bonding
- Small test system
- Good for quick tests and method development

**Use Cases:**
- Method development
- Quick validation runs
- Testing new water models

---

### 2. GO Nanoparticle (Graphene Oxide)
**File:** `GO_nanoparticle.data`  
**Size:** 90 KB  
**Source:** lammps-input-files/GO-nanoparticle  

**Properties:**
- Graphene-based structure
- Oxygen functional groups
- Hydrophilic character
- Large flat surface
- ~thousands of atoms

**Use Cases:**
- Study water layering on flat surfaces
- Hydrogen bonding with oxygen functional groups
- 2D material-water interactions
- Surface hydration

**Scientific Interest:**
- GO is widely used in water purification
- Understanding GO-water interactions crucial for applications
- Strong hydrogen bonding expected

---

### 3. Amorphous Carbon Nanoparticle
**File:** `amorphous_carbon.data`  
**Size:** 664 KB (largest)  
**Source:** lammps-input-files/amorphous-carbon  

**Properties:**
- Disordered carbon structure
- No crystalline order
- Rough surface
- Many surface sites
- ~thousands of atoms

**Use Cases:**
- Study solvent structure around irregular surfaces
- Compare with graphene (ordered) vs amorphous (disordered)
- Realistic nanoparticle morphology
- Surface roughness effects

**Scientific Interest:**
- Amorphous carbon common in environmental nanoparticles
- Surface roughness affects solvent layering
- Hydrophobic character

---

## Comparative Study Plan

### Research Questions

1. **Surface Chemistry Effects**
   - How does surface composition (Si/C vs pure C vs GO) affect water structure?
   - Role of oxygen functional groups (GO) in water ordering

2. **Surface Topology Effects**
   - Flat (GO) vs curved (SiC) vs rough (amorphous) surfaces
   - How does surface roughness affect solvent layering?

3. **Hydrophilicity/Hydrophobicity**
   - SiC: Moderate
   - GO: Hydrophilic (due to O groups)
   - Amorphous C: Hydrophobic

4. **Size Effects**
   - Small (SiC, 8 atoms) vs Large (GO, amorphous C, thousands)
   - Curvature effects on water structure

### Analysis Metrics

For each nanoparticle, calculate:

1. **Radial Distribution Functions (RDF)**
   - g(r) for NP-O (water oxygen)
   - g(r) for NP-H (water hydrogen)
   - g(r) for O-O, O-H (bulk water structure)

2. **Coordination Numbers**
   - First shell water count
   - Second shell water count
   - Comparison to bulk

3. **Hydrogen Bonding**
   - H-bonds per water molecule vs distance from NP
   - H-bond network disruption near surface

4. **Water Orientation**
   - Dipole angle distribution vs distance from NP
   - Preferential orientation near surface

5. **Density Profiles**
   - Water density as function of distance from NP
   - Layering structure

### Recommended Workflow

```bash
# For each nanoparticle:

# 1. Generate solvated system (100-1000 waters)
python3 prepare_system.py <nanoparticle.data> 500 auto

# 2. Run equilibration (NVT, 100 ps)
mpirun -np 4 lmp_mpi -in equilibrate.in

# 3. Run production (NPT, 500 ps - 1 ns)
mpirun -np 4 lmp_mpi -in production.in

# 4. Analyze trajectories
python3 analyze_rdf.py trajectory.lammpstrj
python3 analyze_coordination.py trajectory.lammpstrj
python3 analyze_hbonds.py trajectory.lammpstrj
python3 analyze_orientation.py trajectory.lammpstrj

# 5. Compare results
python3 compare_nanoparticles.py
```

### Expected Timeline

**Per Nanoparticle:**
- System preparation: 10 minutes
- Equilibration: 30 minutes - 2 hours
- Production: 2-8 hours
- Analysis: 1-2 hours

**Total for 3 Nanoparticles:** 1-2 days

### Output Structure

```
solvent_effects/
├── input_files/
│   ├── sic_nanoparticle.data
│   ├── GO_nanoparticle.data
│   └── amorphous_carbon.data
├── output/
│   ├── sic/
│   │   ├── trajectories/
│   │   ├── rdf/
│   │   ├── coordination/
│   │   └── analysis/
│   ├── GO/
│   │   └── ...
│   └── amorphous_carbon/
│       └── ...
└── analysis/
    ├── rdf_comparison.png
    ├── coordination_comparison.png
    └── summary_table.csv
```

## Water Model Recommendation

Based on extensive testing (see `TIP5P_STATUS_REPORT.md`):

**Use TIP4P/2005:**
- ✅ Proven to work in LAMMPS
- ✅ Full electrostatics
- ✅ MPI parallelization
- ✅ SHAKE constraints work reliably
- ✅ ~90% of TIP5P quality
- ✅ 30 minutes to working simulation

**Avoid TIP5P** (for now):
- ❌ Domain decomposition issues
- ❌ rigid/small compatibility problems
- ❌ Unknown time to resolution
- ⚠️ May work in future LAMMPS versions

## Next Steps

1. **Implement TIP4P/2005 water model**
   - Copy H2O_TIP4P.txt from LAMMPS molecules
   - Adapt water-co2 example
   - Create universal solvation script

2. **Generate solvated systems**
   - SiC + 500 waters
   - GO + 1000 waters
   - Amorphous C + 1000 waters

3. **Run simulations**
   - Equilibrate each system
   - Production runs (500 ps - 1 ns)

4. **Analyze and compare**
   - Calculate all metrics
   - Create comparison plots
   - Write summary report

## References

- **GO nanoparticle:** lammps-input-files/inputs/GO-nanoparticle/README.md
- **Amorphous carbon:** lammps-input-files/inputs/amorphous-carbon/README.md
- **TIP4P/2005:** Abascal & Vega, J. Chem. Phys. 123, 234505 (2005)
- **TIP5P:** Rick, J. Chem. Phys. 120, 6085 (2004)

## Contact & Status

**Status:** Nanoparticles ready, water model decision pending  
**Next action:** Implement TIP4P/2005 OR continue debugging TIP5P  
**Estimated time to first results:** 4-8 hours (with TIP4P)

---
*Last updated: 2025-01-XX*  
*Nanoparticles copied from lammps-input-files repository*
