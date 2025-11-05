# TIP5P Solvation Study - Quick Reference

## 🚀 Quick Start

```bash
cd solvent_effects/shell_scripts

# Test run (2 minutes)
./run_tip5p_simulation.sh -n 100 -t 10 -c 4

# Production run (4-6 hours)
./run_tip5p_simulation.sh -n 5000 -t 500 -c 10
```

---

## 📂 Project Structure

```
solvent_effects/
├── README.md                      # Project overview
├── input_files/
│   ├── sic_nanoparticle.data     # SiC nanoparticle (8 atoms)
│   └── solvation_tip5p.in        # LAMMPS input script (260 lines)
├── python_scripts/
│   └── prepare_tip5p_system.py   # Water placement (478 lines)
├── shell_scripts/
│   └── run_tip5p_simulation.sh   # Main automation (500 lines)
├── docs/
│   ├── SETUP_GUIDE.md            # Complete guide (543 lines)
│   └── IMPLEMENTATION_SUMMARY.md # Technical details (525 lines)
├── output/                        # Simulation results (auto-created)
└── analysis/                      # Analysis results (auto-created)
```

**Total: 2455 lines of code and documentation**

---

## 🎯 Common Commands

### Run Simulations

```bash
cd shell_scripts

# Quick test
./run_tip5p_simulation.sh -n 100 -t 10 -c 4

# Standard production
./run_tip5p_simulation.sh -n 5000 -t 500 -c 10

# Large system
./run_tip5p_simulation.sh -n 10000 -t 1000 -c 20

# Solvation shell only
./run_tip5p_simulation.sh -n 2000 -s shell -t 500 -c 10

# Custom box size
./run_tip5p_simulation.sh -n 5000 -b 60.0 -t 500 -c 10

# Help
./run_tip5p_simulation.sh --help
```

### Visualize

```bash
# Find latest output
cd ../output
ls -lt | head

# Visualize with VMD
vmd production_tip5p_*/production.lammpstrj
```

### Analyze

```bash
# Check energy drift
grep "^[0-9]" output/production_*/log.lammps | awk '{print $1, $4}' > energy.dat

# Plot temperature
grep "^[0-9]" output/production_*/log.lammps | awk '{print $1, $2}' > temp.dat

# Count frames
grep -c "ITEM: TIMESTEP" output/production_*/production.lammpstrj
```

---

## 🔧 Command Line Options

```
./run_tip5p_simulation.sh [options]

Options:
  -n, --nwaters N      Number of water molecules (default: 3000)
  -b, --boxsize SIZE   Box size in Å (default: auto)
  -s, --strategy TYPE  'full_box' or 'shell' (default: full_box)
  -t, --time TIME      Production time in ps (default: 500)
  -c, --cores N        MPI cores (default: 10)
  -h, --help           Show help
```

---

## 📊 Output Files

Each simulation creates:

```
output/production_tip5p_<N>waters_<T>ps_<timestamp>/
├── production.lammpstrj          # Trajectory (VMD)
├── production_custom.dump        # Detailed trajectory
├── temperature.dat               # Temperature vs time
├── pressure.dat                  # Pressure vs time
├── energy.dat                    # Energy components
├── final_configuration.data      # Final structure
├── restart.*.lmp                 # Restart files
├── log.lammps                    # LAMMPS log
└── simulation_info.txt           # Parameters
```

---

## 🧪 TIP5P Parameters

### Geometry
- O-H bond: **0.9572 Å**
- H-O-H angle: **104.52°**
- O-L distance: **0.70 Å** (lone pairs)
- L-O-L angle: **109.47°**

### Charges
- O: **q = 0.0**
- H: **q = +0.241**
- L: **q = -0.241**

### LJ (oxygen)
- ε = **0.16 kcal/mol**
- σ = **3.12 Å**

---

## ✅ Validation Checklist

After simulation, check:

- [ ] **Energy drift < 5%**
  ```bash
  # Check in log file or output
  ```

- [ ] **Temperature stable at 300 ± 10 K**
  ```bash
  grep "^[0-9]" log.lammps | awk '{print $2}' | tail -100
  ```

- [ ] **No errors in log.lammps**
  ```bash
  grep -i "error\|warning" log.lammps
  ```

- [ ] **Trajectory looks reasonable**
  ```bash
  vmd production.lammpstrj
  ```

- [ ] **Density approximately correct**
  ```bash
  grep "Density" simulation_info.txt
  ```

---

## ⚠️ Troubleshooting

### LAMMPS not found
```bash
which lmp
which lmp_mpi
export PATH=/path/to/lammps:$PATH
```

### NumPy not found
```bash
pip3 install numpy
```

### High energy drift
- Increase equilibration time (edit `.in` file)
- Reduce timestep from 0.5 to 0.2 fs
- Check for atom overlaps in initial structure

### Slow simulation
- Use more cores: `-c 20`
- Reduce output frequency (edit `.in` file)
- Use smaller system: `-n 2000 -s shell`

### Can't place all waters
- Increase box size: `-b 60.0`
- Reduce water count
- Use shell strategy: `-s shell`

---

## 📚 Documentation

- **Setup Guide:** `docs/SETUP_GUIDE.md` - Complete installation and usage
- **Implementation:** `docs/IMPLEMENTATION_SUMMARY.md` - Technical details
- **Project Overview:** `README.md` - Goals and features
- **LAMMPS Docs:** https://docs.lammps.org/Howto_tip5p.html

---

## 🎓 Expected Improvements

| Property | Old (SPC/E) | New (TIP5P) |
|----------|-------------|-------------|
| Energy drift | 115% | <5% |
| Electrostatics | None | Full PPPM |
| H-bonding | Not possible | Accurate |
| Sim time | 125 ps | 500-1000 ps |
| Temperature | 300 ± 3 K | 300 ± 1 K |
| Analysis | Basic | Comprehensive |

---

## 🏃 Typical Workflow

1. **Test:**
   ```bash
   ./run_tip5p_simulation.sh -n 100 -t 10 -c 4
   ```

2. **Validate:**
   - Check energy drift < 5%
   - Verify temperature stable
   - Look at VMD trajectory

3. **Production:**
   ```bash
   ./run_tip5p_simulation.sh -n 5000 -t 500 -c 10
   ```

4. **Analyze:**
   - RDF calculations
   - Coordination numbers
   - H-bond analysis
   - Water orientation

5. **Compare:**
   - Old vs new results
   - Different parameters
   - Literature values

---

## 💡 Pro Tips

- **Start small:** Test with 100 waters before large runs
- **Monitor:** Check log files during long runs
- **Backup:** Output directories are timestamped
- **Parallelize:** Use `-c 20` for faster runs
- **Document:** Keep notes on parameter choices

---

## 📞 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `prepare_tip5p_system.py` | 478 | Place waters |
| `solvation_tip5p.in` | 260 | LAMMPS protocol |
| `run_tip5p_simulation.sh` | 500 | Automation |
| `SETUP_GUIDE.md` | 543 | Documentation |
| `IMPLEMENTATION_SUMMARY.md` | 525 | Technical details |

---

**Status:** ✅ Ready for testing  
**Last Updated:** November 4, 2025

---

*For detailed information, see `docs/SETUP_GUIDE.md`*
