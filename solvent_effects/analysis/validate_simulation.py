#!/usr/bin/env python3
"""
Comprehensive Simulation Validation Script
==========================================

Validates TIP4P/2005 simulations by checking:
1. Thermodynamic stability (temperature, energy, pressure)
2. Structural validity (RDF from LAMMPS output)
3. Trajectory completeness
4. Simulation quality metrics

Usage:
    python validate_simulation.py OUTPUT_DIR
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

class SimulationValidator:
    """Validates simulation quality and completeness"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.results = {}
        self.issues = []
        self.warnings = []
        
    def check_files(self):
        """Check if all required output files exist"""
        print("=" * 70)
        print("FILE VALIDATION")
        print("=" * 70)
        
        required_files = [
            ('trajectory.lammpstrj', 'Trajectory'),
            ('log.lammps', 'LAMMPS log'),
            ('final_config.data', 'Final configuration'),
            ('final.restart', 'Restart file'),
            ('rdf_np_water.dat', 'RDF data')
        ]
        
        all_present = True
        for filename, description in required_files:
            filepath = self.output_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  ✓ {description:25s} {filename:30s} ({size/1024:.1f} KB)")
            else:
                print(f"  ✗ {description:25s} {filename:30s} MISSING")
                all_present = False
                self.issues.append(f"Missing file: {filename}")
        
        print()
        return all_present
    
    def parse_log(self):
        """Parse LAMMPS log file"""
        print("=" * 70)
        print("THERMODYNAMIC ANALYSIS")
        print("=" * 70)
        
        log_file = self.output_dir / 'log.lammps'
        
        data = {
            'step': [], 'time': [], 'temp': [], 'press': [],
            'pe': [], 'ke': [], 'etotal': [], 'vol': [], 'density': []
        }
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        reading_data = False
        for line in lines:
            line = line.strip()
            
            if line.startswith('Step'):
                reading_data = True
                continue
            
            if reading_data:
                if line.startswith('Loop') or line.startswith('Setting') or line == '':
                    reading_data = False
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 9:
                        data['step'].append(float(parts[0]))
                        data['time'].append(float(parts[1]))
                        data['temp'].append(float(parts[2]))
                        data['press'].append(float(parts[3]))
                        data['pe'].append(float(parts[4]))
                        data['ke'].append(float(parts[5]))
                        data['etotal'].append(float(parts[6]))
                        data['vol'].append(float(parts[7]))
                        data['density'].append(float(parts[8]))
                except (ValueError, IndexError):
                    continue
        
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
        
        self.results['thermo'] = data
        
        # Calculate statistics
        if len(data['temp']) > 0:
            temp_avg = np.mean(data['temp'])
            temp_std = np.std(data['temp'])
            
            # Energy drift
            if len(data['etotal']) > 1:
                energy_drift = abs(data['etotal'][-1] - data['etotal'][0])
                energy_drift_pct = (energy_drift / abs(data['etotal'][0])) * 100 if data['etotal'][0] != 0 else 0
            else:
                energy_drift = 0
                energy_drift_pct = 0
            
            print(f"\nTemperature:")
            print(f"  Average: {temp_avg:.2f} ± {temp_std:.2f} K")
            print(f"  Target:  300.0 K")
            
            temp_dev = abs(temp_avg - 300.0)
            if temp_dev < 20:
                print(f"  Status:  ✓ EXCELLENT (within ±20 K)")
            elif temp_dev < 50:
                print(f"  Status:  ✓ GOOD (within ±50 K)")
                self.warnings.append(f"Temperature deviation: {temp_dev:.1f} K")
            else:
                print(f"  Status:  ⚠ WARNING (deviation: {temp_dev:.1f} K)")
                self.warnings.append(f"Large temperature deviation: {temp_dev:.1f} K")
            
            print(f"\nEnergy Drift:")
            print(f"  Total:   {energy_drift:.2f} kcal/mol")
            print(f"  Percent: {energy_drift_pct:.2f}%")
            
            if energy_drift_pct < 1:
                print(f"  Status:  ✓ EXCELLENT (< 1%)")
            elif energy_drift_pct < 5:
                print(f"  Status:  ✓ GOOD (< 5%)")
            elif energy_drift_pct < 10:
                print(f"  Status:  ⚠ ACCEPTABLE (< 10%)")
                self.warnings.append(f"Energy drift: {energy_drift_pct:.1f}%")
            else:
                print(f"  Status:  ❌ HIGH (> 10%)")
                self.issues.append(f"High energy drift: {energy_drift_pct:.1f}%")
            
            # Simulation length
            sim_time = data['time'][-1] if len(data['time']) > 0 else 0
            print(f"\nSimulation Time: {sim_time:.2f} ps")
            
            if sim_time < 50:
                print(f"  Status:  ⚠ VERY SHORT (< 50 ps)")
                self.warnings.append(f"Short simulation: {sim_time:.2f} ps")
            elif sim_time < 100:
                print(f"  Status:  ⚠ SHORT (< 100 ps, minimum for testing)")
            elif sim_time < 1000:
                print(f"  Status:  ✓ GOOD (test/equilibration scale)")
            else:
                print(f"  Status:  ✓ EXCELLENT (production scale)")
        
        print()
    
    def analyze_rdf(self):
        """Analyze RDF from LAMMPS output"""
        print("=" * 70)
        print("RADIAL DISTRIBUTION FUNCTION (RDF)")
        print("=" * 70)
        
        rdf_file = self.output_dir / 'rdf_np_water.dat'
        
        if not rdf_file.exists():
            print("  ✗ RDF file not found")
            self.issues.append("Missing RDF file")
            return
        
        # Parse RDF file
        data = []
        with open(rdf_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Format: timestep bin_number bin_coord value1 value2 ...
                        # We want the g(r) values
                        r = float(parts[1])  # bin coordinate
                        g_r = float(parts[2])  # First pair type RDF
                        data.append([r, g_r])
                except (ValueError, IndexError):
                    continue
        
        if len(data) == 0:
            print("  ✗ No RDF data found")
            self.issues.append("Empty RDF file")
            return
        
        data = np.array(data)
        r = data[:, 0]
        g_r = data[:, 1]
        
        # Find first peak (within first 6 Å)
        mask = r < 6.0
        if np.any(mask):
            peak_idx = np.argmax(g_r[mask])
            peak_r = r[mask][peak_idx]
            peak_g = g_r[mask][peak_idx]
            
            print(f"\nFirst Solvation Shell:")
            print(f"  Peak position: {peak_r:.2f} Å")
            print(f"  Peak height:   {peak_g:.3f}")
            
            if peak_g > 1.5:
                print(f"  Status:  ✓ STRONG ordering (g > 1.5)")
            elif peak_g > 1.0:
                print(f"  Status:  ✓ MODERATE ordering (g > 1.0)")
            else:
                print(f"  Status:  ⚠ WEAK ordering (g < 1.0)")
                self.warnings.append(f"Weak RDF peak: g = {peak_g:.3f}")
            
            # Check if bulk behavior is reached
            bulk_region = g_r[r > 10.0]
            if len(bulk_region) > 0:
                bulk_avg = np.mean(bulk_region)
                print(f"\nBulk Region (r > 10 Å):")
                print(f"  Average g(r): {bulk_avg:.3f}")
                
                if abs(bulk_avg - 1.0) < 0.2:
                    print(f"  Status:  ✓ EXCELLENT (approaches 1.0)")
                elif abs(bulk_avg - 1.0) < 0.5:
                    print(f"  Status:  ✓ GOOD")
                else:
                    print(f"  Status:  ⚠ POOR (should approach 1.0)")
                    self.warnings.append(f"RDF bulk value: {bulk_avg:.3f} (should be ~1.0)")
        
        self.results['rdf'] = (r, g_r)
        print()
    
    def check_trajectory(self):
        """Check trajectory file"""
        print("=" * 70)
        print("TRAJECTORY VALIDATION")
        print("=" * 70)
        
        traj_file = self.output_dir / 'trajectory.lammpstrj'
        
        if not traj_file.exists():
            print("  ✗ Trajectory file not found")
            self.issues.append("Missing trajectory file")
            return
        
        # Count frames
        with open(traj_file, 'r') as f:
            frames = 0
            for line in f:
                if line.strip() == "ITEM: TIMESTEP":
                    frames += 1
        
        print(f"\nTrajectory Frames: {frames}")
        
        if frames < 10:
            print(f"  Status:  ⚠ VERY FEW frames (< 10)")
            self.warnings.append(f"Only {frames} trajectory frames")
        elif frames < 50:
            print(f"  Status:  ⚠ FEW frames (< 50)")
        elif frames < 100:
            print(f"  Status:  ✓ ADEQUATE (50-100 frames)")
        else:
            print(f"  Status:  ✓ GOOD (> 100 frames)")
        
        self.results['n_frames'] = frames
        print()
    
    def generate_plots(self):
        """Generate validation plots"""
        print("=" * 70)
        print("GENERATING VALIDATION PLOTS")
        print("=" * 70)
        
        if 'thermo' not in self.results or 'rdf' not in self.results:
            print("  ⚠ Insufficient data for plotting")
            return
        
        thermo = self.results['thermo']
        r, g_r = self.results['rdf']
        
        # Create 3-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Temperature plot
        ax = axes[0]
        ax.plot(thermo['time'], thermo['temp'], 'b-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=300, color='r', linestyle='--', linewidth=2, label='Target (300 K)')
        ax.fill_between(thermo['time'], 280, 320, alpha=0.2, color='green', label='±20 K')
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title('Temperature Stability', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy plot
        ax = axes[1]
        ax.plot(thermo['time'], thermo['etotal'], 'g-', linewidth=1.5, label='Total')
        ax.plot(thermo['time'], thermo['pe'], 'r-', linewidth=1.5, alpha=0.7, label='Potential')
        ax.plot(thermo['time'], thermo['ke'], 'b-', linewidth=1.5, alpha=0.7, label='Kinetic')
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
        ax.set_title('Energy Conservation', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RDF plot
        ax = axes[2]
        ax.plot(r, g_r, 'b-', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Bulk (g=1)')
        ax.set_xlabel('Distance (Å)', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title('Radial Distribution Function: NP - Water O', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'validation_report.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        
        plt.close()
    
    def generate_report(self):
        """Generate text report"""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        # Overall status
        if len(self.issues) == 0:
            if len(self.warnings) == 0:
                print("\n  ✓✓ SIMULATION QUALITY: EXCELLENT")
                status = "EXCELLENT"
            else:
                print("\n  ✓ SIMULATION QUALITY: GOOD (with minor warnings)")
                status = "GOOD"
        else:
            print("\n  ⚠ SIMULATION QUALITY: NEEDS IMPROVEMENT")
            status = "NEEDS IMPROVEMENT"
        
        # List issues
        if self.issues:
            print(f"\n  Critical Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"    ❌ {issue}")
        
        # List warnings
        if self.warnings:
            print(f"\n  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"    ⚠ {warning}")
        
        # Recommendations
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        
        if 'thermo' in self.results:
            sim_time = self.results['thermo']['time'][-1] if len(self.results['thermo']['time']) > 0 else 0
            
            if sim_time < 100:
                print("  • Run longer simulation (at least 100 ps for testing, 1 ns for structure)")
            elif sim_time < 1000:
                print("  • For production: extend to 5-10 ns")
        
        if status != "EXCELLENT":
            print("  • Check equilibration: temperature should stabilize around 300 K")
            print("  • Verify energy conservation: < 5% drift is good, < 1% is excellent")
            print("  • Ensure RDF shows clear solvation shell structure")
        
        print("\n  For detailed analysis, use:")
        print(f"    python analyze_tip4p_simple.py {self.output_dir}/")
        print()
        
        # Save report
        report_file = self.output_dir / 'validation_summary.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SIMULATION VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Directory: {self.output_dir}\n\n")
            f.write(f"Overall Status: {status}\n\n")
            
            if self.issues:
                f.write(f"Critical Issues ({len(self.issues)}):\n")
                for issue in self.issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if self.warnings:
                f.write(f"Warnings ({len(self.warnings)}):\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            if 'thermo' in self.results:
                thermo = self.results['thermo']
                f.write("Thermodynamic Properties:\n")
                f.write(f"  Temperature: {np.mean(thermo['temp']):.2f} ± {np.std(thermo['temp']):.2f} K\n")
                f.write(f"  Simulation time: {thermo['time'][-1]:.2f} ps\n")
                f.write(f"  Trajectory frames: {self.results.get('n_frames', 'N/A')}\n")
        
        print(f"  ✓ Report saved: {report_file}\n")
        
        return status


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_simulation.py OUTPUT_DIR")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "SIMULATION VALIDATION" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    validator = SimulationValidator(output_dir)
    
    # Run all checks
    validator.check_files()
    validator.parse_log()
    validator.analyze_rdf()
    validator.check_trajectory()
    validator.generate_plots()
    status = validator.generate_report()
    
    print("=" * 70)
    print()
    
    # Exit with appropriate code
    if status == "NEEDS IMPROVEMENT":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
