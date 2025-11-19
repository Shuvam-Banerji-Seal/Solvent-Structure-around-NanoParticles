#!/usr/bin/env python3
"""
Master Analysis Script - C60 Nanoparticle Solvation Study
==========================================================

This script orchestrates all analysis modules:
1. Thermodynamic analysis
2. Equilibration and stability analysis  
3. RDF and structural analysis
4. Trajectory-based dynamics analysis (will be created separately)

Run this script to perform complete analysis.

Author: Scientific Analysis Suite
Date: November 2025
"""

import subprocess
import sys
from pathlib import Path
import time

# Define paths
BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")
CODES_DIR = BASE_DIR / "analysis" / "codes"

def run_analysis_script(script_name, description):
    """Run an analysis script and report results"""
    script_path = CODES_DIR / script_name
    
    if not script_path.exists():
        print(f"\n⚠️  WARNING: {script_name} not found, skipping...")
        return False
    
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Analysis failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False

def main():
    """Run all analysis modules"""
    print("="*70)
    print("C60 NANOPARTICLE SOLVATION STUDY - COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Analysis codes: {CODES_DIR}")
    
    # List of analysis modules
    # Analysis scripts to run
    analyses = [
        ("01_thermodynamic_analysis.py", "Thermodynamic Properties Analysis"),
        ("02_equilibration_stability_analysis.py", "Equilibration & Stability Assessment"),
        ("03_rdf_structural_analysis.py", "Radial Distribution Functions"),
        ("04_comprehensive_water_structure_CUDA.py", "Comprehensive Water Structure (CUDA)"),
        ("05_plot_water_structure.py", "Water Structure Visualization"),
    ]
    
    results = {}
    total_start = time.time()
    
    # Run each analysis
    for script, description in analyses:
        success = run_analysis_script(script, description)
        results[description] = "✓ Success" if success else "✗ Failed"
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    for analysis, status in results.items():
        print(f"  {status:12} - {analysis}")
    
    print(f"\nTotal time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print("="*70)
    
    # Check if all succeeded
    all_success = all("Success" in status for status in results.values())
    
    if all_success:
        print("\n🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print(f"\nResults saved to: {BASE_DIR / 'analysis' / 'plots'}")
    else:
        print("\n⚠️  Some analyses failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
