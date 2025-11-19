#!/usr/bin/env python3
"""
Simulation Status Checker
==========================

Quickly check the status of all 6 epsilon simulations.

Author: Scientific Analysis Suite
Date: November 2025
"""

import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path("/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2")
EPSILON_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
TIMESTEP = 2.0  # fs
PRODUCTION_START = 600000
TOTAL_PRODUCTION_STEPS = 2000000

def check_simulation_status():
    """Check completion status of all simulations"""
    
    print("="*80)
    print(" "*25 + "SIMULATION STATUS CHECK")
    print("="*80)
    print()
    
    status_data = []
    
    for eps in EPSILON_VALUES:
        if eps == 0.0:
            eps_dir = BASE_DIR / "epsilon_0.0"
        else:
            eps_dir = BASE_DIR / f"epsilon_{eps:.2f}"
        log_file = eps_dir / "equilibration.log"
        
        if not log_file.exists():
            status_data.append({
                'Epsilon': f"{eps:.2f}",
                'Status': 'NOT STARTED',
                'Current_Step': 0,
                'Progress': '0%',
                'Prod_Time_ns': 0.0,
                'Remaining_ns': 4.0
            })
            continue
        
        # Read last line of log file to get current step
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Find last line with step number
            current_step = 0
            for line in reversed(lines):
                parts = line.strip().split()
                if len(parts) > 0 and parts[0].isdigit():
                    current_step = int(parts[0])
                    break
            
            # Calculate progress
            if current_step < PRODUCTION_START:
                stage = "EQUILIBRATION"
                progress = (current_step / PRODUCTION_START) * 100
                prod_time = 0.0
            else:
                stage = "PRODUCTION"
                prod_steps = current_step - PRODUCTION_START
                progress = (prod_steps / TOTAL_PRODUCTION_STEPS) * 100
                prod_time = (prod_steps * TIMESTEP) / 1e6  # ns
            
            remaining = 4.0 - prod_time
            
            if current_step >= (PRODUCTION_START + TOTAL_PRODUCTION_STEPS):
                status = "✓ COMPLETE"
            elif stage == "PRODUCTION":
                status = "⚡ RUNNING (PRODUCTION)"
            else:
                status = "⚡ RUNNING (EQUILIBRATION)"
            
            status_data.append({
                'Epsilon': f"{eps:.2f}",
                'Status': status,
                'Current_Step': current_step,
                'Progress': f"{progress:.1f}%",
                'Prod_Time_ns': f"{prod_time:.2f}",
                'Remaining_ns': f"{remaining:.2f}"
            })
            
        except Exception as e:
            status_data.append({
                'Epsilon': f"{eps:.2f}",
                'Status': 'ERROR',
                'Current_Step': '?',
                'Progress': '?',
                'Prod_Time_ns': '?',
                'Remaining_ns': '?'
            })
    
    # Create DataFrame for nice display
    df = pd.DataFrame(status_data)
    
    # Print table
    print(df.to_string(index=False))
    print()
    print("="*80)
    
    # Summary
    complete_count = sum(1 for row in status_data if '✓' in row['Status'])
    running_count = sum(1 for row in status_data if '⚡' in row['Status'])
    
    print(f"\nSummary: {complete_count} complete, {running_count} running, {6 - complete_count - running_count} not started")
    
    if complete_count == 6:
        print("\n🎉 ALL SIMULATIONS COMPLETE! Ready for full analysis.")
        print(f"\nRun analyses with:")
        print(f"  cd {BASE_DIR}/analysis/codes")
        print(f"  python run_all_analyses.py")
        return True
    elif running_count > 0:
        # Estimate remaining time
        avg_prod_time = sum(float(row['Prod_Time_ns']) for row in status_data if row['Prod_Time_ns'] != '?') / len([r for r in status_data if r['Prod_Time_ns'] != '?'])
        avg_remaining = sum(float(row['Remaining_ns']) for row in status_data if row['Remaining_ns'] != '?') / len([r for r in status_data if r['Remaining_ns'] != '?'])
        print(f"\n⏱️  Average progress: {avg_prod_time:.2f} / 4.00 ns")
        print(f"  Estimated remaining: ~{avg_remaining:.2f} ns ({avg_remaining/4.0*100:.0f}% of production)")
        return False
    else:
        print("\n⚠️  No simulations running. Check system status.")
        return False

def main():
    """Main entry point"""
    complete = check_simulation_status()
    sys.exit(0 if complete else 1)

if __name__ == "__main__":
    main()
