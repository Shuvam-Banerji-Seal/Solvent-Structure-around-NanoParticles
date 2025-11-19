#!/usr/bin/env python3
"""
PPM SNAPSHOT ANALYSIS
=====================

Analyzes 200 PPM snapshot images from production runs:
- Extracts statistics from snapshots (brightness, texture)
- Creates image montages
- Tracks visual changes across epsilon values
- Generates statistics from image data

PPM files contain:
- production_610000.ppm through production_2570000.ppm
- One every ~10,000 timesteps during production
- Raw bitmap format from LAMMPS

Author: AI Analysis Suite
Date: 2024-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import imageio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

class PPMAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.epsilon_dirs = {
            0.0: 'epsilon_0.0',
            0.05: 'epsilon_0.05',
            0.1: 'epsilon_0.10',
            0.15: 'epsilon_0.15',
            0.2: 'epsilon_0.20',
            0.25: 'epsilon_0.25'
        }
        self.plots_dir = self.base_dir / 'analysis' / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_snapshots(self):
        """
        Analyze all PPM snapshots
        Extract brightness, contrast, and structure metrics
        """
        print("\n" + "="*80)
        print("PPM SNAPSHOT ANALYSIS")
        print("="*80)
        
        results = {}
        
        for eps in self.epsilon_values:
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            ppm_files = sorted(eps_dir.glob('production_*.ppm'))
            
            if not ppm_files:
                print(f"  ⚠ ε={eps}: No PPM files found")
                continue
            
            print(f"\n[ε={eps}] Analyzing {len(ppm_files)} snapshots...")
            
            brightness_list = []
            contrast_list = []
            timestamps = []
            
            for ppm_file in tqdm(ppm_files[::5], desc=f"ε={eps}"):  # Every 5th image
                try:
                    # Read PPM file
                    img = imageio.imread(str(ppm_file))
                    
                    # Convert to grayscale if RGB
                    if len(img.shape) == 3:
                        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
                    else:
                        img_gray = img
                    
                    # Calculate metrics
                    brightness = img_gray.mean()
                    contrast = img_gray.std()
                    
                    brightness_list.append(brightness)
                    contrast_list.append(contrast)
                    
                    # Extract timestep from filename
                    ts = int(ppm_file.stem.split('_')[1])
                    timestamps.append(ts)
                
                except Exception as e:
                    print(f"    Error reading {ppm_file.name}: {e}")
                    continue
            
            if len(brightness_list) > 0:
                df = pd.DataFrame({
                    'timestep': timestamps,
                    'brightness': brightness_list,
                    'contrast': contrast_list
                })
                
                results[eps] = {
                    'data': df,
                    'mean_brightness': np.mean(brightness_list),
                    'mean_contrast': np.mean(contrast_list),
                    'n_snapshots': len(ppm_files)
                }
                
                print(f"  Mean brightness: {results[eps]['mean_brightness']:.1f}")
                print(f"  Mean contrast: {results[eps]['mean_contrast']:.1f}")
        
        self.results_ppm = results
        
        # Save summary
        if results:
            summary = pd.DataFrame({
                eps: {
                    'mean_brightness': results[eps]['mean_brightness'],
                    'mean_contrast': results[eps]['mean_contrast'],
                    'n_snapshots': results[eps]['n_snapshots']
                } for eps in results.keys()
            }).T
            summary.to_csv(self.plots_dir / 'ppm_snapshot_statistics.csv')
            print(f"\n✓ Statistics saved")
    
    def create_montage(self):
        """Create montage of representative snapshots"""
        if not hasattr(self, 'results_ppm'):
            return
        
        print("\nCreating snapshot montages...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, eps in enumerate(self.epsilon_values):
            if eps not in self.results_ppm:
                axes[idx].text(0.5, 0.5, f'No data for ε={eps}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            eps_dir = self.base_dir / self.epsilon_dirs[eps]
            ppm_files = sorted(eps_dir.glob('production_*.ppm'))
            
            # Get middle snapshot
            if len(ppm_files) > 0:
                mid_idx = len(ppm_files) // 2
                img = imageio.imread(str(ppm_files[mid_idx]))
                
                axes[idx].imshow(img)
                axes[idx].set_title(f'ε={eps:.2f}, t={ppm_files[mid_idx].stem.split("_")[1]} steps')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '23_ppm_snapshot_montage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 23_ppm_snapshot_montage.png")
    
    def plot_snapshot_metrics(self):
        """Plot brightness and contrast evolution"""
        if not hasattr(self, 'results_ppm'):
            return
        
        print("\nGenerating snapshot metric plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        eps_list = sorted(self.results_ppm.keys())
        brightness_means = [self.results_ppm[eps]['mean_brightness'] for eps in eps_list]
        contrast_means = [self.results_ppm[eps]['mean_contrast'] for eps in eps_list]
        
        ax1.plot(eps_list, brightness_means, 'o-', linewidth=2, markersize=10)
        ax1.set_xlabel('ε (kcal/mol)')
        ax1.set_ylabel('Mean Brightness')
        ax1.set_title('Snapshot Brightness vs Hydrophobicity')
        ax1.grid(alpha=0.3)
        
        ax2.plot(eps_list, contrast_means, 's-', linewidth=2, markersize=10, color='red')
        ax2.set_xlabel('ε (kcal/mol)')
        ax2.set_ylabel('Mean Contrast (std dev)')
        ax2.set_title('Snapshot Contrast vs Hydrophobicity')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / '24_ppm_metrics.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 24_ppm_metrics.png")

def main():
    print("="*80)
    print("PPM SNAPSHOT ANALYSIS")
    print("="*80)
    
    base_dir = Path('/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2')
    analyzer = PPMAnalyzer(base_dir)
    
    try:
        analyzer.analyze_snapshots()
        analyzer.create_montage()
        analyzer.plot_snapshot_metrics()
        
        print("\n" + "="*80)
        print("✓ PPM SNAPSHOT ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()