import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import sys

# Add current directory to path to import compare_generated
sys.path.append(str(Path(__file__).parent))
from compare_generated import load_thermo, load_rdf

def setup_plotting():
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_rdf_comparison(epsilons, base_dir, output_dir):
    """Plot RDF comparisons for selected epsilons."""
    pairs = ['CC', 'CO', 'OO']
    
    for eps in epsilons:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'RDF Comparison - Epsilon {eps:.2f}', fontsize=16)
        
        actual_dir = base_dir / f"solvent_effects/epsilon_{eps:.2f}"
        gen_dir = base_dir / f"ml_integration/advanced/generated/epsilon_{eps:.2f}"
        
        for i, pair in enumerate(pairs):
            ax = axes[i]
            
            # Load data
            act_file = actual_dir / f"rdf_{pair}.dat"
            gen_file = gen_dir / f"rdf_{pair}.dat"
            
            if act_file.exists() and gen_file.exists():
                rdf_act = load_rdf(act_file)
                rdf_gen = load_rdf(gen_file)
                
                ax.plot(rdf_act[:, 0], rdf_act[:, 1], 'k-', label='Actual', linewidth=2, alpha=0.7)
                ax.plot(rdf_gen[:, 0], rdf_gen[:, 1], 'r--', label='Predicted', linewidth=2)
                
                ax.set_title(f'{pair} Pair')
                ax.set_xlabel('r (Å)')
                ax.set_ylabel('g(r)')
                ax.legend()
                ax.set_xlim(0, 12)
            else:
                ax.text(0.5, 0.5, "Missing Data", ha='center', va='center')
                
        plt.tight_layout()
        plt.savefig(output_dir / f"rdf_comparison_eps_{eps:.2f}.png", dpi=150)
        plt.close()

def plot_thermo_distributions(epsilons, base_dir, output_dir):
    """Plot thermodynamic property distributions."""
    props = ['Temp', 'Press', 'PE', 'Density']
    # Map to dataframe columns
    col_map = {'Temp': 'Temp', 'Press': 'Press', 'PE': 'PE', 'Density': 'Dens'}
    
    for eps in epsilons:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Thermodynamics Distributions - Epsilon {eps:.2f}', fontsize=16)
        axes = axes.flatten()
        
        actual_file = base_dir / f"solvent_effects/epsilon_{eps:.2f}/production_detailed_thermo.dat"
        gen_file = base_dir / f"ml_integration/advanced/generated/epsilon_{eps:.2f}/production_detailed_thermo.dat"
        
        if actual_file.exists() and gen_file.exists():
            df_act = load_thermo(actual_file)
            df_gen = load_thermo(gen_file)
            
            for i, prop in enumerate(props):
                col = col_map[prop]
                ax = axes[i]
                
                sns.kdeplot(data=df_act, x=col, ax=ax, color='black', label='Actual', fill=True, alpha=0.1)
                sns.kdeplot(data=df_gen, x=col, ax=ax, color='red', label='Predicted', linestyle='--')
                
                ax.set_title(prop)
                ax.legend()
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "Missing Data", ha='center', va='center')
                
        plt.tight_layout()
        plt.savefig(output_dir / f"thermo_dist_eps_{eps:.2f}.png", dpi=150)
        plt.close()

def plot_metrics_summary(log_dir, output_dir):
    """Plot summary of metrics across all epsilons."""
    csv_path = log_dir / "batch_metrics.csv"
    if not csv_path.exists():
        print("Metrics file not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. RDF MSE vs Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(df['epsilon'], df['RDF_CC_MSE'], 'o-', label='C-C')
    plt.plot(df['epsilon'], df['RDF_CO_MSE'], 's-', label='C-O')
    plt.plot(df['epsilon'], df['RDF_OO_MSE'], '^-', label='O-O')
    
    # Mark training region
    plt.axvspan(0, 0.5, alpha=0.1, color='green', label='Training Region')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Squared Error')
    plt.title('RDF Prediction Error vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(output_dir / "summary_rdf_mse.png", dpi=150)
    plt.close()
    
    # 2. Thermo MSE vs Epsilon (Normalized/Relative if possible, but log scale works)
    plt.figure(figsize=(10, 6))
    cols = ['Thermo_Temp_MSE', 'Thermo_Press_MSE', 'Thermo_PE_MSE', 'Thermo_Dens_MSE']
    labels = ['Temperature', 'Pressure', 'Potential Energy', 'Density']
    
    for col, label in zip(cols, labels):
        if col in df.columns:
            plt.plot(df['epsilon'], df[col], 'o-', label=label)
            
    plt.axvspan(0, 0.5, alpha=0.1, color='green', label='Training Region')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.title('Thermodynamic Prediction Error vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(output_dir / "summary_thermo_mse.png", dpi=150)
    plt.close()

def main():
    setup_plotting()
    
    base_dir = Path("/store/shuvam/learning_solvent_effects")
    log_dir = base_dir / "ml_integration/advanced/logs"
    output_dir = log_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Select representative epsilons
    # 0.25 (Training), 0.50 (Boundary), 0.80 (Extrapolation), 1.10 (Far Extrapolation)
    epsilons = [0.25, 0.50, 0.80, 1.10]
    
    print("📊 Generating RDF comparison plots...")
    plot_rdf_comparison(epsilons, base_dir, output_dir)
    
    print("📊 Generating Thermodynamic distribution plots...")
    plot_thermo_distributions(epsilons, base_dir, output_dir)
    
    print("📊 Generating Summary Metrics plots...")
    plot_metrics_summary(log_dir, output_dir)
    
    print(f"✅ All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
