import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path to import compare_generated
sys.path.append(str(Path(__file__).parent))
from compare_generated import compare_thermo, compare_rdf

def run_batch_comparison_vae():
    epsilons = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    generated_base_dir = Path("../generated_vae")
    output_base_dir = Path("../logs_vae/comparison")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting VAE batch comparison analysis for {len(epsilons)} epsilons...")
    print(f"   Generated dir: {generated_base_dir}")
    print(f"   Output dir: {output_base_dir}")
    
    results = []
    
    for eps in epsilons:
        print(f"\nAnalyzing epsilon {eps:.2f}...")
        
        gen_dir = generated_base_dir / f"epsilon_{eps:.2f}"
        actual_dir = Path(f"/store/shuvam/learning_solvent_effects/solvent_effects/epsilon_{eps:.2f}")
        
        if not gen_dir.exists():
            print(f"⚠️  Generated directory not found: {gen_dir}")
            continue
            
        if not actual_dir.exists():
            print(f"⚠️  Actual directory not found: {actual_dir}")
            continue
            
        try:
            # Compare thermodynamics (actual first, then generated)
            thermo_metrics = compare_thermo(
                actual_dir / "production_detailed_thermo.dat",
                gen_dir / "production_detailed_thermo.dat"
            )
            
            # Compare RDFs (returns dict with CC, CO, OO keys)
            rdf_metrics = compare_rdf(actual_dir, gen_dir)
            
            # Aggregate results
            res = {
                'epsilon': eps,
                'thermo_mse': {k: v['MSE'] for k, v in thermo_metrics.items()},
                'rdf_mse': {k: v['MSE'] for k, v in rdf_metrics.items() if v},
                'rdf_corr': {k: v['Corr'] for k, v in rdf_metrics.items() if v}
            }
            results.append(res)
            print(f"✅ Epsilon {eps:.2f} analysis complete")
            
        except Exception as e:
            print(f"❌ Error analyzing epsilon {eps:.2f}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save aggregate results
    if results:
        df = pd.DataFrame(results)
        
        # Flatten dictionaries
        thermo_df = pd.json_normalize(df['thermo_mse']).add_prefix('thermo_mse_')
        rdf_mse_df = pd.json_normalize(df['rdf_mse']).add_prefix('rdf_mse_')
        rdf_corr_df = pd.json_normalize(df['rdf_corr']).add_prefix('rdf_corr_')
        
        final_df = pd.concat([df[['epsilon']], thermo_df, rdf_mse_df, rdf_corr_df], axis=1)
        
        csv_path = output_base_dir / "comparison_summary.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"\n💾 Summary saved to {csv_path}")
        
        # Plot summary
        plot_summary(final_df, output_base_dir)

def plot_summary(df, output_dir):
    # RDF MSE
    plt.figure(figsize=(10, 6))
    for pair in ['CC', 'CO', 'OO']:
        if f'rdf_mse_{pair}' in df.columns:
            plt.plot(df['epsilon'], df[f'rdf_mse_{pair}'], marker='o', label=pair)
    plt.xlabel('Epsilon')
    plt.ylabel('RDF MSE')
    plt.title('RDF Prediction Error vs Epsilon (VAE MODEL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'summary_rdf_mse.png')
    plt.close()
    
    # Thermo MSE (Log scale)
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        if 'thermo_mse' in col:
            name = col.replace('thermo_mse_', '')
            plt.semilogy(df['epsilon'], df[col], marker='o', label=name)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE (Log Scale)')
    plt.title('Thermodynamic Property Error vs Epsilon (VAE MODEL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'summary_thermo_mse.png')
    plt.close()

if __name__ == "__main__":
    run_batch_comparison_vae()
