import subprocess
import sys
from pathlib import Path

def generate_batch_improved():
    epsilons = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    model_path = "../checkpoints_improved/best_model_improved.pt"
    
    print(f"🚀 Starting IMPROVED batch generation for {len(epsilons)} epsilons...")
    
    for eps in epsilons:
        output_dir = f"../generated_improved/epsilon_{eps:.2f}"
        print(f"\nGenerating for epsilon {eps:.2f} -> {output_dir}")
        
        cmd = [
            "python", "generate_files_improved.py",
            "--model", model_path,
            "--epsilon", f"{eps:.2f}",
            "--output", output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Epsilon {eps:.2f} complete")
        except subprocess.CalledProcessError as e:
            print(f"❌ Epsilon {eps:.2f} failed with code {e.returncode}")
            sys.exit(1)

if __name__ == "__main__":
    generate_batch_improved()
