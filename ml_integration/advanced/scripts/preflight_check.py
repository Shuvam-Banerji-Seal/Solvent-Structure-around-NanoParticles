#!/usr/bin/env python3
"""
Pre-Flight Check - Verify Everything Before Training
=====================================================

Rigorous verification of:
- CUDA availability and optimization
- Data loading
- Model forward/backward pass
- File generation
- Memory estimates

Author: Shuvam Banerji Seal
Date: November 2025
"""

import torch
import torch.cuda as cuda
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MDGenerativeModel
from dataset import create_dataloaders
from generate_files import MDFileGenerator
from train_production import PhysicsInformedLoss


def check_cuda():
    """Check CUDA availability and capabilities."""
    print("\n" + "="*70)
    print("  1. CUDA CHECK".center(70))
    print("="*70 + "\n")
    
    if not cuda.is_available():
        print("❌ CUDA not available! Training will be SLOW on CPU.")
        return False
    
    print(f"✅ CUDA available")
    print(f"  GPU: {cuda.get_device_name(0)}")
    print(f"  Memory: {cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version()}")
    print(f"  Compute Capability: {cuda.get_device_capability(0)}")
    
    # Check mixed precision support
    if cuda.get_device_capability(0)[0] >= 7:
        print(f"✅ Mixed Precision (FP16) supported")
    else:
        print(f"⚠️  Mixed Precision may not be optimal (older GPU)")
    
    return True


def check_data_loading():
    """Test data loading."""
    print("\n" + "="*70)
    print("  2. DATA LOADING CHECK".center(70))
    print("="*70 + "\n")
    
    try:
        print("Creating dataloaders (small test)...")
        train_loader, val_loader = create_dataloaders(
            train_epsilon=[0.0, 0.05],
            val_epsilon=[0.10],
            batch_size=1,
            traj_stride=40,
            max_traj_frames=10,
            cache_dir="ml_integration/advanced/data/cache"
        )
        
        print(f"✅ Dataloaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"\n✅ Batch loaded successfully")
        print(f"  Epsilon shape: {batch['epsilon'].shape}")
        if batch['trajectory'][0] is not None:
            print(f"  Trajectory shape: {batch['trajectory'][0].shape}")
        if batch['thermodynamics'][0] is not None:
            print(f"  Thermo keys: {list(batch['thermodynamics'][0].keys())}")
        if batch['rdfs'][0] is not None:
            print(f"  RDF pairs: {list(batch['rdfs'][0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model():
    """Test model creation and forward pass."""
    print("\n" + "="*70)
    print("  3. MODEL CHECK".center(70))
    print("="*70 + "\n")
    
    try:
        device = 'cuda' if cuda.is_available() else 'cpu'
        
        print(f"Creating model on {device}...")
        model = MDGenerativeModel(
            latent_dim=512,
            n_atoms=5541,
            thermo_seq_len=1000,
            rdf_bins=200
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # Forward pass
        print(f"\nTesting forward pass...")
        epsilon = torch.randn(2, 1).to(device)
        outputs = model(epsilon)
        
        print(f"✅ Forward pass successful")
        print(f"  Output shapes:")
        print(f"    Trajectory: {outputs['trajectory'].shape}")
        for key, val in outputs['thermodynamics'].items():
            print(f"    Thermo {key}: {val.shape}")
        for key, val in outputs['rdfs'].items():
            print(f"    RDF {key}: {val.shape}")
        
        # Backward pass
        print(f"\nTesting backward pass...")
        loss = outputs['trajectory'].sum() + outputs['rdfs']['CC'].sum()
        loss.backward()
        
        print(f"✅ Backward pass successful")
        
        # Memory usage
        if device == 'cuda':
            allocated = cuda.memory_allocated(0) / 1024**3
            reserved = cuda.memory_reserved(0) / 1024**3
            print(f"\nGPU Memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_loop():
    """Test training loop components."""
    print("\n" + "="*70)
    print("  4. TRAINING LOOP CHECK".center(70))
    print("="*70 + "\n")
    
    try:
        device = 'cuda' if cuda.is_available() else 'cpu'
        
        # Create components
        model = MDGenerativeModel(latent_dim=64, n_atoms=100, thermo_seq_len=10, rdf_bins=10).to(device)
        criterion = PhysicsInformedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        # Test training step
        epsilon = torch.randn(2, 1).to(device)
        
        with torch.cuda.amp.autocast():
            pred = model(epsilon)
            
            # Create dummy targets
            target = {
                'trajectory': torch.randn(2, 100, 3).to(device),
                'thermodynamics': {
                    'temperature': torch.randn(2, 10).to(device),
                    'pressure': torch.randn(2, 10).to(device),
                    'density': torch.randn(2, 10).to(device),
                    'potential_energy': torch.randn(2, 10).to(device)
                },
                'rdfs': {
                    'CC': {'g_r': torch.randn(2, 10).to(device)},
                    'CO': {'g_r': torch.randn(2, 10).to(device)},
                    'OO': {'g_r': torch.randn(2, 10).to(device)}
                }
            }
            
            loss, losses = criterion(pred, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✅ Training step successful")
        print(f"  Total loss: {loss.item():.6f}")
        print(f"  Component losses:")
        for key, val in losses.items():
            print(f"    {key}: {val.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_generation():
    """Test file generation."""
    print("\n" + "="*70)
    print("  5. FILE GENERATION CHECK".center(70))
    print("="*70 + "\n")
    
    try:
        device = 'cuda' if cuda.is_available() else 'cpu'
        
        # Create a simple model for testing
        print("Creating test model...")
        model = MDGenerativeModel(
            latent_dim=512,
            n_atoms=5541,
            thermo_seq_len=1000,
            rdf_bins=200
        ).to(device)
        
        # Save it temporarily
        test_checkpoint = Path("ml_integration/advanced/data/test_model.pt")
        test_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), test_checkpoint)
        
        # Test generator
        print("Testing file generator...")
        generator = MDFileGenerator(str(test_checkpoint), device=device)
        
        # Generate files
        test_output = Path("ml_integration/advanced/data/test_output")
        generator.generate_all_files(epsilon=0.55, output_dir=str(test_output))
        
        # Check files exist
        expected_files = [
            'production.lammpstrj',
            'production_detailed_thermo.dat',
            'rdf_CC.dat',
            'rdf_CO.dat',
            'rdf_OO.dat'
        ]
        
        all_exist = True
        for fname in expected_files:
            fpath = test_output / fname
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"  ✅ {fname} ({size / 1024:.1f} KB)")
            else:
                print(f"  ❌ {fname} missing")
                all_exist = False
        
        # Cleanup
        test_checkpoint.unlink()
        for f in test_output.glob("*"):
            f.unlink()
        test_output.rmdir()
        
        if all_exist:
            print(f"\n✅ File generation successful")
            return True
        else:
            print(f"\n❌ Some files missing")
            return False
        
    except Exception as e:
        print(f"❌ File generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_resources():
    """Estimate memory and time requirements."""
    print("\n" + "="*70)
    print("  6. RESOURCE ESTIMATES".center(70))
    print("="*70 + "\n")
    
    # Model size
    model_params = 80_500_000
    model_size_mb = model_params * 4 / 1024 / 1024  # FP32
    model_size_fp16_mb = model_params * 2 / 1024 / 1024  # FP16
    
    print(f"Model Size:")
    print(f"  FP32: {model_size_mb:.1f} MB")
    print(f"  FP16: {model_size_fp16_mb:.1f} MB")
    print(f"  Training (with gradients/optimizer): ~{model_size_mb * 4:.1f} MB")
    
    # Data size per epoch
    n_epsilon = 10
    frames_per_eps = 100  # If not using all frames
    atoms = 5541
    data_size_mb = n_epsilon * frames_per_eps * atoms * 3 * 4 / 1024 / 1024
    
    print(f"\nData Size (per epoch):")
    print(f"  With 100 frames/epsilon: ~{data_size_mb:.1f} MB")
    print(f"  With ALL frames (4000): ~{data_size_mb * 40:.1f} MB")
    
    # Total memory estimate
    total_memory_gb = (model_size_mb * 4 + data_size_mb) / 1024
    
    print(f"\nEstimated GPU Memory:")
    print(f"  Minimum: ~{total_memory_gb:.1f} GB")
    print(f"  Recommended: ~{total_memory_gb * 2:.1f} GB (with buffer)")
    
    if cuda.is_available():
        available_gb = cuda.get_device_properties(0).total_memory / 1024**3
        if available_gb >= total_memory_gb * 2:
            print(f"  ✅ Available: {available_gb:.1f} GB (sufficient)")
        else:
            print(f"  ⚠️  Available: {available_gb:.1f} GB (may need to reduce batch size)")
    
    # Training time estimate
    print(f"\nEstimated Training Time:")
    print(f"  100 epochs: ~2-3 hours")
    print(f"  500 epochs: ~10-15 hours")
    print(f"  (With FP16 optimization on A100)")


def main():
    """Run all pre-flight checks."""
    print("\n" + "#"*70)
    print("  PRE-FLIGHT CHECK - ML TRAINING SYSTEM".center(70))
    print("#"*70)
    
    checks = [
        ("CUDA", check_cuda),
        ("Data Loading", check_data_loading),
        ("Model", check_model),
        ("Training Loop", check_training_loop),
        ("File Generation", check_file_generation),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Resource estimates (always runs)
    estimate_resources()
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY".center(70))
    print("="*70 + "\n")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:.<50} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! System is ready for training.\n")
        print("To start training:")
        print("  cd ml_integration/advanced/scripts")
        print("  python train_production.py")
        print("\nTo generate files after training:")
        print("  python generate_files.py --model ../checkpoints/best_model.pt --epsilon 0.80 --output ../generated/epsilon_0.80")
        print()
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED! Please fix issues before training.\n")
        return 1


if __name__ == "__main__":
    exit(main())
