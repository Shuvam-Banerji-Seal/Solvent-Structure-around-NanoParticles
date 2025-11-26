# Training Workflow - Final

## Summary of Changes

I've implemented **Frame-Based Splitting** to address your requirement of training on all epsilons up to 0.50 while maintaining proper validation.

### ✅ Training Strategy (Frame-Based)

Instead of holding out an entire epsilon folder for validation (which prevents learning that epsilon), we now split the **frames** within each simulation:

- **Training Set**: First 80% of frames from **ALL** epsilons (0.00 to 0.50).
- **Validation Set**: Last 20% of frames from **ALL** epsilons (0.00 to 0.50).

**Benefits**:
1. **Full Coverage**: The model learns the physics of `epsilon=0.50` (and all others).
2. **Proper Validation**: We still have unseen data (the last 20% of the trajectory) to monitor overfitting.
3. **Extrapolation Testing**: After training, we generate `epsilon=0.55-1.10` to test generalization to completely new regimes.

### 🐛 Issues Fixed

1. **Validation Epsilon**: Previously set to 0.50 (excluding it from training). Now set to `train_epsilon` (same list), with splitting handled internally.
2. **Data Slicing**: Modified `dataset.py` to slice trajectory and thermodynamics arrays based on `split='train'` or `split='val'`.

### 📊 Current Configuration

```python
# Both sets use the same epsilon list
epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Dataset handles splitting
train_loader = MDSimulationDataset(..., split='train', split_ratio=0.8)
val_loader   = MDSimulationDataset(..., split='val',   split_ratio=0.8)
```

### 🚀 Workflow Steps

1. **[NOW] Train Model**: `python train_production.py`
   - Trains on 80% of 0.0-0.50
   - Validates on 20% of 0.0-0.50
   
2. **[AFTER] Generate Files**: 
   - Generate 0.55-1.10 (Extrapolation)
   - Generate 0.50 (Interpolation check)

3. **[AFTER] Compare Results**:
   - Verify performance on both seen (0.50) and unseen (0.55+) data.
