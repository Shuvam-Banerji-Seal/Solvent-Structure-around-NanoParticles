import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_training(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = np.array(history['train_losses'])
    val_losses = np.array(history['val_losses'])
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Calculate gap
    gap = val_losses - train_losses
    gap_ratio = gap / (train_losses + 1e-8)
    
    # Smooth losses for better trend analysis
    window = 10
    train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
    epochs_smooth = epochs[window-1:]
    
    # Find best epoch based on val loss
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]
    
    print(f"Minimum Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")
    
    # Overfitting Analysis
    # Define overfitting as when val loss starts increasing while train loss decreases or stays flat
    # Or when the gap becomes too large
    
    print("\nOverfitting Analysis:")
    overfitting_starts = []
    for i in range(window, len(val_losses)):
        # Check if val loss is increasing over a window
        if i + 5 < len(val_losses):
            val_trend = np.mean(val_losses[i:i+5]) - np.mean(val_losses[i-5:i])
            train_trend = np.mean(train_losses[i:i+5]) - np.mean(train_losses[i-5:i])
            
            if val_trend > 0 and train_trend <= 0:
                overfitting_starts.append(i)
    
    if overfitting_starts:
        print(f"Potential overfitting detected starting around epoch {overfitting_starts[0]}")
    else:
        print("No clear sign of increasing validation loss (divergence) detected.")

    # Check gap ratio
    high_gap_epochs = np.where(gap_ratio > 0.2)[0] + 1
    if len(high_gap_epochs) > 0:
        print(f"High generalization gap (>20%) detected in {len(high_gap_epochs)} epochs.")
        print(f"First occurrence at epoch {high_gap_epochs[0]}")
    
    # Recommendation
    # We want low val loss AND low gap
    # Let's define a score: val_loss + lambda * gap
    # But simple min val loss is usually the standard unless gap is huge
    
    # Let's look at the epoch 695 the user mentioned
    if 695 <= len(val_losses):
        print(f"\nUser mentioned Epoch 695:")
        print(f"  Train Loss: {train_losses[694]:.6f}")
        print(f"  Val Loss: {val_losses[694]:.6f}")
        print(f"  Gap: {gap[694]:.6f} ({gap_ratio[694]:.2%})")
    
    # Find "stable" best epoch (low val loss, not immediately followed by spike)
    # Actually, the saved best model is just min val loss.
    
    print(f"\nRecommendation:")
    print(f"The model checkpoint 'best_model.pt' corresponds to Epoch {best_epoch}.")
    print(f"If you want to account for overfitting, ensure the gap isn't growing uncontrollably.")
    print(f"At Epoch {best_epoch}:")
    print(f"  Gap: {gap[best_epoch_idx]:.6f} ({gap_ratio[best_epoch_idx]:.2%})")

if __name__ == "__main__":
    analyze_training("/store/shuvam/learning_solvent_effects/ml_integration/advanced/logs/training_history.json")
