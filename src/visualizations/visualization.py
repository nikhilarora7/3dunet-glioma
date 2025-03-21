import matplotlib.pyplot as plt

from src.configuration.config import save_path
def plot_metrics(train_losses, val_losses, dice_scores, save_path):
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots()
    
    # Plot Losses
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, 'r-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    
    # Create second y-axis for Dice Score
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice Score', color='tab:blue')
    ax2.plot(epochs, dice_scores, 'b-', label='Average Dice')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.title("Training Metrics")
    plt.savefig(save_path)
    
    print(f"Plot saved to {save_path}")