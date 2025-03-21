import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch3dunet.unet3d.model import UNet3D
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.configuration.config import data_dir,test_split,val_split,learning_rate,workers,n_epochs,weightpath,checkpoint_savepath,save_path
from src.Dataset.dataset import BraTSDataset
from src.losses.hybridfocalloss import HybridLoss
from src.training.trainer import train_epoch,validate_epoch
from src.visualizations.visualization import plot_metrics
def main():
    patients = os.listdir(data_dir)

    train_val, test_patients = train_test_split(patients, test_size=test_split, random_state=42)
    train_patients, val_patients = train_test_split(train_val, test_size=val_split, random_state=42)
    print(f"Train: {len(train_patients)}, Validation: {len(val_patients)}, Test: {len(test_patients)}")

    train_dataset = BraTSDataset(data_dir, train_patients,augment=True)
    val_dataset = BraTSDataset(data_dir, val_patients,augment=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers)
    print("Train batches:", len(train_loader), "Validation batches:", len(val_loader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(in_channels=3, out_channels=4,f_maps=16).to(device)
    # model.load_state_dict(torch.load(weightpath))
    model.to(device)
    print(f"🔢 Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    criterion = HybridLoss(device=device).to(device)
    num_epochs = n_epochs
    best_combined_tumor_dice = 0.0  # Best Dice score for tumor classes (1, 2, 3)

    train_losses=[]
    val_losses=[]
    dice_scores=[]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training Phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Phase
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)

        # Ensure val_dice is a NumPy array for safe indexing
        val_dice = np.array(val_dice)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Class-wise Dice: {val_dice}")
        
        

        tumor_dice = val_dice[1] + val_dice[2] + val_dice[3]
        avg_tumor_dice = tumor_dice / 3.0

        print(f"🎯 Average Tumor Dice (Classes 1, 2, 3): {avg_tumor_dice:.4f}")

        # Save the model if the average tumor Dice improves
        if avg_tumor_dice > best_combined_tumor_dice:
            best_combined_tumor_dice = avg_tumor_dice
            torch.save(model.state_dict(),checkpoint_savepath)
            print(f"✅ Saved best model with Average Tumor Dice: {avg_tumor_dice:.4f} at :{checkpoint_savepath}")

        scheduler.step(avg_tumor_dice)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores.append(avg_tumor_dice)
    print("Training complete!")
    plot_metrics(train_losses,val_losses,dice_scores,save_path)

if __name__ == "__main__":
    main()