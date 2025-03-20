# Class-wise Dice Calculation
import numpy as np
import torch
from tqdm import tqdm

def calculate_classwise_dice(preds, masks, num_classes=4, epsilon=1e-6):
    dice_scores = []
    for class_id in range(num_classes):
        pred_class = (preds == class_id).float()
        true_class = (masks == class_id).float()

        intersection = (pred_class * true_class).sum()
        union = pred_class.sum() + true_class.sum()

        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())

    return dice_scores


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        # Ensure target shape is (B, H, W, D)
        if masks.dim() == 5:
            masks = masks.squeeze(1)

        optimizer.zero_grad()

        outputs = model(images)  # Model prediction
        loss = criterion(outputs, masks)  # Hybrid Loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


# Updated Validation Function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    total_dice_scores = np.zeros(4)  # For 4 segmentation classes
    num_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)  # Model output: (B, 4, 128, 128, 128)
            loss = criterion(outputs, masks)  # Hybrid Loss
            epoch_loss += loss.item()

            # Convert outputs to class predictions
            preds = torch.argmax(outputs, dim=1)  # Shape: (B, 128, 128, 128)

            # Calculate Dice for each class (0, 1, 2, 3)
            dice_scores = calculate_classwise_dice(preds, masks)
            total_dice_scores += np.array(dice_scores)
            num_samples += 1

    # Average Loss and Dice Scores
    avg_loss = epoch_loss / len(val_loader)
    avg_dice = total_dice_scores / num_samples  # Average Dice per class

    return avg_loss, avg_dice
