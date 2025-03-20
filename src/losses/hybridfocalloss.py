import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=[0.01, 4.0, 1.8, 6.0]):
        """
        Implements Focal Loss to focus on hard-to-classify tumor regions.
        
        Args:
            gamma (float): Controls how much to focus on hard samples.
            alpha (list): Class weight balancing [background, necrotic, edema, enhancing tumor].
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).cuda()  # Move weights to GPU

    def forward(self, preds, targets):
        """
        Compute focal loss.
        Args:
            preds: Model predictions (logits), shape (B, 4, 128, 128, 128).
            targets: Ground truth segmentation mask, shape (B, 128, 128, 128).
        Returns:
            Focal loss scalar.
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='none', weight=self.alpha.to(preds.device))
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights, epsilon=1e-6):
        """
        Implements Weighted Dice Loss for multi-class segmentation.
        
        Args:
            class_weights (list): List of class-specific weights.
            epsilon (float): Small value to prevent division by zero.
        """
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = class_weights
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Compute weighted Dice loss.

        Args:
            pred: Model predictions (softmax probabilities), shape (B, 4, 128, 128, 128).
            target: Ground truth segmentation mask, shape (B, 128, 128, 128).

        Returns:
            Weighted Dice loss scalar.
        """
        target_onehot = self.one_hot_encode(target, num_classes=4)
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities

        dice_loss = 0.0
        for class_id, weight in enumerate(self.class_weights):
            pred_class = pred[:, class_id, :, :, :]
            target_class = target_onehot[:, class_id, :, :, :]

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

            dice_loss += weight * (1 - dice_score)

        return dice_loss

    def one_hot_encode(self, target, num_classes):
        """
        One-hot encodes the target segmentation mask.

        Args:
            target: Ground truth segmentation mask, shape (B, 128, 128, 128).
            num_classes: Number of segmentation classes (4 for BraTS2020).

        Returns:
            One-hot encoded tensor, shape (B, 4, 128, 128, 128).
        """
        if target.dim() == 5:  
            target = target.squeeze(1)
        target = target.long()
        target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        return target_onehot

class HybridLoss(nn.Module):
    def __init__(self, device):
        """
        Combines Focal Loss and Weighted Dice Loss for robust segmentation.
        Args:
            device: CUDA or CPU.
        """
        super(HybridLoss, self).__init__()
        
        # Focal Loss (instead of CrossEntropy)
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.01, 4.0, 1.8, 6.0])  # Adjust weights if needed

        # Weighted Dice Loss
        self.dice_loss = WeightedDiceLoss(class_weights=[0.01, 4.0, 1.8, 6.0])

    def forward(self, preds, targets):
        """
        Compute the combined loss (Focal + Dice).

        Args:
            preds: Model predictions (logits), shape (B, 4, 128, 128, 128).
            targets: Ground truth segmentation mask, shape (B, 128, 128, 128).

        Returns:
            Hybrid loss scalar.
        """
        return self.focal_loss(preds, targets) + self.dice_loss(preds, targets)
