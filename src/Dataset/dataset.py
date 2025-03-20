import torch
from torch.utils.data import Dataset
import torchio as tio
import nibabel as nib
import os
import numpy as np
from src.preprocessing.preprocess import z_score_normalize,center_crop

class BraTSDataset(Dataset):
    def __init__(self, data_dir, patient_list, augment=False):
        self.data_dir = data_dir
        self.patient_list = patient_list
        self.augment = augment  # Enable/disable augmentation

        # Define TorchIO augmentations (only for training)
        if self.augment:
            self.transforms = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Random flipping
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Small scaling and rotation
                tio.RandomElasticDeformation(),  # Elastic deformation
                tio.RandomNoise(std=0.05),  # Add random noise
            ])

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        patient_path = os.path.join(self.data_dir, patient_id)

        # Load MRI modalities
        flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii")).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii")).get_fdata()

        # Load segmentation mask
        seg = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii")).get_fdata()

        # Normalize (Z-score)
        flair = z_score_normalize(flair)
        t1ce = z_score_normalize(t1ce)
        t2 = z_score_normalize(t2)

        # Crop
        flair, t1ce, t2 = center_crop(flair), center_crop(t1ce), center_crop(t2)
        seg = center_crop(seg)

        # Ensure segmentation labels are {0, 1, 2, 3}
        seg[seg == 4] = 3

        # Stack modalities into a tensor (C, H, W, D) → (3, 128, 128, 128)
        image = np.stack([flair, t1ce, t2], axis=0)
        image = torch.tensor(image, dtype=torch.float32)  # Image should be float32
        seg = torch.tensor(seg, dtype=torch.long)  # ✅ Ensure mask is Long (not Float)

        # ✅ FIX: Apply TorchIO augmentations only if augment=True
        if self.augment:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=seg.unsqueeze(0).long())  # ✅ Ensure segmentation is Long
            )
            augmented = self.transforms(subject)  # Apply augmentation
            image, seg = augmented.image.data, augmented.label.data.squeeze(0).long()  # ✅ Convert back to Long

        return image, seg