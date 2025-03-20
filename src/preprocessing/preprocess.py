import numpy as np

"""
X′=(X−μ)/σ
where:
X is the original intensity
μ is the mean intensity
σ is the standard deviation
"""
def z_score_normalize(image, clip_min=-3, clip_max=3):
    # Apply mask for non-zero voxels (ignore background)
    mask = image > 0
    
    # Mean and Std only for foreground
    mean = image[mask].mean()
    std = image[mask].std()
    
    # Z-score normalization (foreground only)
    normalized_image = np.zeros_like(image, dtype=np.float32)
    normalized_image[mask] = (image[mask] - mean) / std
    
    # Clip extreme Z-scores
    normalized_image = np.clip(normalized_image, clip_min, clip_max)
    
    return normalized_image
def center_crop(image, target_shape=(128, 128, 128)):
    x, y, z = image.shape
    cx, cy, cz = target_shape
    
    # Compute start and end indices for cropping
    start_x = (x - cx) // 2
    start_y = (y - cy) // 2
    start_z = (z - cz) // 2

    return image[start_x:start_x + cx, start_y:start_y + cy, start_z:start_z + cz]
