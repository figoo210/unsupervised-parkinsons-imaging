"""
Preprocessing functions for medical image data.
"""

import numpy as np
import torch
import pydicom
import nibabel as nib
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, ball
import matplotlib.pyplot as plt
from typing import Tuple, Union


def load_dicom(file_path: str) -> np.ndarray:
    """
    Load DICOM file and return pixel array.
    
    Args:
        file_path (str): Path to the DICOM file
        
    Returns:
        np.ndarray: Pixel array from DICOM file
    """
    try:
        dicom = pydicom.dcmread(file_path)
        return dicom.pixel_array
    except Exception as e:
        print(f"Error loading DICOM file {file_path}: {e}")
        return None


def load_nifti(file_path: str) -> np.ndarray:
    """
    Load NIfTI file and return data array.
    
    Args:
        file_path (str): Path to the NIfTI file
        
    Returns:
        np.ndarray: Data array from NIfTI file
    """
    try:
        nii = nib.load(file_path)
        return nii.get_fdata()
    except Exception as e:
        print(f"Error loading NIfTI file {file_path}: {e}")
        return None


def extract_slices(volume: np.ndarray, slice_indices: Union[int, list] = None) -> np.ndarray:
    """
    Extract specific slices from a 3D volume.
    
    Args:
        volume (np.ndarray): 3D volume array
        slice_indices (Union[int, list]): Indices of slices to extract
        
    Returns:
        np.ndarray: Extracted slices
    """
    if slice_indices is None:
        # Return middle slice by default
        slice_indices = volume.shape[0] // 2
    
    if isinstance(slice_indices, int):
        return volume[slice_indices]
    elif isinstance(slice_indices, list):
        return volume[slice_indices]
    else:
        raise ValueError("slice_indices must be int or list")


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int] = (64, 128, 128)) -> np.ndarray:
    """
    Resize a 3D volume to target shape using padding and cropping.
    
    Args:
        volume (np.ndarray): Input 3D volume
        target_shape (Tuple[int, int, int]): Target shape (depth, height, width)
        
    Returns:
        np.ndarray: Resized volume
    """
    current_shape = volume.shape
    
    def get_pad_amounts(current_size: int, target_size: int) -> Tuple[int, int]:
        """Calculate padding amounts for each dimension."""
        if current_size >= target_size:
            return 0, 0
        
        total_pad = target_size - current_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        return pad_before, pad_after
    
    def get_crop_amounts(current_size: int, target_size: int) -> Tuple[int, int]:
        """Calculate cropping amounts for each dimension."""
        if current_size <= target_size:
            return 0, current_size
        
        total_crop = current_size - target_size
        crop_before = total_crop // 2
        crop_after = current_size - crop_before
        return crop_before, crop_after
    
    # Process each dimension
    processed_volume = volume.copy()
    
    for i, (current_size, target_size) in enumerate(zip(current_shape, target_shape)):
        if current_size < target_size:
            # Pad
            pad_before, pad_after = get_pad_amounts(current_size, target_size)
            pad_width = [(0, 0)] * processed_volume.ndim
            pad_width[i] = (pad_before, pad_after)
            processed_volume = np.pad(processed_volume, pad_width, mode='constant', constant_values=0)
        elif current_size > target_size:
            # Crop
            crop_before, crop_after = get_crop_amounts(current_size, target_size)
            slices = [slice(None)] * processed_volume.ndim
            slices[i] = slice(crop_before, crop_after)
            processed_volume = processed_volume[tuple(slices)]
    
    return processed_volume


def apply_brain_mask(volume: np.ndarray, threshold_method: str = 'otsu') -> np.ndarray:
    """
    Apply brain mask to remove background.
    
    Args:
        volume (np.ndarray): Input volume
        threshold_method (str): Thresholding method ('otsu' or 'manual')
        
    Returns:
        np.ndarray: Masked volume
    """
    if threshold_method == 'otsu':
        # Calculate Otsu threshold
        threshold = threshold_otsu(volume)
        mask = volume > threshold
        
        # Apply morphological closing to fill holes
        if volume.ndim == 3:
            struct_element = ball(3)
            mask = binary_closing(mask, struct_element)
    else:
        # Simple intensity threshold
        threshold = np.percentile(volume, 10)
        mask = volume > threshold
    
    return volume * mask


def normalize_volume(volume: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Normalize volume intensities.
    
    Args:
        volume (np.ndarray): Input volume
        method (str): Normalization method ('z_score', 'min_max', or 'percentile')
        
    Returns:
        np.ndarray: Normalized volume
    """
    if method == 'z_score':
        mean = np.mean(volume)
        std = np.std(volume)
        return (volume - mean) / (std + 1e-8)
    
    elif method == 'min_max':
        min_val = np.min(volume)
        max_val = np.max(volume)
        return (volume - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'percentile':
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        return (volume - p1) / (p99 - p1 + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def process_volume(volume: np.ndarray, target_shape: Tuple[int, int, int] = (64, 128, 128),
                   apply_mask: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    Complete preprocessing pipeline for a volume.
    
    Args:
        volume (np.ndarray): Input volume
        target_shape (Tuple[int, int, int]): Target shape for resizing
        apply_mask (bool): Whether to apply brain masking
        normalize (bool): Whether to normalize intensities
        
    Returns:
        torch.Tensor: Processed volume as tensor
    """
    # Ensure volume is float32
    volume = volume.astype(np.float32)
    
    # Apply brain mask if requested
    if apply_mask:
        volume = apply_brain_mask(volume)
    
    # Resize volume
    volume = resize_volume(volume, target_shape)
    
    # Normalize intensities
    if normalize:
        volume = normalize_volume(volume, method='percentile')
    
    # Convert to tensor and add channel dimension
    volume_tensor = torch.from_numpy(volume).float()
    volume_tensor = volume_tensor.unsqueeze(0)  # Add channel dimension
    
    return volume_tensor 