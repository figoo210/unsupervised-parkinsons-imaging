import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from typing import Tuple, Dict, Optional, List
from data.data_ingestion import load_dicom
import seaborn as sns
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import time
import gc

def load_mask(mask_path: str = 'rmask_ICV.nii') -> np.ndarray:
    """
    Load and preprocess a brain mask.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        Processed binary mask as numpy array
    """
    mask_header = nib.load(mask_path)
    mask = mask_header.get_fdata() > 0.5
    mask = np.transpose(mask, [2, 1, 0])
    mask = np.flip(mask, axis=1)
    return mask


def extract_slices(volume):
    """
    Given a 3D volume, returns one axial, one coronal, and one sagittal slice.
    Assumes volume shape is (depth, height, width).
    """
    d, h, w = volume.shape
    axial = volume[32, :, :]         # Axial: slice along depth
    coronal = volume[:, 50, :]        # Coronal: slice along height
    sagittal = volume[:, :, 55]       # Sagittal: slice along width
    return axial, coronal, sagittal


def plot_patient_slices(df: pd.DataFrame, 
                        groups: Dict[str, str], 
                        mask: Optional[np.ndarray] = None,
                        random_seed: Optional[int] = None) -> plt.Figure:
    """
    Plot orthogonal slices for a random patient from each group.
    
    Args:
        df: DataFrame with patient data including file_path and label columns
        groups: Dictionary mapping group keys to label values
        mask: Optional brain mask to apply
        random_seed: Optional seed for reproducibility
        
    Returns:
        Matplotlib figure with the plotted slices
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Create a figure with one row per group and three columns for the views
    fig, axes = plt.subplots(nrows=len(groups), ncols=3, figsize=(12, 4 * len(groups)))
    fig.suptitle("Axial, Coronal, and Sagittal Slices for a Random Patient per Group", fontsize=16)

    for i, (group_key, group_label) in enumerate(groups.items()):
        # Filter DataFrame for the current group
        group_df = df[df["label"] == group_label]
        if group_df.empty:
            print(f"No data found for group {group_label}")
            continue

        # Select a random file from the group
        random_file = group_df.sample(1)["file_path"].values[0]
        print(f"Loading file for group {group_label}: {random_file}")

        # Load the DICOM volume using the previously defined load_dicom() function
        volume, _ = load_dicom(random_file)

        # Verify the volume is 3D (if not, skip or raise an error)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape} for file: {random_file}")

        axial, coronal, sagittal = extract_slices(volume)

        # Plot Axial slice
        ax = axes[i, 0]
        ax.imshow(axial, cmap="gray")
        ax.set_title(f"{group_label} - Axial")
        ax.axis("off")

        # Plot Coronal slice
        ax = axes[i, 1]
        ax.imshow(coronal, cmap="gray")
        ax.set_title(f"{group_label} - Coronal")
        ax.axis("off")

        # Plot Sagittal slice
        ax = axes[i, 2]
        ax.imshow(sagittal, cmap="gray")
        ax.set_title(f"{group_label} - Sagittal")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_side_by_side(axial_norm: np.ndarray, axial_masked: np.ndarray) -> plt.Figure:
    """
    Plot two images side-by-side for comparison.

    Args:
        axial_norm: Normalized axial slice
        axial_masked: Masked axial slice

    Returns:
        Matplotlib figure with the two images
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(axial_norm, cmap="gray")
    axes[0].set_title("Normalized Axial Slice")
    axes[0].axis("off")

    axes[1].imshow(axial_masked, cmap="gray")
    axes[1].set_title("Masked Axial Slice")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def plot_intensity_distributions(stats_df):
    """
    Creates violin plots of intensity distributions by group
    """
    plt.figure(figsize=(15, 6))

    # Plot intensity distributions
    plt.subplot(1, 2, 1)
    sns.violinplot(data=stats_df, x='label', y='mean', palette='viridis')
    plt.title('Distribution of Mean Intensities by Group')
    plt.xlabel('Group')
    plt.ylabel('Mean Intensity')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.violinplot(data=stats_df, x='label', y='std', palette='viridis')
    plt.title('Distribution of Intensity Standard Deviations by Group')
    plt.xlabel('Group')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_group_statistics(stats_df):
    """
    Plots statistical summaries by group using lightweight operations
    """
    plt.figure(figsize=(15, 5))

    # Group counts
    plt.subplot(1, 3, 1)
    group_counts = stats_df['label'].value_counts()
    sns.barplot(x=group_counts.index, y=group_counts.values, palette='viridis')
    plt.title('Sample Count by Group')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Box plots - more memory efficient than complex plots
    plt.subplot(1, 3, 2)
    sns.boxplot(data=stats_df, x='label', y='mean', palette='viridis')
    plt.title('Mean Intensity Distribution')
    plt.xlabel('Group')
    plt.ylabel('Mean Intensity')
    plt.xticks(rotation=45)

    # Simple boxplot instead of violin plot
    plt.subplot(1, 3, 3)
    sns.boxplot(data=stats_df, x='label', y='std', palette='viridis')
    plt.title('Intensity Variance Distribution')
    plt.xlabel('Group')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_sample_slices(stats_df, dataloader, samples_per_group=1):
    """
    Visualizes a limited number of samples from each group
    with efficient memory handling, selecting from the pre-processed samples.
    Shows anatomically interesting slices (axial=32, coronal=50, sagittal=55 and 70)
    instead of central slices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get unique groups
    groups = stats_df['label'].unique()

    # Dictionary to hold samples for each group
    samples_data = {}

    # Select paths from stats_df, stratified by group
    selected_paths = {}
    for group in groups:
        group_paths = stats_df[stats_df['label'] == group]['path'].values
        if len(group_paths) > 0:
            selected_paths[group] = random.sample(list(group_paths),
                                                min(samples_per_group, len(group_paths)))

    # Find these samples in the dataloader
    for batch in dataloader:
        volumes = batch['volume']
        paths = batch['path']
        labels = batch['label']

        for i, (vol, path, label) in enumerate(zip(volumes, paths, labels)):
            # Check if this path is in our selected paths
            for group, group_paths in selected_paths.items():
                if path in group_paths:
                    # Store the sample
                    key = f"{group}_{len(samples_data)}"
                    samples_data[key] = vol.cpu().numpy()
                    # Remove from selected_paths to avoid duplicates
                    selected_paths[group].remove(path)

        # Check if we have all samples
        if all(len(paths) == 0 for paths in selected_paths.values()):
            break

    # Define anatomically interesting slices 
    axial_slice_idx = 32      # Axial view - slice 32
    coronal_slice_idx = 50    # Coronal view - slice 50
    sagittal_slice1_idx = 55  # Sagittal view - slice 55
    sagittal_slice2_idx = 70  # Sagittal view - slice 70

    # Visualize the samples
    num_groups = len(groups)
    plt.figure(figsize=(20, 5 * num_groups))  # Wider figure to accommodate 4 slices

    for i, (key, vol) in enumerate(samples_data.items()):
        # Extract label
        label = key.split('_')[0]

        # Get anatomically interesting slices
        vol = vol.squeeze()  # Remove channel dimension
        axial_slice = vol[axial_slice_idx, :, :]
        coronal_slice = vol[:, coronal_slice_idx, :]
        sagittal_slice1 = vol[:, :, sagittal_slice1_idx]
        sagittal_slice2 = vol[:, :, sagittal_slice2_idx]

        # Plot slices (now 4 slices per row)
        plt.subplot(len(samples_data), 4, i*4 + 1)
        plt.imshow(axial_slice, cmap='gray')
        plt.title(f'{label} - Axial (z={axial_slice_idx})')
        plt.axis('off')

        plt.subplot(len(samples_data), 4, i*4 + 2)
        plt.imshow(coronal_slice, cmap='gray')
        plt.title(f'{label} - Coronal (y={coronal_slice_idx})')
        plt.axis('off')

        plt.subplot(len(samples_data), 4, i*4 + 3)
        plt.imshow(sagittal_slice1, cmap='gray')
        plt.title(f'{label} - Sagittal1 (x={sagittal_slice1_idx})')
        plt.axis('off')
        
        plt.subplot(len(samples_data), 4, i*4 + 4)
        plt.imshow(sagittal_slice2, cmap='gray')
        plt.title(f'{label} - Sagittal2 (x={sagittal_slice2_idx})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Plot the slice variance results
def plot_slice_variances(avg_variances):
    """
    Creates line plots for slice-wise variance analysis
    """
    views = ['axial', 'coronal', 'sagittal']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, view in enumerate(views):
        ax = axes[idx]

        for group in avg_variances:
            variances = avg_variances[group][view]
            ax.plot(range(len(variances)), variances, label=group)

        ax.set_title(f'{view.capitalize()} View - Slice-wise Variance')
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('Average Variance')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
