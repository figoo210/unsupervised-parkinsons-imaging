import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
import time
import gc
import random
from data.dataloader import create_dataloaders

def create_memory_efficient_dataloaders(df, batch_size=2, train_split=0.8, mask_path='rmask_ICV.nii'):
    """
    Create train and validation dataloaders with optimized memory usage
    """
    # Reuse the create_dataloaders function
    return create_dataloaders(
        df,
        batch_size=batch_size,
        train_split=train_split,
        on_demand=True,
        mask_path=mask_path
    )


def analyze_dataset_statistics_efficiently(dataloader, max_samples=100, min_samples_per_group=15):
    """
    Analyzes dataset statistics with improved memory efficiency and ensures
    stratified sampling across all patient groups.

    Args:
        dataloader: The dataloader to sample from
        max_samples: Maximum total samples to process
        min_samples_per_group: Minimum samples to collect per group

    Returns:
        Dictionary of statistical measures with proper group representation
    """
    print("Analyzing dataset statistics (stratified, memory-efficient version)...")
    stats = defaultdict(list)
    samples_by_group = defaultdict(int)

    # First pass: Count occurrences of each group
    group_counts = {}
    print("Scanning dataset to count groups...")
    for batch in tqdm(dataloader, desc="Counting groups"):
        labels = batch['label']
        for label in labels:
            if label not in group_counts:
                group_counts[label] = 0
            group_counts[label] += 1

    print(f"Found groups: {group_counts}")

    # Second pass: Collect samples with stratification
    all_samples = []
    all_labels = []
    all_paths = []

    try:
        print("Collecting stratified samples...")
        error_count = 0
        for batch in tqdm(dataloader, desc="Collecting samples"):
            volumes = batch['volume']
            labels = batch['label']
            paths = batch['path']

            # Process each volume in the batch
            for vol_idx, (volume, label, path) in enumerate(zip(volumes, labels, paths)):
                # If we have enough samples from this group, skip unless we need more total samples
                if (samples_by_group[label] >= min_samples_per_group and
                    sum(samples_by_group.values()) >= max_samples):
                    continue

                # Add this sample
                all_samples.append(volume)
                all_labels.append(label)
                all_paths.append(path)
                samples_by_group[label] += 1

            # Memory cleanup after each batch
            del volumes, labels, paths
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check if we've collected enough samples from each group
            if all(samples_by_group[group] >= min_samples_per_group for group in group_counts):
                if sum(samples_by_group.values()) >= max_samples:
                    print(f"Collected sufficient samples from all groups")
                    break
        
    except Exception as e:
        print(f"Error during sample collection: {str(e)}")
        import traceback
        traceback.print_exc()

    # Process collected samples
    print(f"Processing {len(all_samples)} collected samples...")
    print(f"Samples per group: {dict(samples_by_group)}")

    for volume, label, path in zip(all_samples, all_labels, all_paths):
        # Extract statistics
        vol_data = volume.numpy().flatten()

        # Compute statistics
        stats['mean'].append(float(np.mean(vol_data)))
        stats['std'].append(float(np.std(vol_data)))
        stats['min'].append(float(np.min(vol_data)))
        stats['max'].append(float(np.max(vol_data)))
        stats['label'].append(label)
        stats['path'].append(path)

    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame(stats)
    print(f"Successfully analyzed {len(stats_df)} samples")

    # Verify group representation
    group_dist = stats_df['label'].value_counts()
    print("Group distribution in analyzed samples:")
    print(group_dist)

    return stats_df


def analyze_slice_variance(dataloader, num_samples_per_group=5):
    """
    Analyzes slice-wise variance across different views for each patient group
    """
    print("Analyzing slice-wise variance patterns...")

    # Create a new dataloader with num_workers=0 to avoid pickling errors
    single_worker_loader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize storage for variances
    group_variances = {
        'PD': {'axial': [], 'coronal': [], 'sagittal': []},
        'Control': {'axial': [], 'coronal': [], 'sagittal': []},
        'SWEDD': {'axial': [], 'coronal': [], 'sagittal': []}
    }
    sample_counts = {'PD': 0, 'Control': 0, 'SWEDD': 0}

    try:
        for batch in tqdm(single_worker_loader, desc="Computing slice variances"):
            volumes = batch['volume']
            labels = batch['label']

            for volume, label in zip(volumes, labels):
                label = label if isinstance(label, str) else label.item()

                if sample_counts[label] >= num_samples_per_group:
                    continue

                # Get volume data
                vol_data = volume.squeeze().numpy()
                d, h, w = vol_data.shape

                # Compute variance for each slice in each view
                axial_var = [np.var(vol_data[i, :, :]) for i in range(d)]
                coronal_var = [np.var(vol_data[:, i, :]) for i in range(h)]
                sagittal_var = [np.var(vol_data[:, :, i]) for i in range(w)]

                # Store variances
                group_variances[label]['axial'].append(axial_var)
                group_variances[label]['coronal'].append(coronal_var)
                group_variances[label]['sagittal'].append(sagittal_var)

                sample_counts[label] += 1

            # Check if we have enough samples from each group
            if all(count >= num_samples_per_group for count in sample_counts.values()):
                break

            # Memory cleanup
            del volumes, labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during variance analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    # Compute average variances across samples for each group
    avg_variances = {}
    for group in group_variances:
        avg_variances[group] = {
            view: np.mean(variances, axis=0)
            for view, variances in group_variances[group].items()
        }

    return avg_variances
