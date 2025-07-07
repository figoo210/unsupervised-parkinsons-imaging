"""
Data analysis functions for dataset statistics and visualization.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

from .datasets import OnDemandDataset
from ..utils.memory_utils import clear_memory


def create_memory_efficient_dataloaders(df: pd.DataFrame, batch_size: int = 2, 
                                        train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create memory-efficient dataloaders for large datasets.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        batch_size (int): Batch size for the dataloaders
        train_split (float): Fraction of data to use for training
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    # Split data
    train_df, val_df = train_test_split(df, train_size=train_split, random_state=42, 
                                        stratify=df['group'] if 'group' in df.columns else None)
    
    # Create memory-efficient datasets
    train_dataset = OnDemandDataset(train_df)
    val_dataset = OnDemandDataset(val_df)
    
    # Create dataloaders with minimal memory usage
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,  # Disable for memory efficiency
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,  # Disable for memory efficiency
        drop_last=False
    )
    
    return train_loader, val_loader


def analyze_dataset_statistics_efficiently(dataloader: DataLoader, max_samples: int = 100, 
                                          min_samples_per_group: int = 15) -> pd.DataFrame:
    """
    Analyze dataset statistics efficiently by sampling data.
    
    Args:
        dataloader (DataLoader): DataLoader to analyze
        max_samples (int): Maximum number of samples to analyze
        min_samples_per_group (int): Minimum samples per group
        
    Returns:
        pd.DataFrame: DataFrame with statistics
    """
    stats_data = []
    group_counts = defaultdict(int)
    sample_count = 0
    
    # Collect statistics
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing dataset")):
        if sample_count >= max_samples:
            break
            
        for i in range(len(batch['volume'])):
            volume = batch['volume'][i]
            group = batch['group'][i] if 'group' in batch else 'unknown'
            
            # Skip if we have enough samples for this group
            if group_counts[group] >= min_samples_per_group and sample_count >= max_samples:
                continue
            
            # Calculate statistics
            volume_np = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume
            
            stats = {
                'sample_idx': sample_count,
                'group': group,
                'file_path': batch['file_path'][i] if 'file_path' in batch else f'sample_{sample_count}',
                'shape': str(volume_np.shape),
                'mean_intensity': float(np.mean(volume_np)),
                'std_intensity': float(np.std(volume_np)),
                'min_intensity': float(np.min(volume_np)),
                'max_intensity': float(np.max(volume_np)),
                'non_zero_voxels': int(np.count_nonzero(volume_np)),
                'total_voxels': int(volume_np.size),
                'sparsity': float(np.count_nonzero(volume_np) / volume_np.size)
            }
            
            stats_data.append(stats)
            group_counts[group] += 1
            sample_count += 1
            
            # Clear memory periodically
            if sample_count % 10 == 0:
                clear_memory()
    
    return pd.DataFrame(stats_data)


def plot_intensity_distributions(stats_df: pd.DataFrame) -> None:
    """
    Plot intensity distribution statistics by group.
    
    Args:
        stats_df (pd.DataFrame): DataFrame with statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean intensity by group
    sns.boxplot(data=stats_df, x='group', y='mean_intensity', ax=axes[0, 0])
    axes[0, 0].set_title('Mean Intensity by Group')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Standard deviation by group
    sns.boxplot(data=stats_df, x='group', y='std_intensity', ax=axes[0, 1])
    axes[0, 1].set_title('Std Intensity by Group')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Min/Max intensity by group
    sns.boxplot(data=stats_df, x='group', y='min_intensity', ax=axes[1, 0])
    axes[1, 0].set_title('Min Intensity by Group')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=stats_df, x='group', y='max_intensity', ax=axes[1, 1])
    axes[1, 1].set_title('Max Intensity by Group')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_group_statistics(stats_df: pd.DataFrame) -> None:
    """
    Plot group-level statistics.
    
    Args:
        stats_df (pd.DataFrame): DataFrame with statistics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Group counts
    group_counts = stats_df['group'].value_counts()
    group_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Sample Count by Group')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Sparsity by group
    sns.boxplot(data=stats_df, x='group', y='sparsity', ax=axes[1])
    axes[1].set_title('Volume Sparsity by Group')
    axes[1].set_ylabel('Sparsity (non-zero voxels / total voxels)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def analyze_slice_variance(dataloader: DataLoader, num_samples_per_group: int = 5) -> Dict[str, np.ndarray]:
    """
    Analyze variance across slices for different groups.
    
    Args:
        dataloader (DataLoader): DataLoader to analyze
        num_samples_per_group (int): Number of samples per group to analyze
        
    Returns:
        Dict[str, np.ndarray]: Average variances by group
    """
    group_variances = defaultdict(list)
    group_counts = defaultdict(int)
    
    for batch in tqdm(dataloader, desc="Analyzing slice variance"):
        for i in range(len(batch['volume'])):
            volume = batch['volume'][i]
            group = batch['group'][i] if 'group' in batch else 'unknown'
            
            if group_counts[group] >= num_samples_per_group:
                continue
            
            # Calculate variance along each axis
            volume_np = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume
            
            # Remove channel dimension if present
            if volume_np.ndim == 4:
                volume_np = volume_np[0]
            
            # Calculate variance along each slice direction
            axis_variances = []
            for axis in range(3):
                axis_var = np.var(volume_np, axis=axis)
                axis_variances.append(np.mean(axis_var))
            
            group_variances[group].append(axis_variances)
            group_counts[group] += 1
    
    # Calculate average variances
    avg_variances = {}
    for group, variances in group_variances.items():
        avg_variances[group] = np.mean(variances, axis=0)
    
    return avg_variances


def plot_slice_variances(avg_variances: Dict[str, np.ndarray]) -> None:
    """
    Plot slice variances by group and axis.
    
    Args:
        avg_variances (Dict[str, np.ndarray]): Average variances by group
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = list(avg_variances.keys())
    x = np.arange(len(groups))
    width = 0.25
    
    axis_names = ['Axial', 'Coronal', 'Sagittal']
    colors = ['red', 'green', 'blue']
    
    for i, (axis_name, color) in enumerate(zip(axis_names, colors)):
        variances = [avg_variances[group][i] for group in groups]
        ax.bar(x + i * width, variances, width, label=axis_name, color=color, alpha=0.7)
    
    ax.set_xlabel('Group')
    ax.set_ylabel('Average Variance')
    ax.set_title('Slice Variance by Group and Axis')
    ax.set_xticks(x + width)
    ax.set_xticklabels(groups)
    ax.legend()
    
    plt.tight_layout()
    plt.show() 