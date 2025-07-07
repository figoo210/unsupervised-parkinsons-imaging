"""
Dataset classes for medical image data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Callable, Tuple, Dict
from sklearn.model_selection import train_test_split
import gc
from ..utils.memory_utils import clear_memory


class OnDemandDataset(Dataset):
    """
    Dataset that loads data on-demand to save memory.
    """
    
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Callable] = None):
        """
        Initialize the on-demand dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and metadata
            transform (Optional[Callable]): Optional transform to apply to the data
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing volume data and metadata
        """
        row = self.dataframe.iloc[idx]
        
        # Load volume data from file path
        # This would be implemented based on your specific data format
        # For now, returning a placeholder structure
        volume = self._load_volume(row['file_path'])
        
        sample = {
            'volume': volume,
            'file_path': row['file_path'],
            'group': row.get('group', 'unknown'),
            'metadata': {k: v for k, v in row.items() if k not in ['file_path', 'group']}
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_volume(self, file_path: str) -> torch.Tensor:
        """
        Load volume data from file path.
        This is a placeholder - implement based on your data format.
        """
        # Placeholder implementation
        return torch.randn(64, 128, 128)  # Example dimensions


class BatchLoadDataset(Dataset):
    """
    Dataset that preloads data in batches for faster access.
    """
    
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = 32, 
                 transform: Optional[Callable] = None):
        """
        Initialize the batch load dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and metadata
            batch_size (int): Size of batches to preload
            transform (Optional[Callable]): Optional transform to apply to the data
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.transform = transform
        self.current_batch_idx = -1
        self.current_batch_data = None
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing volume data and metadata
        """
        batch_idx = idx // self.batch_size
        
        # Load new batch if needed
        if batch_idx != self.current_batch_idx:
            self._load_batch(batch_idx)
            self.current_batch_idx = batch_idx
        
        # Get item from current batch
        batch_item_idx = idx % self.batch_size
        if batch_item_idx < len(self.current_batch_data):
            sample = self.current_batch_data[batch_item_idx]
        else:
            # Handle edge case for last incomplete batch
            row = self.dataframe.iloc[idx]
            volume = self._load_volume(row['file_path'])
            sample = {
                'volume': volume,
                'file_path': row['file_path'],
                'group': row.get('group', 'unknown'),
                'metadata': {k: v for k, v in row.items() if k not in ['file_path', 'group']}
            }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_batch(self, batch_idx: int):
        """Load a batch of data."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataframe))
        
        batch_data = []
        for idx in range(start_idx, end_idx):
            row = self.dataframe.iloc[idx]
            volume = self._load_volume(row['file_path'])
            
            sample = {
                'volume': volume,
                'file_path': row['file_path'],
                'group': row.get('group', 'unknown'),
                'metadata': {k: v for k, v in row.items() if k not in ['file_path', 'group']}
            }
            batch_data.append(sample)
        
        # Clear previous batch and update
        del self.current_batch_data
        clear_memory()
        self.current_batch_data = batch_data
    
    def _load_volume(self, file_path: str) -> torch.Tensor:
        """
        Load volume data from file path.
        This is a placeholder - implement based on your data format.
        """
        # Placeholder implementation
        return torch.randn(64, 128, 128)  # Example dimensions


def create_dataloaders(df: pd.DataFrame, batch_size: int = 4, train_split: float = 0.8, 
                       on_demand: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        batch_size (int): Batch size for the dataloaders
        train_split (float): Fraction of data to use for training
        on_demand (bool): Whether to use OnDemandDataset or BatchLoadDataset
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    # Split data
    train_df, val_df = train_test_split(df, train_size=train_split, random_state=42)
    
    # Create datasets
    if on_demand:
        train_dataset = OnDemandDataset(train_df)
        val_dataset = OnDemandDataset(val_df)
    else:
        train_dataset = BatchLoadDataset(train_df, batch_size)
        val_dataset = BatchLoadDataset(val_df, batch_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader 