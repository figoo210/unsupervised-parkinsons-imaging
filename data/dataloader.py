import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm
import torch
from data.data_ingestion import load_dicom
from processing.preproccessing import process_volume
from utils.config_helpers import print_memory_stats
import psutil



class OnDemandDataset(Dataset):
    """
    Memory-efficient dataset that loads volumes on demand rather than all at once
    """
    def __init__(self, dataframe, transform=None, mask_path='rmask_ICV.nii'):
        self.df = dataframe
        self.transform = transform
        
        # Load mask once
        try:
            maskH = nib.load(mask_path)
            self.mask = maskH.get_fdata() > 0.5
            self.mask = np.transpose(self.mask, [2, 1, 0])
            self.mask = np.flip(self.mask, axis=1)
            print("✅ Brain mask loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load brain mask: {e}")
            print("⚠️ Creating a dummy mask instead")
            self.mask = np.ones((128, 128, 128), dtype=bool)
        
        # Verify all paths exist
        missing_files = []
        for idx, row in dataframe.iterrows():
            if not os.path.exists(row["file_path"]):
                missing_files.append(row["file_path"])
        
        if missing_files:
            print(f"⚠️ Warning: {len(missing_files)} files not found!")
            print(f"First few missing files: {missing_files[:3]}")
        else:
            print(f"✅ All {len(dataframe)} file paths are valid")

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """Load a single volume on demand"""
        try:
            # Get file path
            file_path = self.df.iloc[idx]["file_path"]
            
            # Load and process volume
            volume, _ = load_dicom(file_path)
            
            # Apply mask
            volume -= volume.min()
            volume = volume * self.mask
            
            # Apply processing
            norm_vol, _, _ = process_volume(volume[9:73, :, :], target_shape=(64, 128, 128))
            
            # Convert to tensor
            volume_tensor = torch.from_numpy(np.expand_dims(norm_vol, axis=0)).float()
            
            return {
                "volume": volume_tensor,
                "label": self.df.iloc[idx]["label"],
                "path": self.df.iloc[idx]["file_path"]
            }
        except Exception as e:
            print(f"Error processing file {self.df.iloc[idx]['file_path']}: {e}")
            # Return a zero tensor as fallback
            return {
                "volume": torch.zeros((1, 64, 128, 128), dtype=torch.float32),
                "label": self.df.iloc[idx]["label"],
                "path": self.df.iloc[idx]["file_path"]
            }

class BatchLoadDataset(Dataset):
    """
    Memory-efficient dataset that processes data in small batches 
    and caches the processed results
    """
    def __init__(self, dataframe, batch_size=32, transform=None, mask_path='rmask_ICV.nii'):
        self.df = dataframe
        self.transform = transform
        self.data_cache = {}  # Cache for processed data
        
        # Load mask once
        try:
            maskH = nib.load(mask_path)
            self.mask = maskH.get_fdata() > 0.5
            self.mask = np.transpose(self.mask, [2, 1, 0])
            self.mask = np.flip(self.mask, axis=1)
            print("✅ Brain mask loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load brain mask: {e}")
            print("⚠️ Creating a dummy mask instead")
            self.mask = np.ones((128, 128, 128), dtype=bool)
        
        # Pre-process data in batches to avoid memory issues
        total_batches = (len(dataframe) + batch_size - 1) // batch_size
        print(f"Pre-processing {len(dataframe)} samples in {total_batches} batches...")
        
        for batch_idx in tqdm(range(total_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataframe))
            
            for idx in range(start_idx, end_idx):
                try:
                    file_path = dataframe.iloc[idx]["file_path"]
                    
                    # Load DICOM
                    volume, _ = load_dicom(file_path)
                    volume -= volume.min()
                    volume = volume * self.mask
                    
                    # Process volume
                    norm_vol, _, _ = process_volume(volume[9:73, :, :], target_shape=(64, 128, 128))
                    
                    # Store in cache (using index as key)
                    self.data_cache[idx] = norm_vol
                    
                except Exception as e:
                    print(f"Error processing file {dataframe.iloc[idx]['file_path']}: {e}")
                    self.data_cache[idx] = np.zeros((64, 128, 128), dtype=np.float32)
            
            # Force garbage collection after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Print memory stats every few batches
            if batch_idx % 5 == 0:
                print_memory_stats()
        
        print("✅ Data pre-processing complete")

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """Get a cached pre-processed volume"""
        try:
            # Get the pre-processed volume from cache
            norm_vol = self.data_cache[idx]
            
            # Convert to tensor
            volume_tensor = torch.from_numpy(np.expand_dims(norm_vol, axis=0)).float()
            
            return {
                "volume": volume_tensor,
                "label": self.df.iloc[idx]["label"],
                "path": self.df.iloc[idx]["file_path"]
            }
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            # Return a zero tensor as fallback
            return {
                "volume": torch.zeros((1, 64, 128, 128), dtype=torch.float32),
                "label": self.df.iloc[idx]["label"],
                "path": self.df.iloc[idx]["file_path"]
            }


def create_dataloaders(df, batch_size=4, train_split=0.8, on_demand=True, mask_path='rmask_ICV.nii'):
    """Create train and validation dataloaders with stratified split"""
    # Stratified split to maintain group distributions
    train_df, val_df = train_test_split(
        df,
        test_size=1-train_split,
        stratify=df['label'],
        random_state=42
    )

    print("\nTraining set distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation set distribution:")
    print(val_df['label'].value_counts())

    # Create datasets with appropriate strategy
    if on_demand:
        print("Using on-demand data loading strategy (lighter on memory but slower)")
        train_dataset = OnDemandDataset(train_df, mask_path=mask_path)
        val_dataset = OnDemandDataset(val_df, mask_path=mask_path)
    else:
        print("Using batch pre-processing strategy (faster but more memory intensive)")
        train_dataset = BatchLoadDataset(train_df, mask_path=mask_path)
        val_dataset = BatchLoadDataset(val_df, mask_path=mask_path)

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 6 to prevent hanging
        pin_memory=True,
        persistent_workers=False  # Changed from True to avoid memory issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Reduced from 6 to prevent hanging
        pin_memory=True,
        persistent_workers=False  # Changed from True to avoid memory issues
    )

    return train_loader, val_loader
