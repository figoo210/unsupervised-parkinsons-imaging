import os
import pandas as pd
import torch
import numpy as np
from data.dataloader import OnDemandDataset, create_dataloaders
from processing.eda import analyze_dataset_statistics_efficiently

def test_dataloader():
    """Test the dataloader with the fixed imports"""
    print("Testing dataloader with fixed imports...")
    
    # Load the validated file paths
    validated_path = os.path.join("output", "Validated", "validated_file_paths.csv")
    if os.path.exists(validated_path):
        df = pd.read_csv(validated_path)
        print(f"Loaded {len(df)} validated file paths")
    else:
        print(f"Validated file paths not found at {validated_path}")
        return
    
    # Create a small sample for testing
    sample_df = df.groupby('label').head(3).reset_index(drop=True)
    print(f"Created sample with {len(sample_df)} files")
    
    # Create a test dataset
    mask_path = os.path.join("data", "masks", "rmask_ICV.nii")
    test_dataset = OnDemandDataset(sample_df, mask_path=mask_path)
    
    # Test loading a few samples
    print("\nTesting individual sample loading:")
    for i in range(min(3, len(test_dataset))):
        try:
            sample = test_dataset[i]
            volume = sample["volume"]
            label = sample["label"]
            path = sample["path"]
            
            print(f"Sample {i}:")
            print(f"  Path: {path}")
            print(f"  Label: {label}")
            print(f"  Volume shape: {volume.shape}")
            print(f"  Volume stats: min={volume.min().item():.4f}, max={volume.max().item():.4f}, "
                  f"mean={volume.mean().item():.4f}, std={volume.std().item():.4f}")
            
            # Check if it's a zero tensor
            if volume.min().item() == 0 and volume.max().item() == 0:
                print("  WARNING: Zero tensor detected!")
            else:
                print("  Volume loaded successfully with non-zero values")
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
    
    # Test the dataset statistics function with a small batch size
    print("\nTesting dataset statistics function:")
    try:
        # Create dataloaders with a small batch size
        train_loader, val_loader = create_dataloaders(
            sample_df, 
            batch_size=2, 
            train_split=0.5,
            on_demand=True,
            mask_path=mask_path
        )
        
        # Analyze dataset statistics
        stats_df = analyze_dataset_statistics_efficiently(
            train_loader,
            max_samples=6,
            min_samples_per_group=1
        )
        
        # Print statistics
        print("\nDataset Statistics:")
        print(stats_df)
        
        # Check for zero values
        zero_means = (stats_df['mean'] == 0).sum()
        if zero_means > 0:
            print(f"WARNING: Found {zero_means} samples with zero mean values")
        else:
            print("SUCCESS: No zero mean values detected in statistics")
            
    except Exception as e:
        print(f"Error testing dataset statistics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()
