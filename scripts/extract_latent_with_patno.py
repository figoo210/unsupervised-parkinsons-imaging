"""
Script to extract latent vectors with PATNO information from file paths.
This creates CSV files that can be properly merged with clinical data.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from tqdm.auto import tqdm

from data.data_ingestion import collect_files, generate_dataframe
from data.dataloader import create_dataloaders
from models.autoencoder.bottleneck_models import (
    BottleneckEncoder,
    ComplexBottleneckDeconvDecoder,
    BottleneckAE,
)


def extract_patno_from_path(file_path):
    """
    Extract PATNO from file path.
    Expected path structure: .../PPMI_Images_XX/PATNO/...
    """
    path_str = str(file_path)
    # Look for pattern: digits after PPMI_Images_XX/
    match = re.search(r'PPMI_Images_[^/\\]+[/\\](\d+)', path_str)
    if match:
        return int(match.group(1))
    return None


def build_full_model(latent_dim=256, device='cuda'):
    """Build the full autoencoder model"""
    target_shape = (64, 128, 128)
    model = BottleneckAE(
        BottleneckEncoder(initial_filters=4, latent_dim=latent_dim, bottleneck_shape=(1,1,1)),
        ComplexBottleneckDeconvDecoder(latent_dim=latent_dim, target_shape=target_shape),
    )
    return model.to(device)


def extract_latent_vectors_with_metadata(loader, encoder, device, dataset_name):
    """
    Extract latent vectors along with file paths and PATNO.
    
    Returns:
        DataFrame with columns: latent_0...latent_255, Label, FilePath, PATNO
    """
    latent_vectors = []
    labels = []
    file_paths = []
    patnos = []
    
    print(f"Extracting latent vectors from {len(loader)} {dataset_name} subjects...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=dataset_name):
            inp = batch['volume'].to(device)
            label = batch['label'][0]
            path = batch['path'][0]
            
            # Extract latent vector
            latent = encoder(inp)
            latent_vector = latent.squeeze().cpu().numpy()
            
            # Extract PATNO from path
            patno = extract_patno_from_path(path)
            
            latent_vectors.append(latent_vector)
            labels.append(label)
            file_paths.append(path)
            patnos.append(patno)
    
    # Create DataFrame
    latent_array = np.array(latent_vectors)
    latent_dim = latent_array.shape[1]
    
    df = pd.DataFrame(latent_array, columns=[f'latent_{i}' for i in range(latent_dim)])
    df['Label'] = labels
    df['FilePath'] = file_paths
    df['PATNO'] = patnos
    df['Dataset'] = dataset_name
    
    print(f"{dataset_name} latent vectors shape: {latent_array.shape}")
    print(f"{dataset_name} labels: {pd.Series(labels).value_counts().to_dict()}")
    print(f"PATNOs extracted: {df['PATNO'].notna().sum()}/{len(df)}")
    
    return df


def main():
    # Configuration
    data_dir = "data/Images"
    mask_path = "data/masks/rmask_ICV.nii"
    batch_size = 1
    latent_dim = 256
    weighted_output_dir = "output/Experiments/BottleneckDeconvWeightedTraining"
    output_dir = "output/Experiments/LatentVectorAnalysis"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nCollecting files...")
    included_files, excluded_files = collect_files(data_dir)
    print(f"Found {len(included_files)} valid files")
    
    # Filter out MSE outliers
    mse_csv = Path("output/Experiments/MSEPerSubject/mse_per_subject.csv")
    if mse_csv.exists():
        mse_df = pd.read_csv(mse_csv)
        
        def detect_outliers_iqr(data, column, multiplier=2.5):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        outlier_mask = np.zeros(len(mse_df), dtype=bool)
        for dataset in ["Validation", "Test"]:
            ds = mse_df[mse_df["Dataset"] == dataset]
            weighted_outliers = detect_outliers_iqr(ds, "Weighted MSE", multiplier=2.5)
            outlier_mask[ds[weighted_outliers].index] = True
        
        outlier_filenames = set(mse_df.loc[outlier_mask, "File Path"].astype(str).tolist())
        
        def _as_path(x):
            return x[0] if isinstance(x, tuple) else x
        
        included_files = [
            p for p in included_files
            if Path(_as_path(p)).name not in outlier_filenames
        ]
        print(f"Excluding weighted-MSE outlier files: {len(outlier_filenames)}")
        print(f"Remaining files after outlier removal: {len(included_files)}")
    
    # Generate dataframe
    print("\nGenerating dataframe...")
    df = generate_dataframe(included_files)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        df, batch_size=batch_size, train_split=0.8,
        on_demand=True, mask_path=mask_path,
        num_workers=0
    )
    
    # Load model
    print("\nLoading weighted model...")
    weighted_model_name = f"deconv_weighted_lat{latent_dim}"
    checkpoint_path = os.path.join(weighted_output_dir, weighted_model_name, f"{weighted_model_name}_best.pth")
    
    full_model = build_full_model(latent_dim=latent_dim, device=device)
    full_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    full_model.eval()
    
    encoder = full_model.encoder
    print("✓ Loaded weighted encoder")
    
    # Extract latent vectors with metadata
    print("\n" + "="*80)
    print("EXTRACTING LATENT VECTORS WITH METADATA")
    print("="*80)
    
    val_df = extract_latent_vectors_with_metadata(val_loader, encoder, device, "Validation")
    train_df = extract_latent_vectors_with_metadata(train_loader, encoder, device, "Train")
    
    # Save to CSV
    val_output_path = os.path.join(output_dir, 'validation_latent_vectors_with_patno.csv')
    train_output_path = os.path.join(output_dir, 'train_latent_vectors_with_patno.csv')
    
    val_df.to_csv(val_output_path, index=False)
    train_df.to_csv(train_output_path, index=False)
    
    print(f"\n✓ Saved validation latent vectors to: {val_output_path}")
    print(f"✓ Saved train latent vectors to: {train_output_path}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nValidation set:")
    print(f"  Total samples: {len(val_df)}")
    print(f"  Unique PATNOs: {val_df['PATNO'].nunique()}")
    print(f"  Missing PATNOs: {val_df['PATNO'].isna().sum()}")
    
    print(f"\nTrain set:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Unique PATNOs: {train_df['PATNO'].nunique()}")
    print(f"  Missing PATNOs: {train_df['PATNO'].isna().sum()}")


if __name__ == "__main__":
    main()
