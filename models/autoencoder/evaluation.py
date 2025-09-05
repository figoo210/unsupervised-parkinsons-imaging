import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm

from models.autoencoder.model import BaseAutoencoder


def load_trained_model(checkpoint_dir, model_name, latent_dim=256):
    """Load best trained model for evaluation"""
    model_path = Path(checkpoint_dir) / f"{model_name}_best.pth"
    metadata_path = Path(checkpoint_dir) / f"{model_name}_metadata.json"

    if not model_path.exists():
        # Try loading checkpoint if best model doesn't exist
        model_path = Path(checkpoint_dir) / f"{model_name}_checkpoint.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        # Load from checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseAutoencoder(latent_dim=latent_dim)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load best model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseAutoencoder(latent_dim=latent_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Load training history
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"train_losses": [], "val_losses": []}

    model.eval()
    model.to(device)

    return model, metadata


def compute_reconstruction_error(model, dataloader):
    """Compute detailed reconstruction error metrics on validation set"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss(reduction='none')

    total_mse = 0
    total_samples = 0
    error_by_label = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            volumes = batch['volume'].to(device)
            labels = batch['label']

            # Get reconstructions
            reconstructed = model(volumes)

            # Compute MSE loss per sample
            mse = criterion(reconstructed, volumes)

            # Average over dimensions
            mse = mse.mean(dim=(1, 2, 3, 4)).cpu().numpy()

            # Track overall error
            total_mse += mse.sum()
            total_samples += volumes.shape[0]

            # Track error by label
            for i, label in enumerate(labels):
                if label not in error_by_label:
                    error_by_label[label] = []
                error_by_label[label].append(mse[i])

            # Memory cleanup
            del volumes, reconstructed, mse
            torch.cuda.empty_cache()

    # Calculate overall metrics
    avg_mse = total_mse / total_samples
    rmse = np.sqrt(avg_mse)

    print("\nReconstruction Error Metrics:")
    print(f"Overall MSE: {avg_mse:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")

    # Calculate metrics by group
    print("\nReconstruction Error by Group:")
    for label, errors in error_by_label.items():
        group_mse = np.mean(errors)
        group_rmse = np.sqrt(group_mse)
        group_std = np.std(errors)
        print(f"{label}:")
        print(f"  MSE: {group_mse:.6f} ± {group_std:.6f}")
        print(f"  RMSE: {group_rmse:.6f}")

    # Plot error distribution by group
    plt.figure(figsize=(10, 6))
    for label, errors in error_by_label.items():
        sns.histplot(errors, alpha=0.5, label=label, bins=20, kde=True)

    plt.title("Reconstruction Error Distribution by Group")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return avg_mse, error_by_label


def extract_latent_vectors(model, dataloader, max_samples=None):
    """Extract latent vectors from all samples in the dataloader"""
    device = next(model.parameters()).device

    latent_vectors = []
    labels = []
    paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent vectors"):
            volumes = batch['volume'].to(device)
            batch_labels = batch['label']
            batch_paths = batch['path']

            # Extract latent vectors
            z = model.encode(volumes)

            # Store results
            latent_vectors.append(z.cpu().numpy())
            labels.extend(batch_labels)
            paths.extend(batch_paths)

            # Memory cleanup
            del volumes, z
            torch.cuda.empty_cache()

            # Check if we have enough samples
            if max_samples and len(labels) >= max_samples:
                latent_vectors = np.vstack(latent_vectors)
                latent_vectors = latent_vectors[:max_samples]
                labels = labels[:max_samples]
                paths = paths[:max_samples]
                break

    # Stack all latent vectors if we didn't break early
    if isinstance(latent_vectors[0], np.ndarray):
        latent_vectors = np.vstack(latent_vectors)

    return latent_vectors, labels, paths


def find_outliers(model, dataloader, threshold_std=2.5):
    """Identify outliers based on reconstruction error"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss(reduction='none')

    all_errors = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Finding outliers"):
            volumes = batch['volume'].to(device)
            paths = batch['path']
            labels = batch['label']

            # Get reconstructions
            reconstructed = model(volumes)

            # Compute MSE loss per sample
            mse = criterion(reconstructed, volumes)

            # Average over dimensions
            mse = mse.mean(dim=(1, 2, 3, 4)).cpu().numpy()

            # Store results
            all_errors.extend(mse)
            all_paths.extend(paths)
            all_labels.extend(labels)

            # Memory cleanup
            del volumes, reconstructed, mse
            torch.cuda.empty_cache()

    # Convert to numpy arrays
    all_errors = np.array(all_errors)

    # Compute statistics
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)

    # Find outliers (samples with error > mean + threshold_std * std)
    threshold = mean_error + threshold_std * std_error
    outlier_indices = np.where(all_errors > threshold)[0]

    print(f"\nOutlier Analysis:")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Error standard deviation: {std_error:.6f}")
    print(f"Outlier threshold: {threshold:.6f}")
    print(f"Found {len(outlier_indices)} outliers out of {len(all_errors)} samples ({len(outlier_indices)/len(all_errors)*100:.2f}%)")

    # Create dictionary of outliers
    outliers = {
        all_paths[i]: {
            'error': all_errors[i],
            'label': all_labels[i],
            'z_score': (all_errors[i] - mean_error) / std_error
        }
        for i in outlier_indices
    }

    # Plot error distribution with outlier threshold
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=30, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_std} std)')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return outliers, all_errors, all_paths, all_labels


def explore_top_dimensions(model, dataloader, dimensions, groups=None):
    """
    Explore the top discriminative dimensions across different patient groups using anatomically relevant slices.
    Ensures consistent scales across all visualizations for a given dimension.

    Parameters:
        model: Trained autoencoder model
        dataloader: DataLoader containing samples
        dimensions: List of dimension indices to explore
        groups: List of groups to include (default is all groups)
    """
    if groups is None:
        # Get all unique groups from the first batch
        for batch in dataloader:
            groups = list(set(batch['label']))
            break

    for dimension in dimensions:
        print(f"\n{'='*80}")
        print(f"Exploring Dimension {dimension}")
        print(f"{'='*80}")
        
        # First pass to determine global min/max difference values
        diff_min_global = float('inf')
        diff_max_global = float('-inf')
        
        # Store results from first pass
        dimension_results = {}
        
        # Scan all groups to find global min/max
        for group in groups:
            print(f"\nAnalyzing dimension {dimension} for group: {group} (first pass)")
            results = visualize_latent_dimension(model, dataloader, dimension, alpha=8.0, group=group)
            
            # Update global min/max
            plus_diff = results['plus_diff']
            minus_diff = results['minus_diff']
            
            curr_min = min(np.min(plus_diff), np.min(minus_diff))
            curr_max = max(np.max(plus_diff), np.max(minus_diff))
            
            diff_min_global = min(diff_min_global, curr_min)
            diff_max_global = max(diff_max_global, curr_max)
            
            # Store results for second pass
            dimension_results[group] = results
        
        # Make the scale symmetric
        diff_abs_max = max(abs(diff_min_global), abs(diff_max_global))
        diff_global_vmin, diff_global_vmax = -diff_abs_max, diff_abs_max
        
        print(f"\nGlobal difference range for dimension {dimension} across all groups: {diff_global_vmin:.3f} to {diff_global_vmax:.3f}")
        
        # Optional: Add a small delay for better visualization
        import time
        time.sleep(1)


