import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import gc

def load_trained_vae(checkpoint_dir, model_name='vae_model_v2', latent_dim=256):
    """Load trained VAE model for evaluation with robust error handling"""
    model_path = Path(checkpoint_dir) / f"{model_name}_best.pth"
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}_checkpoint.pth"
    metadata_path = Path(checkpoint_dir) / f"{model_name}_metadata.json"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create VAE model
    model = VAE(latent_dim=latent_dim)
    
    # Load model weights with error handling
    try:
        if model_path.exists():
            print(f"Loading best VAE model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif checkpoint_path.exists():
            print(f"Best model not found. Loading from checkpoint at {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VAE checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            raise FileNotFoundError(f"No VAE model found at {model_path} or {checkpoint_path}")
    except Exception as e:
        print(f"Error loading VAE model: {str(e)}")
        print("Available files in directory:")
        for file in Path(checkpoint_dir).glob("*"):
            print(f" - {file.name}")
        raise

    # Load training history
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded training history with {len(metadata.get('train_losses', []))} epochs")
    else:
        metadata = {"train_losses": [], "val_losses": [], 
                   "train_recon_losses": [], "train_kl_losses": [],
                   "val_recon_losses": [], "val_kl_losses": []}
        print("No metadata found, using empty history")

    # Move model to device and set to evaluation mode
    model.eval()
    model.to(device)

    return model, metadata


def extract_vae_latent_vectors(model, dataloader, max_samples=200):
    """Extract latent vectors from VAE encoder"""
    device = next(model.parameters()).device

    latent_means = []
    latent_logvars = []
    labels = []
    paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting VAE latent vectors"):
            try:
                volumes = batch['volume'].to(device)
                batch_labels = batch['label']
                batch_paths = batch['path']

                # Extract latent vectors (both mean and log_var)
                mu, log_var = model.encode(volumes)

                # Store results
                latent_means.append(mu.cpu().numpy())
                latent_logvars.append(log_var.cpu().numpy())
                labels.extend(batch_labels)
                paths.extend(batch_paths)

                # Memory cleanup
                del volumes, mu, log_var
                torch.cuda.empty_cache()

                # Check if we have enough samples
                if max_samples and len(labels) >= max_samples:
                    latent_means = np.vstack(latent_means)
                    latent_logvars = np.vstack(latent_logvars)
                    latent_means = latent_means[:max_samples]
                    latent_logvars = latent_logvars[:max_samples]
                    labels = labels[:max_samples]
                    paths = paths[:max_samples]
                    break
            
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

    # Stack all latent vectors if we didn't break early
    if isinstance(latent_means[0], np.ndarray):
        latent_means = np.vstack(latent_means)
        latent_logvars = np.vstack(latent_logvars)

    return latent_means, latent_logvars, labels, paths


def compute_reconstruction_error(model, dataloader):
    """Compute detailed reconstruction error metrics on validation set"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss(reduction='none')

    total_mse = 0
    total_samples = 0
    error_by_label = {}
    kl_by_label = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            volumes = batch['volume'].to(device)
            labels = batch['label']

            # Get reconstructions and latent variables
            reconstructed, mu, log_var = model(volumes)

            # Compute MSE loss per sample
            mse = criterion(reconstructed, volumes)

            # Average over dimensions
            mse = mse.mean(dim=(1, 2, 3, 4)).cpu().numpy()
            
            # Compute KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()

            # Track overall error
            total_mse += mse.sum()
            total_samples += volumes.shape[0]

            # Track error by label
            for i, label in enumerate(labels):
                if label not in error_by_label:
                    error_by_label[label] = []
                    kl_by_label[label] = []
                error_by_label[label].append(mse[i])
                kl_by_label[label].append(kl_div[i])

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
        group_kl = np.mean(kl_by_label[label])
        print(f"{label}:")
        print(f"  MSE: {group_mse:.6f} ± {group_std:.6f}")
        print(f"  RMSE: {group_rmse:.6f}")
        print(f"  KL Divergence: {group_kl:.6f}")

    # Plot error distribution by group
    plt.figure(figsize=(15, 6))
    
    # MSE Distribution
    plt.subplot(1, 2, 1)
    for label, errors in error_by_label.items():
        sns.histplot(errors, alpha=0.5, label=label, bins=20, kde=True)
    plt.title("Reconstruction Error Distribution by Group")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # KL Divergence Distribution
    plt.subplot(1, 2, 2)
    for label, kl_values in kl_by_label.items():
        sns.histplot(kl_values, alpha=0.5, label=label, bins=20, kde=True)
    plt.title("KL Divergence Distribution by Group")
    plt.xlabel("KL Divergence")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return avg_mse, error_by_label, kl_by_label


def find_vae_outliers(model, dataloader, threshold_std=2.5):
    """Identify outliers based on reconstruction error"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss(reduction='none')

    all_errors = []
    all_paths = []
    all_labels = []
    all_kl_divs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Finding outliers"):
            volumes = batch['volume'].to(device)
            paths = batch['path']
            labels = batch['label']

            # Get reconstructions and latent variables
            reconstructed, mu, log_var = model(volumes)

            # Compute MSE loss per sample
            mse = criterion(reconstructed, volumes)

            # Average over dimensions
            mse = mse.mean(dim=(1, 2, 3, 4)).cpu().numpy()
            
            # Compute KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).cpu().numpy()

            # Store results
            all_errors.extend(mse)
            all_paths.extend(paths)
            all_labels.extend(labels)
            all_kl_divs.extend(kl_div)

    # Convert to numpy arrays
    all_errors = np.array(all_errors)
    all_kl_divs = np.array(all_kl_divs)

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
            'kl_div': all_kl_divs[i],
            'z_score': (all_errors[i] - mean_error) / std_error
        }
        for i in outlier_indices
    }

    # Plot error distribution with outlier threshold
    plt.figure(figsize=(15, 6))
    
    # Plot MSE distribution
    plt.subplot(1, 2, 1)
    plt.hist(all_errors, bins=30, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_std} std)')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error vs KL divergence, highlighting outliers
    plt.subplot(1, 2, 2)
    plt.scatter(all_errors, all_kl_divs, alpha=0.5, s=20, label="Normal samples")
    
    # Highlight outliers
    outlier_errors = [all_errors[i] for i in outlier_indices]
    outlier_kl_divs = [all_kl_divs[i] for i in outlier_indices]
    plt.scatter(outlier_errors, outlier_kl_divs, color='r', s=50, label="Outliers")
    
    plt.axvline(threshold, color='r', linestyle='--')
    plt.title("Error vs KL Divergence")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return outliers, all_errors, all_paths, all_labels




