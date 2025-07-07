"""
Evaluation functions for model performance analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from ..utils.memory_utils import clear_memory


def compute_reconstruction_error(model: torch.nn.Module, dataloader, 
                               error_type: str = 'mse') -> Dict[str, float]:
    """
    Compute reconstruction error for the model.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader for evaluation
        error_type (str): Type of error metric ('mse', 'mae', 'rmse')
        
    Returns:
        Dict[str, float]: Error metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction error"):
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
            else:
                volumes = batch.to(device)
            
            # Get reconstructions
            if hasattr(model, 'forward'):
                if 'VAE' in str(type(model)):
                    reconstructions, _, _ = model(volumes)
                else:
                    reconstructions = model(volumes)
            else:
                reconstructions = model(volumes)
            
            # Compute batch errors
            batch_mse = nn.MSELoss()(reconstructions, volumes).item()
            batch_mae = nn.L1Loss()(reconstructions, volumes).item()
            
            total_mse += batch_mse * len(volumes)
            total_mae += batch_mae * len(volumes)
            total_samples += len(volumes)
            
            # Clear memory periodically
            clear_memory()
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'total_samples': total_samples
    }


def evaluate_model_performance(model: torch.nn.Module, train_loader, val_loader,
                             test_loader=None) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive model evaluation across different data splits.
    
    Args:
        model (torch.nn.Module): Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        
    Returns:
        Dict[str, Dict[str, float]]: Performance metrics for each split
    """
    results = {}
    
    # Evaluate on training set
    print("Evaluating on training set...")
    results['train'] = compute_reconstruction_error(model, train_loader)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    results['validation'] = compute_reconstruction_error(model, val_loader)
    
    # Evaluate on test set if provided
    if test_loader:
        print("Evaluating on test set...")
        results['test'] = compute_reconstruction_error(model, test_loader)
    
    return results


def find_outliers(model: torch.nn.Module, dataloader, threshold_percentile: float = 95,
                 method: str = 'reconstruction_error') -> Tuple[List[int], np.ndarray]:
    """
    Find outliers based on reconstruction error or latent space analysis.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader for analysis
        threshold_percentile (float): Percentile threshold for outlier detection
        method (str): Method for outlier detection ('reconstruction_error' or 'latent_space')
        
    Returns:
        Tuple[List[int], np.ndarray]: Outlier indices and scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    if method == 'reconstruction_error':
        scores = []
        sample_indices = []
        
        with torch.no_grad():
            sample_idx = 0
            for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
                if isinstance(batch, dict):
                    volumes = batch['volume'].to(device)
                else:
                    volumes = batch.to(device)
                
                # Get reconstructions
                if 'VAE' in str(type(model)):
                    reconstructions, _, _ = model(volumes)
                else:
                    reconstructions = model(volumes)
                
                # Compute per-sample reconstruction error
                batch_errors = torch.mean((reconstructions - volumes) ** 2, dim=[1, 2, 3, 4])
                
                scores.extend(batch_errors.cpu().numpy())
                sample_indices.extend(range(sample_idx, sample_idx + len(volumes)))
                sample_idx += len(volumes)
                
                clear_memory()
        
        scores = np.array(scores)
        threshold = np.percentile(scores, threshold_percentile)
        outlier_indices = [i for i, score in enumerate(scores) if score > threshold]
        
    elif method == 'latent_space':
        latent_vectors = []
        sample_indices = []
        
        with torch.no_grad():
            sample_idx = 0
            for batch in tqdm(dataloader, desc="Extracting latent vectors"):
                if isinstance(batch, dict):
                    volumes = batch['volume'].to(device)
                else:
                    volumes = batch.to(device)
                
                # Extract latent representations
                if hasattr(model, 'encode'):
                    if 'VAE' in str(type(model)):
                        mu, _ = model.encode(volumes)
                        latent = mu
                    else:
                        latent = model.encode(volumes)
                else:
                    # For autoencoder, use bottleneck representation
                    with torch.no_grad():
                        latent = model.encoder(volumes)
                
                latent_vectors.append(latent.cpu().numpy())
                sample_indices.extend(range(sample_idx, sample_idx + len(volumes)))
                sample_idx += len(volumes)
                
                clear_memory()
        
        latent_vectors = np.vstack(latent_vectors)
        
        # Use DBSCAN for outlier detection in latent space
        scaler = StandardScaler()
        latent_scaled = scaler.fit_transform(latent_vectors)
        
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(latent_scaled)
        
        # Outliers are points labeled as -1
        outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        scores = cluster_labels
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outlier_indices, scores


def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Calculate various reconstruction metrics between original and reconstructed volumes.
    
    Args:
        original (np.ndarray): Original volumes
        reconstructed (np.ndarray): Reconstructed volumes
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Flatten arrays for metric calculation
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    
    # Basic reconstruction metrics
    mse = mean_squared_error(original_flat, reconstructed_flat)
    mae = mean_absolute_error(original_flat, reconstructed_flat)
    rmse = np.sqrt(mse)
    
    # R-squared score
    r2 = r2_score(original_flat, reconstructed_flat)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        max_pixel = np.max(original_flat)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Structural Similarity Index (simplified version)
    # Calculate means and variances
    mu1, mu2 = np.mean(original_flat), np.mean(reconstructed_flat)
    sigma1_sq, sigma2_sq = np.var(original_flat), np.var(reconstructed_flat)
    sigma12 = np.cov(original_flat, reconstructed_flat)[0, 1]
    
    # SSIM constants
    c1, c2 = 0.01**2, 0.03**2
    
    # SSIM calculation
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = numerator / denominator
    
    # Normalized Cross Correlation
    ncc = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'psnr': psnr,
        'ssim': ssim,
        'ncc': ncc if not np.isnan(ncc) else 0.0
    }


def evaluate_reconstruction_quality_by_group(model: torch.nn.Module, dataloader) -> pd.DataFrame:
    """
    Evaluate reconstruction quality separately for different groups in the dataset.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader with group information
        
    Returns:
        pd.DataFrame: Metrics by group
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    group_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating by group"):
            if not isinstance(batch, dict) or 'group' not in batch:
                print("Warning: No group information found in batch")
                continue
            
            volumes = batch['volume'].to(device)
            groups = batch['group']
            
            # Get reconstructions
            if 'VAE' in str(type(model)):
                reconstructions, _, _ = model(volumes)
            else:
                reconstructions = model(volumes)
            
            # Convert to numpy
            volumes_np = volumes.cpu().numpy()
            reconstructions_np = reconstructions.cpu().numpy()
            
            # Calculate metrics for each sample
            for i, group in enumerate(groups):
                if group not in group_metrics:
                    group_metrics[group] = []
                
                # Calculate metrics for this sample
                sample_metrics = calculate_metrics(
                    volumes_np[i:i+1], 
                    reconstructions_np[i:i+1]
                )
                group_metrics[group].append(sample_metrics)
            
            clear_memory()
    
    # Aggregate metrics by group
    results = []
    for group, metrics_list in group_metrics.items():
        if not metrics_list:
            continue
        
        # Calculate mean and std for each metric
        aggregated = {}
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
            if values:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_median'] = np.median(values)
            else:
                aggregated[f'{metric_name}_mean'] = np.nan
                aggregated[f'{metric_name}_std'] = np.nan
                aggregated[f'{metric_name}_median'] = np.nan
        
        aggregated['group'] = group
        aggregated['n_samples'] = len(metrics_list)
        results.append(aggregated)
    
    return pd.DataFrame(results)


def compute_latent_space_metrics(model: torch.nn.Module, dataloader) -> Dict[str, float]:
    """
    Compute metrics related to the latent space representation.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader for analysis
        
    Returns:
        Dict[str, float]: Latent space metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    latent_vectors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent vectors"):
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
            else:
                volumes = batch.to(device)
            
            # Extract latent representations
            if hasattr(model, 'encode'):
                if 'VAE' in str(type(model)):
                    mu, log_var = model.encode(volumes)
                    latent = mu
                    # For VAE, also compute KL divergence
                    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                    avg_kl_div = torch.mean(kl_div).item()
                else:
                    latent = model.encode(volumes)
                    avg_kl_div = None
            else:
                # For autoencoder, use bottleneck representation
                latent = model.encoder(volumes)
                avg_kl_div = None
            
            latent_vectors.append(latent.cpu().numpy())
            clear_memory()
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Compute latent space statistics
    metrics = {
        'latent_dim': latent_vectors.shape[1],
        'latent_mean': np.mean(latent_vectors),
        'latent_std': np.std(latent_vectors),
        'latent_var': np.var(latent_vectors),
        'latent_min': np.min(latent_vectors),
        'latent_max': np.max(latent_vectors),
        'dead_units': np.sum(np.var(latent_vectors, axis=0) < 1e-6),  # Units with very low variance
        'active_units': np.sum(np.var(latent_vectors, axis=0) >= 1e-6)
    }
    
    # Add VAE-specific metrics
    if avg_kl_div is not None:
        metrics['avg_kl_divergence'] = avg_kl_div
    
    # Compute utilization metrics
    latent_var = np.var(latent_vectors, axis=0)
    metrics['utilization_ratio'] = np.mean(latent_var > 1e-6)  # Fraction of dimensions actively used
    metrics['effective_dimension'] = np.sum(latent_var > np.mean(latent_var) * 0.1)  # Effective dimensionality
    
    return metrics
