"""
Model interpretation and feature analysis functions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, mannwhitneyu
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import pandas as pd

from ..utils.memory_utils import clear_memory


def extract_latent_vectors(model: torch.nn.Module, dataloader, 
                          include_groups: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Extract latent vectors from trained model.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader
        include_groups (bool): Whether to extract group information
        
    Returns:
        Tuple[np.ndarray, List[str]]: Latent vectors and group labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    latent_vectors = []
    group_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latent vectors"):
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
                if include_groups and 'group' in batch:
                    groups = batch['group']
                else:
                    groups = ['Unknown'] * len(volumes)
            else:
                volumes = batch.to(device)
                groups = ['Unknown'] * len(volumes)
            
            # Extract latent representations
            if hasattr(model, 'encode'):
                if 'VAE' in str(type(model)):
                    mu, log_var = model.encode(volumes)
                    latent = mu  # Use mean for deterministic representation
                else:
                    latent = model.encode(volumes)
            else:
                # For autoencoder, use bottleneck representation
                latent = model.encoder(volumes)
            
            latent_vectors.append(latent.cpu().numpy())
            group_labels.extend(groups)
            
            clear_memory()
    
    latent_vectors = np.vstack(latent_vectors)
    return latent_vectors, group_labels


def generate_feature_importance_map(model: torch.nn.Module, sample_volume: torch.Tensor,
                                   method: str = 'gradient') -> np.ndarray:
    """
    Generate feature importance map using gradient-based methods.
    
    Args:
        model (torch.nn.Module): Trained model
        sample_volume (torch.Tensor): Input volume to analyze
        method (str): Method for importance calculation ('gradient' or 'integrated_gradient')
        
    Returns:
        np.ndarray: Feature importance map
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    sample_volume = sample_volume.to(device)
    sample_volume.requires_grad_(True)
    
    if method == 'gradient':
        # Standard gradient method
        if 'VAE' in str(type(model)):
            recon, mu, log_var = model(sample_volume)
            # Use reconstruction loss as the target
            loss = nn.MSELoss()(recon, sample_volume)
        else:
            recon = model(sample_volume)
            loss = nn.MSELoss()(recon, sample_volume)
        
        # Compute gradients
        loss.backward()
        importance_map = torch.abs(sample_volume.grad).cpu().numpy()
        
    elif method == 'integrated_gradient':
        # Integrated gradients method
        baseline = torch.zeros_like(sample_volume)
        steps = 50
        
        importance_map = torch.zeros_like(sample_volume)
        
        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (sample_volume - baseline)
            interpolated.requires_grad_(True)
            
            if 'VAE' in str(type(model)):
                recon, mu, log_var = model(interpolated)
                loss = nn.MSELoss()(recon, interpolated)
            else:
                recon = model(interpolated)
                loss = nn.MSELoss()(recon, interpolated)
            
            grads = torch.autograd.grad(loss, interpolated, create_graph=False)[0]
            importance_map += grads
        
        importance_map = importance_map * (sample_volume - baseline) / steps
        importance_map = torch.abs(importance_map).cpu().numpy()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance_map


def analyze_latent_dimensions(latent_vectors: np.ndarray, group_labels: List[str],
                            statistical_test: str = 'ttest') -> pd.DataFrame:
    """
    Analyze latent dimensions for group differences.
    
    Args:
        latent_vectors (np.ndarray): Latent space representations
        group_labels (List[str]): Group labels for each sample
        statistical_test (str): Statistical test to use ('ttest' or 'mannwhitney')
        
    Returns:
        pd.DataFrame: Analysis results for each dimension
    """
    unique_groups = list(set(group_labels))
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    # For simplicity, compare first two groups
    group1, group2 = unique_groups[0], unique_groups[1]
    
    group1_mask = np.array(group_labels) == group1
    group2_mask = np.array(group_labels) == group2
    
    group1_vectors = latent_vectors[group1_mask]
    group2_vectors = latent_vectors[group2_mask]
    
    results = []
    
    for dim in range(latent_vectors.shape[1]):
        dim_data1 = group1_vectors[:, dim]
        dim_data2 = group2_vectors[:, dim]
        
        # Compute basic statistics
        mean1, mean2 = np.mean(dim_data1), np.mean(dim_data2)
        std1, std2 = np.std(dim_data1), np.std(dim_data2)
        
        # Perform statistical test
        if statistical_test == 'ttest':
            stat, p_value = ttest_ind(dim_data1, dim_data2)
        elif statistical_test == 'mannwhitney':
            stat, p_value = mannwhitneyu(dim_data1, dim_data2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown statistical test: {statistical_test}")
        
        # Effect size (Cohen's d for t-test)
        if statistical_test == 'ttest':
            pooled_std = np.sqrt(((len(dim_data1) - 1) * std1**2 + 
                                (len(dim_data2) - 1) * std2**2) / 
                               (len(dim_data1) + len(dim_data2) - 2))
            effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        else:
            effect_size = np.nan
        
        results.append({
            'dimension': dim,
            'group1_mean': mean1,
            'group2_mean': mean2,
            'group1_std': std1,
            'group2_std': std2,
            'mean_difference': mean1 - mean2,
            'statistic': stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        })
    
    df = pd.DataFrame(results)
    
    # Add multiple comparison correction (Benjamini-Hochberg)
    from scipy.stats import false_discovery_control
    corrected_p = false_discovery_control(df['p_value'].values)
    df['p_value_corrected'] = corrected_p
    df['significant_corrected'] = corrected_p < 0.05
    
    return df


def visualize_latent_dimension(latent_vectors: np.ndarray, group_labels: List[str],
                             dimension: int, save_path: Optional[str] = None) -> None:
    """
    Visualize specific latent dimension across groups.
    
    Args:
        latent_vectors (np.ndarray): Latent space representations
        group_labels (List[str]): Group labels
        dimension (int): Dimension to visualize
        save_path (Optional[str]): Path to save the plot
    """
    unique_groups = list(set(group_labels))
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution comparison
    for group in unique_groups:
        mask = np.array(group_labels) == group
        group_data = latent_vectors[mask, dimension]
        
        ax1.hist(group_data, alpha=0.7, label=group, bins=30, density=True)
        ax1.axvline(np.mean(group_data), color='red', linestyle='--', 
                   label=f'{group} mean', alpha=0.8)
    
    ax1.set_xlabel(f'Latent Dimension {dimension} Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Latent Dimension {dimension}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    group_data_list = []
    group_names = []
    
    for group in unique_groups:
        mask = np.array(group_labels) == group
        group_data = latent_vectors[mask, dimension]
        group_data_list.append(group_data)
        group_names.append(group)
    
    ax2.boxplot(group_data_list, labels=group_names)
    ax2.set_ylabel(f'Latent Dimension {dimension} Value')
    ax2.set_title(f'Box Plot of Latent Dimension {dimension}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_reconstruction_attribution(model: torch.nn.Module, sample_volume: torch.Tensor,
                                     target_region: Optional[Tuple] = None) -> np.ndarray:
    """
    Compute attribution of input regions to reconstruction quality.
    
    Args:
        model (torch.nn.Module): Trained model
        sample_volume (torch.Tensor): Input volume
        target_region (Optional[Tuple]): Specific region to analyze (slice indices)
        
    Returns:
        np.ndarray: Attribution map
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    sample_volume = sample_volume.to(device)
    sample_volume.requires_grad_(True)
    
    # Get reconstruction
    if 'VAE' in str(type(model)):
        recon, mu, log_var = model(sample_volume)
    else:
        recon = model(sample_volume)
    
    # Define target for attribution
    if target_region:
        # Focus on specific region
        z_start, z_end, y_start, y_end, x_start, x_end = target_region
        target_output = recon[..., z_start:z_end, y_start:y_end, x_start:x_end]
        target_input = sample_volume[..., z_start:z_end, y_start:y_end, x_start:x_end]
    else:
        # Use entire volume
        target_output = recon
        target_input = sample_volume
    
    # Compute reconstruction loss for target region
    loss = nn.MSELoss()(target_output, target_input)
    
    # Compute gradients
    loss.backward()
    attribution = torch.abs(sample_volume.grad).cpu().numpy()
    
    return attribution


def analyze_latent_interpolation(model: torch.nn.Module, sample1: torch.Tensor, 
                               sample2: torch.Tensor, num_steps: int = 10) -> List[np.ndarray]:
    """
    Analyze latent space interpolation between two samples.
    
    Args:
        model (torch.nn.Module): Trained model
        sample1 (torch.Tensor): First sample
        sample2 (torch.Tensor): Second sample
        num_steps (int): Number of interpolation steps
        
    Returns:
        List[np.ndarray]: Interpolated reconstructions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    sample1 = sample1.to(device)
    sample2 = sample2.to(device)
    
    with torch.no_grad():
        # Encode both samples
        if hasattr(model, 'encode'):
            if 'VAE' in str(type(model)):
                z1, _ = model.encode(sample1)
                z2, _ = model.encode(sample2)
            else:
                z1 = model.encode(sample1)
                z2 = model.encode(sample2)
        else:
            z1 = model.encoder(sample1)
            z2 = model.encoder(sample2)
        
        # Interpolate in latent space
        alphas = np.linspace(0, 1, num_steps)
        interpolated_reconstructions = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode interpolated latent vector
            if hasattr(model, 'decode'):
                recon = model.decode(z_interp)
            else:
                recon = model.decoder(z_interp)
            
            interpolated_reconstructions.append(recon.cpu().numpy())
    
    return interpolated_reconstructions


def compute_latent_space_density(latent_vectors: np.ndarray, group_labels: List[str],
                                method: str = 'gaussian_kde') -> Dict[str, np.ndarray]:
    """
    Compute density estimation in latent space for different groups.
    
    Args:
        latent_vectors (np.ndarray): Latent vectors
        group_labels (List[str]): Group labels
        method (str): Density estimation method
        
    Returns:
        Dict[str, np.ndarray]: Density estimates for each group
    """
    from scipy.stats import gaussian_kde
    
    unique_groups = list(set(group_labels))
    density_estimates = {}
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    for group in unique_groups:
        mask = np.array(group_labels) == group
        group_data = latent_2d[mask]
        
        if len(group_data) > 1:
            if method == 'gaussian_kde':
                kde = gaussian_kde(group_data.T)
                
                # Create grid for density estimation
                x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
                y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()
                
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).reshape(xx.shape)
                
                density_estimates[group] = {
                    'density': density,
                    'xx': xx,
                    'yy': yy,
                    'data_2d': group_data
                }
    
    return density_estimates


def find_most_representative_samples(model: torch.nn.Module, dataloader,
                                   group: str, num_samples: int = 5) -> List[Tuple[int, float]]:
    """
    Find most representative samples for a specific group based on reconstruction quality.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader: Data loader with group information
        group (str): Target group
        num_samples (int): Number of representative samples to find
        
    Returns:
        List[Tuple[int, float]]: List of (sample_index, reconstruction_error) tuples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    group_samples = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Finding representative samples for {group}"):
            if isinstance(batch, dict) and 'group' in batch:
                volumes = batch['volume'].to(device)
                groups = batch['group']
            else:
                continue
            
            # Get reconstructions
            if 'VAE' in str(type(model)):
                reconstructions, _, _ = model(volumes)
            else:
                reconstructions = model(volumes)
            
            # Calculate reconstruction errors for target group
            for i, sample_group in enumerate(groups):
                if sample_group == group:
                    recon_error = torch.mean((reconstructions[i] - volumes[i]) ** 2).item()
                    group_samples.append((sample_idx + i, recon_error))
            
            sample_idx += len(volumes)
            clear_memory()
    
    # Sort by reconstruction error (ascending - better reconstructions first)
    group_samples.sort(key=lambda x: x[1])
    
    # Return the most representative (best reconstructed) samples
    return group_samples[:num_samples]
