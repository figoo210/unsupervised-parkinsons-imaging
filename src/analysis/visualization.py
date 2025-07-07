"""
Visualization functions for model analysis and results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd

from ..utils.memory_utils import clear_memory


def plot_training_history(metadata: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot training history for autoencoder.
    
    Args:
        metadata (Dict): Training metadata containing loss history
        save_path (Optional[str]): Path to save the plot
    """
    train_losses = metadata.get('train_losses', [])
    val_losses = metadata.get('val_losses', [])
    
    if not train_losses:
        print("No training history found in metadata")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'metrics' in metadata and 'learning_rate' in metadata['metrics']:
        learning_rates = [lr for lr in metadata['metrics']['learning_rate'] if lr is not None]
        if learning_rates:
            lr_epochs = range(1, len(learning_rates) + 1)
            ax2.plot(lr_epochs, learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_vae_training_history(metadata: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot training history for VAE with reconstruction and KL losses.
    
    Args:
        metadata (Dict): VAE training metadata containing loss history
        save_path (Optional[str]): Path to save the plot
    """
    train_losses = metadata.get('train_losses', [])
    val_losses = metadata.get('val_losses', [])
    train_recon_losses = metadata.get('train_recon_losses', [])
    train_kl_losses = metadata.get('train_kl_losses', [])
    val_recon_losses = metadata.get('val_recon_losses', [])
    val_kl_losses = metadata.get('val_kl_losses', [])
    
    if not train_losses:
        print("No VAE training history found in metadata")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot total losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Total Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax1.plot(val_epochs, val_losses, 'r-', label='Validation Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('VAE Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot reconstruction losses
    if train_recon_losses:
        ax2.plot(epochs, train_recon_losses, 'g-', label='Training Recon Loss', linewidth=2)
        if val_recon_losses:
            ax2.plot(val_epochs, val_recon_losses, 'orange', label='Validation Recon Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reconstruction Loss')
        ax2.set_title('VAE Reconstruction Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot KL losses
    if train_kl_losses:
        ax3.plot(epochs, train_kl_losses, 'purple', label='Training KL Loss', linewidth=2)
        if val_kl_losses:
            ax3.plot(val_epochs, val_kl_losses, 'brown', label='Validation KL Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('KL Divergence Loss')
        ax3.set_title('VAE KL Divergence Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'metrics' in metadata and 'learning_rate' in metadata['metrics']:
        learning_rates = [lr for lr in metadata['metrics']['learning_rate'] if lr is not None]
        if learning_rates:
            lr_epochs = range(1, len(learning_rates) + 1)
            ax4.plot(lr_epochs, learning_rates, 'navy', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_reconstruction_samples(model: torch.nn.Module, dataloader, num_samples: int = 3,
                                   save_path: Optional[str] = None) -> None:
    """
    Visualize reconstruction samples from autoencoder.
    
    Args:
        model (torch.nn.Module): Trained autoencoder model
        dataloader: Data loader for samples
        num_samples (int): Number of samples to visualize
        save_path (Optional[str]): Path to save the plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    samples_shown = 0
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= num_samples:
                break
            
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
                groups = batch.get('group', ['Unknown'] * len(volumes))
            else:
                volumes = batch.to(device)
                groups = ['Unknown'] * len(volumes)
            
            reconstructions = model(volumes)
            
            for i in range(min(len(volumes), num_samples - samples_shown)):
                original = volumes[i].cpu().numpy()
                recon = reconstructions[i].cpu().numpy()
                
                # Remove channel dimension if present
                if original.ndim == 4:
                    original = original[0]
                    recon = recon[0]
                
                # Get middle slices for different views
                d, h, w = original.shape
                axial_slice = d // 2
                coronal_slice = h // 2
                sagittal_slice = w // 2
                
                row = samples_shown
                
                # Original volume slices
                axes[row, 0].imshow(original[axial_slice], cmap='gray')
                axes[row, 0].set_title(f'Original Axial\nGroup: {groups[i]}')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(original[:, coronal_slice, :], cmap='gray')
                axes[row, 1].set_title('Original Coronal')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(original[:, :, sagittal_slice], cmap='gray')
                axes[row, 2].set_title('Original Sagittal')
                axes[row, 2].axis('off')
                
                # Reconstructed volume slices
                axes[row, 3].imshow(recon[axial_slice], cmap='gray')
                axes[row, 3].set_title('Reconstructed Axial')
                axes[row, 3].axis('off')
                
                axes[row, 4].imshow(recon[:, coronal_slice, :], cmap='gray')
                axes[row, 4].set_title('Reconstructed Coronal')
                axes[row, 4].axis('off')
                
                axes[row, 5].imshow(recon[:, :, sagittal_slice], cmap='gray')
                axes[row, 5].set_title('Reconstructed Sagittal')
                axes[row, 5].axis('off')
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
            
            if samples_shown >= num_samples:
                break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_vae_reconstructions(model: torch.nn.Module, dataloader, num_samples: int = 3,
                                save_path: Optional[str] = None) -> None:
    """
    Visualize VAE reconstructions with uncertainty visualization.
    
    Args:
        model (torch.nn.Module): Trained VAE model
        dataloader: Data loader for samples
        num_samples (int): Number of samples to visualize
        save_path (Optional[str]): Path to save the plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    samples_shown = 0
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= num_samples:
                break
            
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
                groups = batch.get('group', ['Unknown'] * len(volumes))
            else:
                volumes = batch.to(device)
                groups = ['Unknown'] * len(volumes)
            
            reconstructions, mu, log_var = model(volumes)
            
            for i in range(min(len(volumes), num_samples - samples_shown)):
                original = volumes[i].cpu().numpy()
                recon = reconstructions[i].cpu().numpy()
                
                # Remove channel dimension if present
                if original.ndim == 4:
                    original = original[0]
                    recon = recon[0]
                
                # Get middle slices for different views
                d, h, w = original.shape
                axial_slice = d // 2
                coronal_slice = h // 2
                sagittal_slice = w // 2
                
                row = samples_shown
                
                # Original volume slices
                axes[row, 0].imshow(original[axial_slice], cmap='gray')
                axes[row, 0].set_title(f'Original Axial\nGroup: {groups[i]}')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(original[:, coronal_slice, :], cmap='gray')
                axes[row, 1].set_title('Original Coronal')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(original[:, :, sagittal_slice], cmap='gray')
                axes[row, 2].set_title('Original Sagittal')
                axes[row, 2].axis('off')
                
                # Reconstructed volume slices
                axes[row, 3].imshow(recon[axial_slice], cmap='gray')
                axes[row, 3].set_title('VAE Recon Axial')
                axes[row, 3].axis('off')
                
                axes[row, 4].imshow(recon[:, coronal_slice, :], cmap='gray')
                axes[row, 4].set_title('VAE Recon Coronal')
                axes[row, 4].axis('off')
                
                axes[row, 5].imshow(recon[:, :, sagittal_slice], cmap='gray')
                axes[row, 5].set_title('VAE Recon Sagittal')
                axes[row, 5].axis('off')
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
            
            if samples_shown >= num_samples:
                break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_latent_space(latent_vectors: np.ndarray, labels: List[str], 
                          method: str = 'tsne', save_path: Optional[str] = None) -> None:
    """
    Visualize latent space using dimensionality reduction.
    
    Args:
        latent_vectors (np.ndarray): Latent vectors to visualize
        labels (List[str]): Labels for each latent vector
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        save_path (Optional[str]): Path to save the plot
    """
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        embedded = reducer.fit_transform(latent_vectors)
        title = 't-SNE Visualization of Latent Space'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(latent_vectors)
        title = f'PCA Visualization of Latent Space (Explained Variance: {reducer.explained_variance_ratio_.sum():.3f})'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create color map for different groups
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(10, 8))
    
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[color_map[label]], label=label, alpha=0.7, s=50)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_latent_dimension_activation(latent_vectors: np.ndarray, labels: List[str],
                                   save_path: Optional[str] = None) -> None:
    """
    Plot activation patterns across latent dimensions for different groups.
    
    Args:
        latent_vectors (np.ndarray): Latent vectors
        labels (List[str]): Group labels
        save_path (Optional[str]): Path to save the plot
    """
    df_data = []
    for i, label in enumerate(labels):
        for dim in range(latent_vectors.shape[1]):
            df_data.append({
                'dimension': dim,
                'activation': latent_vectors[i, dim],
                'group': label
            })
    
    df = pd.DataFrame(df_data)
    
    # Calculate statistics per dimension and group
    stats = df.groupby(['dimension', 'group'])['activation'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(15, 8))
    
    unique_groups = df['group'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
    
    for i, group in enumerate(unique_groups):
        group_stats = stats[stats['group'] == group]
        plt.errorbar(group_stats['dimension'], group_stats['mean'], 
                    yerr=group_stats['std'], label=group, 
                    color=colors[i], alpha=0.7, capsize=3)
    
    plt.xlabel('Latent Dimension')
    plt.ylabel('Average Activation')
    plt.title('Latent Dimension Activation by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_vae_uncertainty(model: torch.nn.Module, dataloader, group: Optional[str] = None,
                             num_samples: int = 20, save_path: Optional[str] = None) -> None:
    """
    Visualize VAE uncertainty by sampling from the latent distribution.
    
    Args:
        model (torch.nn.Module): Trained VAE model
        dataloader: Data loader for samples
        group (Optional[str]): Specific group to analyze
        num_samples (int): Number of samples for uncertainty estimation
        save_path (Optional[str]): Path to save the plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
                groups = batch.get('group', ['Unknown'] * len(volumes))
            else:
                volumes = batch.to(device)
                groups = ['Unknown'] * len(volumes)
            
            # Find sample from specified group or use first sample
            sample_idx = 0
            if group:
                for i, g in enumerate(groups):
                    if g == group:
                        sample_idx = i
                        break
            
            sample_volume = volumes[sample_idx:sample_idx+1]
            
            # Encode to get latent distribution
            mu, log_var = model.encode(sample_volume)
            
            # Sample multiple times from the latent distribution
            std = torch.exp(0.5 * log_var)
            reconstructions = []
            
            for _ in range(num_samples):
                eps = torch.randn_like(std)
                z = mu + eps * std
                recon = model.decoder(z)
                reconstructions.append(recon.cpu().numpy())
            
            # Calculate mean and std of reconstructions
            reconstructions = np.array(reconstructions)
            mean_recon = np.mean(reconstructions, axis=0)[0, 0]  # Remove batch and channel dims
            std_recon = np.std(reconstructions, axis=0)[0, 0]
            
            # Original volume
            original = sample_volume.cpu().numpy()[0, 0]
            
            # Plot results
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            d, h, w = original.shape
            axial_slice = d // 2
            coronal_slice = h // 2
            sagittal_slice = w // 2
            
            # Original
            axes[0, 0].imshow(original[axial_slice], cmap='gray')
            axes[0, 0].set_title(f'Original Axial\nGroup: {groups[sample_idx]}')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(original[:, coronal_slice, :], cmap='gray')
            axes[0, 1].set_title('Original Coronal')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(original[:, :, sagittal_slice], cmap='gray')
            axes[0, 2].set_title('Original Sagittal')
            axes[0, 2].axis('off')
            
            # Uncertainty (std of reconstructions)
            axes[1, 0].imshow(std_recon[axial_slice], cmap='hot')
            axes[1, 0].set_title('Uncertainty Axial')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(std_recon[:, coronal_slice, :], cmap='hot')
            axes[1, 1].set_title('Uncertainty Coronal')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(std_recon[:, :, sagittal_slice], cmap='hot')
            axes[1, 2].set_title('Uncertainty Sagittal')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            break  # Only process first valid sample
