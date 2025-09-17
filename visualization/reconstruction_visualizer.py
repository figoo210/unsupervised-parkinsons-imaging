import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from utils.logger import get_logger

# Initialize logger
logger = get_logger("reconstruction_visualizer")

class ReconstructionVisualizer:
    """
    Visualizes original and reconstructed 3D volumes with various
    visualization methods to compare reconstruction quality.
    """
    def __init__(self, output_dir="output/Visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_slice_comparison(self, original, reconstructed, 
                                  slice_indices=None, views=None,
                                  title="Original vs Reconstructed", 
                                  save_path=None):
        """
        Visualize comparison of original and reconstructed slices.
        
        Args:
            original: Original volume (batch, 1, D, H, W) or (1, D, H, W) or (D, H, W)
            reconstructed: Reconstructed volume (same shape as original)
            slice_indices: Dictionary of slice indices for each view, e.g., 
                          {'axial': 32, 'coronal': 64, 'sagittal': 64}
            views: List of views to visualize, e.g., ['axial', 'coronal', 'sagittal']
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if tensors
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Ensure 5D shape (batch, channel, D, H, W)
        if original.ndim == 3:  # (D, H, W)
            original = original[np.newaxis, np.newaxis, ...]
        elif original.ndim == 4:  # (1, D, H, W) or (batch, D, H, W)
            if original.shape[0] == 1:  # (1, D, H, W)
                original = original[np.newaxis, ...]
            else:  # (batch, D, H, W)
                original = original[:, np.newaxis, ...]
        
        if reconstructed.ndim == 3:  # (D, H, W)
            reconstructed = reconstructed[np.newaxis, np.newaxis, ...]
        elif reconstructed.ndim == 4:  # (1, D, H, W) or (batch, D, H, W)
            if reconstructed.shape[0] == 1:  # (1, D, H, W)
                reconstructed = reconstructed[np.newaxis, ...]
            else:  # (batch, D, H, W)
                reconstructed = reconstructed[:, np.newaxis, ...]
        
        # Default slice indices if not provided
        if slice_indices is None:
            D, H, W = original.shape[2:]
            slice_indices = {
                'axial': D // 2,
                'coronal': H // 2,
                'sagittal': W // 2
            }
        
        # Default views if not provided
        if views is None:
            views = ['axial', 'coronal', 'sagittal']
        
        # Create figure
        batch_size = original.shape[0]
        num_views = len(views)
        fig, axes = plt.subplots(batch_size, num_views * 3, figsize=(num_views * 6, batch_size * 3))
        
        # Make axes 2D if batch_size is 1
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        # Visualize each sample in the batch
        for b in range(batch_size):
            orig = original[b, 0]  # (D, H, W)
            recon = reconstructed[b, 0]  # (D, H, W)
            
            # Calculate error map
            error = np.abs(orig - recon)
            
            # Visualize each view
            for v, view in enumerate(views):
                if view == 'axial':
                    slice_idx = slice_indices['axial']
                    orig_slice = orig[slice_idx]
                    recon_slice = recon[slice_idx]
                    error_slice = error[slice_idx]
                    view_title = f"Axial (z={slice_idx})"
                elif view == 'coronal':
                    slice_idx = slice_indices['coronal']
                    orig_slice = orig[:, slice_idx, :]
                    recon_slice = recon[:, slice_idx, :]
                    error_slice = error[:, slice_idx, :]
                    view_title = f"Coronal (y={slice_idx})"
                elif view == 'sagittal':
                    slice_idx = slice_indices['sagittal']
                    orig_slice = orig[:, :, slice_idx]
                    recon_slice = recon[:, :, slice_idx]
                    error_slice = error[:, :, slice_idx]
                    view_title = f"Sagittal (x={slice_idx})"
                
                # Plot original slice
                ax = axes[b, v * 3]
                im = ax.imshow(orig_slice, cmap='gray')
                ax.set_title(f"Original\n{view_title}")
                ax.axis('off')
                
                # Plot reconstructed slice
                ax = axes[b, v * 3 + 1]
                im = ax.imshow(recon_slice, cmap='gray')
                ax.set_title(f"Reconstructed\n{view_title}")
                ax.axis('off')
                
                # Plot error map
                ax = axes[b, v * 3 + 2]
                im = ax.imshow(error_slice, cmap='hot')
                ax.set_title(f"Error\n{view_title}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_multiple_reconstructions(self, originals, reconstructions, 
                                          model_names, slice_indices=None,
                                          view='axial', title="Model Comparison",
                                          save_path=None):
        """
        Visualize comparison of reconstructions from multiple models.
        
        Args:
            originals: Original volumes (batch, 1, D, H, W) or (batch, D, H, W)
            reconstructions: List of reconstructed volumes, each with same shape as originals
            model_names: List of model names for reconstructions
            slice_indices: Dictionary of slice indices for each view
            view: View to visualize ('axial', 'coronal', or 'sagittal')
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if tensors
        if isinstance(originals, torch.Tensor):
            originals = originals.detach().cpu().numpy()
        
        reconstructions_np = []
        for recon in reconstructions:
            if isinstance(recon, torch.Tensor):
                recon = recon.detach().cpu().numpy()
            reconstructions_np.append(recon)
        
        # Ensure 5D shape (batch, channel, D, H, W) for originals
        if originals.ndim == 4:  # (batch, D, H, W)
            originals = originals[:, np.newaxis, ...]
        
        # Ensure 5D shape for reconstructions
        for i in range(len(reconstructions_np)):
            if reconstructions_np[i].ndim == 4:  # (batch, D, H, W)
                reconstructions_np[i] = reconstructions_np[i][:, np.newaxis, ...]
        
        # Default slice indices if not provided
        if slice_indices is None:
            D, H, W = originals.shape[2:]
            slice_indices = {
                'axial': D // 2,
                'coronal': H // 2,
                'sagittal': W // 2
            }
        
        # Create figure
        batch_size = originals.shape[0]
        num_models = len(reconstructions_np)
        fig, axes = plt.subplots(batch_size, num_models + 1, figsize=((num_models + 1) * 3, batch_size * 3))
        
        # Make axes 2D if batch_size is 1
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        # Visualize each sample in the batch
        for b in range(batch_size):
            orig = originals[b, 0]  # (D, H, W)
            
            # Get slice based on view
            if view == 'axial':
                slice_idx = slice_indices['axial']
                orig_slice = orig[slice_idx]
                view_title = f"Axial (z={slice_idx})"
            elif view == 'coronal':
                slice_idx = slice_indices['coronal']
                orig_slice = orig[:, slice_idx, :]
                view_title = f"Coronal (y={slice_idx})"
            elif view == 'sagittal':
                slice_idx = slice_indices['sagittal']
                orig_slice = orig[:, :, slice_idx]
                view_title = f"Sagittal (x={slice_idx})"
            
            # Plot original slice
            ax = axes[b, 0]
            ax.imshow(orig_slice, cmap='gray')
            ax.set_title(f"Original\n{view_title}")
            ax.axis('off')
            
            # Plot reconstructed slices from each model
            for m, (recon, model_name) in enumerate(zip(reconstructions_np, model_names)):
                recon_vol = recon[b, 0]  # (D, H, W)
                
                # Get slice based on view
                if view == 'axial':
                    recon_slice = recon_vol[slice_idx]
                elif view == 'coronal':
                    recon_slice = recon_vol[:, slice_idx, :]
                elif view == 'sagittal':
                    recon_slice = recon_vol[:, :, slice_idx]
                
                # Plot reconstructed slice
                ax = axes[b, m + 1]
                ax.imshow(recon_slice, cmap='gray')
                ax.set_title(f"{model_name}")
                ax.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_latent_traversal(self, model, latent_dim_indices, 
                                  num_steps=10, range_scale=3.0,
                                  view='axial', slice_idx=None,
                                  title="Latent Space Traversal",
                                  save_path=None):
        """
        Visualize the effect of traversing specific dimensions in the latent space.
        
        Args:
            model: VAE or Autoencoder model with encode and decode methods
            latent_dim_indices: List of latent dimensions to traverse
            num_steps: Number of steps for traversal
            range_scale: Scale factor for traversal range (in std deviations)
            view: View to visualize ('axial', 'coronal', or 'sagittal')
            slice_idx: Slice index for the specified view
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Default slice index if not provided
        if slice_idx is None:
            if view == 'axial':
                slice_idx = 32  # Default axial slice
            elif view == 'coronal':
                slice_idx = 64  # Default coronal slice
            elif view == 'sagittal':
                slice_idx = 64  # Default sagittal slice
        
        # Create figure
        num_dims = len(latent_dim_indices)
        fig, axes = plt.subplots(num_dims, num_steps, figsize=(num_steps * 1.5, num_dims * 2))
        
        # Make axes 2D if num_dims is 1
        if num_dims == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            # Create base latent vector (zeros)
            z = torch.zeros(1, model.latent_dim, device=device)
            
            # Create traversal values
            traversal_values = torch.linspace(-range_scale, range_scale, num_steps)
            
            # Traverse each specified dimension
            for d, dim_idx in enumerate(latent_dim_indices):
                for s, val in enumerate(traversal_values):
                    # Set the value for the current dimension
                    z_modified = z.clone()
                    z_modified[0, dim_idx] = val
                    
                    # Generate image from latent vector
                    if hasattr(model, 'decode'):
                        generated = model.decode(z_modified)
                    else:
                        generated = model.decoder(z_modified)
                    
                    # Convert to numpy
                    generated = generated.cpu().numpy()[0, 0]  # (D, H, W)
                    
                    # Get slice based on view
                    if view == 'axial':
                        gen_slice = generated[slice_idx]
                    elif view == 'coronal':
                        gen_slice = generated[:, slice_idx, :]
                    elif view == 'sagittal':
                        gen_slice = generated[:, :, slice_idx]
                    
                    # Plot generated slice
                    ax = axes[d, s]
                    ax.imshow(gen_slice, cmap='gray')
                    
                    # Add value as title only for the top row
                    if d == 0:
                        ax.set_title(f"{val.item():.1f}")
                    
                    # Add dimension index only for the leftmost column
                    if s == 0:
                        ax.set_ylabel(f"Dim {dim_idx}")
                    
                    ax.axis('off')
        
        plt.suptitle(f"{title}\n{view.capitalize()} view, slice {slice_idx}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_error_distribution(self, original, reconstructed, title="Error Distribution", save_path=None):
        """
        Visualize the distribution of reconstruction errors.
        
        Args:
            original: Original volume (batch, 1, D, H, W) or (batch, D, H, W)
            reconstructed: Reconstructed volume (same shape as original)
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if tensors
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Ensure 5D shape (batch, channel, D, H, W)
        if original.ndim == 3:  # (D, H, W)
            original = original[np.newaxis, np.newaxis, ...]
        elif original.ndim == 4:  # (1, D, H, W) or (batch, D, H, W)
            if original.shape[0] == 1:  # (1, D, H, W)
                original = original[np.newaxis, ...]
            else:  # (batch, D, H, W)
                original = original[:, np.newaxis, ...]
        
        if reconstructed.ndim == 3:  # (D, H, W)
            reconstructed = reconstructed[np.newaxis, np.newaxis, ...]
        elif reconstructed.ndim == 4:  # (1, D, H, W) or (batch, D, H, W)
            if reconstructed.shape[0] == 1:  # (1, D, H, W)
                reconstructed = reconstructed[np.newaxis, ...]
            else:  # (batch, D, H, W)
                reconstructed = reconstructed[:, np.newaxis, ...]
        
        # Calculate error
        error = original - reconstructed
        squared_error = error ** 2
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot error histogram
        axes[0].hist(error.flatten(), bins=50, alpha=0.7)
        axes[0].set_title("Error Distribution")
        axes[0].set_xlabel("Error")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)
        
        # Plot squared error histogram
        axes[1].hist(squared_error.flatten(), bins=50, alpha=0.7)
        axes[1].set_title("Squared Error Distribution")
        axes[1].set_xlabel("Squared Error")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)
        
        # Plot log squared error histogram
        log_squared_error = np.log10(squared_error.flatten() + 1e-10)
        axes[2].hist(log_squared_error, bins=50, alpha=0.7)
        axes[2].set_title("Log Squared Error Distribution")
        axes[2].set_xlabel("Log10 Squared Error")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        # Calculate and return error statistics
        stats = {
            "mean_error": float(np.mean(error)),
            "std_error": float(np.std(error)),
            "mean_squared_error": float(np.mean(squared_error)),
            "rmse": float(np.sqrt(np.mean(squared_error))),
            "min_error": float(np.min(error)),
            "max_error": float(np.max(error)),
            "median_error": float(np.median(error))
        }
        
        return fig, stats
