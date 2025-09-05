import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F


def plot_training_history(metadata):
    """Plot training and validation loss history"""
    plt.figure(figsize=(12, 5))

    train_losses = metadata["train_losses"]
    val_losses = metadata["val_losses"]

    # Plot full history
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Full Training History')
    plt.legend()
    plt.grid(True)

    # Plot recent history (last 30 epochs or all if < 30)
    plt.subplot(1, 2, 2)
    recent = min(30, len(train_losses))
    if recent > 5:  # Only plot recent history if we have enough epochs
        plt.plot(train_losses[-recent:], label='Train Loss')
        plt.plot(val_losses[-recent:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Last {recent} Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_reconstruction_samples(model, dataloader, num_samples=3):
    """Visualize original vs reconstructed volumes for samples from the dataset using anatomically relevant slices"""
    device = next(model.parameters()).device

    # Get samples from dataloader
    samples = []
    labels = []

    for batch in dataloader:
        volumes = batch['volume']
        batch_labels = batch['label']

        for i in range(min(len(volumes), num_samples - len(samples))):
            samples.append(volumes[i:i+1])
            labels.append(batch_labels[i])

        if len(samples) >= num_samples:
            break

    # Visualize each sample
    with torch.no_grad():
        for idx, (sample, label) in enumerate(zip(samples, labels)):
            # Get original volume
            orig_vol = sample.to(device)

            # Generate reconstruction
            reconstructed = model(orig_vol)

            # Move to CPU for visualization
            orig_vol = orig_vol.cpu().squeeze().numpy()
            recon_vol = reconstructed.cpu().squeeze().numpy()

            # Create figure for this sample
            fig = plt.figure(figsize=(16, 12))
            plt.suptitle(f"Sample {idx+1} - Group: {label}", fontsize=16)

            # Define anatomically relevant slices
            axial_slice = 32      # Axial view - slice 32
            coronal_slice = 50    # Coronal view - slice 50
            sagittal_slice1 = 55  # Sagittal view - slice 55
            sagittal_slice2 = 70  # Sagittal view - slice 70

            # Plot original slices - top row
            plt.subplot(2, 4, 1)
            plt.imshow(orig_vol[axial_slice], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Original - Axial (z={axial_slice})")
            plt.axis('off')
            
            plt.subplot(2, 4, 2)
            plt.imshow(orig_vol[:, coronal_slice, :], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Original - Coronal (y={coronal_slice})")
            plt.axis('off')
            
            plt.subplot(2, 4, 3)
            plt.imshow(orig_vol[:, :, sagittal_slice1], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Original - Sagittal (x={sagittal_slice1})")
            plt.axis('off')
            
            plt.subplot(2, 4, 4)
            plt.imshow(orig_vol[:, :, sagittal_slice2], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Original - Sagittal (x={sagittal_slice2})")
            plt.axis('off')
            
            # Plot reconstructed slices - bottom row
            plt.subplot(2, 4, 5)
            plt.imshow(recon_vol[axial_slice], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Reconstructed - Axial (z={axial_slice})")
            plt.axis('off')
            
            plt.subplot(2, 4, 6)
            plt.imshow(recon_vol[:, coronal_slice, :], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Reconstructed - Coronal (y={coronal_slice})")
            plt.axis('off')
            
            plt.subplot(2, 4, 7)
            plt.imshow(recon_vol[:, :, sagittal_slice1], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Reconstructed - Sagittal (x={sagittal_slice1})")
            plt.axis('off')
            
            plt.subplot(2, 4, 8)
            plt.imshow(recon_vol[:, :, sagittal_slice2], cmap='gray', vmin=0, vmax=3)
            plt.title(f"Reconstructed - Sagittal (x={sagittal_slice2})")
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def visualize_latent_space(latent_vectors, labels, method='tsne'):
    """Visualize latent space using t-SNE or PCA"""
    plt.figure(figsize=(10, 8))

    # Create label-to-color mapping for consistent colors
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        print("Computing t-SNE projection...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors) - 1))
        title = 't-SNE Visualization of Latent Space'
    else:
        print("Computing PCA projection...")
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Latent Space'

    # Apply reduction
    reduced_vecs = reducer.fit_transform(latent_vectors)

    # Create scatter plot
    for label in unique_labels:
        # Get indices where this label appears
        indices = [i for i, l in enumerate(labels) if l == label]

        # Plot these points
        plt.scatter(
            reduced_vecs[indices, 0],
            reduced_vecs[indices, 1],
            label=label,
            color=label_to_color[label],
            alpha=0.7,
            edgecolor='w',
            s=100
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title="Group", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return reduced_vecs


def plot_latent_dimension_activation(latent_vectors, labels):
    """Analyze activation patterns of latent dimensions"""
    # Create a DataFrame with latent dimensions and labels
    import pandas as pd

    # First, convert labels to categorical for better plotting
    unique_labels = list(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    label_indices = [label_map[label] for label in labels]

    # Create DataFrames
    latent_df = pd.DataFrame(latent_vectors)
    latent_df['label'] = labels

    # Compute mean activation by group
    mean_activations = {}
    for label in unique_labels:
        group_vectors = latent_vectors[np.array(labels) == label]
        mean_activations[label] = np.mean(group_vectors, axis=0)

    # Identify top discriminative dimensions
    activation_matrix = np.vstack([mean_activations[label] for label in unique_labels])
    variance = np.var(activation_matrix, axis=0)
    top_dims = np.argsort(variance)[-10:]  # Top 10 dimensions

    # Plot mean activation for top dimensions
    plt.figure(figsize=(14, 6))

    # Plot heatmap
    plt.subplot(1, 2, 1)
    heatmap_data = pd.DataFrame({
        f"Dim {i}": [mean_activations[label][i] for label in unique_labels]
        for i in top_dims
    })
    heatmap_data.index = unique_labels

    sns.heatmap(heatmap_data, cmap='coolwarm', center=0,
               annot=True, fmt=".2f", cbar_kws={'label': 'Mean Activation'})
    plt.title("Mean Activation of Top Discriminative Dimensions")

    # Plot box plots for top 5 dimensions
    plt.subplot(1, 2, 2)

    # Create data for boxplot
    plot_data = []
    labels_for_plot = []
    positions = []

    for i, dim in enumerate(top_dims[:5]):  # Top 5 for clarity
        for j, label in enumerate(unique_labels):
            group_values = latent_vectors[np.array(labels) == label, dim]
            plot_data.append(group_values)
            labels_for_plot.append(f"{label}")
            positions.append(i + j * 0.25)

    # Create boxplot
    boxplot = plt.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.15)

    # Customize boxplot colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels_for_plot) if l == label]
        for idx in indices:
            boxplot['boxes'][idx].set_facecolor(colors[i])

    # Add labels and ticks
    plt.xticks([i + (len(unique_labels) - 1) * 0.125 for i in range(5)],
              [f"Dim {d}" for d in top_dims[:5]])
    plt.title("Distribution of Top 5 Discriminative Dimensions")
    plt.ylabel("Activation Value")

    # Add legend
    for i, label in enumerate(unique_labels):
        plt.plot([], [], 'o', color=colors[i], label=label)
    plt.legend(title="Group")

    plt.tight_layout()
    plt.show()

    return top_dims


def visualize_outliers(model, outliers, threshold=0.01):
    """Visualize the top outliers with highest reconstruction error using anatomically relevant slices"""
    device = next(model.parameters()).device

    # Sort outliers by error (descending)
    sorted_outliers = sorted(outliers.items(), key=lambda x: x[1]['error'], reverse=True)

    # Visualize top outliers
    num_outliers = min(5, len(sorted_outliers))

    for i in range(num_outliers):
        path, info = sorted_outliers[i]
        error = info['error']
        label = info['label']
        z_score = info['z_score']

        try:
            # Load the original volume
            with torch.no_grad():
                # Load DICOM file
                original_volume, _ = load_dicom(path)
                original_volume = original_volume[9:73, :, :]

                # Process volume
                norm_vol, _, _ = process_volume(original_volume, target_shape=(64, 128, 128))

                # Convert to tensor and add batch dimension
                vol_tensor = torch.from_numpy(np.expand_dims(norm_vol, axis=(0, 1))).float().to(device)

                # Get reconstruction
                reconstructed = model(vol_tensor)

                # Move tensors to CPU and remove batch and channel dimensions
                vol_np = vol_tensor.cpu().squeeze().numpy()
                recon_np = reconstructed.cpu().squeeze().numpy()

                # Create figure
                fig = plt.figure(figsize=(16, 12))
                plt.suptitle(f"Outlier {i+1}: {path.split('/')[-1]}\nGroup: {label}, Error: {error:.6f}, Z-score: {z_score:.2f}", fontsize=14)

                # Define anatomically relevant slices
                axial_slice = 32      # Axial view - slice 32
                coronal_slice = 50    # Coronal view - slice 50
                sagittal_slice1 = 55  # Sagittal view - slice 55
                sagittal_slice2 = 70  # Sagittal view - slice 70
                
                # Plot original slices - top row
                plt.subplot(2, 4, 1)
                plt.imshow(vol_np[axial_slice], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Original - Axial (z={axial_slice})")
                plt.axis('off')
                
                plt.subplot(2, 4, 2)
                plt.imshow(vol_np[:, coronal_slice, :], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Original - Coronal (y={coronal_slice})")
                plt.axis('off')
                
                plt.subplot(2, 4, 3)
                plt.imshow(vol_np[:, :, sagittal_slice1], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Original - Sagittal (x={sagittal_slice1})")
                plt.axis('off')
                
                plt.subplot(2, 4, 4)
                plt.imshow(vol_np[:, :, sagittal_slice2], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Original - Sagittal (x={sagittal_slice2})")
                plt.axis('off')
                
                # Plot reconstructed slices - bottom row
                plt.subplot(2, 4, 5)
                plt.imshow(recon_np[axial_slice], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Reconstructed - Axial (z={axial_slice})")
                plt.axis('off')
                
                plt.subplot(2, 4, 6)
                plt.imshow(recon_np[:, coronal_slice, :], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Reconstructed - Coronal (y={coronal_slice})")
                plt.axis('off')
                
                plt.subplot(2, 4, 7)
                plt.imshow(recon_np[:, :, sagittal_slice1], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Reconstructed - Sagittal (x={sagittal_slice1})")
                plt.axis('off')
                
                plt.subplot(2, 4, 8)
                plt.imshow(recon_np[:, :, sagittal_slice2], cmap='gray', vmin=0, vmax=3)
                plt.title(f"Reconstructed - Sagittal (x={sagittal_slice2})")
                plt.axis('off')

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error visualizing outlier {path}: {e}")


def visualize_latent_dimension(model, dataloader, dimension_idx, alpha=5.0, group=None):
    """
    Visualize what a specific latent dimension represents by modifying it
    and observing the effect on brain reconstruction using anatomically relevant slices.
    
    Ensures consistent scales across all views for better comparison.

    Parameters:
        model: Trained autoencoder model
        dataloader: DataLoader containing samples
        dimension_idx: The latent dimension to manipulate (e.g., 231)
        alpha: Strength of the dimension manipulation
        group: Optional filter for specific patient group (e.g., 'PD', 'Control')
    """
    device = next(model.parameters()).device
    model.eval()

    # Find a suitable sample (optionally from specific group)
    for batch in dataloader:
        volumes = batch['volume']
        labels = batch['label']
        paths = batch['path']

        if group is not None:
            # Find samples from the specified group
            group_indices = [i for i, label in enumerate(labels) if label == group]
            if not group_indices:
                continue
            # Use the first matching sample
            idx = group_indices[0]
            sample = volumes[idx:idx+1].to(device)
            sample_label = labels[idx]
            sample_path = paths[idx]
        else:
            # Just use the first sample
            sample = volumes[0:1].to(device)
            sample_label = labels[0]
            sample_path = paths[0]

        break  # Exit after finding a sample

    with torch.no_grad():
        # Encode the sample to get its latent representation
        z = model.encode(sample)

        # Create modified latent vectors
        z_plus = z.clone()
        z_minus = z.clone()

        # Modify the specific dimension
        z_plus[0, dimension_idx] += alpha
        z_minus[0, dimension_idx] -= alpha

        # Decode the original and modified latent vectors
        original_reconstruction = model.decode(z)
        plus_reconstruction = model.decode(z_plus)
        minus_reconstruction = model.decode(z_minus)

        # Move tensors to CPU and convert to numpy for visualization
        original_vol = original_reconstruction.cpu().squeeze().numpy()
        plus_vol = plus_reconstruction.cpu().squeeze().numpy()
        minus_vol = minus_reconstruction.cpu().squeeze().numpy()

        # Calculate the difference maps
        plus_diff = plus_vol - original_vol
        minus_diff = minus_vol - original_vol

        # Set up the figure
        fig = plt.figure(figsize=(16, 15))
        plt.suptitle(f"Visualization of Dimension {dimension_idx} in Brain Space\nPatient Group: {sample_label}", fontsize=16)

        # Define anatomically relevant slices
        slices = {
            'axial': 32,       # Axial z=32
            'coronal': 50,     # Coronal y=50
            'sagittal1': 55,   # Sagittal x=55
            'sagittal2': 70    # Sagittal x=70
        }

        # Create a custom colormap for difference maps
        diff_cmap = plt.cm.RdBu_r  # Red-Blue colormap with red for negative, blue for positive
        
        # Determine consistent scales for brain intensity and difference maps
        brain_vmin, brain_vmax = 0, 3  # Standard scale for brain images
        
        # Find global min/max for difference maps to use consistent scale
        diff_min = min(np.min(plus_diff), np.min(minus_diff))
        diff_max = max(np.max(plus_diff), np.max(minus_diff))
        # Make the range symmetric around zero for better visualization
        diff_abs_max = max(abs(diff_min), abs(diff_max))
        diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

        # Row 1: Original reconstruction - 4 views
        plt.subplot(5, 4, 1)
        plt.imshow(original_vol[slices['axial']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Original (Axial z={slices['axial']})")
        plt.axis('off')

        plt.subplot(5, 4, 2)
        plt.imshow(original_vol[:, slices['coronal'], :], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Original (Coronal y={slices['coronal']})")
        plt.axis('off')

        plt.subplot(5, 4, 3)
        plt.imshow(original_vol[:, :, slices['sagittal1']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Original (Sagittal x={slices['sagittal1']})")
        plt.axis('off')

        plt.subplot(5, 4, 4)
        plt.imshow(original_vol[:, :, slices['sagittal2']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Original (Sagittal x={slices['sagittal2']})")
        plt.axis('off')

        # Row 2: Increased dimension - 4 views
        plt.subplot(5, 4, 5)
        plt.imshow(plus_vol[slices['axial']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} + {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 6)
        plt.imshow(plus_vol[:, slices['coronal'], :], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} + {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 7)
        plt.imshow(plus_vol[:, :, slices['sagittal1']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} + {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 8)
        plt.imshow(plus_vol[:, :, slices['sagittal2']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} + {alpha}")
        plt.axis('off')

        # Row 3: Decreased dimension - 4 views
        plt.subplot(5, 4, 9)
        plt.imshow(minus_vol[slices['axial']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} - {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 10)
        plt.imshow(minus_vol[:, slices['coronal'], :], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} - {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 11)
        plt.imshow(minus_vol[:, :, slices['sagittal1']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} - {alpha}")
        plt.axis('off')

        plt.subplot(5, 4, 12)
        plt.imshow(minus_vol[:, :, slices['sagittal2']], cmap='gray', vmin=brain_vmin, vmax=brain_vmax)
        plt.title(f"Dim {dimension_idx} - {alpha}")
        plt.axis('off')

        # Row 4: Difference map (increased - original) - 4 views
        plt.subplot(5, 4, 13)
        im1 = plt.imshow(plus_diff[slices['axial']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (+)")
        plt.axis('off')

        plt.subplot(5, 4, 14)
        plt.imshow(plus_diff[:, slices['coronal'], :], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (+)")
        plt.axis('off')

        plt.subplot(5, 4, 15)
        plt.imshow(plus_diff[:, :, slices['sagittal1']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (+)")
        plt.axis('off')

        plt.subplot(5, 4, 16)
        plt.imshow(plus_diff[:, :, slices['sagittal2']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (+)")
        plt.axis('off')

        # Row 5: Difference map (decreased - original) - 4 views
        plt.subplot(5, 4, 17)
        im2 = plt.imshow(minus_diff[slices['axial']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (-)")
        plt.axis('off')

        plt.subplot(5, 4, 18)
        plt.imshow(minus_diff[:, slices['coronal'], :], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (-)")
        plt.axis('off')

        plt.subplot(5, 4, 19)
        plt.imshow(minus_diff[:, :, slices['sagittal1']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (-)")
        plt.axis('off')

        plt.subplot(5, 4, 20)
        plt.imshow(minus_diff[:, :, slices['sagittal2']], cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)
        plt.title(f"Difference (-)")
        plt.axis('off')

        # Add colorbar for difference maps - now showing the global range
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.3])
        cbar = plt.colorbar(im1, cax=cbar_ax)
        cbar.set_label(f'Difference Intensity (Range: {diff_vmin:.3f} to {diff_vmax:.3f})')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.9)
        plt.show()

        # Print information about scales used
        print(f"Brain Intensity Scale: {brain_vmin} to {brain_vmax}")
        print(f"Difference Map Scale: {diff_vmin:.3f} to {diff_vmax:.3f}")

        # Return both the sample info and reconstructions for further analysis
        return {
            'label': sample_label,
            'path': sample_path,
            'original': original_vol,
            'plus': plus_vol,
            'minus': minus_vol,
            'plus_diff': plus_diff,
            'minus_diff': minus_diff,
            'diff_min': diff_vmin,
            'diff_max': diff_vmax
        }


def generate_feature_importance_map(model, dataloader, dimension_idx, group=None, num_samples=5):
    """
    Generate a more robust feature importance map for a specific dimension
    by aggregating effects across multiple samples using anatomically relevant slices.
    Uses consistent scales across all views for better comparison.

    Parameters:
        model: Trained autoencoder model
        dataloader: DataLoader containing samples
        dimension_idx: The latent dimension to analyze
        group: Optional filter for specific patient group
        num_samples: Number of samples to aggregate
    """
    device = next(model.parameters()).device
    model.eval()

    # Storage for aggregated results
    aggregated_plus_diff = None
    aggregated_minus_diff = None
    sample_count = 0

    # Find samples (optionally from specific group)
    for batch in dataloader:
        volumes = batch['volume']
        labels = batch['label']

        if group is not None:
            # Find samples from the specified group
            group_indices = [i for i, label in enumerate(labels) if label == group]
            indices = group_indices
        else:
            # Use all samples in batch
            indices = range(len(volumes))

        for idx in indices:
            if sample_count >= num_samples:
                break

            sample = volumes[idx:idx+1].to(device)

            with torch.no_grad():
                # Encode the sample
                z = model.encode(sample)

                # Create modified latent vectors
                z_plus = z.clone()
                z_minus = z.clone()

                # Modify the specific dimension
                z_plus[0, dimension_idx] += 5.0
                z_minus[0, dimension_idx] -= 5.0

                # Decode the vectors
                original_reconstruction = model.decode(z)
                plus_reconstruction = model.decode(z_plus)
                minus_reconstruction = model.decode(z_minus)

                # Calculate the difference maps
                plus_diff = (plus_reconstruction - original_reconstruction).cpu().squeeze().numpy()
                minus_diff = (minus_reconstruction - original_reconstruction).cpu().squeeze().numpy()

                # Aggregate the difference maps
                if aggregated_plus_diff is None:
                    aggregated_plus_diff = plus_diff
                    aggregated_minus_diff = minus_diff
                else:
                    aggregated_plus_diff += plus_diff
                    aggregated_minus_diff += minus_diff

                sample_count += 1

        if sample_count >= num_samples:
            break

    # Average the difference maps
    aggregated_plus_diff /= sample_count
    aggregated_minus_diff /= sample_count

    # Compute absolute importance map (average of plus and minus effects)
    importance_map = (np.abs(aggregated_plus_diff) + np.abs(aggregated_minus_diff)) / 2

    # Define anatomically relevant slices
    axial_slice = 32      # Axial view - slice 32
    coronal_slice = 50    # Coronal view - slice 50
    sagittal_slice1 = 55  # Sagittal view - slice 55
    sagittal_slice2 = 70  # Sagittal view - slice 70
    
    # Determine global max value for importance map for consistent scale
    importance_max = np.max(importance_map)
    
    # Determine global min/max for activation maps for consistent scale
    activation_min = min(np.min(aggregated_plus_diff), np.min(aggregated_minus_diff))
    activation_max = max(np.max(aggregated_plus_diff), np.max(aggregated_minus_diff))
    # Make the activation scale symmetric around zero
    activation_abs_max = max(abs(activation_min), abs(activation_max))
    activation_vmin, activation_vmax = -activation_abs_max, activation_abs_max

    # Visualize the importance map
    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(f"Feature Importance Map for Dimension {dimension_idx}" +
                (f" (Group: {group})" if group else ""), fontsize=16)

    # Plot axial, coronal, and sagittal views for importance map
    plt.subplot(2, 4, 1)
    im1 = plt.imshow(importance_map[axial_slice], cmap='hot', vmin=0, vmax=importance_max)
    plt.title(f"Axial (z={axial_slice})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(importance_map[:, coronal_slice, :], cmap='hot', vmin=0, vmax=importance_max)
    plt.title(f"Coronal (y={coronal_slice})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(importance_map[:, :, sagittal_slice1], cmap='hot', vmin=0, vmax=importance_max)
    plt.title(f"Sagittal (x={sagittal_slice1})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(importance_map[:, :, sagittal_slice2], cmap='hot', vmin=0, vmax=importance_max)
    plt.title(f"Sagittal (x={sagittal_slice2})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Plot activating direction (positive difference) with consistent scale
    plt.subplot(2, 4, 5)
    im2 = plt.imshow(aggregated_plus_diff[axial_slice], cmap='bwr', vmin=activation_vmin, vmax=activation_vmax)
    plt.title(f"Activating Direction - Axial")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(aggregated_plus_diff[:, coronal_slice, :], cmap='bwr', vmin=activation_vmin, vmax=activation_vmax)
    plt.title(f"Activating Direction - Coronal")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(aggregated_plus_diff[:, :, sagittal_slice1], cmap='bwr', vmin=activation_vmin, vmax=activation_vmax)
    plt.title(f"Activating Direction - Sagittal 1")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(aggregated_plus_diff[:, :, sagittal_slice2], cmap='bwr', vmin=activation_vmin, vmax=activation_vmax)
    plt.title(f"Activating Direction - Sagittal 2")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print information about scales used
    print(f"Importance Map Scale: 0 to {importance_max:.3f}")
    print(f"Activation Map Scale: {activation_vmin:.3f} to {activation_vmax:.3f}")

    return importance_map, aggregated_plus_diff, aggregated_minus_diff



