import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from models.autoencoder.model import BaseAutoencoder
from models.experiment_tracker import ExperimentTracker
from utils.logger import get_logger

# Initialize logger
logger = get_logger("latent_optimization")

class LatentSpaceExperiment:
    """
    Runs experiments to optimize latent space dimensions and analyze the impact
    on reconstruction quality.
    """
    def __init__(self, base_dir="output/Experiments/LatentSpace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.trackers = {}
    
    def run_latent_size_experiment(self, train_loader, val_loader, 
                                  latent_dims=[512, 256, 128, 64, 32], 
                                  epochs=10, device=None):
        """
        Run experiments with different latent space dimensions.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            latent_dims: List of latent dimensions to test
            epochs: Number of epochs to train each model
            device: Device to use for training (cuda or cpu)
        
        Returns:
            Dictionary of results for each latent dimension
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Running latent size experiment with dimensions: {latent_dims}")
        logger.info(f"Using device: {device}")
        
        for latent_dim in latent_dims:
            logger.info(f"Training model with latent_dim={latent_dim}")
            
            # Create model with current latent dimension
            model = BaseAutoencoder(latent_dim=latent_dim)
            model.to(device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Create experiment tracker
            tracker = ExperimentTracker(f"LatentDim_{latent_dim}")
            tracker.record_model_info(model)
            
            # Train for specified epochs
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_samples = 0
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for batch in train_loader:
                    # Get data
                    volumes = batch['volume'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    reconstructed = model(volumes)
                    
                    # Calculate loss
                    loss = F.mse_loss(reconstructed, volumes)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    train_loss += loss.item() * volumes.size(0)
                    train_samples += volumes.size(0)
                
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                
                # Calculate average training loss
                avg_train_loss = train_loss / train_samples
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        volumes = batch['volume'].to(device)
                        reconstructed = model(volumes)
                        loss = F.mse_loss(reconstructed, volumes)
                        
                        val_loss += loss.item() * volumes.size(0)
                        val_samples += volumes.size(0)
                        
                        # Store reconstruction samples for the last epoch
                        if epoch == epochs - 1 and val_samples <= 3:
                            tracker.record_reconstruction_samples(
                                volumes, reconstructed, epoch
                            )
                
                # Calculate average validation loss
                avg_val_loss = val_loss / val_samples
                
                # Record epoch metrics
                tracker.record_epoch(
                    epoch, avg_train_loss, avg_val_loss, epoch_time
                )
                
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
            
            # Save experiment results
            tracker.save_experiment()
            self.trackers[latent_dim] = tracker
            
            # Store results for this latent dimension
            self.results[latent_dim] = {
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "model_params": tracker.model_info["trainable_parameters"]
            }
        
        # Save overall results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk"""
        results_file = self.base_dir / "latent_size_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary plots
        self._plot_latent_size_vs_error()
        
        logger.info(f"Saved latent size experiment results to {results_file}")
    
    def _plot_latent_size_vs_error(self):
        """Plot latent size vs. reconstruction error"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        latent_dims = sorted(list(self.results.keys()))
        train_losses = [self.results[dim]["final_train_loss"] for dim in latent_dims]
        val_losses = [self.results[dim]["final_val_loss"] for dim in latent_dims]
        param_counts = [self.results[dim]["model_params"] for dim in latent_dims]
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 6))
        
        # Plot reconstruction error vs. latent dimension
        plt.subplot(1, 2, 1)
        plt.plot(latent_dims, train_losses, 'o-', label="Training Loss")
        plt.plot(latent_dims, val_losses, 's-', label="Validation Loss")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Reconstruction Error (MSE)")
        plt.title("Reconstruction Error vs. Latent Dimension")
        plt.xscale("log", base=2)  # Log scale for x-axis
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot parameter count vs. latent dimension
        plt.subplot(1, 2, 2)
        plt.plot(latent_dims, param_counts, 'o-')
        plt.xlabel("Latent Dimension")
        plt.ylabel("Number of Parameters")
        plt.title("Model Size vs. Latent Dimension")
        plt.xscale("log", base=2)  # Log scale for x-axis
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / "latent_size_vs_error.png")
        plt.close()
    
    @staticmethod
    def analyze_latent_space_compression(model, dataloader, num_samples=100, device=None):
        """
        Analyze how information is compressed in the latent space.
        
        Args:
            model: Trained autoencoder model
            dataloader: DataLoader for input data
            num_samples: Number of samples to analyze
            device: Device to use for inference
        
        Returns:
            Dictionary of analysis results
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        model.to(device)
        
        # Collect latent vectors
        latent_vectors = []
        original_volumes = []
        reconstructed_volumes = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                volumes = batch['volume'].to(device)
                batch_labels = batch['label']
                
                # Get latent vectors and reconstructions
                latent = model.encode(volumes)
                reconstructed = model(volumes)
                
                # Store results
                latent_vectors.append(latent.cpu().numpy())
                original_volumes.append(volumes.cpu().numpy())
                reconstructed_volumes.append(reconstructed.cpu().numpy())
                labels.extend(batch_labels)
                
                # Check if we have enough samples
                if len(labels) >= num_samples:
                    break
        
        # Concatenate results
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
        original_volumes = np.concatenate(original_volumes, axis=0)[:num_samples]
        reconstructed_volumes = np.concatenate(reconstructed_volumes, axis=0)[:num_samples]
        labels = labels[:num_samples]
        
        # Analyze latent space
        latent_dim = latent_vectors.shape[1]
        latent_mean = np.mean(latent_vectors, axis=0)
        latent_std = np.std(latent_vectors, axis=0)
        
        # Calculate variance explained by each latent dimension
        latent_var = np.var(latent_vectors, axis=0)
        total_var = np.sum(latent_var)
        var_explained = latent_var / total_var
        
        # Calculate reconstruction error
        mse = np.mean((original_volumes - reconstructed_volumes) ** 2)
        
        # Create visualization of variance explained
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(latent_dim), var_explained)
        plt.xlabel("Latent Dimension")
        plt.ylabel("Variance Explained")
        plt.title("Variance Explained by Each Latent Dimension")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(np.sort(var_explained)[::-1]))
        plt.xlabel("Number of Dimensions")
        plt.ylabel("Cumulative Variance Explained")
        plt.title("Cumulative Variance Explained")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Return analysis results
        return {
            "latent_dim": latent_dim,
            "mse": mse,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "var_explained": var_explained,
            "cumulative_var": np.cumsum(np.sort(var_explained)[::-1]),
            "visualization": plt.gcf()
        }
