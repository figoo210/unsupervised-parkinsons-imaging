import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from models.vae.model import VAE, VAELoss
from models.experiment_tracker import ExperimentTracker
from utils.logger import get_logger

# Initialize logger
logger = get_logger("vae_beta_experiments")

class BetaExperiment:
    """
    Runs experiments to explore different beta values in VAE and analyze
    the impact on reconstruction quality and latent space regularization.
    """
    def __init__(self, base_dir="output/Experiments/VAEBeta"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.trackers = {}
    
    def run_beta_experiment(self, train_loader, val_loader, 
                           beta_values=[0.0001, 0.001, 0.01, 0.1, 1.0],
                           latent_dim=256, epochs=10, device=None):
        """
        Run experiments with different beta values for VAE.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            beta_values: List of beta values to test
            latent_dim: Latent space dimension
            epochs: Number of epochs to train each model
            device: Device to use for training (cuda or cpu)
        
        Returns:
            Dictionary of results for each beta value
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Running VAE beta experiment with values: {beta_values}")
        logger.info(f"Using device: {device}")
        
        for beta in beta_values:
            logger.info(f"Training VAE with beta={beta}")
            
            # Create VAE model
            model = VAE(latent_dim=latent_dim)
            model.to(device)
            
            # Create VAE loss with current beta
            criterion = VAELoss(beta=beta, beta_warmup_steps=0, free_bits=0.0)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Create experiment tracker
            tracker = ExperimentTracker(f"VAE_Beta_{beta}")
            tracker.record_model_info(model)
            
            # Train for specified epochs
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_recon_loss = 0
                train_kl_loss = 0
                train_samples = 0
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for batch in train_loader:
                    # Get data
                    volumes = batch['volume'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    reconstructed, mu, log_var = model(volumes)
                    
                    # Calculate loss
                    loss, recon_loss, kl_loss, current_beta = criterion(reconstructed, volumes, mu, log_var)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    train_loss += loss.item() * volumes.size(0)
                    train_recon_loss += recon_loss.item() * volumes.size(0)
                    train_kl_loss += kl_loss.item() * volumes.size(0)
                    train_samples += volumes.size(0)
                
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                
                # Calculate average training losses
                avg_train_loss = train_loss / train_samples
                avg_train_recon_loss = train_recon_loss / train_samples
                avg_train_kl_loss = train_kl_loss / train_samples
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kl_loss = 0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        volumes = batch['volume'].to(device)
                        reconstructed, mu, log_var = model(volumes)
                        loss, recon_loss, kl_loss, _ = criterion(reconstructed, volumes, mu, log_var)
                        
                        val_loss += loss.item() * volumes.size(0)
                        val_recon_loss += recon_loss.item() * volumes.size(0)
                        val_kl_loss += kl_loss.item() * volumes.size(0)
                        val_samples += volumes.size(0)
                        
                        # Store reconstruction samples for the last epoch
                        if epoch == epochs - 1 and val_samples <= 3:
                            tracker.record_reconstruction_samples(
                                volumes, reconstructed, epoch
                            )
                
                # Calculate average validation losses
                avg_val_loss = val_loss / val_samples
                avg_val_recon_loss = val_recon_loss / val_samples
                avg_val_kl_loss = val_kl_loss / val_samples
                
                # Record epoch metrics
                tracker.record_epoch(
                    epoch, avg_train_loss, avg_val_loss, epoch_time,
                    train_recon_loss=avg_train_recon_loss, val_recon_loss=avg_val_recon_loss
                )
                
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, "
                           f"train_recon={avg_train_recon_loss:.6f}, train_kl={avg_train_kl_loss:.6f}")
            
            # Save experiment results
            tracker.save_experiment()
            self.trackers[beta] = tracker
            
            # Store results for this beta value
            self.results[beta] = {
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "final_train_recon_loss": avg_train_recon_loss,
                "final_val_recon_loss": avg_val_recon_loss,
                "final_train_kl_loss": avg_train_kl_loss,
                "final_val_kl_loss": avg_val_kl_loss
            }
        
        # Save overall results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk"""
        results_file = self.base_dir / "beta_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary plots
        self._plot_beta_vs_losses()
        
        logger.info(f"Saved beta experiment results to {results_file}")
    
    def _plot_beta_vs_losses(self):
        """Plot beta value vs. different loss components"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        beta_values = sorted(list(self.results.keys()))
        train_losses = [self.results[beta]["final_train_loss"] for beta in beta_values]
        val_losses = [self.results[beta]["final_val_loss"] for beta in beta_values]
        train_recon_losses = [self.results[beta]["final_train_recon_loss"] for beta in beta_values]
        val_recon_losses = [self.results[beta]["final_val_recon_loss"] for beta in beta_values]
        train_kl_losses = [self.results[beta]["final_train_kl_loss"] for beta in beta_values]
        val_kl_losses = [self.results[beta]["final_val_kl_loss"] for beta in beta_values]
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot total loss vs. beta
        plt.subplot(2, 2, 1)
        plt.plot(beta_values, train_losses, 'o-', label="Training Loss")
        plt.plot(beta_values, val_losses, 's-', label="Validation Loss")
        plt.xlabel("Beta Value")
        plt.ylabel("Total Loss")
        plt.title("Total Loss vs. Beta Value")
        plt.xscale("log")  # Log scale for x-axis
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot reconstruction loss vs. beta
        plt.subplot(2, 2, 2)
        plt.plot(beta_values, train_recon_losses, 'o-', label="Training Recon Loss")
        plt.plot(beta_values, val_recon_losses, 's-', label="Validation Recon Loss")
        plt.xlabel("Beta Value")
        plt.ylabel("Reconstruction Loss (MSE)")
        plt.title("Reconstruction Loss vs. Beta Value")
        plt.xscale("log")  # Log scale for x-axis
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot KL loss vs. beta
        plt.subplot(2, 2, 3)
        plt.plot(beta_values, train_kl_losses, 'o-', label="Training KL Loss")
        plt.plot(beta_values, val_kl_losses, 's-', label="Validation KL Loss")
        plt.xlabel("Beta Value")
        plt.ylabel("KL Divergence Loss")
        plt.title("KL Divergence Loss vs. Beta Value")
        plt.xscale("log")  # Log scale for x-axis
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot KL loss vs. reconstruction loss
        plt.subplot(2, 2, 4)
        plt.scatter(train_recon_losses, train_kl_losses, label="Training", marker='o')
        plt.scatter(val_recon_losses, val_kl_losses, label="Validation", marker='s')
        # Add beta value annotations
        for i, beta in enumerate(beta_values):
            plt.annotate(f"β={beta}", (train_recon_losses[i], train_kl_losses[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel("Reconstruction Loss (MSE)")
        plt.ylabel("KL Divergence Loss")
        plt.title("KL Divergence vs. Reconstruction Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.base_dir / "beta_vs_losses.png")
        plt.close()


class AdvancedVAEExperiment:
    """
    Runs experiments with advanced VAE configurations like cyclical annealing
    and free bits approach.
    """
    def __init__(self, base_dir="output/Experiments/AdvancedVAE"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.trackers = {}
    
    def run_advanced_vae_experiment(self, train_loader, val_loader, 
                                   configs=[
                                       {"name": "baseline", "beta": 0.01, "warmup": 0, "free_bits": 0.0},
                                       {"name": "cyclical", "beta": 0.01, "warmup": 1000, "free_bits": 0.0},
                                       {"name": "free_bits", "beta": 0.01, "warmup": 0, "free_bits": 3.0},
                                       {"name": "combined", "beta": 0.01, "warmup": 1000, "free_bits": 3.0}
                                   ],
                                   latent_dim=256, epochs=10, device=None):
        """
        Run experiments with different advanced VAE configurations.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            configs: List of configuration dictionaries
            latent_dim: Latent space dimension
            epochs: Number of epochs to train each model
            device: Device to use for training (cuda or cpu)
        
        Returns:
            Dictionary of results for each configuration
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Running advanced VAE experiment with {len(configs)} configurations")
        logger.info(f"Using device: {device}")
        
        for config in configs:
            config_name = config["name"]
            beta = config["beta"]
            warmup = config["warmup"]
            free_bits = config["free_bits"]
            
            logger.info(f"Training VAE with config: {config_name} (beta={beta}, warmup={warmup}, free_bits={free_bits})")
            
            # Create VAE model
            model = VAE(latent_dim=latent_dim)
            model.to(device)
            
            # Create VAE loss with current configuration
            criterion = VAELoss(beta=beta, beta_warmup_steps=warmup, free_bits=free_bits)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # Create experiment tracker
            tracker = ExperimentTracker(f"VAE_{config_name}")
            tracker.record_model_info(model)
            
            # Train for specified epochs
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_recon_loss = 0
                train_kl_loss = 0
                train_samples = 0
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for batch in train_loader:
                    # Get data
                    volumes = batch['volume'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    reconstructed, mu, log_var = model(volumes)
                    
                    # Calculate loss
                    loss, recon_loss, kl_loss, current_beta = criterion(reconstructed, volumes, mu, log_var)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    train_loss += loss.item() * volumes.size(0)
                    train_recon_loss += recon_loss.item() * volumes.size(0)
                    train_kl_loss += kl_loss.item() * volumes.size(0)
                    train_samples += volumes.size(0)
                
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                
                # Calculate average training losses
                avg_train_loss = train_loss / train_samples
                avg_train_recon_loss = train_recon_loss / train_samples
                avg_train_kl_loss = train_kl_loss / train_samples
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kl_loss = 0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        volumes = batch['volume'].to(device)
                        reconstructed, mu, log_var = model(volumes)
                        loss, recon_loss, kl_loss, _ = criterion(reconstructed, volumes, mu, log_var)
                        
                        val_loss += loss.item() * volumes.size(0)
                        val_recon_loss += recon_loss.item() * volumes.size(0)
                        val_kl_loss += kl_loss.item() * volumes.size(0)
                        val_samples += volumes.size(0)
                        
                        # Store reconstruction samples for the last epoch
                        if epoch == epochs - 1 and val_samples <= 3:
                            tracker.record_reconstruction_samples(
                                volumes, reconstructed, epoch
                            )
                
                # Calculate average validation losses
                avg_val_loss = val_loss / val_samples
                avg_val_recon_loss = val_recon_loss / val_samples
                avg_val_kl_loss = val_kl_loss / val_samples
                
                # Record epoch metrics
                tracker.record_epoch(
                    epoch, avg_train_loss, avg_val_loss, epoch_time,
                    train_recon_loss=avg_train_recon_loss, val_recon_loss=avg_val_recon_loss
                )
                
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, "
                           f"train_recon={avg_train_recon_loss:.6f}, train_kl={avg_train_kl_loss:.6f}")
            
            # Save experiment results
            tracker.save_experiment()
            self.trackers[config_name] = tracker
            
            # Store results for this configuration
            self.results[config_name] = {
                "config": config,
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "final_train_recon_loss": avg_train_recon_loss,
                "final_val_recon_loss": avg_val_recon_loss,
                "final_train_kl_loss": avg_train_kl_loss,
                "final_val_kl_loss": avg_val_kl_loss
            }
        
        # Save overall results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk"""
        results_file = self.base_dir / "advanced_vae_results.json"
        with open(results_file, 'w') as f:
            json.dump({k: {**v, "config": dict(v["config"])} for k, v in self.results.items()}, f, indent=2)
        
        # Generate summary plots
        self._plot_comparison()
        
        logger.info(f"Saved advanced VAE experiment results to {results_file}")
    
    def _plot_comparison(self):
        """Plot comparison of different VAE configurations"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        config_names = list(self.results.keys())
        train_losses = [self.results[name]["final_train_loss"] for name in config_names]
        val_losses = [self.results[name]["final_val_loss"] for name in config_names]
        train_recon_losses = [self.results[name]["final_train_recon_loss"] for name in config_names]
        val_recon_losses = [self.results[name]["final_val_recon_loss"] for name in config_names]
        train_kl_losses = [self.results[name]["final_train_kl_loss"] for name in config_names]
        val_kl_losses = [self.results[name]["final_val_kl_loss"] for name in config_names]
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot total loss comparison
        plt.subplot(2, 2, 1)
        x = np.arange(len(config_names))
        width = 0.35
        plt.bar(x - width/2, train_losses, width, label="Training Loss")
        plt.bar(x + width/2, val_losses, width, label="Validation Loss")
        plt.xlabel("Configuration")
        plt.ylabel("Total Loss")
        plt.title("Total Loss Comparison")
        plt.xticks(x, config_names)
        plt.legend()
        
        # Plot reconstruction loss comparison
        plt.subplot(2, 2, 2)
        plt.bar(x - width/2, train_recon_losses, width, label="Training Recon Loss")
        plt.bar(x + width/2, val_recon_losses, width, label="Validation Recon Loss")
        plt.xlabel("Configuration")
        plt.ylabel("Reconstruction Loss (MSE)")
        plt.title("Reconstruction Loss Comparison")
        plt.xticks(x, config_names)
        plt.legend()
        
        # Plot KL loss comparison
        plt.subplot(2, 2, 3)
        plt.bar(x - width/2, train_kl_losses, width, label="Training KL Loss")
        plt.bar(x + width/2, val_kl_losses, width, label="Validation KL Loss")
        plt.xlabel("Configuration")
        plt.ylabel("KL Divergence Loss")
        plt.title("KL Divergence Loss Comparison")
        plt.xticks(x, config_names)
        plt.legend()
        
        # Plot KL loss vs. reconstruction loss
        plt.subplot(2, 2, 4)
        plt.scatter(train_recon_losses, train_kl_losses, label="Training", marker='o')
        plt.scatter(val_recon_losses, val_kl_losses, label="Validation", marker='s')
        # Add configuration name annotations
        for i, name in enumerate(config_names):
            plt.annotate(name, (train_recon_losses[i], train_kl_losses[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel("Reconstruction Loss (MSE)")
        plt.ylabel("KL Divergence Loss")
        plt.title("KL Divergence vs. Reconstruction Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.base_dir / "advanced_vae_comparison.png")
        plt.close()
