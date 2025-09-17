import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

from data.data_ingestion import collect_files, generate_dataframe
from data.dataloader import create_dataloaders
from models.autoencoder.direct_ae import DirectAutoencoder
from models.autoencoder.model_variants import get_model_variant
from models.vae.model import VAE, VAELoss
from models.experiment_tracker import ExperimentTracker
from models.latent_optimization import LatentSpaceExperiment
from models.vae.beta_experiments import BetaExperiment, AdvancedVAEExperiment
from visualization.reconstruction_visualizer import ReconstructionVisualizer
from utils.logger import get_logger

# Initialize logger
logger = get_logger("experiment_runner")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run autoencoder architecture experiments")
    parser.add_argument("--experiment", type=str, default="architecture", 
                       choices=["architecture", "latent", "beta", "advanced_vae"],
                       help="Type of experiment to run")
    parser.add_argument("--data_dir", type=str, default="data/Images",
                       help="Directory containing the image data")
    parser.add_argument("--mask_path", type=str, default="data/masks/rmask_ICV.nii",
                       help="Path to brain mask file")
    parser.add_argument("--output_dir", type=str, default="output/Experiments",
                       help="Directory to save experiment results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs for training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for training (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(data_dir, mask_path, batch_size):
    """Prepare data for experiments"""
    logger.info("Preparing data...")
    
    # Collect files
    included_files, excluded_files = collect_files(data_dir)
    
    # Generate dataframe
    df = generate_dataframe(included_files)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        df, batch_size=batch_size, train_split=0.8, 
        on_demand=True, mask_path=mask_path
    )
    
    logger.info(f"Data preparation complete. Train: {len(train_loader.dataset)}, "
               f"Val: {len(val_loader.dataset)}")
    
    return train_loader, val_loader

def run_architecture_experiment(train_loader, val_loader, output_dir, epochs, device):
    """Run architecture comparison experiment"""
    logger.info("Running architecture comparison experiment...")
    
    # Define model variants to test
    variants = ["direct", "light", "grouped", "efficient"]
    
    # Create output directory
    output_dir = Path(output_dir) / "Architecture"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    trackers = {}
    
    # Train and evaluate each model variant
    for variant in variants:
        logger.info(f"Training {variant} model...")
        
        # Create model
        model = get_model_variant(variant)
        model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create experiment tracker
        tracker = ExperimentTracker(f"AE_{variant}")
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
        trackers[variant] = tracker
        
        # Store results for this variant
        results[variant] = {
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss,
            "model_params": tracker.model_info["trainable_parameters"],
            "epoch_time": np.mean(tracker.training_history["epoch_times"])
        }
    
    # Save overall results
    results_file = output_dir / "architecture_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate comparison plots
    _plot_architecture_comparison(results, output_dir)
    
    # Generate reconstruction comparisons
    _generate_reconstruction_comparisons(trackers, output_dir)
    
    logger.info(f"Architecture experiment complete. Results saved to {output_dir}")
    
    return results

def _plot_architecture_comparison(results, output_dir):
    """Plot comparison of different architectures"""
    variants = list(results.keys())
    val_losses = [results[v]["final_val_loss"] for v in variants]
    param_counts = [results[v]["model_params"] for v in variants]
    epoch_times = [results[v]["epoch_time"] for v in variants]
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 5))
    
    # Plot validation loss comparison
    plt.subplot(1, 3, 1)
    plt.bar(variants, val_losses)
    plt.xlabel("Architecture")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("Validation Loss Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot parameter count comparison
    plt.subplot(1, 3, 2)
    plt.bar(variants, param_counts)
    plt.xlabel("Architecture")
    plt.ylabel("Number of Parameters")
    plt.title("Model Size Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot epoch time comparison
    plt.subplot(1, 3, 3)
    plt.bar(variants, epoch_times)
    plt.xlabel("Architecture")
    plt.ylabel("Epoch Time (seconds)")
    plt.title("Training Time Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png")
    plt.close()

def _generate_reconstruction_comparisons(trackers, output_dir):
    """Generate reconstruction comparisons for different architectures"""
    variants = list(trackers.keys())
    
    # Get reconstruction samples from each tracker
    originals = []
    reconstructions = []
    
    for variant in variants:
        tracker = trackers[variant]
        # Get the last epoch's reconstruction samples
        last_epoch = max(tracker.reconstruction_samples.keys())
        samples = tracker.reconstruction_samples[last_epoch]
        
        originals.append(samples["original"])
        reconstructions.append(samples["reconstructed"])
    
    # Use the first tracker's originals for comparison
    original = originals[0]
    
    # Create visualizer
    visualizer = ReconstructionVisualizer(output_dir=output_dir)
    
    # Generate comparison visualization
    visualizer.visualize_multiple_reconstructions(
        original, reconstructions, variants,
        title="Architecture Comparison",
        save_path=output_dir / "reconstruction_comparison.png"
    )
    
    # Generate detailed visualizations for each variant
    for i, variant in enumerate(variants):
        visualizer.visualize_slice_comparison(
            original, reconstructions[i],
            title=f"{variant} Reconstruction",
            save_path=output_dir / f"{variant}_reconstruction.png"
        )

def run_experiments(args):
    """Run the specified experiment"""
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare data
    train_loader, val_loader = prepare_data(args.data_dir, args.mask_path, args.batch_size)
    
    # Run the specified experiment
    if args.experiment == "architecture":
        run_architecture_experiment(
            train_loader, val_loader, args.output_dir, args.epochs, device
        )
    elif args.experiment == "latent":
        latent_exp = LatentSpaceExperiment(base_dir=Path(args.output_dir) / "LatentSpace")
        latent_exp.run_latent_size_experiment(
            train_loader, val_loader, 
            latent_dims=[512, 256, 128, 64, 32],
            epochs=args.epochs, device=device
        )
    elif args.experiment == "beta":
        beta_exp = BetaExperiment(base_dir=Path(args.output_dir) / "VAEBeta")
        beta_exp.run_beta_experiment(
            train_loader, val_loader,
            beta_values=[0.0001, 0.001, 0.01, 0.1, 1.0],
            epochs=args.epochs, device=device
        )
    elif args.experiment == "advanced_vae":
        adv_exp = AdvancedVAEExperiment(base_dir=Path(args.output_dir) / "AdvancedVAE")
        adv_exp.run_advanced_vae_experiment(
            train_loader, val_loader,
            configs=[
                {"name": "baseline", "beta": 0.01, "warmup": 0, "free_bits": 0.0},
                {"name": "cyclical", "beta": 0.01, "warmup": 1000, "free_bits": 0.0},
                {"name": "free_bits", "beta": 0.01, "warmup": 0, "free_bits": 3.0},
                {"name": "combined", "beta": 0.01, "warmup": 1000, "free_bits": 3.0}
            ],
            epochs=args.epochs, device=device
        )

if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
