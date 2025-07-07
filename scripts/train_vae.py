#!/usr/bin/env python3
"""
Main training script for Variational Autoencoder (VAE) models.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.utils import configure_gpu, print_memory_stats
from src.data import create_dataloaders, OnDemandDataset
from src.models import VAE
from src.training import VAEConfig, train_vae


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VAE Model')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to CSV file containing data metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing the image data')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension size')
    parser.add_argument('--input_shape', type=str, default='64,128,128',
                       help='Input volume shape (D,H,W)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # VAE-specific arguments
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for KL divergence weighting')
    parser.add_argument('--beta_warmup_steps', type=int, default=10,
                       help='Number of epochs for beta warmup')
    parser.add_argument('--free_bits', type=float, default=0.0,
                       help='Free bits for KL divergence')
    parser.add_argument('--reconstruction_loss', type=str, default='mse',
                       choices=['mse', 'l1', 'bce'],
                       help='Reconstruction loss function')
    
    # Training configuration
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int, default=1,
                       help='Validate every N epochs')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--model_name', type=str, default='vae',
                       help='Model name for saving')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (if None, uses timestamp)')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directory and configuration."""
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.model_name}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "samples").mkdir(exist_ok=True)  # For generated samples
    
    return experiment_dir, experiment_name


def load_data(args):
    """Load and prepare datasets."""
    print("Loading data...")
    
    # Load metadata
    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df)} samples from {args.data_csv}")
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        df=df,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        target_shape=input_shape,
        num_workers=args.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader, input_shape


def create_model(args, input_shape):
    """Create and initialize VAE model."""
    print("Creating VAE model...")
    
    model = VAE(
        input_shape=input_shape,
        latent_dim=args.latent_dim
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: {input_shape}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta: {args.beta}")
    print(f"Beta warmup steps: {args.beta_warmup_steps}")
    print(f"Free bits: {args.free_bits}")
    
    return model


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup experiment
    experiment_dir, experiment_name = setup_experiment(args)
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {experiment_dir}")
    
    # Configure GPU
    device = configure_gpu()
    if args.device != 'auto':
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print_memory_stats()
    
    # Load data
    train_loader, val_loader, input_shape = load_data(args)
    
    # Create model
    model = create_model(args, input_shape)
    
    # Create VAE training configuration
    config = VAEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=str(device),
        use_amp=args.use_amp,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        free_bits=args.free_bits,
        reconstruction_loss=args.reconstruction_loss,
        early_stopping_patience=args.early_stopping_patience,
        save_every=args.save_every,
        validate_every=args.validate_every,
        checkpoint_dir=str(experiment_dir / "checkpoints"),
        model_name=args.model_name
    )
    
    # Save configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save command line arguments
    args_path = experiment_dir / "args.json"
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Starting VAE training...")
    print(f"Configuration saved to: {config_path}")
    
    # Train model
    try:
        training_history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Save training history
        history_path = experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"VAE training completed successfully!")
        print(f"Training history saved to: {history_path}")
        print(f"Best validation loss: {training_history['best_val_loss']:.6f}")
        print(f"Best epoch: {training_history['best_epoch']}")
        
        # Print final loss breakdown
        final_recon_loss = training_history['val_recon_losses'][-1] if training_history['val_recon_losses'] else 0
        final_kl_loss = training_history['val_kl_losses'][-1] if training_history['val_kl_losses'] else 0
        print(f"Final validation reconstruction loss: {final_recon_loss:.6f}")
        print(f"Final validation KL loss: {final_kl_loss:.6f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 