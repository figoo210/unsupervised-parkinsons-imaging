#!/usr/bin/env python3
"""
Main training script for autoencoder models.
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
from src.models import BaseAutoencoder
from src.training import TrainingConfig, train_autoencoder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Autoencoder Model')
    
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
    parser.add_argument('--loss_function', type=str, default='mse',
                       choices=['mse', 'l1', 'smooth_l1'],
                       help='Loss function')
    
    # Training configuration
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int, default=1,
                       help='Validate every N epochs')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--model_name', type=str, default='autoencoder',
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
    """Create and initialize model."""
    print("Creating model...")
    
    model = BaseAutoencoder(
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
    
    # Create training configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        loss_function=args.loss_function,
        device=str(device),
        use_amp=args.use_amp,
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
    
    print("Starting training...")
    print(f"Configuration saved to: {config_path}")
    
    # Train model
    try:
        training_history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Save training history
        history_path = experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"Training completed successfully!")
        print(f"Training history saved to: {history_path}")
        print(f"Best validation loss: {training_history['best_val_loss']:.6f}")
        print(f"Best epoch: {training_history['best_epoch']}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 