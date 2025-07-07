#!/usr/bin/env python3
"""
Main evaluation script for trained autoencoder and VAE models.
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
import numpy as np
from torch.utils.data import DataLoader

from src.utils import configure_gpu, print_memory_stats
from src.data import create_dataloaders, OnDemandDataset
from src.models import BaseAutoencoder, VAE
from src.training import CheckpointHandler, VAECheckpointHandler
from src.analysis import (
    evaluate_model_performance,
    compute_reconstruction_error,
    evaluate_reconstruction_quality_by_group,
    find_outliers,
    extract_latent_vectors,
    visualize_latent_space,
    plot_training_history,
    plot_vae_training_history,
    visualize_reconstruction_samples,
    visualize_vae_reconstructions
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['autoencoder', 'vae'],
                       help='Type of model to evaluate')
    parser.add_argument('--config_path', type=str,
                       help='Path to training configuration file')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to CSV file containing data metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing the image data')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='all',
                       choices=['train', 'val', 'test', 'all'],
                       help='Data split to evaluate')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Analysis arguments
    parser.add_argument('--compute_metrics', action='store_true',
                       help='Compute detailed reconstruction metrics')
    parser.add_argument('--find_outliers', action='store_true',
                       help='Find outliers in the dataset')
    parser.add_argument('--outlier_method', type=str, default='reconstruction_error',
                       choices=['reconstruction_error', 'latent_space'],
                       help='Method for outlier detection')
    parser.add_argument('--outlier_threshold', type=float, default=95,
                       help='Percentile threshold for outlier detection')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--latent_analysis', action='store_true',
                       help='Perform latent space analysis')
    parser.add_argument('--latent_method', type=str, default='tsne',
                       choices=['tsne', 'pca'],
                       help='Method for latent space visualization')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def load_model_and_config(args):
    """Load trained model and configuration."""
    print(f"Loading model from: {args.checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Try to get configuration from checkpoint or separate file
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("No configuration found. Please provide --config_path")
    
    # Parse input shape and latent dimension
    if isinstance(config.get('input_shape'), list):
        input_shape = tuple(config['input_shape'])
    else:
        # Default fallback
        input_shape = (64, 128, 128)
    
    latent_dim = config.get('latent_dim', 256)
    
    # Create model
    if args.model_type == 'autoencoder':
        model = BaseAutoencoder(input_shape=input_shape, latent_dim=latent_dim)
    elif args.model_type == 'vae':
        model = VAE(input_shape=input_shape, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    print(f"Model type: {args.model_type}")
    print(f"Input shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    return model, config, checkpoint


def load_data(args, config):
    """Load and prepare datasets."""
    print("Loading data...")
    
    # Load metadata
    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df)} samples from {args.data_csv}")
    
    # Parse input shape from config
    if isinstance(config.get('input_shape'), list):
        input_shape = tuple(config['input_shape'])
    else:
        input_shape = (64, 128, 128)
    
    # Create dataloaders based on split
    if args.split == 'all':
        # Create train/val split for comprehensive evaluation
        train_loader, val_loader = create_dataloaders(
            df=df,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            target_shape=input_shape,
            num_workers=args.num_workers
        )
        return {'train': train_loader, 'val': val_loader}
    
    else:
        # Create single dataloader for specific split
        train_loader, val_loader = create_dataloaders(
            df=df,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            target_shape=input_shape,
            num_workers=args.num_workers
        )
        
        if args.split == 'train':
            return {'train': train_loader}
        elif args.split == 'val':
            return {'val': val_loader}


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure GPU
    device = configure_gpu()
    if args.device != 'auto':
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print_memory_stats()
    
    # Load model and configuration
    model, config, checkpoint = load_model_and_config(args)
    model = model.to(device)
    model.eval()
    
    # Load data
    dataloaders = load_data(args, config)
    
    # Save evaluation configuration
    eval_config = {
        'checkpoint_path': args.checkpoint_path,
        'model_type': args.model_type,
        'data_csv': args.data_csv,
        'data_dir': args.data_dir,
        'split': args.split,
        'batch_size': args.batch_size,
        'device': str(device),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / 'evaluation_config.json', 'w') as f:
        json.dump(eval_config, f, indent=2)
    
    results = {}
    
    # 1. Basic performance evaluation
    print("\n" + "="*50)
    print("BASIC PERFORMANCE EVALUATION")
    print("="*50)
    
    if args.split == 'all':
        performance = evaluate_model_performance(
            model, dataloaders['train'], dataloaders['val']
        )
    else:
        performance = {}
        for split_name, dataloader in dataloaders.items():
            print(f"Evaluating on {split_name} set...")
            performance[split_name] = compute_reconstruction_error(model, dataloader)
    
    results['performance'] = performance
    
    # Print results
    for split_name, metrics in performance.items():
        print(f"\n{split_name.upper()} SET METRICS:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  Samples: {metrics['total_samples']}")
    
    # 2. Group-wise evaluation (if group information available)
    if args.compute_metrics:
        print("\n" + "="*50)
        print("GROUP-WISE EVALUATION")
        print("="*50)
        
        for split_name, dataloader in dataloaders.items():
            print(f"Computing group-wise metrics for {split_name} set...")
            try:
                group_metrics = evaluate_reconstruction_quality_by_group(model, dataloader)
                results[f'{split_name}_group_metrics'] = group_metrics.to_dict()
                
                print(f"\nGroup-wise metrics saved for {split_name} set")
                print(group_metrics[['group', 'n_samples', 'mse_mean', 'mae_mean']].to_string(index=False))
                
                # Save detailed group metrics
                group_metrics.to_csv(output_dir / f'{split_name}_group_metrics.csv', index=False)
                
            except Exception as e:
                print(f"Could not compute group-wise metrics: {e}")
    
    # 3. Outlier detection
    if args.find_outliers:
        print("\n" + "="*50)
        print("OUTLIER DETECTION")
        print("="*50)
        
        for split_name, dataloader in dataloaders.items():
            print(f"Finding outliers in {split_name} set using {args.outlier_method}...")
            outlier_indices, scores = find_outliers(
                model, dataloader, 
                threshold_percentile=args.outlier_threshold,
                method=args.outlier_method
            )
            
            results[f'{split_name}_outliers'] = {
                'indices': outlier_indices,
                'method': args.outlier_method,
                'threshold_percentile': args.outlier_threshold,
                'num_outliers': len(outlier_indices)
            }
            
            print(f"Found {len(outlier_indices)} outliers in {split_name} set")
            print(f"Outlier rate: {len(outlier_indices)/len(dataloader.dataset)*100:.2f}%")
    
    # 4. Latent space analysis
    if args.latent_analysis:
        print("\n" + "="*50)
        print("LATENT SPACE ANALYSIS")
        print("="*50)
        
        for split_name, dataloader in dataloaders.items():
            print(f"Extracting latent vectors for {split_name} set...")
            latent_vectors, group_labels = extract_latent_vectors(model, dataloader)
            
            print(f"Latent vectors shape: {latent_vectors.shape}")
            print(f"Unique groups: {set(group_labels)}")
            
            # Save latent vectors
            np.save(output_dir / f'{split_name}_latent_vectors.npy', latent_vectors)
            with open(output_dir / f'{split_name}_group_labels.json', 'w') as f:
                json.dump(group_labels, f)
            
            # Visualize latent space if requested
            if args.visualize and len(set(group_labels)) > 1:
                print(f"Visualizing latent space for {split_name} set...")
                save_path = output_dir / f'{split_name}_latent_space_{args.latent_method}.png' if args.save_plots else None
                visualize_latent_space(latent_vectors, group_labels, method=args.latent_method, save_path=save_path)
    
    # 5. Visualization
    if args.visualize:
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Plot training history
        if 'train_losses' in checkpoint:
            print("Plotting training history...")
            if args.model_type == 'vae':
                save_path = output_dir / 'vae_training_history.png' if args.save_plots else None
                plot_vae_training_history(checkpoint, save_path=save_path)
            else:
                save_path = output_dir / 'training_history.png' if args.save_plots else None
                plot_training_history(checkpoint, save_path=save_path)
        
        # Visualize reconstructions
        for split_name, dataloader in dataloaders.items():
            print(f"Visualizing reconstructions for {split_name} set...")
            if args.model_type == 'vae':
                save_path = output_dir / f'{split_name}_vae_reconstructions.png' if args.save_plots else None
                visualize_vae_reconstructions(model, dataloader, num_samples=args.num_samples, save_path=save_path)
            else:
                save_path = output_dir / f'{split_name}_reconstructions.png' if args.save_plots else None
                visualize_reconstruction_samples(model, dataloader, num_samples=args.num_samples, save_path=save_path)
    
    # Save all results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str to handle numpy types
    
    print(f"\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)
    print(f"Results saved to: {output_dir}")
    print(f"Detailed results: {results_path}")
    
    if args.save_plots:
        print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main() 