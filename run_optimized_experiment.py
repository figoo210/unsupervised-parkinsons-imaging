import os
import torch
import argparse
from pathlib import Path
from experiment_runner import run_architecture_experiment, prepare_data, set_seed
from utils.logger import get_logger

# Initialize logger
logger = get_logger("optimized_experiment")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run optimized autoencoder architecture experiments")
    parser.add_argument("--data_dir", type=str, default="data/Images",
                       help="Directory containing the image data")
    parser.add_argument("--mask_path", type=str, default="data/masks/rmask_ICV.nii",
                       help="Path to brain mask file")
    parser.add_argument("--output_dir", type=str, default="output/Experiments/Optimized",
                       help="Directory to save experiment results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs for training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for training (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--variants", type=str, nargs='+', 
                       default=["optimized", "grouped_latent"],
                       help="Model variants to test")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run architecture experiment with the specified variants
    run_architecture_experiment(
        train_loader, val_loader, args.output_dir, args.epochs, device
    )
    
    logger.info(f"Experiment completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
