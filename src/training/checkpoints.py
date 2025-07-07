"""
Model checkpointing utilities for saving and loading training states.
"""

import os
import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ..utils.memory_utils import clear_memory


class CheckpointHandler:
    """
    Handles saving and loading of model checkpoints for autoencoder training.
    """
    
    def __init__(self, checkpoint_dir: str, model_name: str):
        """
        Initialize CheckpointHandler.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            model_name (str): Name of the model for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, model, optimizer, scheduler, epoch: int, train_losses: list, 
             val_losses: list, is_best: bool = False, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch (int): Current epoch
            train_losses (list): Training loss history
            val_losses (list): Validation loss history
            is_best (bool): Whether this is the best model so far
            **kwargs: Additional metadata to save
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'timestamp': time.time(),
            'model_name': self.model_name,
            **kwargs
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save metadata as JSON
        metadata = {
            'epoch': epoch,
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
            'timestamp': time.time(),
            'is_best': is_best,
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        
        metadata_path = self.checkpoint_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        clear_memory()
        
    def load(self, model, optimizer=None, scheduler=None, device=None, 
             checkpoint_type: str = "best") -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load model on
            checkpoint_type (str): Type of checkpoint to load ("best", "latest", or epoch number)
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine checkpoint path
        if checkpoint_type == "best":
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        elif checkpoint_type == "latest":
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        else:
            # Assume it's an epoch number
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{checkpoint_type}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob(f"{self.model_name}_*.pth"):
            if "epoch" in path.name:
                epoch = int(path.name.split("epoch_")[1].split(".pth")[0])
                checkpoints.append(epoch)
        return sorted(checkpoints)
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last (int): Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            to_remove = checkpoints[:-keep_last]
            for epoch in to_remove:
                checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: epoch {epoch}")


class VAECheckpointHandler:
    """
    Specialized checkpoint handler for VAE models with additional VAE-specific metadata.
    """
    
    def __init__(self, checkpoint_dir: str, model_name: str):
        """
        Initialize VAE CheckpointHandler.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            model_name (str): Name of the model for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, model, optimizer, scheduler, epoch: int, train_losses: list, 
             val_losses: list, train_recon_losses: list, train_kl_losses: list,
             val_recon_losses: list, val_kl_losses: list, is_best: bool = False, **kwargs):
        """
        Save VAE model checkpoint with reconstruction and KL loss tracking.
        
        Args:
            model: VAE model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch (int): Current epoch
            train_losses (list): Training total loss history
            val_losses (list): Validation total loss history
            train_recon_losses (list): Training reconstruction loss history
            train_kl_losses (list): Training KL loss history
            val_recon_losses (list): Validation reconstruction loss history
            val_kl_losses (list): Validation KL loss history
            is_best (bool): Whether this is the best model so far
            **kwargs: Additional metadata to save
        """
        # Prepare VAE checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_recon_losses': train_recon_losses,
            'train_kl_losses': train_kl_losses,
            'val_recon_losses': val_recon_losses,
            'val_kl_losses': val_kl_losses,
            'timestamp': time.time(),
            'model_name': self.model_name,
            'latent_dim': model.latent_dim,
            **kwargs
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Best VAE model saved at epoch {epoch}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save detailed metadata as JSON
        metadata = {
            'epoch': epoch,
            'train_total_loss': train_losses[-1] if train_losses else None,
            'val_total_loss': val_losses[-1] if val_losses else None,
            'train_recon_loss': train_recon_losses[-1] if train_recon_losses else None,
            'train_kl_loss': train_kl_losses[-1] if train_kl_losses else None,
            'val_recon_loss': val_recon_losses[-1] if val_recon_losses else None,
            'val_kl_loss': val_kl_losses[-1] if val_kl_losses else None,
            'timestamp': time.time(),
            'is_best': is_best,
            'latent_dim': model.latent_dim,
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        
        metadata_path = self.checkpoint_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"VAE checkpoint saved: {checkpoint_path}")
        clear_memory()
        
    def load(self, model, optimizer=None, scheduler=None, device=None, 
             checkpoint_type: str = "best") -> Dict[str, Any]:
        """
        Load VAE model checkpoint.
        
        Args:
            model: VAE model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load model on
            checkpoint_type (str): Type of checkpoint to load ("best", "latest", or epoch number)
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine checkpoint path
        if checkpoint_type == "best":
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        elif checkpoint_type == "latest":
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        else:
            # Assume it's an epoch number
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{checkpoint_type}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"VAE checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available VAE checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob(f"{self.model_name}_*.pth"):
            if "epoch" in path.name:
                epoch = int(path.name.split("epoch_")[1].split(".pth")[0])
                checkpoints.append(epoch)
        return sorted(checkpoints)
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Remove old VAE checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last (int): Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            to_remove = checkpoints[:-keep_last]
            for epoch in to_remove:
                checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old VAE checkpoint: epoch {epoch}") 