import os
import json
import time
from pathlib import Path
import torch
import torch.optim as optim
import torch.cuda.amp as amp

class TrainingConfig:
    """Training configuration optimized for NVIDIA 4070Ti"""
    def __init__(self, **kwargs):
        # Model parameters
        self.latent_dim = kwargs.get('latent_dim', 256)
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.batch_size = kwargs.get('batch_size', 4)
        self.accumulation_steps = kwargs.get('accumulation_steps', 4)
        self.epochs = kwargs.get('epochs', 100)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Optimization
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        
        # Dataloader parameters
        self.num_workers = kwargs.get('num_workers', 2)
        self.pin_memory = kwargs.get('pin_memory', True)
        
        # Checkpoint parameters
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.model_name = kwargs.get('model_name', 'autoencoder')
        self.save_interval = kwargs.get('save_interval', 5)
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Print configuration summary
        print(f"\n{'='*50}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*50}")
        print(f"Model: {self.model_name} with latent dim {self.latent_dim}")
        print(f"Batch size: {self.batch_size} × {self.accumulation_steps} steps = {self.batch_size * self.accumulation_steps} effective")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        print(f"Epochs: {self.epochs} with patience {self.early_stopping_patience}")
        print(f"Dataloader workers: {self.num_workers}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*50}\n")
        
    @classmethod
    def quick_test(cls, latent_dim=256):
        """Quick testing configuration with minimal epochs and small batch size
        
        Args:
            latent_dim: Dimension of the latent space
            
        Returns:
            TrainingConfig: Configuration optimized for quick testing
        """
        return cls(
            latent_dim=latent_dim,
            batch_size=2,
            accumulation_steps=2,  # Effective batch size = 4
            learning_rate=1e-4,
            epochs=5,
            early_stopping_patience=2,
            use_mixed_precision=True,
            num_workers=2,
            model_name=f"autoencoder_test_ld{latent_dim}"
        )
    
    @classmethod
    def full_training(cls, latent_dim=256):
        """Full training configuration with optimal parameters for production model
        
        Args:
            latent_dim: Dimension of the latent space
            
        Returns:
            TrainingConfig: Configuration optimized for full training
        """
        return cls(
            latent_dim=latent_dim,
            batch_size=8,
            accumulation_steps=8,  # Effective batch size = 64
            learning_rate=1e-4,
            epochs=200,
            early_stopping_patience=15,
            use_mixed_precision=True,
            num_workers=4,
            model_name=f"autoencoder_full_ld{latent_dim}"
        )
        
    @classmethod
    def medium_training(cls, latent_dim=256):
        """Medium training configuration balancing speed and quality
        
        Args:
            latent_dim: Dimension of the latent space
            
        Returns:
            TrainingConfig: Configuration optimized for medium training
        """
        return cls(
            latent_dim=latent_dim,
            batch_size=4,
            accumulation_steps=4,  # Effective batch size = 16
            learning_rate=1e-4,
            epochs=50,
            early_stopping_patience=10,
            use_mixed_precision=True,
            num_workers=2,
            model_name=f"autoencoder_medium_ld{latent_dim}"
        )


class EarlyStopping:
    """Early stopping handler with patience"""
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        print(f"DEBUG: EarlyStopping - Current val_loss: {val_loss:.6f}, Best loss: {self.best_loss:.6f}")
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                improvement = self.best_loss - val_loss
                print(f"Validation loss improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            print(f"DEBUG: EarlyStopping - Loss improved, counter reset to {self.counter}")
            return True  # Model improved
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            print(f"DEBUG: EarlyStopping - No improvement, counter increased to {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best epoch was {self.best_epoch}.")
                print(f"DEBUG: EarlyStopping - Patience exceeded, early_stop set to {self.early_stop}")
            return False  # Model didn't improve


class CheckpointHandler:
    """Handles saving and loading of model checkpoints"""
    def __init__(self, checkpoint_dir, model_name):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.checkpoint_path = self.checkpoint_dir / f"{model_name}_checkpoint.pth"
        self.best_model_path = self.checkpoint_dir / f"{model_name}_best.pth"
        self.metadata_path = self.checkpoint_dir / f"{model_name}_metadata.json"
        
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, train_losses, val_losses, is_best=False):
        """Save model checkpoint and training metadata"""
        try:
            print(f"DEBUG: CheckpointHandler - Saving checkpoint for epoch {epoch}")
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            # Always save latest checkpoint
            torch.save(checkpoint, self.checkpoint_path)
            print(f"DEBUG: CheckpointHandler - Saved checkpoint to {self.checkpoint_path}")
            
            # Save best model separately if this is the best model
            if is_best:
                torch.save(model.state_dict(), self.best_model_path)
                print(f"Saved best model to {self.best_model_path}")

            # Save metadata
            metadata = {
                'last_epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"DEBUG: CheckpointHandler - Saved metadata to {self.metadata_path}")
            return True
        except Exception as e:
            print(f"DEBUG: CheckpointHandler - Error saving checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load(self, model, optimizer=None, scheduler=None, device=None):
        """Load model checkpoint and return training metadata"""
        try:
            print(f"DEBUG: CheckpointHandler - Attempting to load checkpoint from {self.checkpoint_path}")
            if not self.checkpoint_path.exists():
                print(f"No checkpoint found at {self.checkpoint_path}")
                return None

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            # Load checkpoint
            print(f"DEBUG: CheckpointHandler - Loading checkpoint file")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            
            # Load model state
            print(f"DEBUG: CheckpointHandler - Loading model state")
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optionally load optimizer and scheduler states
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                print(f"DEBUG: CheckpointHandler - Loading optimizer state")
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
                print(f"DEBUG: CheckpointHandler - Loading scheduler state")
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            return {
                'epoch': checkpoint['epoch'],
                'train_losses': checkpoint['train_losses'],
                'val_losses': checkpoint['val_losses']
            }
        except Exception as e:
            print(f"DEBUG: CheckpointHandler - Error loading checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

