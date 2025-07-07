"""
Training configuration classes for model training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """
    Configuration class for autoencoder training.
    """
    
    # Model parameters
    latent_dim: int = 256
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Optimizer parameters
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    
    # Scheduler parameters
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    model_name: str = "autoencoder"
    
    # Data parameters
    train_split: float = 0.8
    num_workers: int = 0
    pin_memory: bool = False
    
    # Loss parameters
    loss_function: str = "mse"  # "mse", "l1", "smooth_l1"
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    validate_every: int = 1
    
    # Device
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class VAEConfig:
    """
    Configuration class for VAE training.
    """
    
    # Model parameters
    latent_dim: int = 256
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Optimizer parameters
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    
    # Scheduler parameters
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    model_name: str = "vae_model"
    
    # Data parameters
    train_split: float = 0.8
    num_workers: int = 0
    pin_memory: bool = False
    
    # VAE-specific loss parameters
    beta: float = 0.0005  # Weight for KL divergence
    beta_warmup_steps: int = 5000
    free_bits: float = 3.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    validate_every: int = 1
    
    # Device
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VAEConfig':
        """Create config from dictionary."""
        return cls(**config_dict) 