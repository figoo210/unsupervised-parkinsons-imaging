"""
Training utilities for the medical image analysis project.
"""

from .config import TrainingConfig, VAEConfig
from .callbacks import (
    EarlyStopping, 
    VAEEarlyStopping, 
    LearningRateSchedulerCallback, 
    MetricTracker
)
from .checkpoints import CheckpointHandler, VAECheckpointHandler  
from .optimizers import (
    create_optimizer, 
    create_scheduler, 
    create_vae_optimizer, 
    create_vae_scheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from .trainer import train_autoencoder, train_vae

__all__ = [
    # Configuration
    'TrainingConfig',
    'VAEConfig',
    
    # Callbacks
    'EarlyStopping',
    'VAEEarlyStopping',
    'LearningRateSchedulerCallback',
    'MetricTracker',
    
    # Checkpoints
    'CheckpointHandler',
    'VAECheckpointHandler',
    
    # Optimizers and Schedulers
    'create_optimizer',
    'create_scheduler',
    'create_vae_optimizer', 
    'create_vae_scheduler',
    'get_cosine_schedule_with_warmup',
    'get_linear_schedule_with_warmup',
    'get_polynomial_decay_schedule_with_warmup',
    
    # Training Loops
    'train_autoencoder',
    'train_vae'
]
