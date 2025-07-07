"""
Optimizer and learning rate scheduler utilities for model training.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import math
from typing import Union
from .config import TrainingConfig, VAEConfig


def create_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model (torch.nn.Module): Model to optimize
        config (TrainingConfig): Training configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    elif config.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        config (TrainingConfig): Training configuration
        
    Returns:
        Learning rate scheduler or None
    """
    if config.scheduler.lower() == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )
    elif config.scheduler.lower() == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.scheduler_patience,
            gamma=config.scheduler_factor
        )
    elif config.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler_min_lr
        )
    elif config.scheduler.lower() == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.epochs // 10,  # 10% warmup
            num_training_steps=config.epochs,
            min_lr=config.scheduler_min_lr
        )
    elif config.scheduler.lower() == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    return scheduler


def create_vae_optimizer(model: torch.nn.Module, config: VAEConfig) -> torch.optim.Optimizer:
    """
    Create optimizer for VAE training based on configuration.
    
    Args:
        model (torch.nn.Module): VAE model to optimize
        config (VAEConfig): VAE training configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    elif config.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_vae_scheduler(optimizer: torch.optim.Optimizer, config: VAEConfig):
    """
    Create learning rate scheduler for VAE training based on configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        config (VAEConfig): VAE training configuration
        
    Returns:
        Learning rate scheduler or None
    """
    if config.scheduler.lower() == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )
    elif config.scheduler.lower() == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.scheduler_patience,
            gamma=config.scheduler_factor
        )
    elif config.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler_min_lr
        )
    elif config.scheduler.lower() == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.epochs // 10,  # 10% warmup
            num_training_steps=config.epochs,
            min_lr=config.scheduler_min_lr
        )
    elif config.scheduler.lower() == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    return scheduler


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, 
                                   num_training_steps: int, min_lr: float = 0.0, 
                                   last_epoch: int = -1):
    """
    Create a learning rate scheduler that linearly increases the learning rate from 0 to the 
    initial lr set in the optimizer during the warmup period, and then decreases it following 
    a cosine annealing schedule.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate
        num_warmup_steps (int): The number of steps for the warmup phase
        num_training_steps (int): The total number of training steps
        min_lr (float): Minimum learning rate
        last_epoch (int): The index of last epoch
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Ensure minimum learning rate
        return max(min_lr / optimizer.param_groups[0]['lr'], cosine_lr)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, 
                                   num_training_steps: int, last_epoch: int = -1):
    """
    Create a learning rate scheduler that linearly increases the learning rate from 0 to the 
    initial lr set in the optimizer during the warmup period, and then linearly decreases it 
    to 0.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate
        num_warmup_steps (int): The number of steps for the warmup phase
        num_training_steps (int): The total number of training steps
        last_epoch (int): The index of last epoch
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int,
                                             num_training_steps: int, lr_end: float = 1e-7,
                                             power: float = 1.0, last_epoch: int = -1):
    """
    Create a learning rate scheduler that linearly increases the learning rate from 0 to the 
    initial lr set in the optimizer during the warmup period, and then decreases it following 
    a polynomial decay.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate
        num_warmup_steps (int): The number of steps for the warmup phase
        num_training_steps (int): The total number of training steps
        lr_end (float): The end learning rate
        power (float): Power of the polynomial
        last_epoch (int): The index of last epoch
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler
    """
    lr_init = optimizer.param_groups[0]['lr']
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch) 