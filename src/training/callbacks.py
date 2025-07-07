"""
Training callbacks for early stopping and other training utilities.
"""

import torch
from typing import Optional


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0, verbose: bool = True):
        """
        Initialize EarlyStopping.
        
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch number
            
        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Epoch {epoch}: Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epoch {epoch}: Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {epoch} epochs")
                    
        return self.early_stop


class VAEEarlyStopping:
    """
    Early stopping callback specifically designed for VAE training.
    Monitors total loss instead of just reconstruction loss.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0, verbose: bool = True):
        """
        Initialize VAE EarlyStopping.
        
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Current validation total loss
            epoch (int): Current epoch number
            
        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Epoch {epoch}: VAE validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epoch {epoch}: VAE validation loss did not improve. Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"VAE early stopping triggered after {epoch} epochs")
                    
        return self.early_stop


class LearningRateSchedulerCallback:
    """
    Callback for learning rate scheduling during training.
    """
    
    def __init__(self, scheduler, metric_name: str = "val_loss", verbose: bool = True):
        """
        Initialize LR Scheduler Callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
            metric_name (str): Name of the metric to monitor
            verbose (bool): Whether to print messages
        """
        self.scheduler = scheduler
        self.metric_name = metric_name
        self.verbose = verbose
        
    def __call__(self, metrics: dict, epoch: int):
        """
        Update learning rate based on metrics.
        
        Args:
            metrics (dict): Dictionary of training metrics
            epoch (int): Current epoch number
        """
        if hasattr(self.scheduler, 'step'):
            if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                # ReduceLROnPlateau needs a metric
                metric_value = metrics.get(self.metric_name)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
                    if self.verbose:
                        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                        print(f"Epoch {epoch}: Learning rate: {current_lr:.2e}")
            else:
                # Other schedulers just need step()
                self.scheduler.step()
                if self.verbose:
                    current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}: Learning rate: {current_lr:.2e}")


class MetricTracker:
    """
    Callback to track and store training metrics.
    """
    
    def __init__(self):
        """Initialize MetricTracker."""
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
    def update(self, epoch: int, **kwargs):
        """
        Update metrics for the current epoch.
        
        Args:
            epoch (int): Current epoch number
            **kwargs: Metric values to update
        """
        for metric_name, value in kwargs.items():
            if metric_name in self.metrics:
                # Ensure we have enough entries
                while len(self.metrics[metric_name]) <= epoch:
                    self.metrics[metric_name].append(None)
                self.metrics[metric_name][epoch] = value
                
    def get_best_epoch(self, metric_name: str = 'val_loss', minimize: bool = True) -> int:
        """
        Get the epoch with the best metric value.
        
        Args:
            metric_name (str): Name of the metric
            minimize (bool): Whether lower values are better
            
        Returns:
            int: Best epoch number
        """
        if metric_name not in self.metrics:
            return 0
            
        values = [v for v in self.metrics[metric_name] if v is not None]
        if not values:
            return 0
            
        if minimize:
            best_value = min(values)
        else:
            best_value = max(values)
            
        return self.metrics[metric_name].index(best_value)
        
    def get_metric_history(self, metric_name: str) -> list:
        """
        Get the history of a specific metric.
        
        Args:
            metric_name (str): Name of the metric
            
        Returns:
            list: List of metric values
        """
        return [v for v in self.metrics.get(metric_name, []) if v is not None] 