"""
Main training loops for autoencoder and VAE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from tqdm import tqdm
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .config import TrainingConfig, VAEConfig
from .callbacks import EarlyStopping, VAEEarlyStopping, MetricTracker
from .checkpoints import CheckpointHandler, VAECheckpointHandler
from .optimizers import create_optimizer, create_scheduler, create_vae_optimizer, create_vae_scheduler
from ..utils.memory_utils import clear_memory
from ..models.vae import VAELoss


def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                     config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """
    Train autoencoder model.
    
    Args:
        model (nn.Module): Autoencoder model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (Optional[TrainingConfig]): Training configuration
        
    Returns:
        Dict[str, Any]: Training history and metadata
    """
    if config is None:
        config = TrainingConfig()
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Setup loss function
    if config.loss_function == "mse":
        criterion = nn.MSELoss()
    elif config.loss_function == "l1":
        criterion = nn.L1Loss()
    elif config.loss_function == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    checkpoint_handler = CheckpointHandler(config.checkpoint_dir, config.model_name)
    metric_tracker = MetricTracker()
    
    # Setup mixed precision
    scaler = amp.GradScaler() if config.use_amp else None
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Starting autoencoder training for {config.epochs} epochs")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # Get data
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
            else:
                volumes = batch.to(device)
            
            optimizer.zero_grad()
            
            if config.use_amp and scaler:
                with amp.autocast():
                    outputs = model(volumes)
                    loss = criterion(outputs, volumes)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(volumes)
                loss = criterion(outputs, volumes)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            if batch_idx % config.log_interval == 0:
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{train_loss/train_batches:.6f}'
                })
            
            # Clear memory periodically
            if batch_idx % 10 == 0:
                clear_memory()
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if epoch % config.validate_every == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
                for batch in val_pbar:
                    if isinstance(batch, dict):
                        volumes = batch['volume'].to(device)
                    else:
                        volumes = batch.to(device)
                    
                    if config.use_amp:
                        with amp.autocast():
                            outputs = model(volumes)
                            loss = criterion(outputs, volumes)
                    else:
                        outputs = model(volumes)
                        loss = criterion(outputs, volumes)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
        else:
            avg_val_loss = val_losses[-1] if val_losses else float('inf')
        
        # Update learning rate
        if scheduler:
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
        
        # Update metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        metric_tracker.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time
        )
        
        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or is_best:
            checkpoint_handler.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                is_best=is_best,
                config=config.to_dict()
            )
        
        # Early stopping check
        if early_stopping(avg_val_loss, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
    
    # Save final checkpoint
    checkpoint_handler.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        is_best=False,
        config=config.to_dict()
    )
    
    # Cleanup old checkpoints
    checkpoint_handler.cleanup_old_checkpoints(keep_last=5)
    
    print("Training completed!")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': metric_tracker.get_best_epoch('val_loss'),
        'final_epoch': epoch,
        'config': config.to_dict(),
        'metrics': metric_tracker.metrics
    }


def train_vae(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
              config: Optional[VAEConfig] = None) -> Dict[str, Any]:
    """
    Train VAE model.
    
    Args:
        model (nn.Module): VAE model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (Optional[VAEConfig]): VAE training configuration
        
    Returns:
        Dict[str, Any]: Training history and metadata
    """
    if config is None:
        config = VAEConfig()
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = create_vae_optimizer(model, config)
    scheduler = create_vae_scheduler(optimizer, config)
    
    # Setup VAE loss function
    vae_loss = VAELoss(
        beta=config.beta,
        beta_warmup_steps=config.beta_warmup_steps,
        free_bits=config.free_bits
    )
    
    # Setup callbacks
    early_stopping = VAEEarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    checkpoint_handler = VAECheckpointHandler(config.checkpoint_dir, config.model_name)
    metric_tracker = MetricTracker()
    
    # Setup mixed precision
    scaler = amp.GradScaler() if config.use_amp else None
    
    # Training history
    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    print(f"Starting VAE training for {config.epochs} epochs")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Beta: {config.beta}, Beta warmup: {config.beta_warmup_steps}, Free bits: {config.free_bits}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # Get data
            if isinstance(batch, dict):
                volumes = batch['volume'].to(device)
            else:
                volumes = batch.to(device)
            
            optimizer.zero_grad()
            
            if config.use_amp and scaler:
                with amp.autocast():
                    recon_volumes, mu, log_var = model(volumes)
                    total_loss, recon_loss, kl_loss = vae_loss(recon_volumes, volumes, mu, log_var)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_volumes, mu, log_var = model(volumes)
                total_loss, recon_loss, kl_loss = vae_loss(recon_volumes, volumes, mu, log_var)
                total_loss.backward()
                optimizer.step()
            
            train_total_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_batches += 1
            
            # Update progress bar
            if batch_idx % config.log_interval == 0:
                train_pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
            
            # Clear memory periodically
            if batch_idx % 10 == 0:
                clear_memory()
        
        avg_train_total_loss = train_total_loss / train_batches
        avg_train_recon_loss = train_recon_loss / train_batches
        avg_train_kl_loss = train_kl_loss / train_batches
        
        train_losses.append(avg_train_total_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_kl_losses.append(avg_train_kl_loss)
        
        # Validation phase
        if epoch % config.validate_every == 0:
            model.eval()
            val_total_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
                for batch in val_pbar:
                    if isinstance(batch, dict):
                        volumes = batch['volume'].to(device)
                    else:
                        volumes = batch.to(device)
                    
                    if config.use_amp:
                        with amp.autocast():
                            recon_volumes, mu, log_var = model(volumes)
                            total_loss, recon_loss, kl_loss = vae_loss(recon_volumes, volumes, mu, log_var)
                    else:
                        recon_volumes, mu, log_var = model(volumes)
                        total_loss, recon_loss, kl_loss = vae_loss(recon_volumes, volumes, mu, log_var)
                    
                    val_total_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        'total': f'{total_loss.item():.4f}',
                        'recon': f'{recon_loss.item():.4f}',
                        'kl': f'{kl_loss.item():.4f}'
                    })
            
            avg_val_total_loss = val_total_loss / val_batches
            avg_val_recon_loss = val_recon_loss / val_batches
            avg_val_kl_loss = val_kl_loss / val_batches
            
            val_losses.append(avg_val_total_loss)
            val_recon_losses.append(avg_val_recon_loss)
            val_kl_losses.append(avg_val_kl_loss)
        else:
            avg_val_total_loss = val_losses[-1] if val_losses else float('inf')
            avg_val_recon_loss = val_recon_losses[-1] if val_recon_losses else 0
            avg_val_kl_loss = val_kl_losses[-1] if val_kl_losses else 0
        
        # Update learning rate
        if scheduler:
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(avg_val_total_loss)
                else:
                    scheduler.step()
        
        # Update metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        metric_tracker.update(
            epoch=epoch,
            train_loss=avg_train_total_loss,
            val_loss=avg_val_total_loss,
            train_recon_loss=avg_train_recon_loss,
            val_recon_loss=avg_val_recon_loss,
            train_kl_loss=avg_train_kl_loss,
            val_kl_loss=avg_val_kl_loss,
            learning_rate=current_lr,
            epoch_time=epoch_time
        )
        
        # Check for best model
        is_best = avg_val_total_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_total_loss
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or is_best:
            checkpoint_handler.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                train_recon_losses=train_recon_losses,
                train_kl_losses=train_kl_losses,
                val_recon_losses=val_recon_losses,
                val_kl_losses=val_kl_losses,
                is_best=is_best,
                config=config.to_dict()
            )
        
        # Early stopping check
        if early_stopping(avg_val_total_loss, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d} | "
              f"Total: {avg_train_total_loss:.4f}/{avg_val_total_loss:.4f} | "
              f"Recon: {avg_train_recon_loss:.4f}/{avg_val_recon_loss:.4f} | "
              f"KL: {avg_train_kl_loss:.4f}/{avg_val_kl_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
    
    # Save final checkpoint
    checkpoint_handler.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        train_recon_losses=train_recon_losses,
        train_kl_losses=train_kl_losses,
        val_recon_losses=val_recon_losses,
        val_kl_losses=val_kl_losses,
        is_best=False,
        config=config.to_dict()
    )
    
    # Cleanup old checkpoints
    checkpoint_handler.cleanup_old_checkpoints(keep_last=5)
    
    print("VAE training completed!")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_recon_losses': train_recon_losses,
        'train_kl_losses': train_kl_losses,
        'val_recon_losses': val_recon_losses,
        'val_kl_losses': val_kl_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': metric_tracker.get_best_epoch('val_loss'),
        'final_epoch': epoch,
        'config': config.to_dict(),
        'metrics': metric_tracker.metrics
    } 