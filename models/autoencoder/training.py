import os
import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from models.autoencoder.config import TrainingConfig, EarlyStopping, CheckpointHandler
from models.autoencoder.scheduler import create_scheduler
from models.autoencoder.optimizer import create_optimizer
from utils.config_helpers import print_memory_stats

# Progress bar display options:
# If you're getting duplicate progress bar output or too much output,
# you can pass disable_epoch_bars=True when calling train_autoencoder() 
# to only show the total progress bar and not the per-epoch bars.


def train_autoencoder(model, train_loader, val_loader, config=None, disable_epoch_bars=True):
    """Optimized training loop with GPU memory management and progress tracking
    
    Args:
        model: The autoencoder model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        disable_epoch_bars: If True, only show the total progress bar, not per-epoch bars
    """
    if config is None:
        config = TrainingConfig()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize components
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    checkpoint_handler = CheckpointHandler(config.checkpoint_dir, config.model_name)

    # Mixed precision setup
    scaler = amp.GradScaler('cuda', enabled=config.use_mixed_precision)

    # Training tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    # Load checkpoint if available
    start_epoch = 0
    checkpoint_data = checkpoint_handler.load(model, optimizer, scheduler, device)
    if checkpoint_data:
        start_epoch = checkpoint_data['epoch'] + 1
        train_losses = checkpoint_data['train_losses']
        val_losses = checkpoint_data['val_losses']
        print(f"Resuming training from epoch {start_epoch}")

    # Calculate total steps for progress tracking
    total_steps = len(train_loader) * config.epochs

    # Training loop
    try:
        print("DEBUG: Starting training loop")
        # Create progress bar for total training - single line display
        total_pbar = tqdm(total=total_steps, desc="Total Progress", position=0, leave=True, dynamic_ncols=True)
        total_pbar.update(start_epoch * len(train_loader))
        
        for epoch in range(start_epoch, config.epochs):
            try:
                print(f"DEBUG: Starting epoch {epoch+1}/{config.epochs}")
                # Training phase
                model.train()
                epoch_loss = 0
                optimizer.zero_grad()  # Zero gradients at epoch start

                # Update total progress bar to show current phase
                total_pbar.set_description(f"Progress [E{epoch+1}/{config.epochs}|Train]")
                
                # Compact epoch progress bar - can be disabled if too verbose
                train_pbar = tqdm(train_loader, 
                                desc=f'E{epoch+1}/{config.epochs}|Train',
                                leave=False, 
                                position=1,
                                dynamic_ncols=True,
                                mininterval=1.0,  # Update at most once per second
                                disable=disable_epoch_bars)

                for batch_idx, batch in enumerate(train_pbar):
                    try:
                        # Move data to device
                        volumes = batch['volume'].to(device, non_blocking=True)
                        
                        # Mixed precision forward pass
                        with amp.autocast('cuda', enabled=config.use_mixed_precision):
                            reconstructed = model(volumes)
                            loss = criterion(reconstructed, volumes)
                            # Scale loss by accumulation steps
                            loss = loss / config.accumulation_steps

                        # Mixed precision backward pass
                        scaler.scale(loss).backward()

                        # Gradient accumulation
                        if (batch_idx + 1) % config.accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                            # Clip gradients to prevent exploding gradients
                            if config.gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                            
                            # Update weights
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                        # Track loss (using the non-scaled loss for reporting)
                        batch_loss = loss.item() * config.accumulation_steps
                        epoch_loss += batch_loss
                        
                        # Update progress bars with concise format - only update display when needed
                        if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:  # Update less frequently
                            train_pbar.set_postfix(loss=f"{batch_loss:.6f}")
                        total_pbar.update(1)

                        # Memory cleanup
                        del volumes, reconstructed, loss
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"\nOOM in batch {batch_idx}. Cleaning up...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            # Skip this batch and continue
                            continue
                        print(f"DEBUG: Error in batch {batch_idx}: {str(e)}")
                        raise e

                # Close training progress bar
                train_pbar.close()
                
                # Calculate average training loss
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation phase
                model.eval()
                val_loss = 0
                
                # Update total progress bar to show current phase
                total_pbar.set_description(f"Progress [E{epoch+1}/{config.epochs}|Val]")
                
                # Compact validation progress bar - can be disabled if too verbose
                val_pbar = tqdm(val_loader, 
                              desc=f'E{epoch+1}/{config.epochs}|Val',
                              leave=False,
                              position=1,
                              dynamic_ncols=True,
                              mininterval=1.0,  # Update at most once per second
                              disable=disable_epoch_bars)

                with torch.no_grad():
                    for batch in val_pbar:
                        try:
                            volumes = batch['volume'].to(device)
                            reconstructed = model(volumes)
                            loss = criterion(reconstructed, volumes)
                            val_loss += loss.item()
                            
                            # Update validation progress less frequently
                            # Use a counter instead of trying to extract batch index
                            if val_pbar.n % 5 == 0:
                                val_pbar.set_postfix(loss=f"{loss.item():.6f}")

                            # Memory cleanup
                            del volumes, reconstructed, loss
                            torch.cuda.empty_cache()

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print("\nOOM during validation. Cleaning up...")
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            print(f"DEBUG: Error in validation batch: {str(e)}")
                            raise e

                # Close validation progress bar
                val_pbar.close()

                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # Update learning rate
                scheduler.step(avg_val_loss)

                # Check if this is the best model
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
            except Exception as e:
                print(f"DEBUG: Error during epoch {epoch+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Make sure to close progress bars in case of exception
                if 'train_pbar' in locals():
                    train_pbar.close()
                if 'val_pbar' in locals():
                    val_pbar.close()

            # Outside of the try block for each epoch
            try:
                # Save checkpoint
                if (epoch + 1) % config.save_interval == 0 or is_best or (epoch + 1 == config.epochs):
                    checkpoint_handler.save(
                        model, optimizer, scheduler,
                        epoch, train_losses, val_losses,
                        is_best=is_best
                    )

                # Print epoch summary
                elapsed_time = time.time() - start_time
                time_per_epoch = elapsed_time / (epoch - start_epoch + 1) if epoch >= start_epoch else 0
                est_time_left = time_per_epoch * (config.epochs - epoch - 1)
                
                # Single line epoch summary
                print(f"Epoch {epoch+1}/{config.epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.8f} | ETA: {est_time_left/60:.2f}m")
                
                try:
                    print("DEBUG: About to call print_memory_stats()")
                    print_memory_stats()
                    print("DEBUG: Successfully called print_memory_stats()")
                except Exception as e:
                    print(f"DEBUG: Error in print_memory_stats(): {str(e)}")
                    # Continue training even if memory stats fail
                
                # Early stopping check
                print("DEBUG: Checking early stopping criteria")
                if early_stopping(avg_val_loss, epoch):
                    if early_stopping.early_stop:
                        print("\nEarly stopping triggered!")
                        break
                print(f"DEBUG: Completed epoch {epoch+1}/{config.epochs}, continuing to next epoch")
            except Exception as e:
                print(f"DEBUG: Error after epoch training/validation: {str(e)}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Still save the model
        checkpoint_handler.save(
            model, optimizer, scheduler,
            epoch, train_losses, val_losses,
            is_best=False
        )

    finally:
        print("DEBUG: Entering finally block")
        # Close progress bars
        if 'total_pbar' in locals():
            total_pbar.close()
        
        print("DEBUG: About to plot training history")
        # Plot training history
        try:
            plt.figure(figsize=(12, 5))
            
            # Plot full history
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Full Training History')
            plt.legend()
            plt.grid(True)
            
            # Plot recent history (last 30 epochs or full history if < 30 epochs)
            plt.subplot(1, 2, 2)
            recent = min(30, len(train_losses))
            if recent > 5:  # Only plot recent if we have enough epochs
                plt.plot(train_losses[-recent:], label='Train Loss')
                plt.plot(val_losses[-recent:], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Last {recent} Epochs')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            print(f"DEBUG: Saving plot to {os.path.join(config.checkpoint_dir, f'{config.model_name}_training_history.png')}")
            plt.savefig(os.path.join(config.checkpoint_dir, f"{config.model_name}_training_history.png"))
            plt.show()
            print("DEBUG: Plot saved and displayed successfully")
        except Exception as e:
            print(f"DEBUG: Error in plotting: {str(e)}")
            import traceback
            traceback.print_exc()

        # Print final training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return train_losses, val_losses, model