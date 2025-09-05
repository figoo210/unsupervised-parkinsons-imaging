import torch
from torch.cuda import amp
from models.vae.model import VAELoss
from models.vae.optimizer import create_vae_optimizer
from models.vae.scheduler import create_vae_scheduler
from models.vae.config import VAEConfig, VAECheckpointHandler, VAEEarlyStopping
from utils.config_helpers import print_memory_stats
from tqdm import tqdm
import time
import gc


def train_vae(model, train_loader, val_loader, config=None):
    """Memory-optimized VAE training loop with mixed precision and gradient accumulation"""
    if config is None:
        config = VAEConfig()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize components
    criterion = VAELoss(beta=config.beta, beta_warmup_steps=config.beta_warmup_steps)
    optimizer = create_vae_optimizer(model, config)
    scheduler = create_vae_scheduler(optimizer, config)
    early_stopping = VAEEarlyStopping(patience=config.early_stopping_patience)
    checkpoint_handler = VAECheckpointHandler(config.checkpoint_dir, config.model_name)

    # Mixed precision setup
    scaler = amp.GradScaler(enabled=config.use_mixed_precision)

    # Training tracking variables
    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_recon_losses = []
    val_kl_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    # Load checkpoint if available
    start_epoch = 0
    checkpoint_data = checkpoint_handler.load(model, optimizer, scheduler, device)
    if checkpoint_data:
        start_epoch = checkpoint_data['epoch'] + 1
        train_losses = checkpoint_data.get('train_losses', [])
        val_losses = checkpoint_data.get('val_losses', [])
        train_recon_losses = checkpoint_data.get('train_recon_losses', [])
        train_kl_losses = checkpoint_data.get('train_kl_losses', [])
        val_recon_losses = checkpoint_data.get('val_recon_losses', [])
        val_kl_losses = checkpoint_data.get('val_kl_losses', [])
        print(f"Resuming training from epoch {start_epoch}")

    # Calculate total steps for progress tracking
    total_steps = len(train_loader) * config.epochs

    # Training loop
    try:
        # Create progress bar for total training
        total_pbar = tqdm(total=total_steps, desc="Total Progress", position=0)
        total_pbar.update(start_epoch * len(train_loader))
        
        for epoch in range(start_epoch, config.epochs):
            # Training phase
            model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            current_beta = 0
            optimizer.zero_grad()  # Zero gradients at epoch start

            train_pbar = tqdm(train_loader, 
                            desc=f'Epoch {epoch+1}/{config.epochs} [Train]',
                            leave=False, 
                            position=1)

            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Move data to device
                    volumes = batch['volume'].to(device, non_blocking=True)
                    
                    # Mixed precision forward pass
                    with amp.autocast(enabled=config.use_mixed_precision):
                        reconstructed, mu, log_var = model(volumes)
                        loss, recon_loss, kl_loss, beta = criterion(reconstructed, volumes, mu, log_var)
                        current_beta = beta
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
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    
                    # Update progress bars
                    train_pbar.set_postfix({
                        'loss': f"{batch_loss:.6f}",
                        'recon': f"{recon_loss.item():.6f}",
                        'kl': f"{kl_loss.item():.6f}",
                        'beta': f"{beta:.6f}"
                    })
                    total_pbar.update(1)

                    # Memory cleanup
                    del volumes, reconstructed, mu, log_var, loss, recon_loss, kl_loss
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM in batch {batch_idx}. Cleaning up...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        # Skip this batch and continue
                        continue
                    else:
                        print(f"RuntimeError: {str(e)}")
                        raise e
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise e

            # Calculate average training losses
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_recon_loss = epoch_recon_loss / len(train_loader)
            avg_train_kl_loss = epoch_kl_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            train_recon_losses.append(avg_train_recon_loss)
            train_kl_losses.append(avg_train_kl_loss)

            # Validation phase
            model.eval()
            val_loss = 0
            val_recon_loss_sum = 0
            val_kl_loss_sum = 0
            
            val_pbar = tqdm(val_loader, 
                          desc=f'Epoch {epoch+1}/{config.epochs} [Val]',
                          leave=False,
                          position=1)

            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        volumes = batch['volume'].to(device)
                        reconstructed, mu, log_var = model(volumes)
                        loss, recon_loss, kl_loss, _ = criterion(reconstructed, volumes, mu, log_var)
                        
                        val_loss += loss.item()
                        val_recon_loss_sum += recon_loss.item()
                        val_kl_loss_sum += kl_loss.item()
                        
                        val_pbar.set_postfix({
                            'loss': f"{loss.item():.6f}",
                            'recon': f"{recon_loss.item():.6f}",
                            'kl': f"{kl_loss.item():.6f}"
                        })

                        # Memory cleanup
                        del volumes, reconstructed, mu, log_var, loss, recon_loss, kl_loss
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("\nOOM during validation. Cleaning up...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            print(f"RuntimeError during validation: {str(e)}")
                            raise e
                    except Exception as e:
                        print(f"Unexpected error during validation: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        raise e

            # Calculate average validation losses
            avg_val_loss = val_loss / len(val_loader)
            avg_val_recon_loss = val_recon_loss_sum / len(val_loader)
            avg_val_kl_loss = val_kl_loss_sum / len(val_loader)
            
            val_losses.append(avg_val_loss)
            val_recon_losses.append(avg_val_recon_loss)
            val_kl_losses.append(avg_val_kl_loss)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Check if this is the best model
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss

            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0 or is_best or (epoch + 1 == config.epochs):
                # Save additional loss components
                checkpoint_handler.save(
                    model, optimizer, scheduler,
                    epoch, train_losses, val_losses, 
                    train_recon_losses, train_kl_losses,
                    val_recon_losses, val_kl_losses,
                    is_best=is_best
                )

            # Print epoch summary
            elapsed_time = time.time() - start_time
            time_per_epoch = elapsed_time / (epoch - start_epoch + 1) if epoch >= start_epoch else 0
            est_time_left = time_per_epoch * (config.epochs - epoch - 1)
            
            print(f"\nEpoch {epoch+1}/{config.epochs} completed in {time_per_epoch:.2f}s")
            print(f"Train Loss: {avg_train_loss:.6f} (Recon: {avg_train_recon_loss:.6f}, KL: {avg_train_kl_loss:.6f})")
            print(f"Val Loss: {avg_val_loss:.6f} (Recon: {avg_val_recon_loss:.6f}, KL: {avg_val_kl_loss:.6f})")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8f}")
            print(f"Beta value: {current_beta:.6f}")
            print(f"Est. time remaining: {est_time_left/60:.2f} minutes")
            print_memory_stats()
            
            # Early stopping check
            if early_stopping(avg_val_loss, epoch):
                if early_stopping.early_stop:
                    print("\nEarly stopping triggered!")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Still save the model
        checkpoint_handler.save(
            model, optimizer, scheduler,
            epoch, train_losses, val_losses,
            train_recon_losses, train_kl_losses,
            val_recon_losses, val_kl_losses,
            is_best=False
        )

    finally:
        # Close progress bars
        if 'total_pbar' in locals():
            total_pbar.close()
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Loss History')
        plt.legend()
        plt.grid(True)
        
        # Plot reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(train_recon_losses, label='Train Recon Loss')
        plt.plot(val_recon_losses, label='Val Recon Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss History')
        plt.legend()
        plt.grid(True)
        
        # Plot KL loss
        plt.subplot(2, 2, 3)
        plt.plot(train_kl_losses, label='Train KL Loss')
        plt.plot(val_kl_losses, label='Val KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.title('KL Divergence Loss History')
        plt.legend()
        plt.grid(True)
        
        # Plot recent total loss (last 30 epochs or all if < 30)
        plt.subplot(2, 2, 4)
        recent = min(30, len(train_losses))
        if recent > 5:  # Only plot recent history if we have enough epochs
            plt.plot(train_losses[-recent:], label='Train Loss')
            plt.plot(val_losses[-recent:], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss')
            plt.title(f'Last {recent} Epochs')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.checkpoint_dir, f"{config.model_name}_training_history.png"))
        plt.show()

        # Print final training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_recon_losses': train_recon_losses,
            'train_kl_losses': train_kl_losses,
            'val_recon_losses': val_recon_losses,
            'val_kl_losses': val_kl_losses,
            'best_val_loss': best_val_loss,
            'model': model
        }