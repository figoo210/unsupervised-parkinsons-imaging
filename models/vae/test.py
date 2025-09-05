import torch
from models.autoencoder.model import BaseAutoencoder
from utils.config_helpers import print_memory_stats
from models.vae.model import VAE, VAELoss


# Test VAE with dummy data
def test_vae(batch_size=2, latent_dim=256):
    """Test the VAE architecture and memory usage with dummy data."""
    print("\nTesting VAE Architecture...")

    try:
        # Create model and move to GPU
        model = VAE(latent_dim=latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create loss function
        vae_loss = VAELoss(beta=0.005)

        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nVAE Model Parameters: {num_params:,}")

        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, 64, 128, 128, device=device)

        # Test forward pass
        print("\nTesting forward pass...")
        print_memory_stats()
        with torch.no_grad():
            recon, mu, log_var = model(dummy_input)

        # Test loss calculation
        loss, recon_loss, kl_loss, beta = vae_loss(recon, dummy_input, mu, log_var)

        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Output shape: {recon.shape}")
        print(f"\nLoss values:")
        print(f"Total loss: {loss.item():.6f}")
        print(f"Reconstruction loss: {recon_loss.item():.6f}")
        print(f"KL loss: {kl_loss.item():.6f}")
        print(f"Beta value: {beta:.6f}")

        # Check memory usage
        print("\nGPU Memory After Forward Pass:")
        print_memory_stats()

        # Clean up
        del model, dummy_input, recon, mu, log_var
        torch.cuda.empty_cache()

        print("\nVAE test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing VAE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    """Test the autoencoder with dummy data and verify memory usage."""
    print("\nTesting Autoencoder Architecture...")

    try:
        # Create model and move to GPU
        model = BaseAutoencoder(latent_dim=256)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel Parameters: {num_params:,}")

        # Create dummy input (64x128x128 volume)
        dummy_input = torch.randn(batch_size, 1, 64, 128, 128, device=device)

        # Print initial memory usage
        print("\nInitial GPU Memory Usage:")
        print_memory_stats()

        # Test encoding
        print("\nTesting encoder...")
        with torch.no_grad():
            latent = model.encode(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Latent shape: {latent.shape}")

        # Test full forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            output = model(dummy_input)

        # Print output shape and final memory usage
        print(f"Output shape: {output.shape}")
        print("\nFinal GPU Memory Usage:")
        print_memory_stats()

        # Verify shapes
        assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} vs {dummy_input.shape}"

        # Clean up
        del model, dummy_input, output, latent
        torch.cuda.empty_cache()

        print("\nAutoencoder test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing autoencoder: {str(e)}")
        import traceback
        traceback.print_exc()
        return False