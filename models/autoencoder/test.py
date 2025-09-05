import torch
from models.autoencoder.model import BaseAutoencoder
from utils.config_helpers import print_memory_stats

# Memory and GPU test function
def test_autoencoder(batch_size=2):
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