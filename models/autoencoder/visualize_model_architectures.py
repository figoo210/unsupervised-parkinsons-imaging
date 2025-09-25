import torch
import torch.nn as nn
from models.autoencoder.model_variants import get_model_variant
from torchsummary import summary
import os
import sys

def visualize_model_architecture(model_name):
    """
    Visualize the architecture of a specific model variant
    
    Args:
        model_name: Name of the model variant to visualize
    """
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    # Create model instance
    try:
        model = get_model_variant(model_name)
        model.eval()
        
        # Move to CPU for visualization
        device = torch.device("cpu")
        model = model.to(device)
        
        # Print model summary for 3D input (1, 64, 128, 128)
        summary(model, input_size=(1, 64, 128, 128), device="cpu")
        
        # Print the model's structure
        print("\nDetailed Model Structure:")
        print(model)
        
        print(f"\n{'='*80}\n")
        return True
    except Exception as e:
        print(f"Error visualizing {model_name} model: {str(e)}")
        return False

def main():
    # Define model variants to visualize
    model_variants = ["direct", "light", "grouped", "efficient", "optimized", "grouped_latent"]
    
    # Create output directory for saving visualizations
    os.makedirs("model_visualizations", exist_ok=True)
    
    # Visualize each model
    for variant in model_variants:
        success = visualize_model_architecture(variant)
        if not success:
            print(f"Failed to visualize {variant} model")

if __name__ == "__main__":
    main()
