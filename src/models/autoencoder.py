"""
Autoencoder model architectures for medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Basic convolutional block with conv, batch norm, and activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        """
        Initialize ConvBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvBlock."""
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """
    3D CNN Encoder for autoencoder.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize Encoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(Encoder, self).__init__()
        
        # Encoder layers
        self.conv1 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers to latent space
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, D, H, W)
            
        Returns:
            torch.Tensor: Latent representation
        """
        # Convolutional layers
        x = self.conv1(x)  # (B, 32, D, H, W)
        x = self.conv2(x)  # (B, 64, D/2, H/2, W/2)
        x = self.conv3(x)  # (B, 128, D/4, H/4, W/4)
        x = self.conv4(x)  # (B, 256, D/8, H/8, W/8)
        x = self.conv5(x)  # (B, 512, D/16, H/16, W/16)
        
        # Global pooling and flatten
        x = self.global_pool(x)  # (B, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (B, latent_dim)
        
        return x


class Decoder(nn.Module):
    """
    3D CNN Decoder for autoencoder.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize Decoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(Decoder, self).__init__()
        
        # Fully connected layers from latent space
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512 * 4 * 8 * 8)  # For reshaping to (512, 4, 8, 8)
        self.dropout = nn.Dropout(0.2)
        
        # Transposed convolution layers (decoder)
        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(256)
        
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        
        self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        
        self.deconv5 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Reconstructed volume
        """
        # Fully connected layers
        x = F.relu(self.fc1(z))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Reshape for convolutional layers
        x = x.view(x.size(0), 512, 4, 8, 8)
        
        # Transposed convolutions
        x = F.relu(self.bn1(self.deconv1(x)))  # (B, 256, 8, 16, 16)
        x = F.relu(self.bn2(self.deconv2(x)))  # (B, 128, 16, 32, 32)
        x = F.relu(self.bn3(self.deconv3(x)))  # (B, 64, 32, 64, 64)
        x = F.relu(self.bn4(self.deconv4(x)))  # (B, 32, 64, 128, 128)
        x = torch.sigmoid(self.deconv5(x))     # (B, 1, 64, 128, 128)
        
        return x


class BaseAutoencoder(nn.Module):
    """
    Complete autoencoder combining encoder and decoder.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize BaseAutoencoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(BaseAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x (torch.Tensor): Input volume
            
        Returns:
            torch.Tensor: Reconstructed volume
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        return self.decoder(z)


def test_autoencoder(batch_size: int = 2) -> None:
    """
    Test autoencoder with random input.
    
    Args:
        batch_size (int): Batch size for testing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and test input
    model = BaseAutoencoder(latent_dim=256).to(device)
    test_input = torch.randn(batch_size, 1, 64, 128, 128).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Test encoding
    latent = model.encode(test_input)
    print(f"Latent shape: {latent.shape}")
    
    # Test full forward pass
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Test decoding
    decoded = model.decode(latent)
    print(f"Decoded shape: {decoded.shape}")
    
    print("Autoencoder test completed successfully!")


if __name__ == "__main__":
    test_autoencoder() 