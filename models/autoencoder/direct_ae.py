import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Memory-efficient convolutional block with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))  # inplace ReLU for memory efficiency
        ]))

    def forward(self, x):
        return self.block(x)

class DirectEncoder(nn.Module):
    """3D Encoder network without bottleneck, optimized for 64×128×128 input volumes."""
    def __init__(self, output_channels=256):
        super().__init__()

        # Initial feature extraction
        self.init_conv = ConvBlock(1, 16)  # (1, 64, 128, 128) -> (16, 64, 128, 128)

        # Downsampling path with progressive channel increase
        self.down1 = nn.Sequential(
            ConvBlock(16, 32, stride=2),    # -> (32, 32, 64, 64)
            ConvBlock(32, 32)               # -> (32, 32, 64, 64)
        )

        self.down2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),    # -> (64, 16, 32, 32)
            ConvBlock(64, 64)               # -> (64, 16, 32, 32)
        )

        self.down3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),   # -> (128, 8, 16, 16)
            ConvBlock(128, 128)             # -> (128, 8, 16, 16)
        )

        self.down4 = nn.Sequential(
            ConvBlock(128, output_channels, stride=2),  # -> (output_channels, 4, 8, 8)
            ConvBlock(output_channels, output_channels) # -> (output_channels, 4, 8, 8)
        )

    def forward(self, x):
        x = self.init_conv(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4

class DirectDecoder(nn.Module):
    """3D Decoder network optimized for 64×128×128 output volumes."""
    def __init__(self, input_channels=256):
        super().__init__()

        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(input_channels, 128),
            ConvBlock(128, 128)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(64, 32),
            ConvBlock(32, 32)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(32, 16),
            ConvBlock(16, 16)
        )

        # Final convolution
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final convolution
        x = self.final_conv(x)
        return x

class DirectAutoencoder(nn.Module):
    """
    Memory-optimized 3D Autoencoder for 64×128×128 medical volumes.
    This version connects encoder directly to decoder without a latent bottleneck.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.encoder = DirectEncoder(output_channels=channels)
        self.decoder = DirectDecoder(input_channels=channels)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction
    
    def get_encoded_shape(self, input_shape=(1, 64, 128, 128)):
        """Get the shape of the encoded representation for a given input shape"""
        device = next(self.parameters()).device
        dummy_input = torch.zeros(input_shape).to(device)
        with torch.no_grad():
            encoded = self.encoder(dummy_input)
        return encoded.shape
