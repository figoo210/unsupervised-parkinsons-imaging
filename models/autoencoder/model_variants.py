import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.autoencoder.direct_ae import ConvBlock, DirectAutoencoder
from models.autoencoder.optimized_ae import OptimizedAutoencoder, GroupedConvAutoencoder as GroupedLatentAutoencoder

class LightAutoencoder(nn.Module):
    """
    Lightweight 3D Autoencoder with reduced channel counts
    for 64×128×128 medical volumes.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial feature extraction
            ConvBlock(1, 8),  # (1, 64, 128, 128) -> (8, 64, 128, 128)
            
            # Downsampling path with reduced channels
            ConvBlock(8, 16, stride=2),    # -> (16, 32, 64, 64)
            ConvBlock(16, 16),             # -> (16, 32, 64, 64)
            
            ConvBlock(16, 32, stride=2),   # -> (32, 16, 32, 32)
            ConvBlock(32, 32),             # -> (32, 16, 32, 32)
            
            ConvBlock(32, 64, stride=2),   # -> (64, 8, 16, 16)
            ConvBlock(64, 64),             # -> (64, 8, 16, 16)
            
            ConvBlock(64, 128, stride=2),  # -> (128, 4, 8, 8)
            ConvBlock(128, 128)            # -> (128, 4, 8, 8)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsampling path
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(64, 32),
            ConvBlock(32, 32),
            
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(32, 16),
            ConvBlock(16, 16),
            
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(16, 8),
            ConvBlock(8, 8),
            
            # Final convolution
            nn.Conv3d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class GroupedConvAutoencoder(nn.Module):
    """
    Memory-efficient 3D Autoencoder using grouped convolutions
    for 64×128×128 medical volumes.
    """
    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups
        
        # Encoder
        self.init_conv = ConvBlock(1, 16)  # Regular conv for first layer
        
        # Downsampling with grouped convolutions
        self.down1 = nn.Sequential(
            self._grouped_conv_block(16, 32, stride=2),    # -> (32, 32, 64, 64)
            self._grouped_conv_block(32, 32)               # -> (32, 32, 64, 64)
        )
        
        self.down2 = nn.Sequential(
            self._grouped_conv_block(32, 64, stride=2),    # -> (64, 16, 32, 32)
            self._grouped_conv_block(64, 64)               # -> (64, 16, 32, 32)
        )
        
        self.down3 = nn.Sequential(
            self._grouped_conv_block(64, 128, stride=2),   # -> (128, 8, 16, 16)
            self._grouped_conv_block(128, 128)             # -> (128, 8, 16, 16)
        )
        
        self.down4 = nn.Sequential(
            self._grouped_conv_block(128, 256, stride=2),  # -> (256, 4, 8, 8)
            self._grouped_conv_block(256, 256)             # -> (256, 4, 8, 8)
        )
        
        # Decoder with grouped convolutions
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            self._grouped_conv_block(256, 128),
            self._grouped_conv_block(128, 128)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            self._grouped_conv_block(128, 64),
            self._grouped_conv_block(64, 64)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            self._grouped_conv_block(64, 32),
            self._grouped_conv_block(32, 32)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            self._grouped_conv_block(32, 16),
            self._grouped_conv_block(16, 16)
        )
        
        # Final convolution (regular conv)
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)
    
    def _grouped_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """Create a block with grouped convolution, batch norm, and ReLU"""
        # Ensure channels are divisible by groups
        assert in_channels % self.groups == 0, f"Input channels {in_channels} not divisible by {self.groups} groups"
        assert out_channels % self.groups == 0, f"Output channels {out_channels} not divisible by {self.groups} groups"
        
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=self.groups)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    
    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        encoded = self.down4(x)
        
        # Decoder
        x = self.up1(encoded)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        reconstructed = self.final_conv(x)
        
        return reconstructed


class DepthwiseSeparableConvBlock(nn.Module):
    """Memory-efficient depthwise separable convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            # Depthwise convolution (one filter per input channel)
            ('depthwise', nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)),
            ('bn1', nn.BatchNorm3d(in_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            # Pointwise convolution (1x1x1 convolution for channel mixing)
            ('pointwise', nn.Conv3d(in_channels, out_channels, kernel_size=1)),
            ('bn2', nn.BatchNorm3d(out_channels)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)


class EfficientAutoencoder(nn.Module):
    """
    Efficient 3D Autoencoder using depthwise separable convolutions
    for 64×128×128 medical volumes.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.init_conv = ConvBlock(1, 16)  # Regular conv for first layer
        
        # Downsampling with depthwise separable convolutions
        self.down1 = nn.Sequential(
            DepthwiseSeparableConvBlock(16, 32, stride=2),    # -> (32, 32, 64, 64)
            DepthwiseSeparableConvBlock(32, 32)               # -> (32, 32, 64, 64)
        )
        
        self.down2 = nn.Sequential(
            DepthwiseSeparableConvBlock(32, 64, stride=2),    # -> (64, 16, 32, 32)
            DepthwiseSeparableConvBlock(64, 64)               # -> (64, 16, 32, 32)
        )
        
        self.down3 = nn.Sequential(
            DepthwiseSeparableConvBlock(64, 128, stride=2),   # -> (128, 8, 16, 16)
            DepthwiseSeparableConvBlock(128, 128)             # -> (128, 8, 16, 16)
        )
        
        self.down4 = nn.Sequential(
            DepthwiseSeparableConvBlock(128, 256, stride=2),  # -> (256, 4, 8, 8)
            DepthwiseSeparableConvBlock(256, 256)             # -> (256, 4, 8, 8)
        )
        
        # Decoder with depthwise separable convolutions
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConvBlock(256, 128),
            DepthwiseSeparableConvBlock(128, 128)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConvBlock(128, 64),
            DepthwiseSeparableConvBlock(64, 64)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConvBlock(64, 32),
            DepthwiseSeparableConvBlock(32, 32)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            DepthwiseSeparableConvBlock(32, 16),
            DepthwiseSeparableConvBlock(16, 16)
        )
        
        # Final convolution (regular conv)
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        encoded = self.down4(x)
        
        # Decoder
        x = self.up1(encoded)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        reconstructed = self.final_conv(x)
        
        return reconstructed


def get_model_variant(variant_name, **kwargs):
    """
    Factory function to get a specific autoencoder variant.
    
    Args:
        variant_name: Name of the model variant
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An instance of the requested model variant
    """
    variants = {
        "direct": DirectAutoencoder,
        "light": LightAutoencoder,
        "grouped": GroupedConvAutoencoder,
        "efficient": EfficientAutoencoder,
        "optimized": OptimizedAutoencoder,
        "grouped_latent": GroupedLatentAutoencoder
    }
    
    if variant_name not in variants:
        raise ValueError(f"Unknown model variant: {variant_name}. Available variants: {list(variants.keys())}")
    
    return variants[variant_name](**kwargs)
