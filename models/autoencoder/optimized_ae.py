import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class OptimizedConvBlock(nn.Module):
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

class ChannelReductionBlock(nn.Module):
    """Block that reduces channels at the end of each resolution level using 1x1 convolutions."""
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()
        self.out_channels = in_channels // reduction_factor
        self.block = nn.Sequential(OrderedDict([
            ('conv1x1', nn.Conv3d(in_channels, self.out_channels, kernel_size=1)),
            ('bn', nn.BatchNorm3d(self.out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    
    def forward(self, x):
        return self.block(x)

class OptimizedEncoder(nn.Module):
    """3D Encoder network with optimized architecture for 64×128×128 input volumes."""
    def __init__(self, initial_filters=4):
        super().__init__()
        
        # Start directly with strided convolution as requested
        # 1->4 strided, 4->32 non-strided
        self.down1 = nn.Sequential(
            OptimizedConvBlock(1, initial_filters, stride=2),      # -> (4, 32, 64, 64)
            OptimizedConvBlock(initial_filters, 32),               # -> (32, 32, 64, 64)
            ChannelReductionBlock(32)                              # -> (8, 32, 64, 64)
        )
        
        # Second downsampling block
        self.down2 = nn.Sequential(
            OptimizedConvBlock(8, 16, stride=2),                   # -> (16, 16, 32, 32)
            OptimizedConvBlock(16, 16),                            # -> (16, 16, 32, 32)
            ChannelReductionBlock(16)                              # -> (4, 16, 32, 32)
        )
        
        # Third downsampling block
        self.down3 = nn.Sequential(
            OptimizedConvBlock(4, 32, stride=2),                   # -> (32, 8, 16, 16)
            OptimizedConvBlock(32, 32),                            # -> (32, 8, 16, 16)
            ChannelReductionBlock(32)                              # -> (8, 8, 16, 16)
        )
        
        # Fourth downsampling block
        self.down4 = nn.Sequential(
            OptimizedConvBlock(8, 64, stride=2),                   # -> (64, 4, 8, 8)
            OptimizedConvBlock(64, 512),                           # -> (512, 4, 8, 8)
            # No channel reduction to maintain 512 channels for latent space
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4

class LatentSpaceDecoder(nn.Module):
    """
    Decoder that implements the specific bottleneck architecture with grouped convolutions:
    latent-space → Reshape(Nlat-space, 1, 1, 1) → Upsampling3D(Nlat-space, Nx, Ny, Nz) 
    → Conv3D(Nlat-space, Nlat-space, (Nx, Ny, Nz), groups = Nlat-space) 
    → Conv3D(Nlat-space, Nchan, 1) → Activation
    """
    def __init__(self, latent_dim=512, output_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(latent_dim, 128),
            OptimizedConvBlock(128, 64),
            ChannelReductionBlock(64)  # -> 16 channels
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(16, 32),
            OptimizedConvBlock(32, 32),
            ChannelReductionBlock(32)  # -> 8 channels
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(8, 16),
            OptimizedConvBlock(16, 16),
            ChannelReductionBlock(16)   # -> 4 channels
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(4, 8),
            OptimizedConvBlock(8, 4)   # Keep at 4 channels before final conv
        )
        
        # Final convolution
        self.final_conv = nn.Conv3d(4, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        return x

class GroupedLatentDecoder(nn.Module):
    """
    Decoder that implements the specific bottleneck architecture with grouped convolutions
    as described in the optimization request.
    
    The grouped convolution uses a 4x8x8 kernel size with padding='same' to learn the
    "base image" from the latent space, which helps recover spatial patterns more effectively.
    """
    def __init__(self, latent_dim=512, output_shape=(64, 128, 128), output_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        # Initial target shape after first upsampling
        self.initial_shape = (output_shape[0]//8, output_shape[1]//8, output_shape[2]//8)  # (8, 16, 16)
        
        # Reshape and upsample to initial shape
        self.reshape_upsample = nn.Upsample(size=self.initial_shape, mode='trilinear', align_corners=False)
        
        # Grouped convolution with 4x8x8 kernel to learn the "base image"
        # Each channel gets its own filter that spans the entire initial volume
        self.grouped_conv = nn.Conv3d(
            latent_dim, latent_dim, 
            kernel_size=(4, 8, 8), padding='same',
            groups=latent_dim  # Each channel processed independently
        )
        
        # 1x1 convolution to mix channels
        self.channel_mixer = nn.Conv3d(latent_dim, 32, kernel_size=1)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
        
        # Remaining upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(32, 16)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(16, 8)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            OptimizedConvBlock(8, 4)
        )
        
        # Final convolution
        self.final_conv = nn.Conv3d(4, output_channels, kernel_size=1)

    def forward(self, x):
        # Get batch size and shape
        batch_size = x.size(0)
        
        # Check if input is already in 5D format or needs flattening
        if len(x.shape) == 5:  # Already in format [B, C, D, H, W]
            # Just ensure the channel dimension is correct
            if x.size(1) != self.latent_dim:
                raise ValueError(f"Expected {self.latent_dim} channels in input, got {x.size(1)}")
            
            # Flatten spatial dimensions
            x = x.flatten(2).mean(dim=2).view(batch_size, self.latent_dim, 1, 1, 1)
        else:
            # Reshape to (batch_size, latent_dim, 1, 1, 1)
            x = x.view(batch_size, self.latent_dim, 1, 1, 1)
        
        # Upsample to initial shape
        x = self.reshape_upsample(x)
        
        # Apply grouped convolution
        x = self.grouped_conv(x)
        
        # Mix channels with 1x1 conv
        x = self.channel_mixer(x)
        
        # Activation
        x = self.activation(x)
        
        # Continue with regular upsampling path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final_conv(x)
        
        return x

class OptimizedAutoencoder(nn.Module):
    """
    Optimized 3D Autoencoder implementing all requested optimizations:
    1. Starting directly with strided convolution and fewer filters
    2. Reducing channels at the end of each block with 1x1 convolutions
    3. Direct connection between encoder and decoder (no bottleneck)
    """
    def __init__(self, initial_filters=4):
        super().__init__()
        self.encoder = OptimizedEncoder(initial_filters=initial_filters)
        self.decoder = LatentSpaceDecoder(latent_dim=512)  # 512 is output from encoder's down4
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

class GroupedConvAutoencoder(nn.Module):
    """
    Autoencoder with optimized encoder and the specific grouped convolution decoder
    as described in the optimization request.
    
    Uses a larger latent dimension (512) to provide sufficient capacity for learning
    complex spatial patterns in the data.
    """
    def __init__(self, initial_filters=4, output_shape=(64, 128, 128)):
        super().__init__()
        self.encoder = OptimizedEncoder(initial_filters=initial_filters)
        self.decoder = GroupedLatentDecoder(latent_dim=512, output_shape=output_shape)
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
