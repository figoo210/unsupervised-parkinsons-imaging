"""
Bottleneck Autoencoder Models
Extracted models: BottleneckEncoder and ComplexBottleneckDeconvDecoder
"""
import math
import torch
import torch.nn as nn
from collections import OrderedDict


def _compute_ladder_shapes(initial_shape, target_shape, stages=4):
    """Plan intermediate spatial sizes so the ladder lands exactly on `target_shape`."""
    current = tuple(initial_shape)
    target = tuple(target_shape)
    shapes = []

    for stage in range(stages):
        remaining = stages - stage - 1
        next_dims = []
        for dim_current, dim_target in zip(current, target):
            if dim_current == dim_target or remaining < 0:
                next_dims.append(dim_target)
                continue

            if remaining == 0:
                next_dims.append(dim_target)
                continue

            min_dim = math.ceil(dim_target / (2 ** remaining))
            proposed = dim_current * 2
            next_dims.append(min(dim_target, max(proposed, min_dim)))

        current = tuple(next_dims)
        shapes.append(current)

    if shapes:
        shapes[-1] = target
    return shapes


class BaseConvBlock(nn.Module):
    """3D convolution followed by batch normalization and inplace ReLU for compact feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)


class BaseChannelReductionBlock(nn.Module):
    """Channel compressor that applies a 1×1×1 convolution before normalization and activation."""
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


class BaseEncoder(nn.Module):
    """Four-stage 3D encoder that downsamples (1, 64, 128, 128) volumes to a latent tensor of shape (128, 4, 8, 8)."""
    def __init__(self, initial_filters=4):
        super().__init__()
        
        # Stage 1: spatial stride 2, expand to 32 channels, then compress to 8
        # Input tensor: (1, 64, 128, 128)
        self.down1 = nn.Sequential(
            BaseConvBlock(1, initial_filters, stride=2),      # -> (4, 32, 64, 64)
            BaseConvBlock(initial_filters, 32),               # -> (32, 32, 64, 64)
            BaseChannelReductionBlock(32)                     # -> (8, 32, 64, 64)
        )
        
        # Stage 2: repeat downsampling while keeping 8-channel output
        # Input tensor: (8, 32, 64, 64)
        self.down2 = nn.Sequential(
            BaseConvBlock(8, 32, stride=2),                   # -> (32, 16, 32, 32)
            BaseConvBlock(32, 32),                            # -> (32, 16, 32, 32)
            BaseChannelReductionBlock(32)                     # -> (8, 16, 32, 32)
        )
        
        # Stage 3: widen to 64 features before compressing to 16 channels
        # Input tensor: (8, 16, 32, 32)
        self.down3 = nn.Sequential(
            BaseConvBlock(8, 64, stride=2),                   # -> (64, 8, 16, 16)
            BaseConvBlock(64, 64),                            # -> (64, 8, 16, 16)
            BaseChannelReductionBlock(64)                     # -> (16, 8, 16, 16)
        )
        
        # Stage 4: reach latent representation without channel reduction
        # Input tensor: (16, 8, 16, 16)
        self.down4 = nn.Sequential(
            BaseConvBlock(16, 128, stride=2),                 # -> (128, 4, 8, 8)
            BaseConvBlock(128, 128),                          # -> (128, 4, 8, 8)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4


class BottleneckEncoder(nn.Module):
    """Base encoder followed by spatial squeeze to produce latent vectors.

    Keeps the four downsampling stages but collapses spatial support with
    adaptive average pooling so the decoder receives true bottleneck codes.
    """
    def __init__(self, initial_filters=4, latent_dim=128, bottleneck_shape=(1,1,1)):
        super().__init__()
        self.backbone = BaseEncoder(initial_filters=initial_filters)
        # Conv3d projects from 128 channels to latent_dim while collapsing spatial dims
        self.global_pool = nn.Conv3d(128, latent_dim, kernel_size=(4,8,8), padding=0)
        self.latent_dim = latent_dim

    def forward(self, x):
        # Extract feature volume using the shared backbone.
        x = self.backbone(x)
        # Collapse spatial dimensions and project to latent_dim
        x = self.global_pool(x)  # -> (B, latent_dim, 1, 1, 1)
        return x


class ComplexBottleneckDeconvDecoder(nn.Module):
    """ConvTranspose3d variant: densly learn spatial expansion from the latent vector."""
    def __init__(
        self,
        latent_dim=128,
        initial_shape=None,
        target_shape=(64, 128, 128),
        mid_channels=64,
        output_channels=1,
    ):
        super().__init__()
        self.target_shape = target_shape
        stages = 4
        if initial_shape is None:
            self.initial_shape = tuple(max(1, s // (2 ** stages)) for s in target_shape)
        else:
            self.initial_shape = initial_shape
        self.deconv = nn.ConvTranspose3d(latent_dim, mid_channels, kernel_size=self.initial_shape)
        self.act = nn.ReLU(inplace=True)
        ladder_shapes = _compute_ladder_shapes(self.initial_shape, target_shape)
        ladder_channels = [mid_channels, 32, 16, 8, 4]
        self.ladder = nn.ModuleList()
        in_channels = ladder_channels[0]
        current_shape = self.initial_shape
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            current_shape = target
        self.refine_conv = BaseConvBlock(in_channels, in_channels)
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        x = self.refine_conv(x)
        return self.final_conv(x)


class BottleneckAE(nn.Module):
    """Wrapper for bottleneck-style encoder that outputs (B, latent_dim, bz,by,bx) and dec expects that"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class BottleneckVAE(nn.Module):
    """Variational Bottleneck Autoencoder with reparameterization trick.
    
    Extends the bottleneck architecture with a probabilistic latent space.
    The encoder outputs mu and log_var instead of a deterministic latent code.
    """
    def __init__(self, encoder, decoder, latent_dim=128):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
        # Project encoder output (latent_dim, 1, 1, 1) to mu and log_var
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to enable backpropagation through sampling."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encode to bottleneck representation (B, latent_dim, 1, 1, 1)
        bottleneck = self.encoder(x)
        
        # Flatten to (B, latent_dim)
        flat = bottleneck.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Project to mu and log_var
        mu = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Reshape back to (B, latent_dim, 1, 1, 1) for decoder
        z_reshaped = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Decode
        reconstruction = self.decoder(z_reshaped)
        
        return reconstruction, mu, log_var

    def encode(self, x):
        """Encode input to latent parameters without sampling."""
        bottleneck = self.encoder(x)
        flat = bottleneck.squeeze(-1).squeeze(-1).squeeze(-1)
        mu = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)
        return mu, log_var

    def generate(self, z=None, num_samples=1):
        """Generate samples from latent space or random samples."""
        device = next(self.parameters()).device
        
        if z is None:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Reshape for decoder
        z_reshaped = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        with torch.no_grad():
            samples = self.decoder(z_reshaped)
        
        return samples
