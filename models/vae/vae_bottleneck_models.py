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
    """
    Standard 4-stage encoder with increasing channel depth (32->64->128->256).
    Removed the 'ChannelReductionBlock' to preserve image details/texture.
    """
    def __init__(self, initial_filters=32): # Increased default from 4 to 32
        super().__init__()
        
        # Stage 1: (1 -> 32)
        # We keep the 32 channels. No reduction to 8.
        self.down1 = nn.Sequential(
            BaseConvBlock(1, 32, stride=2),      # -> (32, 32, 64, 64)
            BaseConvBlock(32, 32)                # -> (32, 32, 64, 64)
        )
        
        # Stage 2: (32 -> 64)
        # Standard doubling of channels as spatial size halves
        self.down2 = nn.Sequential(
            BaseConvBlock(32, 64, stride=2),     # -> (64, 16, 32, 32)
            BaseConvBlock(64, 64)                # -> (64, 16, 32, 32)
        )
        
        # Stage 3: (64 -> 128)
        self.down3 = nn.Sequential(
            BaseConvBlock(64, 128, stride=2),    # -> (128, 8, 16, 16)
            BaseConvBlock(128, 128)              # -> (128, 8, 16, 16)
        )
        
        # Stage 4: (128 -> 256)
        # High capacity for the bottleneck features
        self.down4 = nn.Sequential(
            BaseConvBlock(128, 256, stride=2),   # -> (256, 4, 8, 8)
            BaseConvBlock(256, 256)              # -> (256, 4, 8, 8)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4


class BottleneckEncoder(nn.Module):
    def __init__(self, initial_filters=32, latent_dim=128, bottleneck_shape=(1,1,1)):
        super().__init__()
        self.backbone = BaseEncoder(initial_filters=initial_filters)
        
        # NOTE: Input channels are now 256 (from Stage 4 above), not 128.
        self.global_pool = nn.Conv3d(256, latent_dim, kernel_size=(4,8,8), padding=0)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
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
