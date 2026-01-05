import math
import torch
import torch.nn as nn
from collections import OrderedDict

# --- 1. Helper Functions & Basic Blocks (Unchanged) ---
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
    """3D convolution followed by batch normalization and inplace ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    def forward(self, x):
        return self.block(x)

# --- 2. THE HIGH-CAPACITY ENCODER (Fixes Blur) ---
class BaseEncoder(nn.Module):
    """
    Standard 4-stage encoder with increasing channel depth (32->64->128->256).
    NO Reduction Blocks.
    """
    def __init__(self, initial_filters=32):
        super().__init__()
        
        # Stage 1: (1 -> 32)
        self.down1 = nn.Sequential(
            BaseConvBlock(1, 32, stride=2),
            BaseConvBlock(32, 32)
        )
        
        # Stage 2: (32 -> 64)
        self.down2 = nn.Sequential(
            BaseConvBlock(32, 64, stride=2),
            BaseConvBlock(64, 64)
        )
        
        # Stage 3: (64 -> 128)
        self.down3 = nn.Sequential(
            BaseConvBlock(64, 128, stride=2),
            BaseConvBlock(128, 128)
        )
        
        # Stage 4: (128 -> 256)
        self.down4 = nn.Sequential(
            BaseConvBlock(128, 256, stride=2),
            BaseConvBlock(256, 256)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return d4

# --- 3. THE OPTIMIZED BOTTLENECK (Prof's 1st Email Note) ---
class BottleneckEncoder(nn.Module):
    """
    Encoder that outputs (B, 2*latent_dim, 1, 1, 1).
    We double the latent dim here to hold both Mu and LogVar in one tensor.
    """
    def __init__(self, initial_filters=32, latent_dim=128, bottleneck_shape=(1,1,1)):
        super().__init__()
        self.backbone = BaseEncoder(initial_filters=initial_filters)
        
        # PROFESSOR'S OPTIMIZATION:
        # Instead of 128 channels -> Linear -> Linear
        # We go 256 channels -> Conv3d -> 2 * Latent Dim
        self.global_pool = nn.Conv3d(256, latent_dim * 2, kernel_size=(4,8,8), padding=0)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.backbone(x)
        # Output shape: (B, 2*latent_dim, 1, 1, 1)
        x = self.global_pool(x)
        return x

class ComplexBottleneckDeconvDecoder(nn.Module):
    """Standard Decoder (Unchanged from your previous reliable version)"""
    def __init__(self, latent_dim=128, initial_shape=None, target_shape=(64, 128, 128), mid_channels=64, output_channels=1):
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
        for target, out_channels in zip(ladder_shapes, ladder_channels[1:]):
            self.ladder.append(nn.Sequential(
                nn.Upsample(size=target, mode='trilinear', align_corners=False),
                BaseConvBlock(in_channels, out_channels),
            ))
            in_channels = out_channels
            
        self.refine_conv = BaseConvBlock(in_channels, in_channels)
        self.final_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.act(x)
        for stage in self.ladder:
            x = stage(x)
        x = self.refine_conv(x)
        return self.final_conv(x)

# --- 4. THE OPTIMIZED VAE WRAPPER ---
class BottleneckVAE(nn.Module):
    """
    VAE that performs the splitting of the bottleneck tensor into Mu and LogVar.
    NO Linear Layers! (Fully Convolutional)
    """
    def __init__(self, encoder, decoder, latent_dim=128):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
        # PROFESSOR'S OPTIMIZATION:
        # Removed self.fc_mu and self.fc_log_var
        # The splitting now happens in forward()

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # 1. Encode -> Get (B, 2*latent_dim, 1, 1, 1)
        bottleneck = self.encoder(x)
        
        # 2. Split into Mu and LogVar along channel dimension
        # chunk(tensor, chunks=2, dim=1) splits the 256 channels into 128 and 128
        mu, log_var = torch.chunk(bottleneck, 2, dim=1)
        
        # 3. Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # 4. Decode
        # z is already (B, latent_dim, 1, 1, 1), so no reshaping needed!
        reconstruction = self.decoder(z)
        
        # Flatten mu/log_var for Loss Calculation functions that expect 2D
        # (Optional: depends on how your loss function is written, but usually safe to flatten here)
        mu_flat = mu.view(mu.size(0), -1)
        log_var_flat = log_var.view(log_var.size(0), -1)
        
        return reconstruction, mu_flat, log_var_flat