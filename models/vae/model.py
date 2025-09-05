import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class VAE(nn.Module):
    """
    3D Variational Autoencoder optimized for 64×128×128 medical volumes with
    memory-efficient implementation for NVIDIA 4070Ti.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        # Enable CUDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to enable backpropagation through sampling.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encode input to get mean and variance of latent distribution
        mu, log_var = self.encoder(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Decode sampled latent vector
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    def encode(self, x):
        """Encode input to latent parameters without sampling"""
        mu, log_var = self.encoder(x)
        return mu, log_var

    def generate(self, z=None, num_samples=1):
        """Generate samples from latent space or random samples"""
        device = next(self.parameters()).device

        if z is None:
            # Generate random latent vectors
            z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode latent vectors to generate samples
        with torch.no_grad():
            samples = self.decoder(z)

        return samples


class ConvBlock(nn.Module):
    """Memory-efficient convolutional block with batch normalization and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))  # inplace ReLU for memory efficiency
        ]))

    def forward(self, x):
        return self.block(x)


class VAEEncoder(nn.Module):
    """3D Encoder network with probabilistic latent space."""
    def __init__(self, latent_dim=256):
        super().__init__()

        # Initial feature extraction
        self.init_conv = ConvBlock(1, 16)  # 64×128×128 -> same size

        # Downsampling path with progressive channel increase
        self.down1 = nn.Sequential(
            ConvBlock(16, 32, stride=2),    # -> 32×64×64
            ConvBlock(32, 32)
        )

        self.down2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),    # -> 16×32×32
            ConvBlock(64, 64)
        )

        self.down3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),   # -> 8×16×16
            ConvBlock(128, 128)
        )

        self.down4 = nn.Sequential(
            ConvBlock(128, 256, stride=2),  # -> 4×8×8
            ConvBlock(256, 256)
        )

        # Project to latent parameters
        self.flatten_size = 256 * 4 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)

        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # Flatten and project to latent parameters
        flat = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)

        return mu, log_var


class VAEDecoder(nn.Module):
    """3D Decoder network without enhanced capacity - matching the saved model."""
    def __init__(self, latent_dim=256):
        super().__init__()

        self.flatten_size = 256 * 4 * 8 * 8
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Upsampling path without enhanced capacity (matches the saved model)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(256, 128),  # Original channel size from saved model
            ConvBlock(128, 128)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(128, 64),   # Original channel size from saved model
            ConvBlock(64, 64)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(64, 32),    # Original channel size from saved model
            ConvBlock(32, 32)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(32, 16),    # Original channel size from saved model
            ConvBlock(16, 16)
        )

        # No refinement layer in the original model

        # Final convolution
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, z):
        # Reshape from latent space
        x = self.fc(z)
        x = x.view(-1, 256, 4, 8, 8)

        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final convolution (no activation to allow for negative values if needed)
        x = self.final_conv(x)

        return x


# Custom loss function for VAE
class VAELoss(nn.Module):
    """
    VAE loss combining reconstruction loss and KL divergence
    with beta parameter to control the trade-off and free bits approach.
    """
    def __init__(self, beta=0.0005, beta_warmup_steps=5000, free_bits=3.0):
        super().__init__()
        self.beta_base = beta
        self.beta_warmup_steps = beta_warmup_steps
        self.current_step = 0
        self.free_bits = free_bits  # Free bits parameter to prevent posterior collapse

    def forward(self, recon_x, x, mu, log_var):
        """
        Calculate VAE loss with enhanced beta annealing and free bits.
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence with free bits
        # Instead of penalizing all KL divergence, allow some "free bits"
        kl_raw = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        # Apply free bits: only penalize when KL is above threshold
        kl_free = torch.maximum(kl_raw - self.free_bits, torch.zeros_like(kl_raw))
        kl_loss = torch.mean(kl_free)

        # Calculate beta with cyclical warmup
        # Cyclical annealing gives the model periods to focus on reconstruction
        if self.beta_warmup_steps > 0:
            # Implement cyclical annealing
            cycle_length = self.beta_warmup_steps // 2
            cycle_position = (self.current_step % cycle_length) / cycle_length
            if self.current_step // cycle_length % 2 == 0:  # Even cycles: warmup
                beta = self.beta_base * cycle_position
            else:  # Odd cycles: constant
                beta = self.beta_base
        else:
            beta = self.beta_base

        # Increment step counter
        self.current_step += 1

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss, beta

