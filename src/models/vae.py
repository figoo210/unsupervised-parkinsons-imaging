"""
Variational Autoencoder (VAE) model architectures for medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Basic convolutional block for VAE with conv, batch norm, and activation.
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


class VAEEncoder(nn.Module):
    """
    VAE Encoder that outputs mean and log variance for the latent distribution.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize VAE Encoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(VAEEncoder, self).__init__()
        
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
        self.fc_mu = nn.Linear(512, latent_dim)      # Mean
        self.fc_log_var = nn.Linear(512, latent_dim) # Log variance
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, D, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of latent distribution
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
        
        # Output mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    VAE Decoder that reconstructs images from latent representations.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize VAE Decoder.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(VAEDecoder, self).__init__()
        
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
        Forward pass through VAE decoder.
        
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


class VAE(nn.Module):
    """
    Complete Variational Autoencoder combining encoder and decoder.
    """
    
    def __init__(self, latent_dim: int = 256):
        """
        Initialize VAE.
        
        Args:
            latent_dim (int): Dimension of the latent space
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x (torch.Tensor): Input volume
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed volume, mu, log_var
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def generate(self, z: torch.Tensor = None, num_samples: int = 1) -> torch.Tensor:
        """
        Generate new samples from the VAE.
        
        Args:
            z (torch.Tensor, optional): Latent vectors. If None, samples from standard normal.
            num_samples (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated samples
        """
        if z is None:
            device = next(self.parameters()).device
            z = torch.randn(num_samples, self.latent_dim).to(device)
        
        return self.decoder(z)


class VAELoss(nn.Module):
    """
    VAE Loss function combining reconstruction loss and KL divergence.
    """
    
    def __init__(self, beta: float = 0.0005, beta_warmup_steps: int = 5000, 
                 free_bits: float = 3.0):
        """
        Initialize VAE Loss.
        
        Args:
            beta (float): Weight for KL divergence term
            beta_warmup_steps (int): Number of steps to warm up beta
            free_bits (float): Free bits for KL divergence
        """
        super(VAELoss, self).__init__()
        self.beta = beta
        self.beta_warmup_steps = beta_warmup_steps
        self.free_bits = free_bits
        self.step = 0
        
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, 
                mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss, KL loss
        """
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        
        # Apply free bits
        if self.free_bits > 0:
            kl_loss = torch.max(kl_loss, torch.tensor(self.free_bits).to(kl_loss.device))
        
        # Beta warmup
        if self.beta_warmup_steps > 0:
            current_beta = min(self.beta, self.beta * (self.step / self.beta_warmup_steps))
        else:
            current_beta = self.beta
        
        # Total loss
        total_loss = recon_loss + current_beta * kl_loss
        
        self.step += 1
        
        return total_loss, recon_loss, kl_loss


def test_vae(batch_size: int = 2, latent_dim: int = 256) -> None:
    """
    Test VAE with random input.
    
    Args:
        batch_size (int): Batch size for testing
        latent_dim (int): Latent dimension
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and test input
    model = VAE(latent_dim=latent_dim).to(device)
    test_input = torch.randn(batch_size, 1, 64, 128, 128).to(device)
    loss_fn = VAELoss()
    
    print(f"Input shape: {test_input.shape}")
    
    # Test encoding
    mu, log_var = model.encode(test_input)
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Test full forward pass
    recon_x, mu, log_var = model(test_input)
    print(f"Reconstruction shape: {recon_x.shape}")
    
    # Test loss computation
    total_loss, recon_loss, kl_loss = loss_fn(recon_x, test_input, mu, log_var)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test generation
    generated = model.generate(num_samples=batch_size)
    print(f"Generated shape: {generated.shape}")
    
    print("VAE test completed successfully!")


if __name__ == "__main__":
    test_vae() 