"""VAE encoder: image to latent distribution parameters."""

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """CNN encoder that maps images to latent distribution parameters (μ, log_σ)."""

    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 256,
        latent_dim: int = 128,
        base_channels: int = 64,
    ):
        """
        Initialize the VAE encoder.

        Args:
            image_channels: Number of input image channels (3 for RGB)
            image_size: Size of square input images
            latent_dim: Dimension of the latent space
            base_channels: Base number of channels (will be multiplied in deeper layers)
        """
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Calculate the spatial size after convolutions
        # With 5 layers of stride-2 convolutions: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.final_spatial_size = image_size // (2 ** 5)
        self.final_channels = base_channels * 8
        self.flatten_dim = self.final_spatial_size ** 2 * self.final_channels

        # Convolutional encoder
        self.encoder = nn.Sequential(
            # 3 x 256 x 256 -> 64 x 128 x 128
            nn.Conv2d(image_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 128 x 128 -> 128 x 64 x 64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 64 x 64 -> 256 x 32 x 32
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 32 x 32 -> 512 x 16 x 16
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 x 16 x 16 -> 512 x 8 x 8
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters.

        Args:
            images: Input images of shape (batch_size, 3, image_size, image_size)

        Returns:
            Tuple of (mu, logvar) each of shape (batch_size, latent_dim)
            - mu: Mean of latent distribution
            - logvar: Log variance of latent distribution
        """
        # Pass through convolutional layers
        encoded = self.encoder(images)

        # Flatten: (batch_size, flatten_dim)
        flattened = encoded.view(encoded.size(0), -1)

        # Get distribution parameters
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, sigma) using N(0, 1).

        Args:
            mu: Mean of shape (batch_size, latent_dim)
            logvar: Log variance of shape (batch_size, latent_dim)

        Returns:
            Sampled latent code of shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
