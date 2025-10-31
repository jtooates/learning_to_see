"""VAE decoder: latent code + caption embedding to image."""

import torch
import torch.nn as nn


class VAEDecoder(nn.Module):
    """Decoder that generates images from latent codes and caption embeddings."""

    def __init__(
        self,
        latent_dim: int = 128,
        caption_dim: int = 256,
        image_channels: int = 3,
        image_size: int = 256,
        base_channels: int = 64,
    ):
        """
        Initialize the VAE decoder.

        Args:
            latent_dim: Dimension of the latent code
            caption_dim: Dimension of caption embeddings
            image_channels: Number of output image channels (3 for RGB)
            image_size: Size of square output images
            base_channels: Base number of channels
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.caption_dim = caption_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.base_channels = base_channels

        # Starting spatial size (will be upsampled to image_size)
        # With 5 transposed convs: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.initial_spatial_size = image_size // (2 ** 5)  # 8 for 256x256
        self.initial_channels = base_channels * 8
        self.initial_dim = self.initial_spatial_size ** 2 * self.initial_channels

        # Project concatenated [latent, caption] to initial feature map
        combined_dim = latent_dim + caption_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, self.initial_dim),
            nn.ReLU(inplace=True),
        )

        # Transposed convolutional decoder
        self.decoder = nn.Sequential(
            # 512 x 8 x 8 -> 512 x 16 x 16
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 8,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            # 512 x 16 x 16 -> 256 x 32 x 32
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 256 x 32 x 32 -> 128 x 64 x 64
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # 128 x 64 x 64 -> 64 x 128 x 128
            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 64 x 128 x 128 -> 3 x 256 x 256
            nn.ConvTranspose2d(
                base_channels, image_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(
        self, latent_code: torch.Tensor, caption_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate images from latent codes and caption embeddings.

        Args:
            latent_code: Latent codes of shape (batch_size, latent_dim)
            caption_embedding: Caption embeddings of shape (batch_size, caption_dim)

        Returns:
            Generated images of shape (batch_size, 3, image_size, image_size)
        """
        # Concatenate latent code and caption embedding
        combined = torch.cat([latent_code, caption_embedding], dim=1)

        # Project to initial feature map dimension
        projected = self.fc(combined)

        # Reshape to spatial feature map
        # (batch_size, initial_channels, initial_spatial_size, initial_spatial_size)
        feature_map = projected.view(
            -1, self.initial_channels, self.initial_spatial_size, self.initial_spatial_size
        )

        # Pass through transposed convolutions
        generated_images = self.decoder(feature_map)

        return generated_images
