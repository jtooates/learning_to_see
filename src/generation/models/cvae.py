"""Conditional VAE model for caption-to-image generation."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.generation.models.text_encoder import TextEncoder
from src.generation.models.vae_encoder import VAEEncoder
from src.generation.models.vae_decoder import VAEDecoder


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for text-to-image generation."""

    def __init__(
        self,
        vocab_size: int,
        image_size: int = 256,
        image_channels: int = 3,
        latent_dim: int = 128,
        caption_dim: int = 256,
        base_channels: int = 64,
        text_embedding_dim: int = 128,
        text_hidden_dim: int = 256,
    ):
        """
        Initialize the Conditional VAE.

        Args:
            vocab_size: Size of the caption vocabulary
            image_size: Size of square images
            image_channels: Number of image channels (3 for RGB)
            latent_dim: Dimension of latent space
            caption_dim: Dimension of caption embeddings
            base_channels: Base number of channels in conv layers
            text_embedding_dim: Word embedding dimension
            text_hidden_dim: LSTM hidden dimension
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.caption_dim = caption_dim

        # Text encoder: tokens -> caption embedding
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            output_dim=caption_dim,
        )

        # VAE encoder: image -> latent distribution parameters
        self.vae_encoder = VAEEncoder(
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )

        # VAE decoder: [latent, caption] -> image
        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,
            caption_dim=caption_dim,
            image_channels=image_channels,
            image_size=image_size,
            base_channels=base_channels,
        )

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode caption tokens to embeddings.

        Args:
            token_ids: Token indices of shape (batch_size, seq_length)

        Returns:
            Caption embeddings of shape (batch_size, caption_dim)
        """
        return self.text_encoder(token_ids)

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters.

        Args:
            images: Images of shape (batch_size, 3, image_size, image_size)

        Returns:
            Tuple of (mu, logvar) each of shape (batch_size, latent_dim)
        """
        return self.vae_encoder(images)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            mu: Mean of shape (batch_size, latent_dim)
            logvar: Log variance of shape (batch_size, latent_dim)

        Returns:
            Sampled latent code of shape (batch_size, latent_dim)
        """
        return self.vae_encoder.reparameterize(mu, logvar)

    def decode(
        self, latent_code: torch.Tensor, caption_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent code and caption to image.

        Args:
            latent_code: Latent codes of shape (batch_size, latent_dim)
            caption_embedding: Caption embeddings of shape (batch_size, caption_dim)

        Returns:
            Generated images of shape (batch_size, 3, image_size, image_size)
        """
        return self.vae_decoder(latent_code, caption_embedding)

    def forward(
        self, images: torch.Tensor, token_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model (for training).

        Args:
            images: Input images of shape (batch_size, 3, image_size, image_size)
            token_ids: Caption token indices of shape (batch_size, seq_length)

        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed images
                - 'mu': Latent distribution mean
                - 'logvar': Latent distribution log variance
                - 'caption_embedding': Caption embeddings
        """
        # Encode caption
        caption_embedding = self.encode_text(token_ids)

        # Encode image to latent parameters
        mu, logvar = self.encode_image(images)

        # Sample latent code
        latent_code = self.reparameterize(mu, logvar)

        # Decode to reconstruct image
        reconstructed = self.decode(latent_code, caption_embedding)

        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'caption_embedding': caption_embedding,
        }

    def generate(
        self, token_ids: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate images from captions (inference mode).

        Args:
            token_ids: Caption token indices of shape (batch_size, seq_length)
            num_samples: Number of samples to generate per caption

        Returns:
            Generated images of shape (batch_size * num_samples, 3, image_size, image_size)
        """
        self.eval()
        with torch.no_grad():
            # Encode caption
            caption_embedding = self.encode_text(token_ids)

            # Repeat caption embedding for multiple samples
            if num_samples > 1:
                caption_embedding = caption_embedding.repeat_interleave(num_samples, dim=0)

            # Sample random latent codes from standard normal
            batch_size = caption_embedding.size(0)
            latent_code = torch.randn(
                batch_size, self.latent_dim, device=caption_embedding.device
            )

            # Generate images
            generated = self.decode(latent_code, caption_embedding)

        return generated

    def reconstruct(
        self, images: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct images (for evaluation).

        Args:
            images: Input images of shape (batch_size, 3, image_size, image_size)
            token_ids: Caption token indices of shape (batch_size, seq_length)

        Returns:
            Reconstructed images of shape (batch_size, 3, image_size, image_size)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, token_ids)
            return outputs['reconstructed']
