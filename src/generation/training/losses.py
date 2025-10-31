"""Loss functions for VAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VAELoss(nn.Module):
    """Combined loss for Conditional VAE training."""

    def __init__(
        self,
        reconstruction_loss: str = "mse",
        kl_weight: float = 0.001,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 10,
    ):
        """
        Initialize VAE loss.

        Args:
            reconstruction_loss: Type of reconstruction loss ("mse" or "l1")
            kl_weight: Weight for KL divergence term
            kl_annealing: Whether to gradually increase KL weight
            kl_annealing_epochs: Number of epochs to anneal KL weight
        """
        super().__init__()

        self.reconstruction_loss = reconstruction_loss
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.current_epoch = 0

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstructed: Reconstructed images (batch_size, 3, H, W)
            target: Target images (batch_size, 3, H, W)
            mu: Latent distribution mean (batch_size, latent_dim)
            logvar: Latent distribution log variance (batch_size, latent_dim)

        Returns:
            Dictionary with 'total_loss', 'recon_loss', 'kl_loss'
        """
        # Reconstruction loss
        if self.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
        elif self.reconstruction_loss == "l1":
            recon_loss = F.l1_loss(reconstructed, target, reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        # KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / target.size(0)  # Normalize by batch size

        # Apply KL weight (with optional annealing)
        kl_weight = self._get_kl_weight()
        weighted_kl_loss = kl_weight * kl_loss

        # Total loss
        total_loss = recon_loss + weighted_kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl_loss': weighted_kl_loss,
            'kl_weight': torch.tensor(kl_weight),
        }

    def _get_kl_weight(self) -> float:
        """Get current KL weight (with annealing if enabled)."""
        if not self.kl_annealing:
            return self.kl_weight

        # Linear annealing from 0 to kl_weight over kl_annealing_epochs
        if self.current_epoch >= self.kl_annealing_epochs:
            return self.kl_weight
        else:
            return self.kl_weight * (self.current_epoch / self.kl_annealing_epochs)

    def step_epoch(self):
        """Call this at the end of each epoch for KL annealing."""
        self.current_epoch += 1


class PerceptualLoss(nn.Module):
    """Optional perceptual loss using pre-trained VGG (not implemented yet)."""

    def __init__(self):
        super().__init__()
        # Could implement VGG-based perceptual loss here if needed
        raise NotImplementedError("Perceptual loss not yet implemented")
