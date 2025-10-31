"""Evaluation metrics for image generation."""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute evaluation metrics between generated and target images.

    Args:
        generated: Generated images of shape (batch_size, 3, H, W)
        target: Target images of shape (batch_size, 3, H, W)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Mean Squared Error
    mse = F.mse_loss(generated, target).item()
    metrics['mse'] = mse

    # Mean Absolute Error (L1)
    mae = F.l1_loss(generated, target).item()
    metrics['mae'] = mae

    # Peak Signal-to-Noise Ratio (PSNR)
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    # Assuming images in [0, 1] range, MAX = 1
    if mse > 0:
        psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
        metrics['psnr'] = psnr.item()
    else:
        metrics['psnr'] = float('inf')

    return metrics


def compute_pixel_accuracy(
    generated: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute pixel-wise accuracy (percentage of pixels within threshold).

    Args:
        generated: Generated images of shape (batch_size, 3, H, W)
        target: Target images of shape (batch_size, 3, H, W)
        threshold: Threshold for considering pixels as matching

    Returns:
        Pixel accuracy as a float
    """
    diff = torch.abs(generated - target)
    accurate = (diff < threshold).float()
    accuracy = accurate.mean().item()
    return accuracy
