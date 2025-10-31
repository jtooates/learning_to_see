"""Visualization utilities for image generation."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor image to numpy array for visualization.

    Args:
        image: Tensor of shape (3, H, W) in [0, 1]

    Returns:
        Numpy array of shape (H, W, 3) in [0, 255]
    """
    # Move to CPU and convert to numpy
    image = image.cpu().detach()

    # Clamp to [0, 1]
    image = torch.clamp(image, 0, 1)

    # Convert to (H, W, 3)
    image = image.permute(1, 2, 0).numpy()

    # Scale to [0, 255]
    image = (image * 255).astype(np.uint8)

    return image


def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    captions: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 8,
):
    """
    Visualize original vs reconstructed images.

    Args:
        original: Original images (batch_size, 3, H, W)
        reconstructed: Reconstructed images (batch_size, 3, H, W)
        captions: List of captions
        save_path: Path to save figure
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, original.size(0))

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original
        orig_img = denormalize_image(original[i])
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original\n{captions[i]}", fontsize=8)
        axes[i, 0].axis('off')

        # Reconstructed
        recon_img = denormalize_image(reconstructed[i])
        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title(f"Reconstructed", fontsize=8)
        axes[i, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_generation(
    generated: torch.Tensor,
    captions: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 8,
):
    """
    Visualize generated images from captions.

    Args:
        generated: Generated images (batch_size, 3, H, W)
        captions: List of captions
        save_path: Path to save figure
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, generated.size(0))

    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        img = denormalize_image(generated[i])
        axes[i].imshow(img)
        axes[i].set_title(f"{captions[i]}", fontsize=8, wrap=True)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: List[dict],
    val_losses: Optional[List[dict]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot training curves.

    Args:
        train_losses: List of training loss dictionaries
        val_losses: List of validation loss dictionaries
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(epochs, [d['total_loss'] for d in train_losses], label='Train')
    if val_losses:
        axes[0].plot(epochs, [d['val_total_loss'] for d in val_losses], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Reconstruction loss
    axes[1].plot(epochs, [d['recon_loss'] for d in train_losses], label='Train')
    if val_losses:
        axes[1].plot(epochs, [d['val_recon_loss'] for d in val_losses], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    # KL loss
    axes[2].plot(epochs, [d['kl_loss'] for d in train_losses], label='Train')
    if val_losses:
        axes[2].plot(epochs, [d['val_kl_loss'] for d in val_losses], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()
