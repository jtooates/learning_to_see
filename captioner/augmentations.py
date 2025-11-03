"""Data augmentation transforms for robust captioning."""
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Optional
import random


class CaptionerAugmentation(nn.Module):
    """Comprehensive augmentation pipeline for vision captioner.

    Applies:
    - Random resized crop
    - Horizontal flip
    - Small affine transforms
    - Gaussian blur
    - JPEG compression simulation
    - Color jitter
    - Additive Gaussian noise
    - Cutout
    """

    def __init__(self,
                 image_size: int = 64,
                 scale: tuple = (0.8, 1.0),
                 flip_prob: float = 0.5,
                 rotation_degrees: float = 5.0,
                 blur_sigma: tuple = (0.1, 1.5),
                 jpeg_quality: tuple = (30, 70),
                 noise_std: float = 0.02,
                 cutout_holes: int = 2,
                 cutout_size: int = 8):
        """Initialize augmentation pipeline.

        Args:
            image_size: Target image size
            scale: Range for random resized crop
            flip_prob: Probability of horizontal flip
            rotation_degrees: Max rotation in degrees
            blur_sigma: Range for Gaussian blur sigma
            jpeg_quality: Range for JPEG compression quality
            noise_std: Standard deviation for Gaussian noise
            cutout_holes: Number of cutout holes
            cutout_size: Size of each cutout hole
        """
        super().__init__()
        self.image_size = image_size
        self.scale = scale
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.blur_sigma = blur_sigma
        self.jpeg_quality = jpeg_quality
        self.noise_std = noise_std
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size

        # Color jitter (light)
        self.color_jitter = T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentations.

        Args:
            image: Input image tensor (C, H, W) in [0, 1]

        Returns:
            Augmented image tensor (C, H, W) in [0, 1]
        """
        # Random resized crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=(0.95, 1.05)
        )
        image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size))

        # Horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)

        # Small affine (rotation)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = TF.rotate(image, angle)

        # Color jitter
        image = self.color_jitter(image)

        # Gaussian blur
        if random.random() < 0.5:
            sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            kernel_size = int(2 * round(2 * sigma) + 1)  # Ensure odd
            image = TF.gaussian_blur(image, kernel_size, sigma)

        # JPEG compression (simulate by quantizing)
        if random.random() < 0.5:
            quality_factor = random.uniform(0.3, 0.7)  # Lower = more compression
            image = self._simulate_jpeg(image, quality_factor)

        # Additive Gaussian noise
        noise = torch.randn_like(image) * self.noise_std
        image = image + noise
        image = torch.clamp(image, 0, 1)

        # Cutout
        for _ in range(random.randint(1, self.cutout_holes)):
            image = self._apply_cutout(image)

        return image

    def _simulate_jpeg(self, image: torch.Tensor, quality: float) -> torch.Tensor:
        """Simulate JPEG compression by quantizing.

        Args:
            image: Input image (C, H, W)
            quality: Quality factor in [0, 1]

        Returns:
            Quantized image
        """
        # Simple quantization to simulate compression artifacts
        levels = int(128 * quality + 32)  # 32-160 levels
        image = (image * levels).round() / levels
        return torch.clamp(image, 0, 1)

    def _apply_cutout(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random cutout.

        Args:
            image: Input image (C, H, W)

        Returns:
            Image with cutout applied
        """
        C, H, W = image.shape
        size = self.cutout_size

        # Random position
        y = random.randint(0, H - size)
        x = random.randint(0, W - size)

        # Fill with gray
        image[:, y:y+size, x:x+size] = 0.5

        return image


def get_train_augmentation(image_size: int = 64, strong: bool = True) -> CaptionerAugmentation:
    """Get training augmentation pipeline.

    Args:
        image_size: Target image size
        strong: If True, use strong augmentation; else use light

    Returns:
        Augmentation transform
    """
    if strong:
        return CaptionerAugmentation(
            image_size=image_size,
            scale=(0.8, 1.0),
            flip_prob=0.5,
            rotation_degrees=5.0,
            blur_sigma=(0.1, 1.5),
            jpeg_quality=(30, 70),
            noise_std=0.02,
            cutout_holes=2,
            cutout_size=8
        )
    else:
        # Light augmentation
        return CaptionerAugmentation(
            image_size=image_size,
            scale=(0.9, 1.0),
            flip_prob=0.3,
            rotation_degrees=2.0,
            blur_sigma=(0.1, 0.5),
            jpeg_quality=(50, 90),
            noise_std=0.01,
            cutout_holes=1,
            cutout_size=4
        )


def get_eval_transform(image_size: int = 64) -> nn.Module:
    """Get evaluation transform (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Identity transform
    """
    return nn.Identity()
