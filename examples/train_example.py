"""Example training script demonstrating dataset usage."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.data.dataset import ShapeSceneDataset, collate_fn


def example_training_loop():
    """Demonstrate how to use the dataset in a training loop."""

    # Create dataset
    dataset = ShapeSceneDataset(
        size=100,
        canvas_size=256,
        min_objects=1,
        max_objects=3,
        seed=42,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Set to 0 for simplicity; increase for performance
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print()

    # Simulate training for a few batches
    print("Simulating training loop...")
    for batch_idx, (images, captions) in enumerate(dataloader):
        if batch_idx >= 3:  # Only show first 3 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Number of captions: {len(captions)}")
        print(f"  Sample captions:")
        for i, caption in enumerate(captions[:3]):  # Show first 3
            print(f"    {i+1}. {caption}")

        # Here you would typically:
        # 1. Forward pass through your model
        # 2. Compute loss
        # 3. Backward pass
        # 4. Update weights

    print("\nTraining loop demonstration complete!")


def example_image_captioning():
    """Example setup for image captioning task (image -> text)."""
    print("\n" + "="*60)
    print("Image Captioning Task Setup")
    print("="*60)

    dataset = ShapeSceneDataset(size=1000, canvas_size=256, seed=42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    print("In this task, you would:")
    print("1. Input: Image tensor (3, 256, 256)")
    print("2. Model: CNN encoder + LSTM/Transformer decoder")
    print("3. Output: Generated caption text")
    print("4. Loss: Cross-entropy on caption tokens")


def example_image_generation():
    """Example setup for image generation task (text -> image)."""
    print("\n" + "="*60)
    print("Image Generation Task Setup")
    print("="*60)

    dataset = ShapeSceneDataset(size=1000, canvas_size=256, seed=42)

    print(f"Total samples: {len(dataset)}")
    print()
    print("In this task, you would:")
    print("1. Input: Caption text (encoded as tokens or embeddings)")
    print("2. Model: Text encoder + GAN/Diffusion/VAE decoder")
    print("3. Output: Generated image (3, 256, 256)")
    print("4. Loss: Reconstruction loss (L1/L2) or adversarial loss")


if __name__ == "__main__":
    example_training_loop()
    example_image_captioning()
    example_image_generation()
