"""Generate and visualize sample scenes."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from src.data.dataset import ShapeSceneDataset


def visualize_samples(num_samples: int = 6, canvas_size: int = 256):
    """
    Generate and visualize sample scenes.

    Args:
        num_samples: Number of samples to generate
        canvas_size: Size of the canvas
    """
    # Create dataset
    dataset = ShapeSceneDataset(
        size=num_samples,
        canvas_size=canvas_size,
        min_objects=1,
        max_objects=3,
        seed=42,
    )

    # Create figure
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Generate and display samples
    for i in range(num_samples):
        image_array, caption = dataset.get_raw_sample(i)

        axes[i].imshow(image_array)
        axes[i].set_title(f"Sample {i}\n{caption}", fontsize=10, wrap=True)
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_scenes.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to sample_scenes.png")
    plt.show()


def print_sample_info():
    """Print information about a few samples."""
    dataset = ShapeSceneDataset(size=5, canvas_size=256, seed=42)

    print("Sample Dataset Information:")
    print(f"Dataset size: {len(dataset)}")
    print(f"Canvas size: {dataset.canvas_size}x{dataset.canvas_size}")
    print()

    for i in range(min(5, len(dataset))):
        image, caption = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Caption: {caption}")
        print()


if __name__ == "__main__":
    print("Generating sample scenes...")
    print()
    print_sample_info()
    print()
    visualize_samples(num_samples=9)
