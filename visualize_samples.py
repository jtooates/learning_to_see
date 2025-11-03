"""Generate and visualize example caption/image pairs."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import random

from dsl.parser import SceneParser
from dsl.canonicalize import to_canonical
from dsl.splits import enumerate_all_samples
from render.renderer import SceneRenderer


def create_visualization(n_examples: int = 10, seed: int = 42, output_path: str = "examples.png"):
    """Generate and visualize n example caption/image pairs.

    Args:
        n_examples: Number of examples to generate
        seed: Random seed for reproducibility
        output_path: Path to save the output PNG
    """
    # Set seed
    random.seed(seed)

    # Initialize components
    parser = SceneParser()
    renderer = SceneRenderer(width=64, height=64, seed=seed)

    # Sample some scene graphs
    all_graphs = enumerate_all_samples()
    sampled_graphs = random.sample(all_graphs, n_examples)

    # Generate images and captions
    examples = []
    for graph in sampled_graphs:
        try:
            # Get canonical text
            caption = to_canonical(graph)

            # Render image
            image, meta = renderer.render(graph)

            examples.append((image, caption))
        except Exception as e:
            print(f"Failed to render: {e}")
            continue

    # Create visualization
    n_examples = len(examples)
    n_cols = 2
    n_rows = (n_examples + n_cols - 1) // n_cols

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 3 * n_rows))
    fig.suptitle('Synthetic Scene Dataset Examples', fontsize=16, fontweight='bold', y=0.995)

    for idx, (image, caption) in enumerate(examples):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        # Display image
        ax.imshow(np.array(image))
        ax.axis('off')

        # Add caption as title
        ax.set_title(caption, fontsize=10, pad=10, wrap=True)

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    print(f"Generated {n_examples} examples")

    # Also create a detailed version with metadata
    create_detailed_visualization(examples[:4], "examples_detailed.png")


def create_detailed_visualization(examples, output_path: str = "examples_detailed.png"):
    """Create a more detailed visualization showing bounding boxes and metadata.

    Args:
        examples: List of (image, caption) tuples
        output_path: Path to save the output PNG
    """
    n_examples = len(examples)

    # Create figure
    fig = plt.figure(figsize=(14, 3.5 * n_examples))
    fig.suptitle('Detailed Scene Examples with Metadata', fontsize=16, fontweight='bold', y=0.995)

    # Re-render with metadata to show bboxes
    parser = SceneParser()
    renderer = SceneRenderer(width=64, height=64, seed=42)

    for idx, (image, caption) in enumerate(examples):
        # Parse the caption to get scene graph
        scene_graph = parser.parse(caption)

        # Render to get metadata
        _, meta = renderer.render(scene_graph)

        # Create subplot with 2 columns: original and annotated
        ax1 = plt.subplot(n_examples, 2, idx * 2 + 1)
        ax2 = plt.subplot(n_examples, 2, idx * 2 + 2)

        # Original image
        ax1.imshow(np.array(image))
        ax1.axis('off')
        ax1.set_title('Original Image', fontsize=9)

        # Annotated image with bboxes
        img_array = np.array(image)
        ax2.imshow(img_array)
        ax2.axis('off')
        ax2.set_title('With Bounding Boxes', fontsize=9)

        # Draw bounding boxes and labels
        for inst in meta.instances:
            x0, y0, x1, y1 = inst.bbox
            width = x1 - x0
            height = y1 - y0

            # Choose color for bbox
            colors = {'red': 'red', 'green': 'green', 'blue': 'blue', 'yellow': 'gold'}
            edge_color = colors.get(inst.color, 'white')

            # Draw rectangle
            rect = Rectangle((x0, y0), width, height,
                           linewidth=1.5, edgecolor=edge_color,
                           facecolor='none', linestyle='--')
            ax2.add_patch(rect)

            # Add label
            label = f"{inst.color} {inst.shape}"
            ax2.text(x0, y0 - 2, label, fontsize=6, color=edge_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Add caption below both images
        fig.text(0.5, 1 - (idx + 0.85) / n_examples, caption,
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Detailed visualization saved to: {output_path}")


if __name__ == '__main__':
    # Generate examples
    create_visualization(n_examples=10, seed=42, output_path="examples.png")

    print("\nExample captions generated:")

    # Also print some statistics
    all_graphs = enumerate_all_samples()
    print(f"\nDataset statistics:")
    print(f"  Total possible scenes: {len(all_graphs)}")

    # Count by type
    count_sent = sum(1 for g in all_graphs if len(g['relations']) == 0)
    rel_sent = sum(1 for g in all_graphs if len(g['relations']) > 0)
    print(f"  COUNT sentences: {count_sent}")
    print(f"  RELATION sentences: {rel_sent}")

    # Count by relation type
    rel_types = {}
    for g in all_graphs:
        for rel in g.get('relations', []):
            rel_type = rel['type']
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

    print(f"\n  Relations breakdown:")
    for rel_type, count in sorted(rel_types.items()):
        print(f"    {rel_type}: {count}")
