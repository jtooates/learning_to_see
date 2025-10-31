# Usage Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.data.dataset import ShapeSceneDataset

# Create dataset
dataset = ShapeSceneDataset(size=1000, canvas_size=256, seed=42)

# Get a sample
image, caption = dataset[0]
print(f"Caption: {caption}")
print(f"Image shape: {image.shape}")  # torch.Size([3, 256, 256])
```

### Using with PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from src.data.dataset import ShapeSceneDataset, collate_fn

# Create dataset
dataset = ShapeSceneDataset(size=1000, canvas_size=256)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# Iterate through batches
for images, captions in dataloader:
    # images: torch.Tensor of shape (batch_size, 3, 256, 256)
    # captions: list of strings
    pass
```

### Customization

```python
# Customize scene complexity
dataset = ShapeSceneDataset(
    size=1000,
    canvas_size=512,           # Larger images
    min_objects=2,             # At least 2 object groups
    max_objects=5,             # Up to 5 object groups
    min_quantity=1,            # At least 1 shape per group
    max_quantity=6,            # Up to 6 shapes per group
    relationship_probability=0.9,  # Higher chance of relationships
    seed=42,                   # For reproducibility
)
```

### Generating Individual Components

```python
from src.data.generator import SceneGenerator
from src.data.caption import CaptionGenerator
from src.data.renderer import SceneRenderer

# Create generators
scene_gen = SceneGenerator(canvas_size=256, seed=42)
caption_gen = CaptionGenerator(seed=42)
renderer = SceneRenderer(canvas_size=256)

# Generate a scene
scene = scene_gen.generate_scene()

# Generate caption
caption = caption_gen.generate_caption(scene)

# Render image
image = renderer.render(scene)  # Returns PIL Image
image_array = renderer.render_to_array(scene)  # Returns numpy array
```

### Visualization

```python
# Run the example script
python examples/generate_samples.py

# This will:
# 1. Generate 9 sample scenes
# 2. Display them in a grid
# 3. Save to 'sample_scenes.png'
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/data/test_scene.py

# Run with coverage
pytest --cov=src --cov-report=html
```

## Example Captions

The generator creates varied captions like:

- "a large blue square above 3 small red triangles"
- "2 medium green circles to the left of a yellow rectangle"
- "a small purple triangle below 4 large orange squares"
- "a pink triangle (medium), with 3 small yellow squares left of it"

## Project Structure

```
src/
├── data/           # Data generation
│   ├── scene.py    # Scene representation
│   ├── generator.py # Scene generation
│   ├── caption.py  # Caption generation
│   ├── renderer.py # Image rendering
│   └── dataset.py  # PyTorch Dataset
├── captioning/     # Future: image → caption models
└── generation/     # Future: caption → image models
```

## Next Steps

Once you have the data generator working, you can:

1. **Train an Image Captioning Model** (image → caption)
   - Use a CNN encoder (ResNet, ViT) for images
   - Use an LSTM or Transformer decoder for captions
   - Optimize with cross-entropy loss

2. **Train an Image Generation Model** (caption → image)
   - Use text encoder (BERT, CLIP)
   - Use GAN, VAE, or Diffusion model for generation
   - Optimize with reconstruction or adversarial loss

## Tips

- Start with small canvas sizes (256x256) for faster iteration
- Use `seed` parameter for reproducibility during development
- Increase `num_workers` in DataLoader for faster data loading
- Use `get_raw_sample()` for visualization (returns numpy array)
- Adjust complexity parameters to match your model's capacity
