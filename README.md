# Shape Scene Generator

A synthetic data generator for vision-language machine learning tasks that creates (image, caption) pairs depicting 2D geometric scenes.

## Overview

This project generates synthetic training data for:
- **Image captioning models**: Learning to map images → captions
- **Image generation models**: Learning to map captions → images

The generator creates scenes with various 2D shapes (circles, squares, triangles, rectangles) that have different:
- Colors (red, blue, green, yellow, etc.)
- Sizes (small, medium, large)
- Quantities (1-N objects)
- Spatial relationships (above, below, left of, right of, etc.)

## Features

- Declarative scene representation
- Natural language caption generation with word order variation
- Photorealistic 2D shape rendering
- PyTorch Dataset wrapper for easy integration with training pipelines
- Configurable generation parameters

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data.dataset import ShapeSceneDataset

# Create dataset
dataset = ShapeSceneDataset(size=1000, canvas_size=256)

# Get a sample
image, caption = dataset[0]
print(f"Caption: {caption}")
print(f"Image shape: {image.shape}")  # (3, 256, 256)
```

## Project Structure

```
src/
├── data/           # Data generation components
│   ├── scene.py    # Scene representation
│   ├── generator.py # Scene generation logic
│   ├── caption.py  # Caption generation
│   ├── renderer.py # Image rendering
│   └── dataset.py  # PyTorch Dataset
├── captioning/     # Future: image → caption models
└── generation/     # ✅ Caption → image generation (Conditional VAE)
    ├── models/     # Model architectures
    ├── training/   # Training loop, losses, metrics
    └── utils/      # Tokenizer, visualization
```

## Example Outputs

The generator creates diverse scenes like:
- "a large blue square above 3 small red triangles"
- "2 medium green circles to the left of a yellow rectangle"
- "a small purple triangle below 4 large orange squares"

## Image Generation Model

This project includes a **Conditional VAE** model for caption-to-image generation:

### Quick Start

```bash
# Test the model architecture
python scripts/test_model.py

# Train the model
python scripts/train_generator.py
```

### Model Architecture

- **Text Encoder**: LSTM-based encoder for captions
- **VAE Encoder**: CNN encoder (image → latent distribution)
- **VAE Decoder**: Transposed CNN decoder (caption + latent → image)
- **Loss**: Reconstruction loss (MSE/L1) + KL divergence

### Training

The model trains on synthetic (image, caption) pairs and learns to:
1. **Reconstruct** images from captions (with image as reference)
2. **Generate** novel images from captions alone

See `USAGE.md` for detailed documentation.

## License

MIT
