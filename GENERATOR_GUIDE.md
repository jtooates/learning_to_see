# Image Generation Model Guide

## Overview

This guide covers the **Conditional VAE** model for caption-to-image generation in the shape scene project.

## Architecture

### Components

1. **Text Encoder** (`src/generation/models/text_encoder.py`)
   - LSTM-based encoder
   - Input: Token sequences (batch_size, seq_length)
   - Output: Caption embeddings (batch_size, 256)
   - Handles variable-length captions with bidirectional LSTM

2. **VAE Encoder** (`src/generation/models/vae_encoder.py`)
   - CNN-based image encoder
   - Input: Images (batch_size, 3, 256, 256)
   - Output: Latent distribution parameters μ, log_σ (batch_size, 128)
   - 5 convolutional layers with stride-2 downsampling

3. **VAE Decoder** (`src/generation/models/vae_decoder.py`)
   - Transposed CNN decoder
   - Input: [Caption embedding (256) + Latent code (128)]
   - Output: Generated images (batch_size, 3, 256, 256)
   - 5 transposed convolutional layers with stride-2 upsampling

4. **Conditional VAE** (`src/generation/models/cvae.py`)
   - Combines all components
   - Training mode: Encode image → sample latent → decode with caption
   - Inference mode: Sample random latent → decode with caption

### Model Parameters

- Total parameters: ~37 million
- Latent dimension: 128
- Caption embedding dimension: 256
- Base channels: 64 (multiplied in deeper layers)

## Training

### Loss Function

```python
Total Loss = Reconstruction Loss + β * KL Divergence

where:
- Reconstruction Loss: MSE or L1 between generated and real image
- KL Divergence: Regularizes latent space to be close to N(0, 1)
- β: KL weight (starts at 0.001, anneals over first 10 epochs)
```

### Hyperparameters

```python
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
LATENT_DIM = 128
CAPTION_DIM = 256
KL_WEIGHT = 0.001
```

### Training Script

```bash
python scripts/train_generator.py
```

This will:
1. Create dataset (10,000 samples)
2. Build tokenizer vocabulary
3. Split into train/val (90%/10%)
4. Train for 50 epochs
5. Save checkpoints every 10 epochs
6. Generate visualizations

### Output Files

- `checkpoints/` - Model checkpoints
- `checkpoints/tokenizer_vocab.json` - Tokenizer vocabulary
- `training_curves.png` - Loss curves over training
- `reconstruction.png` - Original vs reconstructed images
- `generated_samples.png` - Images generated from text only

## Usage

### Training

```python
from torch.utils.data import DataLoader
from src.data.dataset import ShapeSceneDataset
from src.generation.models.cvae import ConditionalVAE
from src.generation.training.trainer import Trainer
from src.generation.training.losses import VAELoss
from src.generation.utils.tokenizer import CaptionTokenizer

# Create dataset and tokenizer
dataset = ShapeSceneDataset(size=10000, canvas_size=256)
tokenizer = CaptionTokenizer()
tokenizer.fit(captions)

# Create model
model = ConditionalVAE(
    vocab_size=tokenizer.get_vocab_size(),
    image_size=256,
    latent_dim=128,
    caption_dim=256,
)

# Create trainer and train
trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device)
trainer.train(num_epochs=50)
```

### Inference (Generation)

```python
# Load trained model
model = ConditionalVAE(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenize caption
caption = "a large blue square above 3 small red circles"
tokens = torch.tensor(tokenizer.encode(caption)).unsqueeze(0)

# Generate image
with torch.no_grad():
    generated_image = model.generate(tokens, num_samples=1)

# generated_image: (1, 3, 256, 256) in [0, 1] range
```

### Reconstruction

```python
# Given an image and its caption
image = ...  # (1, 3, 256, 256)
tokens = ...  # (1, seq_length)

# Reconstruct
with torch.no_grad():
    reconstructed = model.reconstruct(image, tokens)
```

## Tokenizer

The custom word-level tokenizer (`src/generation/utils/tokenizer.py`):

- Vocabulary size: ~50-100 words (very small for this domain)
- Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
- Max sequence length: 32 tokens
- Handles: colors, shapes, sizes, numbers, relationships, articles

Example vocabulary:
```
0: <PAD>
1: <UNK>
2: <SOS>
3: <EOS>
4: a
5: above
6: an
7: and
8: blue
...
```

## Evaluation Metrics

1. **Reconstruction Loss**: How well the model reconstructs training images
2. **MSE/MAE**: Pixel-wise error metrics
3. **Visual Quality**: Subjective assessment of generated images
4. **Caption Consistency**: Check if generated images match caption attributes

## Tips for Training

1. **Start small**: Begin with 1,000 samples to verify the pipeline works
2. **Monitor KL weight**: Should gradually increase to prevent posterior collapse
3. **Check reconstructions**: Should look good before expecting good generations
4. **Adjust β**: If reconstructions are blurry, decrease KL weight
5. **Learning rate**: Start with 1e-4, reduce if training is unstable
6. **Batch size**: 32 works well, increase if you have GPU memory

## Common Issues

### Posterior Collapse
- **Symptom**: KL divergence goes to zero, all generations look the same
- **Fix**: Decrease KL weight or use KL annealing

### Blurry Reconstructions
- **Symptom**: Reconstructed images are blurry
- **Fix**: Decrease KL weight, try L1 loss instead of MSE

### Mode Collapse
- **Symptom**: Model generates similar images for different captions
- **Fix**: Increase training data diversity, adjust caption embedding dimension

### Out of Memory
- **Symptom**: CUDA OOM errors
- **Fix**: Reduce batch size, reduce base_channels, use smaller image size

## Future Improvements

1. **Better Architecture**: Try transformer-based text encoder
2. **GAN Loss**: Add adversarial loss for sharper images
3. **Perceptual Loss**: Use VGG features for better quality
4. **Diffusion Models**: Replace VAE with diffusion for higher quality
5. **Attention**: Add cross-attention between text and image features
6. **Conditioning**: Condition on multiple attributes separately

## File Structure

```
src/generation/
├── models/
│   ├── text_encoder.py      # LSTM caption encoder
│   ├── vae_encoder.py       # Image → latent
│   ├── vae_decoder.py       # [Caption, latent] → image
│   └── cvae.py              # Full Conditional VAE
├── training/
│   ├── losses.py            # VAE loss function
│   ├── trainer.py           # Training loop
│   └── metrics.py           # Evaluation metrics
└── utils/
    ├── tokenizer.py         # Caption tokenization
    └── visualization.py     # Plotting utilities

scripts/
└── train_generator.py       # Main training script
```

## References

- Conditional VAE: Sohn et al. (2015) "Learning Structured Output Representation using Deep Conditional Generative Models"
- VAE: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
