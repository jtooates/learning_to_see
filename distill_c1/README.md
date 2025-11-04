# C1: Renderer Distillation (Text → Image)

This module implements a text-to-image decoder by distilling knowledge from the procedural renderer. The decoder learns to generate 64×64 images from DSL captions using only pixel-level supervision from the renderer.

## Overview

**Goal**: Train a neural network to generate images from text by learning from a procedural renderer (teacher), without requiring a large dataset or pretrained models.

**Approach**:
1. Text encoder (Transformer) converts DSL tokens → 512-d embedding
2. Image decoder (FiLM-conditioned CNN) generates 64×64 RGB images
3. Training uses pixel losses (L1+L2) + TV regularization + optional random perceptual loss

## Architecture

### Text Encoder (`text_encoder.py`)
- **Token embeddings**: vocab_size → 256
- **Positional encodings**: Learned 1D positional embeddings
- **[CLS] token**: Prepended for pooling
- **Transformer**: 4 layers, d_model=256, nhead=4, dim_ff=512, Pre-LN
- **Output projection**: LayerNorm([CLS]) → Linear(256→512) + SiLU
- **Output**: 512-d text embedding

**Parameters**: ~2.4M

### Image Decoder (`decoder.py`)
- **Stem**: Linear(512 → 256×8×8)
- **Stage 1** (8→16): ResBlock(256)×2 + FiLM → Upsample → Conv 256→128
- **Stage 2** (16→32): ResBlock(128)×2 + FiLM → MHSA(4 heads) → Upsample → Conv 128→64
- **Stage 3** (32→64): Upsample → ResBlock(64)×2 + FiLM
- **Head**: Conv 64→3 + Tanh → [-1, 1]

**Key components**:
- **ResBlock**: Conv3×3→GN→SiLU→Conv3×3→GN + skip
- **FiLM**: Feature-wise modulation (γ⊙x + β)
- **MHSA**: Single self-attention layer at 32×32 for global coherence
- **Upsampling**: Nearest-neighbor + Conv (avoids checkerboard artifacts)

**Parameters**: ~8.5M

## Loss Functions (`losses.py`)

1. **Pixel losses**:
   - L1 + L2 (equal weight)
   - Applied between predicted and target images in [-1, 1]

2. **Total Variation (TV)**:
   - Anisotropic TV for smoothness
   - Weight: 1e-5

3. **Random Perceptual loss** (optional):
   - TinyRandNet: 3-layer frozen random CNN
   - Provides weak perceptual bias without pretrained models
   - Weight: 1e-3

**Combined loss**:
```
L = 0.5*L1 + 0.5*L2 + λ_tv*TV + λ_perc*RandPerc
```

## Training (`trainer.py`)

**Optimizer**:
- AdamW (lr=3e-4, weight_decay=0.05, betas=(0.9, 0.999))
- OneCycleLR scheduler with cosine decay
- Warmup: 1000 steps
- Gradient clipping: 1.0

**Features**:
- Automatic Mixed Precision (AMP) for faster training
- Exponential Moving Average (EMA) with decay 0.999
- Evaluation every 2000 steps
- Checkpoint saving (best.pt, last.pt, ema_best.pt)

**Expected training time**: ~6-8 hours on GPU for 100k steps

## Usage

### 1. Generate Data

First, generate synthetic scene data:

```bash
python -m data.gen \
  --out_dir data/scenes \
  --n 6000 \
  --split_strategy random \
  --seed 42
```

### 2. Train Model

**Using the training script**:
```bash
bash scripts/train_distill.sh
```

**Or directly with Python**:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1 \
  --steps 100000 \
  --batch 192 \
  --lr 3e-4 \
  --tv 1e-5 \
  --perc 1e-3 \
  --seed 1337
```

**Options**:
- `--no_perc`: Disable random perceptual loss
- `--no_amp`: Disable automatic mixed precision
- `--eval_every N`: Change evaluation frequency
- `--resume CKPT`: Resume from checkpoint

### 3. Evaluate Model

**Using the evaluation script**:
```bash
bash scripts/eval_distill.sh
```

**Or directly with Python**:
```bash
python -m distill_c1.eval_distill \
  --data_dir data/scenes \
  --ckpt runs/distill_c1/ema_best.pt \
  --report runs/distill_c1/report.json \
  --save_images runs/distill_c1/eval_images \
  --counterfactual
```

**Options**:
- `--split`: Evaluate on 'val' or 'test' split
- `--counterfactual`: Perform counterfactual sensitivity analysis
- `--max_samples N`: Limit number of samples

### 4. Run Tests

```bash
python -m distill_c1.tests_distill
```

Tests cover:
- Shape correctness
- Gradient flow
- Determinism
- Loss function toggling
- EMA updates
- Architecture validation

## Outputs

**During training**:
- `runs/distill_c1/best.pt`: Best model checkpoint (by SSIM)
- `runs/distill_c1/last.pt`: Latest checkpoint
- `runs/distill_c1/ema_best.pt`: Best EMA checkpoint (use for inference)
- `runs/distill_c1/log.json`: Training log with metrics
- `runs/distill_c1/samples/step_*.png`: Visualization grids

**After evaluation**:
- `report.json`: PSNR, SSIM, counterfactual sensitivity results
- `eval_images/grid.png`: 3-row grid (teacher, student, diff)
- `eval_images/comparison.png`: Side-by-side comparison

## Acceptance Criteria

The model should achieve:
- ✅ **PSNR ≥ 24 dB** on validation set
- ✅ **SSIM ≥ 0.92** on validation set
- ✅ **Localized changes** for counterfactual edits (color/shape/number/relation)
- ✅ **No checkerboard artifacts** (using upsample+conv)
- ✅ **Good compositional generalization** on holdout splits
- ✅ **Stable frequency spectrum** (low TV loss)

## Files

```
distill_c1/
├── __init__.py              # Module exports
├── text_encoder.py          # Transformer text encoder
├── decoder.py               # FiLM-conditioned CNN decoder
├── losses.py                # Loss functions
├── metrics.py               # Evaluation metrics
├── vis.py                   # Visualization utilities
├── trainer.py               # Training loop with EMA/AMP
├── train_distill.py         # CLI training script
├── eval_distill.py          # CLI evaluation script
├── tests_distill.py         # Unit tests
├── config_default.yaml      # Default configuration
└── README.md                # This file

scripts/
├── train_distill.sh         # Training shell script
└── eval_distill.sh          # Evaluation shell script
```

## Example Results

Expected validation metrics after training:
```
PSNR: 26-28 dB
SSIM: 0.93-0.95
```

The model should successfully:
1. Generate clean 64×64 images matching the renderer style
2. Respect color, shape, and spatial constraints from captions
3. Show localized visual changes when captions are edited
4. Generalize to held-out compositional combinations

## Next Steps

After achieving the acceptance criteria, this decoder can be used as a warm-start for:
- **Phase C2**: Fine-tuning with caption loss only (strict regime)
- **Phase C3**: Adding adversarial losses for realism
- **Phase C4**: Extending to larger images or more complex scenes

## Configuration

See `config_default.yaml` for all hyperparameters. Key settings:

```yaml
trainer:
  lr: 3e-4
  batch: 192
  steps: 100000

loss:
  tv: 1e-5
  perc: 1e-3

model:
  emb_dim: 512
  base_ch: 256
  attn_heads: 4
```

## Troubleshooting

**Out of memory**:
- Reduce batch size: `--batch 128` or `--batch 64`
- Reduce base channels: `--base_ch 192`

**Training unstable**:
- Check TV weight isn't too high
- Reduce learning rate: `--lr 1e-4`
- Enable gradient clipping (default: 1.0)

**Poor PSNR/SSIM**:
- Train longer (>100k steps)
- Check data quality (run `visualize_samples.py`)
- Ensure renderer target images are in correct range [-1, 1]

**Checkerboard artifacts**:
- Verify decoder uses nearest-neighbor upsample + conv (not ConvTranspose2d)
- Tests should catch this: `test_no_checkerboard_artifacts()`

## References

- FiLM: [Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871)
- ConvNeXt: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- Total Variation: Standard regularization for image generation
