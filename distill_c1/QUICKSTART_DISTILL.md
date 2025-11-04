# Quick Start Guide: Renderer Distillation

This guide will get you training a text-to-image model in 5 minutes.

## Prerequisites

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU (recommended)
- Generated scene data (see Step 1)

## Step-by-Step

### 1. Generate Training Data (if not already done)

```bash
python -m data.gen \
  --out_dir data/scenes \
  --n 6000 \
  --split_strategy random \
  --seed 42
```

This creates ~6000 scene images with captions.

**Expected output**: `data/scenes/` directory with sharded data files.

### 2. Verify Data

```bash
python visualize_samples.py --data_dir data/scenes --n 16
```

This shows a grid of sample images to verify data quality.

### 3. Run Tests

```bash
python -m pytest distill_c1/tests_distill.py -v
```

**Expected**: All 19 tests pass ✅

### 4. Train the Model

**Quick training (for testing)**:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1_test \
  --steps 1000 \
  --batch 64 \
  --eval_every 500
```

**Full training (for results)**:
```bash
bash scripts/train_distill.sh
```

Or manually:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1 \
  --steps 100000 \
  --batch 192 \
  --lr 3e-4 \
  --seed 1337
```

**Expected time**: 6-8 hours on GPU for 100k steps

**Monitor training**:
- Watch console output for loss/metrics
- Check `runs/distill_c1/samples/` for generated images
- View `runs/distill_c1/log.json` for detailed metrics

### 5. Evaluate the Model

```bash
bash scripts/eval_distill.sh
```

Or manually:
```bash
python -m distill_c1.eval_distill \
  --data_dir data/scenes \
  --ckpt runs/distill_c1/ema_best.pt \
  --report runs/distill_c1/report.json \
  --save_images runs/distill_c1/eval_images \
  --counterfactual
```

**Expected output**:
```
PSNR: 26-28 dB
SSIM: 0.93-0.95
```

### 6. View Results

**Evaluation report**:
```bash
cat runs/distill_c1/report.json
```

**Visualizations**:
```bash
open runs/distill_c1/eval_images/grid.png
open runs/distill_c1/eval_images/comparison.png
```

**Training samples**:
```bash
open runs/distill_c1/samples/step_0100000.png
```

## Common Commands

### Resume Training
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1 \
  --steps 100000 \
  --resume runs/distill_c1/last.pt
```

### Train with Different Settings

**Smaller batch (less memory)**:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1_small \
  --steps 100000 \
  --batch 64
```

**Without perceptual loss**:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1_no_perc \
  --steps 100000 \
  --no_perc
```

**On CPU (slow)**:
```bash
python -m distill_c1.train_distill \
  --data_dir data/scenes \
  --save_dir runs/distill_c1_cpu \
  --steps 1000 \
  --batch 16 \
  --device cpu
```

### Evaluate on Test Set
```bash
python -m distill_c1.eval_distill \
  --data_dir data/scenes \
  --ckpt runs/distill_c1/ema_best.pt \
  --report runs/distill_c1/test_report.json \
  --split test
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch 128  # or --batch 64
```

Or reduce model size:
```bash
--base_ch 192  # instead of 256
```

### Training is Slow

Enable AMP (should be on by default):
```bash
--use_amp
```

Increase number of workers:
```bash
--num_workers 8
```

### Poor Results

Train longer:
```bash
--steps 150000
```

Check data:
```bash
python visualize_samples.py --data_dir data/scenes
```

Adjust learning rate:
```bash
--lr 1e-4  # lower lr for stability
```

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Data generation | 10-20 min | `data/scenes/` |
| Tests | <1 min | All pass ✅ |
| Quick training (1k steps) | 5-10 min | Sanity check |
| Full training (100k steps) | 6-8 hours | Production model |
| Evaluation | 2-5 min | Metrics + images |

## Directory Structure After Training

```
learning_to_see/
├── data/
│   └── scenes/           # Generated data
│       ├── images_*.pt
│       ├── texts_*.jsonl
│       └── splits.json
│
├── runs/
│   └── distill_c1/       # Training outputs
│       ├── best.pt       # Best checkpoint
│       ├── ema_best.pt   # Best EMA checkpoint (use this!)
│       ├── last.pt       # Latest checkpoint
│       ├── log.json      # Training log
│       ├── report.json   # Evaluation report
│       ├── samples/      # Generated samples
│       │   └── step_*.png
│       └── eval_images/  # Evaluation visualizations
│           ├── grid.png
│           └── comparison.png
│
└── distill_c1/           # Source code
    ├── text_encoder.py
    ├── decoder.py
    └── ...
```

## Next Steps After Training

1. **Check acceptance criteria**:
   - PSNR ≥ 24 dB ✅
   - SSIM ≥ 0.92 ✅
   - Review counterfactual results

2. **Experiment with prompts**:
   - Modify captions in DSL format
   - Test compositional combinations
   - Try held-out test set

3. **Fine-tune if needed**:
   - Adjust hyperparameters
   - Train on more data
   - Try different splits

4. **Proceed to Phase C2**:
   - Use decoder as warm-start
   - Switch to caption loss only
   - Fine-tune with strict regime

## Getting Help

- **Documentation**: See `distill_c1/README.md`
- **Configuration**: See `distill_c1/config_default.yaml`
- **Tests**: Run `python -m pytest distill_c1/tests_distill.py -v`
- **Code**: All modules have docstrings and examples

## Minimal Working Example

For the absolute quickest test:

```bash
# Generate small dataset
python -m data.gen --out_dir data/scenes_test --n 500 --seed 42

# Quick training
python -m distill_c1.train_distill \
  --data_dir data/scenes_test \
  --save_dir runs/test \
  --steps 1000 \
  --batch 32 \
  --eval_every 500

# Evaluate
python -m distill_c1.eval_distill \
  --data_dir data/scenes_test \
  --ckpt runs/test/ema_best.pt \
  --report runs/test/report.json
```

This completes in ~10 minutes and verifies everything works!
