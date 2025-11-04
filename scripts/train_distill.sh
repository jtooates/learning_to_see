#!/bin/bash
#
# Training script for renderer distillation
#
# Usage:
#   bash scripts/train_distill.sh
#
# Customize the parameters below as needed.

set -e  # Exit on error

# Configuration
DATA_DIR="data/scenes"
SAVE_DIR="runs/distill_c1"
STEPS=100000
BATCH=192
LR=3e-4
WD=0.05
TV_WEIGHT=1e-5
PERC_WEIGHT=1e-3
WARMUP=1000
EVAL_EVERY=2000
SEED=1337
NUM_WORKERS=4

# Model settings
EMB_DIM=512
BASE_CH=256
ATTN_HEADS=4

# Parse command line arguments (optional)
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --no_perc)
      NO_PERC="--no_perc"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Renderer Distillation Training"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "Steps: $STEPS"
echo "Batch size: $BATCH"
echo "Learning rate: $LR"
echo "Seed: $SEED"
echo "========================================"

# Run training
python -m distill_c1.train_distill \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --wd "$WD" \
  --tv "$TV_WEIGHT" \
  --perc "$PERC_WEIGHT" \
  --warmup "$WARMUP" \
  --eval_every "$EVAL_EVERY" \
  --emb_dim "$EMB_DIM" \
  --base_ch "$BASE_CH" \
  --attn_heads "$ATTN_HEADS" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --use_amp \
  $NO_PERC

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: $SAVE_DIR"
echo "========================================"
