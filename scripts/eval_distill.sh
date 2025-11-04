#!/bin/bash
#
# Evaluation script for renderer distillation
#
# Usage:
#   bash scripts/eval_distill.sh
#
# Customize the parameters below as needed.

set -e  # Exit on error

# Configuration
DATA_DIR="data/scenes"
CKPT="runs/distill_c1/ema_best.pt"
REPORT="runs/distill_c1/report.json"
SAVE_IMAGES="runs/distill_c1/eval_images"
BATCH=64
SPLIT="val"
NUM_WORKERS=4

# Parse command line arguments (optional)
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --save_images)
      SAVE_IMAGES="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --counterfactual)
      COUNTERFACTUAL="--counterfactual"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Renderer Distillation Evaluation"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Data directory: $DATA_DIR"
echo "Split: $SPLIT"
echo "Report: $REPORT"
echo "Save images: $SAVE_IMAGES"
echo "========================================"

# Run evaluation
python -m distill_c1.eval_distill \
  --data_dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --report "$REPORT" \
  --save_images "$SAVE_IMAGES" \
  --batch "$BATCH" \
  --split "$SPLIT" \
  --num_workers "$NUM_WORKERS" \
  $COUNTERFACTUAL

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Report saved to: $REPORT"
echo "Images saved to: $SAVE_IMAGES"
echo "========================================"
