"""
Training script for renderer distillation.

CLI:
  python -m distill_c1.train_distill \\
    --data_dir data/scenes \\
    --save_dir runs/distill_c1 \\
    --steps 100000 \\
    --batch 192 \\
    --lr 3e-4 \\
    --tv 1e-5 \\
    --perc 1e-3 \\
    --seed 1337
"""

import argparse
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

from .text_encoder import build_text_encoder
from .decoder import build_decoder
from .trainer import DistillTrainer, DistillDataset
from dsl.tokens import Vocab


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train text-to-image distillation model')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing scene data')
    parser.add_argument('--save_dir', type=str, default='runs/distill_c1',
                        help='Directory to save checkpoints and logs')

    # Training
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--batch', type=int, default=192,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')

    # Loss weights
    parser.add_argument('--tv', type=float, default=1e-5,
                        help='TV loss weight')
    parser.add_argument('--perc', type=float, default=1e-3,
                        help='Perceptual loss weight')
    parser.add_argument('--no_perc', action='store_true',
                        help='Disable perceptual loss')

    # Model
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--base_ch', type=int, default=256,
                        help='Base number of channels in decoder')
    parser.add_argument('--attn_heads', type=int, default=4,
                        help='Number of attention heads')

    # Training settings
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable automatic mixed precision')
    parser.add_argument('--ema', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--eval_every', type=int, default=2000,
                        help='Evaluate every N steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Other
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("Renderer Distillation Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Total steps: {args.steps:,}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Use AMP: {args.use_amp}")
    print(f"Use perceptual loss: {not args.no_perc}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = Vocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DistillDataset(
        data_dir=args.data_dir,
        split='train',
        vocab=vocab,
        max_seq_len=32,
    )

    val_dataset = DistillDataset(
        data_dir=args.data_dir,
        split='val',
        vocab=vocab,
        max_seq_len=32,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    # Build models
    print("\nBuilding models...")
    text_encoder = build_text_encoder(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        emb_dim=args.emb_dim,
    )

    decoder = build_decoder(
        emb_dim=args.emb_dim,
        base_ch=args.base_ch,
        attn_heads=args.attn_heads,
    )

    # Count parameters
    text_enc_params = sum(p.numel() for p in text_encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = text_enc_params + decoder_params

    print(f"Text encoder parameters: {text_enc_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DistillTrainer(
        text_encoder=text_encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=args.device,
        lr=args.lr,
        weight_decay=args.wd,
        grad_clip=args.grad_clip,
        tv_weight=args.tv,
        perc_weight=args.perc,
        use_perc=not args.no_perc,
        use_amp=args.use_amp,
        ema_decay=args.ema,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...\n")
    trainer.train(total_steps=args.steps, warmup_steps=args.warmup)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best SSIM: {trainer.best_ssim:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
