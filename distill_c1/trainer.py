"""
Trainer for Renderer Distillation

Handles:
- Training loop with AMP and EMA
- Validation and metrics tracking
- Checkpointing
- Visualization during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import time
from typing import Optional, Dict, Any
from copy import deepcopy

from .losses import DistillationLoss
from .metrics import compute_psnr, compute_ssim, MetricsTracker
from .vis import save_grid
from dsl.tokens import Vocab


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a shadow copy of model weights that are updated with EMA.

    Args:
        model: Model to track
        decay: EMA decay rate (default: 0.999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    @torch.no_grad()
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class DistillDataset(Dataset):
    """
    Dataset wrapper for distillation.

    Loads scene data and provides (token_ids, pad_mask, graph, target_image).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        vocab: Optional[Vocab] = None,
        max_seq_len: int = 32,
    ):
        from data.dataset import SceneDataset

        self.dataset = SceneDataset(
            data_dir=data_dir,
            split=split,
            vocab=vocab,
            max_seq_len=max_seq_len,
            image_transforms=None,  # No augmentation for distillation
        )
        self.vocab = self.dataset.vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Normalize images from [0, 1] to [-1, 1]
        image = item['image'] * 2.0 - 1.0

        # Get graph from dataset
        shard_idx = idx // self.dataset.shard_size
        local_idx = idx % self.dataset.shard_size
        shard = self.dataset._load_shard(shard_idx)
        graph = shard['graphs'][local_idx]

        return {
            'token_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'graph': graph,
            'image': image,  # [-1, 1]
            'text': item['text'],
        }


def distill_collate_fn(batch):
    """
    Custom collate function for DistillDataset.

    Handles variable-length sequences and graph objects.
    """
    token_ids = torch.stack([item['token_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    graphs = [item['graph'] for item in batch]  # Keep as list
    texts = [item['text'] for item in batch]

    return {
        'token_ids': token_ids,
        'attention_mask': attention_mask,
        'image': images,
        'graph': graphs,
        'text': texts,
    }


class DistillTrainer:
    """
    Trainer for renderer distillation.

    Args:
        text_encoder: Text encoder model
        decoder: Image decoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        grad_clip: Gradient clipping value
        tv_weight: TV loss weight
        perc_weight: Perceptual loss weight
        use_perc: Whether to use perceptual loss
        use_amp: Whether to use automatic mixed precision
        ema_decay: EMA decay rate
        save_dir: Directory to save checkpoints and logs
        eval_every: Evaluate every N steps
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        decoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab: Vocab,
        device: str = 'cuda',
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        betas: tuple = (0.9, 0.999),
        grad_clip: float = 1.0,
        tv_weight: float = 1e-5,
        perc_weight: float = 1e-3,
        use_perc: bool = True,
        use_amp: bool = True,
        ema_decay: float = 0.999,
        save_dir: str = 'runs/distill_c1',
        eval_every: int = 2000,
    ):
        self.text_encoder = text_encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.save_dir = Path(save_dir)
        self.eval_every = eval_every

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'samples').mkdir(exist_ok=True)

        # Loss function
        self.criterion = DistillationLoss(
            tv_weight=tv_weight,
            perc_weight=perc_weight,
            use_perc=use_perc,
        ).to(device)

        # Optimizer
        params = list(text_encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        # Learning rate scheduler (cosine decay with warmup)
        self.scheduler = None  # Will be set in train()

        # AMP scaler
        self.scaler = GradScaler() if use_amp else None

        # EMA for decoder
        self.ema = EMA(decoder, decay=ema_decay)

        # Metrics tracking
        self.step = 0
        self.best_ssim = 0.0
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # Training log
        self.log = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of loss values
        """
        self.text_encoder.train()
        self.decoder.train()

        # Move to device
        token_ids = batch['token_ids'].to(self.device)
        target_images = batch['image'].to(self.device)

        # Forward pass with AMP
        if self.use_amp:
            with autocast('cuda'):
                # Encode text
                e = self.text_encoder(token_ids, pad_id=self.vocab.pad_id)

                # Decode to image
                pred_images = self.decoder(e)

                # Compute loss
                loss, loss_dict = self.criterion(pred_images, target_images)
        else:
            # Encode text
            e = self.text_encoder(token_ids, pad_id=self.vocab.pad_id)

            # Decode to image
            pred_images = self.decoder(e)

            # Compute loss
            loss, loss_dict = self.criterion(pred_images, target_images)

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.text_encoder.parameters()) + list(self.decoder.parameters()),
                self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.text_encoder.parameters()) + list(self.decoder.parameters()),
                self.grad_clip
            )
            self.optimizer.step()

        # Update EMA
        self.ema.update()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return loss_dict

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dictionary of metrics
        """
        self.text_encoder.eval()
        self.decoder.eval()

        # Apply EMA weights for evaluation
        self.ema.apply_shadow()

        metrics = MetricsTracker()

        # Collect images for visualization
        vis_teacher = []
        vis_student = []

        for i, batch in enumerate(self.val_loader):
            token_ids = batch['token_ids'].to(self.device)
            target_images = batch['image'].to(self.device)

            # Forward pass
            e = self.text_encoder(token_ids, pad_id=self.vocab.pad_id)
            pred_images = self.decoder(e)

            # Compute loss
            loss, loss_dict = self.criterion(pred_images, target_images)

            # Compute metrics
            psnr = compute_psnr(pred_images, target_images)
            ssim = compute_ssim(pred_images, target_images)

            metrics.update(**loss_dict, psnr=psnr, ssim=ssim)

            # Collect first batch for visualization
            if i == 0:
                vis_teacher = target_images[:8].cpu()
                vis_student = pred_images[:8].cpu()

            if i >= 10:  # Limit validation batches
                break

        # Restore original weights
        self.ema.restore()

        # Save visualization
        if len(vis_teacher) > 0:
            save_grid(
                teacher=vis_teacher,
                student=vis_student,
                path=str(self.save_dir / 'samples' / f'step_{self.step:07d}.png'),
                nrow=8,
            )

        return metrics.get_averages()

    def train(self, total_steps: int, warmup_steps: int = 1000):
        """
        Main training loop.

        Args:
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps for learning rate
        """
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )

        print(f"Training for {total_steps} steps...")
        print(f"Evaluating every {self.eval_every} steps")
        print(f"Saving to {self.save_dir}")

        start_time = time.time()
        self.step = 0

        while self.step < total_steps:
            for batch in self.train_loader:
                # Training step
                loss_dict = self.train_step(batch)
                self.train_metrics.update(**loss_dict)

                self.step += 1

                # Log training metrics
                if self.step % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.step / elapsed
                    eta = (total_steps - self.step) / steps_per_sec

                    train_avg = self.train_metrics.get_averages()
                    print(f"Step {self.step}/{total_steps} | "
                          f"Loss: {train_avg['total']:.6f} | "
                          f"L1: {train_avg['l1']:.6f} | "
                          f"TV: {train_avg['tv']:.6f} | "
                          f"Steps/s: {steps_per_sec:.2f} | "
                          f"ETA: {eta/60:.1f}m")

                    self.train_metrics.reset()

                # Evaluation
                if self.step % self.eval_every == 0 or self.step == total_steps:
                    val_metrics = self.evaluate()

                    print(f"\n=== Validation at step {self.step} ===")
                    print(f"PSNR: {val_metrics['psnr']:.2f} dB")
                    print(f"SSIM: {val_metrics['ssim']:.4f}")
                    print(f"Loss: {val_metrics['total']:.6f}")
                    print("=" * 40 + "\n")

                    # Save checkpoint
                    self.save_checkpoint('last.pt', val_metrics)

                    # Save best model
                    if val_metrics['ssim'] > self.best_ssim:
                        self.best_ssim = val_metrics['ssim']
                        self.save_checkpoint('best.pt', val_metrics)
                        self.save_ema_checkpoint('ema_best.pt', val_metrics)
                        print(f"✓ New best SSIM: {self.best_ssim:.4f}\n")

                    # Log metrics
                    self.log.append({
                        'step': self.step,
                        'metrics': val_metrics,
                    })

                    # Save log
                    with open(self.save_dir / 'log.json', 'w') as f:
                        json.dump(self.log, f, indent=2)

                if self.step >= total_steps:
                    break

        print(f"\nTraining complete! Best SSIM: {self.best_ssim:.4f}")

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'text_encoder': self.text_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'ema': self.ema.state_dict(),
            'metrics': metrics,
            'best_ssim': self.best_ssim,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def save_ema_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save checkpoint with EMA weights."""
        # Apply EMA weights
        self.ema.apply_shadow()

        checkpoint = {
            'step': self.step,
            'text_encoder': self.text_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, self.save_dir / filename)

        # Restore original weights
        self.ema.restore()

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

        self.step = checkpoint['step']
        self.best_ssim = checkpoint.get('best_ssim', 0.0)

        print(f"Loaded checkpoint from step {self.step}")


if __name__ == '__main__':
    print("Trainer module ready. Use train_distill.py for training.")
