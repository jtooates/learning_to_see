"""Training loop for captioner with AdamW, AMP, and scheduled sampling."""
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import json
from typing import Optional, Dict
from tqdm import tqdm

from .model import build_captioner
from .metrics import CaptioningMetrics, evaluate_model
from .augmentations import get_train_augmentation, get_eval_transform
from dsl.tokens import Vocab


class CaptionerTrainer:
    """Trainer for captioner model with full training loop."""

    def __init__(self,
                 model,
                 vocab: Vocab,
                 train_loader,
                 val_loader,
                 device: str = 'cuda',
                 lr: float = 3e-4,
                 weight_decay: float = 0.01,
                 max_epochs: int = 100,
                 warmup_epochs: int = 5,
                 use_amp: bool = True,
                 scheduled_sampling_k: float = 10.0,
                 checkpoint_dir: str = 'checkpoints',
                 log_interval: int = 50):
        """Initialize trainer.

        Args:
            model: Captioner model
            vocab: Vocabulary object
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            max_epochs: Maximum number of epochs
            warmup_epochs: Number of warmup epochs
            use_amp: Whether to use automatic mixed precision
            scheduled_sampling_k: Decay factor for scheduled sampling
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval in steps
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp
        self.scheduled_sampling_k = scheduled_sampling_k
        self.log_interval = log_interval

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler (OneCycleLR)
        total_steps = len(train_loader) * max_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_epochs / max_epochs,
            anneal_strategy='cos'
        )

        # AMP scaler
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_exact_match = 0.0

        # Augmentation
        self.train_aug = get_train_augmentation(image_size=64, strong=True)
        self.eval_aug = get_eval_transform(image_size=64)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Compute teacher forcing ratio with scheduled sampling
        # Linearly decay from 1.0 to 0.0 using sigmoid schedule
        progress = self.epoch / self.max_epochs
        teacher_forcing_ratio = self._scheduled_sampling_ratio(progress)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['input_ids'].to(self.device)

            # Apply augmentation
            if self.train_aug is not None:
                with torch.no_grad():
                    images = torch.stack([self.train_aug(img) for img in images])

            # Forward pass with AMP
            if self.use_amp:
                with autocast('cuda'):
                    logits, loss = self.model(
                        images=images,
                        targets=targets,
                        teacher_forcing_ratio=teacher_forcing_ratio
                    )
            else:
                logits, loss = self.model(
                    images=images,
                    targets=targets,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                'tf_ratio': f'{teacher_forcing_ratio:.2f}'
            })

            # Periodic logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, "
                      f"lr={self.scheduler.get_last_lr()[0]:.6f}")

        return {
            'train_loss': total_loss / num_batches,
            'teacher_forcing_ratio': teacher_forcing_ratio
        }

    def _scheduled_sampling_ratio(self, progress: float) -> float:
        """Compute teacher forcing ratio using scheduled sampling.

        Uses inverse sigmoid decay: epsilon_i = k / (k + exp(i/k))

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Teacher forcing ratio
        """
        k = self.scheduled_sampling_k
        ratio = k / (k + torch.exp(torch.tensor(progress * k)))
        return ratio.item()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Compute loss on validation set
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            targets = batch['input_ids'].to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    logits, loss = self.model(
                        images=images,
                        targets=targets,
                        teacher_forcing_ratio=1.0  # Always use ground truth
                    )
            else:
                logits, loss = self.model(
                    images=images,
                    targets=targets,
                    teacher_forcing_ratio=1.0
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Compute captioning metrics
        metrics_obj = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            vocab=self.vocab,
            device=self.device,
            use_constraints=True,
            max_length=32
        )

        metrics = metrics_obj.compute()
        metrics['val_loss'] = avg_loss

        return metrics

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss:      {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:        {val_metrics['val_loss']:.4f}")
            print(f"  Exact Match:     {val_metrics['exact_match']:.4f}")
            print(f"  Token Accuracy:  {val_metrics['token_accuracy']:.4f}")
            print(f"  Color F1:        {val_metrics['color_f1']:.4f}")
            print(f"  Shape F1:        {val_metrics['shape_f1']:.4f}")

            # Save checkpoint
            self.save_checkpoint(
                filename=f'checkpoint_epoch_{epoch}.pt',
                metrics=val_metrics
            )

            # Save best model
            if val_metrics['exact_match'] > self.best_exact_match:
                self.best_exact_match = val_metrics['exact_match']
                self.save_checkpoint(
                    filename='best_model.pt',
                    metrics=val_metrics
                )
                print(f"  → Saved best model (exact_match={self.best_exact_match:.4f})")

            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']

        print("\nTraining complete!")
        print(f"Best exact match: {self.best_exact_match:.4f}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_exact_match': self.best_exact_match,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

        # Save metrics separately as JSON
        if metrics is not None:
            metrics_path = self.checkpoint_dir / filename.replace('.pt', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_exact_match = checkpoint['best_exact_match']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.epoch}")


def train_captioner(train_loader,
                    val_loader,
                    vocab: Vocab,
                    device: str = 'cuda',
                    lr: float = 3e-4,
                    weight_decay: float = 0.01,
                    max_epochs: int = 100,
                    warmup_epochs: int = 5,
                    embed_dim: int = 256,
                    hidden_dim: int = 512,
                    dropout: float = 0.3,
                    drop_path_rate: float = 0.1,
                    label_smoothing: float = 0.1,
                    use_amp: bool = True,
                    checkpoint_dir: str = 'checkpoints') -> CaptionerTrainer:
    """Train captioner model.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        vocab: Vocabulary object
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum epochs
        warmup_epochs: Warmup epochs
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        dropout: Dropout rate
        drop_path_rate: Drop path rate for encoder
        label_smoothing: Label smoothing factor
        use_amp: Use automatic mixed precision
        checkpoint_dir: Checkpoint directory

    Returns:
        Trained CaptionerTrainer object
    """
    # Build model
    model = build_captioner(
        vocab_size=vocab.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        encoder_dim=256,  # ConvNeXt output dim
        attention_dim=256,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        label_smoothing=label_smoothing
    )

    # Create trainer
    trainer = CaptionerTrainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir
    )

    # Train
    trainer.train()

    return trainer
