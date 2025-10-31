"""Trainer class for Conditional VAE."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Callable
from pathlib import Path
import json
from tqdm import tqdm

from src.generation.models.cvae import ConditionalVAE
from src.generation.training.losses import VAELoss
from src.generation.training.metrics import compute_metrics


class Trainer:
    """Trainer for Conditional VAE model."""

    def __init__(
        self,
        model: ConditionalVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        criterion: VAELoss,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        scheduler: Optional[_LRScheduler] = None,
        tokenize_fn: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The Conditional VAE model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            scheduler: Learning rate scheduler (optional)
            tokenize_fn: Function to tokenize captions
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.tokenize_fn = tokenize_fn

        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(device)

        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, captions) in enumerate(pbar):
            # Move images to device
            images = images.to(self.device)

            # Tokenize captions
            if self.tokenize_fn is not None:
                token_ids = torch.stack([
                    torch.tensor(self.tokenize_fn(caption), dtype=torch.long)
                    for caption in captions
                ]).to(self.device)
            else:
                # Assume captions are already tokenized
                token_ids = captions.to(self.device)

            # Forward pass
            outputs = self.model(images, token_ids)

            # Compute loss
            loss_dict = self.criterion(
                outputs['reconstructed'],
                images,
                outputs['mu'],
                outputs['logvar'],
            )

            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': loss_dict['recon_loss'].item(),
                'kl': loss_dict['kl_loss'].item(),
            })

        # Average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
        }

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of average validation losses and metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0

        for images, captions in tqdm(self.val_loader, desc="Validation"):
            # Move images to device
            images = images.to(self.device)

            # Tokenize captions
            if self.tokenize_fn is not None:
                token_ids = torch.stack([
                    torch.tensor(self.tokenize_fn(caption), dtype=torch.long)
                    for caption in captions
                ]).to(self.device)
            else:
                token_ids = captions.to(self.device)

            # Forward pass
            outputs = self.model(images, token_ids)

            # Compute loss
            loss_dict = self.criterion(
                outputs['reconstructed'],
                images,
                outputs['mu'],
                outputs['logvar'],
            )

            # Compute metrics
            metrics = compute_metrics(outputs['reconstructed'], images)

            # Track
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_mse += metrics['mse']
            total_mae += metrics['mae']
            num_batches += 1

        # Average
        avg_results = {
            'val_total_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches,
            'val_mse': total_mse / num_batches,
            'val_mae': total_mae / num_batches,
        }

        return avg_results

    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)

            # Validate
            val_results = self.validate()
            if val_results:
                self.val_losses.append(val_results)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f} "
                  f"(Recon: {train_losses['recon_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f})")
            if val_results:
                print(f"  Val Loss: {val_results['val_total_loss']:.4f} "
                      f"(MSE: {val_results['val_mse']:.4f})")

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Step KL annealing
            self.criterion.step_epoch()

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint("final_model.pt")
        print("\nTraining complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: {checkpoint_path}")
