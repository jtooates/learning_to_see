"""Main training script for the image generation model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data.dataset import ShapeSceneDataset, collate_fn
from src.generation.models.cvae import ConditionalVAE
from src.generation.training.losses import VAELoss
from src.generation.training.trainer import Trainer
from src.generation.utils.tokenizer import CaptionTokenizer
from src.generation.utils.visualization import (
    visualize_reconstruction, visualize_generation, plot_training_curves
)


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    LATENT_DIM = 128
    CAPTION_DIM = 256
    IMAGE_SIZE = 256
    DATASET_SIZE = 10000
    VAL_SPLIT = 0.1

    KL_WEIGHT = 0.001
    KL_ANNEALING = True
    RECONSTRUCTION_LOSS = "mse"  # or "l1"

    CHECKPOINT_DIR = "checkpoints"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("Shape Scene Image Generator Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Caption dim: {CAPTION_DIM}")
    print()

    # Create dataset
    print("Creating dataset...")
    full_dataset = ShapeSceneDataset(
        size=DATASET_SIZE,
        canvas_size=IMAGE_SIZE,
        seed=42,
    )

    # Split into train and validation
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Build tokenizer vocabulary
    print("Building tokenizer vocabulary...")
    tokenizer = CaptionTokenizer(max_length=32)

    # Sample captions to build vocabulary
    sample_captions = []
    for i in range(min(1000, len(full_dataset))):
        _, caption = full_dataset[i]
        sample_captions.append(caption)

    tokenizer.fit(sample_captions)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Save tokenizer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tokenizer.save_vocab(os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.json"))
    print("Tokenizer vocabulary saved")
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Create model
    print("Creating model...")
    model = ConditionalVAE(
        vocab_size=tokenizer.get_vocab_size(),
        image_size=IMAGE_SIZE,
        latent_dim=LATENT_DIM,
        caption_dim=CAPTION_DIM,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Create loss function
    criterion = VAELoss(
        reconstruction_loss=RECONSTRUCTION_LOSS,
        kl_weight=KL_WEIGHT,
        kl_annealing=KL_ANNEALING,
        kl_annealing_epochs=10,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler=scheduler,
        tokenize_fn=tokenizer.encode,
    )

    # Train
    print("Starting training...")
    print()
    trainer.train(num_epochs=NUM_EPOCHS, save_every=10)

    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path="training_curves.png"
    )

    # Generate some samples
    print("\nGenerating sample images...")
    model.eval()

    # Get some validation samples
    val_images, val_captions = next(iter(val_loader))
    val_images = val_images[:8].to(DEVICE)
    val_captions = val_captions[:8]

    # Tokenize captions
    val_tokens = torch.stack([
        torch.tensor(tokenizer.encode(caption), dtype=torch.long)
        for caption in val_captions
    ]).to(DEVICE)

    # Reconstruct
    with torch.no_grad():
        reconstructed = model.reconstruct(val_images, val_tokens)

    visualize_reconstruction(
        val_images,
        reconstructed,
        val_captions,
        save_path="reconstruction.png",
        num_samples=8,
    )

    # Generate from captions only
    test_captions = [
        "a large blue square above 3 small red circles",
        "2 medium green triangles",
        "a small yellow rectangle left of a large purple square",
        "4 orange circles",
    ]

    test_tokens = torch.stack([
        torch.tensor(tokenizer.encode(caption), dtype=torch.long)
        for caption in test_captions
    ]).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(test_tokens, num_samples=1)

    visualize_generation(
        generated,
        test_captions,
        save_path="generated_samples.png",
    )

    print("\nTraining complete! Check the output images.")


if __name__ == "__main__":
    main()
