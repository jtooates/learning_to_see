"""
Visualization Utilities for Distillation

Creates grids comparing teacher (renderer) vs student (decoder) outputs,
along with difference maps for inspection.
"""

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from pathlib import Path


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Denormalize images from [-1, 1] to [0, 1].

    Args:
        images: Images in [-1, 1] range

    Returns:
        Images in [0, 1] range
    """
    return (images + 1.0) / 2.0


def normalize_diff(diff: torch.Tensor) -> torch.Tensor:
    """
    Normalize difference map for visualization.

    Maps differences to [0, 1] range for display.

    Args:
        diff: Difference tensor

    Returns:
        Normalized difference in [0, 1]
    """
    # Take absolute difference
    diff = torch.abs(diff)

    # Normalize to [0, 1]
    diff_min = diff.min()
    diff_max = diff.max()

    if diff_max - diff_min > 1e-6:
        diff = (diff - diff_min) / (diff_max - diff_min)
    else:
        diff = torch.zeros_like(diff)

    return diff


def save_grid(
    teacher: torch.Tensor,
    student: torch.Tensor,
    diff: torch.Tensor = None,
    path: str = 'grid.png',
    nrow: int = 8,
    captions: list = None,
) -> None:
    """
    Save a 3-row grid comparing teacher, student, and difference images.

    Args:
        teacher: Teacher (renderer) images [B, 3, H, W] in [-1, 1]
        student: Student (decoder) images [B, 3, H, W] in [-1, 1]
        diff: Optional pre-computed difference [B, 3, H, W]
        path: Output path for grid image
        nrow: Number of images per row
        captions: Optional list of caption strings
    """
    # Denormalize for display
    teacher_vis = denormalize(teacher.cpu())
    student_vis = denormalize(student.cpu())

    # Compute difference if not provided
    if diff is None:
        diff = teacher - student

    diff_vis = normalize_diff(diff.cpu())

    # Stack all rows: teacher, student, difference
    all_images = torch.cat([teacher_vis, student_vis, diff_vis], dim=0)

    # Create grid
    grid = vutils.make_grid(all_images, nrow=nrow, padding=2, normalize=False)

    # Convert to PIL image
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)

    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(path)


def save_comparison(
    teacher: torch.Tensor,
    student: torch.Tensor,
    path: str = 'comparison.png',
    max_images: int = 16,
) -> None:
    """
    Save a side-by-side comparison of teacher vs student images.

    Alternates between teacher and student images for direct comparison.

    Args:
        teacher: Teacher images [B, 3, H, W] in [-1, 1]
        student: Student images [B, 3, H, W] in [-1, 1]
        path: Output path
        max_images: Maximum number of image pairs to show
    """
    B = min(teacher.shape[0], max_images)

    # Denormalize
    teacher_vis = denormalize(teacher[:B].cpu())
    student_vis = denormalize(student[:B].cpu())

    # Interleave teacher and student
    interleaved = []
    for i in range(B):
        interleaved.append(teacher_vis[i])
        interleaved.append(student_vis[i])

    interleaved = torch.stack(interleaved)

    # Create grid (2 columns per pair)
    grid = vutils.make_grid(interleaved, nrow=2, padding=2, normalize=False)

    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(path)


def save_individual_images(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = 'img',
    captions: list = None,
) -> None:
    """
    Save individual images to separate files.

    Args:
        images: Images [B, 3, H, W] in [-1, 1]
        save_dir: Directory to save images
        prefix: Filename prefix
        captions: Optional list of caption strings (used in filename)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images_vis = denormalize(images.cpu())

    for i, img in enumerate(images_vis):
        # Convert to PIL
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Generate filename
        if captions and i < len(captions):
            # Sanitize caption for filename
            caption_clean = captions[i].replace(' ', '_').replace('.', '')[:50]
            filename = f"{prefix}_{i:04d}_{caption_clean}.png"
        else:
            filename = f"{prefix}_{i:04d}.png"

        # Save
        img_pil.save(save_dir / filename)


def create_montage(
    images_dict: dict,
    path: str = 'montage.png',
    nrow: int = 8,
) -> None:
    """
    Create a montage with multiple rows, each labeled with a key.

    Args:
        images_dict: Dictionary mapping labels to image tensors [B, 3, H, W]
        path: Output path
        nrow: Number of images per row
    """
    all_rows = []

    for label, images in images_dict.items():
        # Denormalize
        images_vis = denormalize(images.cpu())
        all_rows.append(images_vis)

    # Concatenate all rows
    all_images = torch.cat(all_rows, dim=0)

    # Create grid
    grid = vutils.make_grid(all_images, nrow=nrow, padding=2, normalize=False)

    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(path)


def visualize_counterfactual(
    original_img: torch.Tensor,
    edited_img: torch.Tensor,
    original_text: str,
    edited_text: str,
    path: str = 'counterfactual.png',
) -> None:
    """
    Visualize counterfactual edits with captions.

    Args:
        original_img: Original image [1, 3, H, W]
        edited_img: Edited image [1, 3, H, W]
        original_text: Original caption
        edited_text: Edited caption
        path: Output path
    """
    from PIL import ImageDraw, ImageFont

    # Denormalize
    orig_vis = denormalize(original_img.squeeze(0).cpu())
    edit_vis = denormalize(edited_img.squeeze(0).cpu())

    # Compute difference
    diff = torch.abs(original_img - edited_img).squeeze(0).cpu()
    diff_vis = normalize_diff(diff)

    # Stack horizontally
    combined = torch.cat([orig_vis, edit_vis, diff_vis], dim=2)

    # Convert to PIL
    combined_np = combined.permute(1, 2, 0).numpy()
    combined_np = (combined_np * 255).astype(np.uint8)
    img = Image.fromarray(combined_np)

    # Add text labels
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw captions
    draw.text((5, 5), f"Original: {original_text}", fill=(255, 255, 255), font=font)
    draw.text((orig_vis.shape[2] + 5, 5), f"Edited: {edited_text}", fill=(255, 255, 255), font=font)
    draw.text((orig_vis.shape[2] * 2 + 5, 5), "Difference", fill=(255, 255, 255), font=font)

    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


if __name__ == '__main__':
    # Quick test
    batch_size = 8
    H, W = 64, 64

    # Create dummy images
    teacher = torch.randn(batch_size, 3, H, W) * 0.5
    student = teacher + torch.randn_like(teacher) * 0.1  # Slightly different

    # Test save_grid
    print("Testing save_grid...")
    save_grid(teacher, student, path='/tmp/test_grid.png')
    print("✓ Grid saved to /tmp/test_grid.png")

    # Test save_comparison
    print("\nTesting save_comparison...")
    save_comparison(teacher, student, path='/tmp/test_comparison.png', max_images=4)
    print("✓ Comparison saved to /tmp/test_comparison.png")

    # Test save_individual_images
    print("\nTesting save_individual_images...")
    captions = [f"Caption {i}" for i in range(batch_size)]
    save_individual_images(teacher, save_dir='/tmp/test_images', prefix='teacher', captions=captions)
    print("✓ Individual images saved to /tmp/test_images")

    # Test create_montage
    print("\nTesting create_montage...")
    images_dict = {
        'teacher': teacher[:4],
        'student': student[:4],
    }
    create_montage(images_dict, path='/tmp/test_montage.png', nrow=4)
    print("✓ Montage saved to /tmp/test_montage.png")

    # Test visualize_counterfactual
    print("\nTesting visualize_counterfactual...")
    visualize_counterfactual(
        original_img=teacher[0:1],
        edited_img=student[0:1],
        original_text="There is one red ball.",
        edited_text="There is one blue ball.",
        path='/tmp/test_counterfactual.png',
    )
    print("✓ Counterfactual saved to /tmp/test_counterfactual.png")

    print("\n✓ All visualization tests passed")
