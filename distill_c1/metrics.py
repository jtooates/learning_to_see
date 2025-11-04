"""
Evaluation Metrics for Image Generation

Implements:
1. PSNR (Peak Signal-to-Noise Ratio)
2. SSIM (Structural Similarity Index)
3. Counterfactual sensitivity analysis
"""

import torch
import torch.nn.functional as F
import math


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 20 * log10(MAX) - 10 * log10(MSE)

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        max_val: Maximum pixel value range (default: 2.0 for [-1, 1])

    Returns:
        Average PSNR in dB
    """
    mse = F.mse_loss(pred, target)

    if mse == 0:
        return float('inf')

    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 2.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Simplified implementation using Gaussian weighting.

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        window_size: Size of Gaussian window (default: 11)
        max_val: Maximum pixel value range (default: 2.0 for [-1, 1])

    Returns:
        Average SSIM score (0 to 1, higher is better)
    """
    B, C, H, W = pred.shape

    # Constants for stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    # Create 2D Gaussian kernel
    kernel_1d = gauss.unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float()
    kernel = kernel_2d.expand(C, 1, window_size, window_size).contiguous()
    kernel = kernel.to(pred.device)

    # Compute local statistics
    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def counterfactual_sensitivity(
    text_encoder,
    decoder,
    vocab,
    parser,
    renderer,
    original_text: str,
    token_edit: str,
    device: str = 'cpu',
) -> dict:
    """
    Compute counterfactual sensitivity to token edits.

    Given an original caption, applies a token edit (e.g., changing color),
    generates images for both captions, and measures the difference.

    Args:
        text_encoder: Text encoder model
        decoder: Image decoder model
        vocab: Vocabulary object
        parser: Scene parser
        renderer: Scene renderer
        original_text: Original caption
        token_edit: Type of edit ('color', 'shape', 'number', 'relation')
        device: Device to run on

    Returns:
        Dictionary with:
            - 'delta_l2': L2 difference between images
            - 'delta_psnr': PSNR difference
            - 'original_text': Original caption
            - 'edited_text': Edited caption
            - 'bbox': Expected change region (if available)
    """
    from dsl.parser import SceneParser
    from dsl.canonicalize import to_canonical
    import copy

    # Parse original scene
    scene_graph = parser.parse(original_text)

    # Create edited scene graph
    edited_graph = copy.deepcopy(scene_graph)

    # Apply edit based on type
    if token_edit == 'color' and edited_graph['objects']:
        # Change color of first object
        obj = edited_graph['objects'][0]
        colors = ['red', 'green', 'blue', 'yellow']
        current_color = obj['color']
        new_color = [c for c in colors if c != current_color][0]
        obj['color'] = new_color

    elif token_edit == 'shape' and edited_graph['objects']:
        # Change shape of first object
        obj = edited_graph['objects'][0]
        shapes = ['ball', 'cube', 'block']
        current_shape = obj['shape']
        new_shape = [s for s in shapes if s != current_shape][0]
        obj['shape'] = new_shape

    elif token_edit == 'number' and edited_graph['sentence_type'] == 'count':
        # Change count
        current_count = edited_graph['count']
        new_count = (current_count % 5) + 1  # Cycle through 1-5
        edited_graph['count'] = new_count

    elif token_edit == 'relation' and edited_graph['sentence_type'] == 'relational':
        # Change relation
        relations = ['left_of', 'right_of', 'on', 'in_front_of']
        if 'relation' in edited_graph:
            current_rel = edited_graph['relation']
            new_rel = [r for r in relations if r != current_rel][0]
            edited_graph['relation'] = new_rel

    # Convert back to canonical text
    edited_text = to_canonical(edited_graph)

    # Tokenize both captions
    original_tokens = vocab.encode(original_text, add_special_tokens=True)
    edited_tokens = vocab.encode(edited_text, add_special_tokens=True)

    # Pad to same length
    max_len = max(len(original_tokens), len(edited_tokens))
    original_tokens = original_tokens + [vocab.pad_id] * (max_len - len(original_tokens))
    edited_tokens = edited_tokens + [vocab.pad_id] * (max_len - len(edited_tokens))

    # Convert to tensors
    original_ids = torch.tensor([original_tokens], dtype=torch.long, device=device)
    edited_ids = torch.tensor([edited_tokens], dtype=torch.long, device=device)

    # Generate images
    text_encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Original image
        e_orig = text_encoder(original_ids, vocab.pad_id)
        img_orig = decoder(e_orig)  # [1, 3, 64, 64]

        # Edited image
        e_edit = text_encoder(edited_ids, vocab.pad_id)
        img_edit = decoder(e_edit)  # [1, 3, 64, 64]

    # Compute differences
    delta_l2 = F.mse_loss(img_orig, img_edit).item()
    psnr_orig = compute_psnr(img_orig, torch.zeros_like(img_orig))
    psnr_edit = compute_psnr(img_edit, torch.zeros_like(img_edit))
    delta_psnr = abs(psnr_orig - psnr_edit)

    # Extract bounding box from scene graph (if available)
    bbox = None
    if edited_graph['objects']:
        obj = edited_graph['objects'][0]
        if 'bbox' in obj:
            bbox = obj['bbox']

    return {
        'delta_l2': delta_l2,
        'delta_psnr': delta_psnr,
        'original_text': original_text,
        'edited_text': edited_text,
        'bbox': bbox,
        'img_orig': img_orig.cpu(),
        'img_edit': img_edit.cpu(),
    }


class MetricsTracker:
    """
    Tracks metrics during training and evaluation.

    Maintains running averages of PSNR, SSIM, and loss components.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: Metric name-value pairs
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def get_averages(self) -> dict:
        """
        Get average values of all metrics.

        Returns:
            Dictionary of averaged metrics
        """
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages

    def get_summary(self) -> str:
        """
        Get a formatted summary string of metrics.

        Returns:
            Formatted string of metrics
        """
        averages = self.get_averages()
        summary_parts = []
        for key, value in sorted(averages.items()):
            if 'psnr' in key.lower():
                summary_parts.append(f"{key}: {value:.2f} dB")
            elif 'ssim' in key.lower():
                summary_parts.append(f"{key}: {value:.4f}")
            else:
                summary_parts.append(f"{key}: {value:.6f}")
        return " | ".join(summary_parts)


if __name__ == '__main__':
    # Quick test
    batch_size = 4
    H, W = 64, 64

    # Test PSNR
    pred = torch.randn(batch_size, 3, H, W) * 0.5
    target = torch.randn(batch_size, 3, H, W) * 0.5

    psnr = compute_psnr(pred, target)
    print(f"PSNR: {psnr:.2f} dB")

    # Test SSIM
    ssim = compute_ssim(pred, target)
    print(f"SSIM: {ssim:.4f}")

    # Test with identical images (should be perfect)
    psnr_perfect = compute_psnr(target, target)
    ssim_perfect = compute_ssim(target, target)
    print(f"\nPerfect reconstruction:")
    print(f"  PSNR: {psnr_perfect:.2f} dB (should be inf or very high)")
    print(f"  SSIM: {ssim_perfect:.4f} (should be ~1.0)")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(psnr=25.0, ssim=0.85, loss=0.01)
    tracker.update(psnr=26.0, ssim=0.87, loss=0.009)
    tracker.update(psnr=24.5, ssim=0.86, loss=0.011)

    print("\nMetrics Tracker:")
    print(tracker.get_summary())

    averages = tracker.get_averages()
    print(f"\nAverage PSNR: {averages['psnr']:.2f} dB")
    print(f"Average SSIM: {averages['ssim']:.4f}")

    print("\n✓ All metrics computed successfully")
