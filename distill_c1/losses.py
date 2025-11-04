"""
Loss Functions for Renderer Distillation

Implements:
1. Pixel losses: L1 and L2 between predicted and target images
2. Total Variation (TV) loss: Anisotropic TV for smoothness
3. Random Perceptual loss: Frozen random CNN features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_losses(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute L1 and L2 pixel losses.

    Args:
        pred: Predicted images [B, 3, H, W] in [-1, 1]
        target: Target images [B, 3, H, W] in [-1, 1]

    Returns:
        Dictionary with 'l1' and 'l2' losses
    """
    l1_loss = F.l1_loss(pred, target)
    l2_loss = F.mse_loss(pred, target)

    return {
        'l1': l1_loss,
        'l2': l2_loss,
    }


def tv_loss(images: torch.Tensor) -> torch.Tensor:
    """
    Compute anisotropic Total Variation loss for smoothness.

    Penalizes high-frequency artifacts by computing differences
    between adjacent pixels.

    Args:
        images: Images [B, C, H, W]

    Returns:
        Scalar TV loss
    """
    # Horizontal differences
    h_diff = images[:, :, 1:, :] - images[:, :, :-1, :]
    # Vertical differences
    v_diff = images[:, :, :, 1:] - images[:, :, :, :-1]

    # Anisotropic TV: sum of absolute differences
    tv = torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(v_diff))

    return tv


class TinyRandNet(nn.Module):
    """
    Tiny random CNN for perceptual loss.

    A 3-layer convolutional network with FROZEN random weights.
    Used to provide a weak perceptual bias without requiring pretrained models.

    Architecture:
        Conv1: 3→16, 3×3, stride 1
        Conv2: 16→32, 3×3, stride 2
        Conv3: 32→64, 3×3, stride 2

    Output feature maps at three scales:
        - Level 1: [B, 16, 64, 64]
        - Level 2: [B, 32, 32, 32]
        - Level 3: [B, 64, 16, 16]
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.act = nn.ReLU(inplace=True)

        # Initialize with random weights and freeze
        self._init_random_weights()
        self._freeze()

    def _init_random_weights(self):
        """Initialize with random weights (Kaiming normal)."""
        for module in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(module.bias)

    def _freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        """
        Extract features at multiple scales.

        Args:
            x: Input images [B, 3, H, W] in [-1, 1]

        Returns:
            List of feature maps [feat1, feat2, feat3]
        """
        # Normalize from [-1, 1] to [0, 1] for stability
        x = (x + 1.0) / 2.0

        # Extract features
        feat1 = self.act(self.conv1(x))  # [B, 16, 64, 64]
        feat2 = self.act(self.conv2(feat1))  # [B, 32, 32, 32]
        feat3 = self.act(self.conv3(feat2))  # [B, 64, 16, 16]

        return [feat1, feat2, feat3]


def rand_perc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: TinyRandNet,
) -> torch.Tensor:
    """
    Compute random perceptual loss using frozen random CNN.

    Extracts features from both predicted and target images,
    then computes L2 distance between features at each level.

    Args:
        pred: Predicted images [B, 3, H, W] in [-1, 1]
        target: Target images [B, 3, H, W] in [-1, 1]
        net: Frozen TinyRandNet

    Returns:
        Scalar perceptual loss (sum of L2 distances across levels)
    """
    # Extract features
    pred_feats = net(pred)
    target_feats = net(target)

    # Compute L2 loss at each level
    loss = 0.0
    for pred_feat, target_feat in zip(pred_feats, target_feats):
        loss += F.mse_loss(pred_feat, target_feat)

    return loss


class DistillationLoss(nn.Module):
    """
    Combined loss for renderer distillation.

    Combines pixel losses (L1 + L2), TV regularization, and
    optional random perceptual loss.

    Args:
        l1_weight: Weight for L1 pixel loss (default: 0.05)
        l2_weight: Weight for L2 pixel loss (default: 0.05)
        tv_weight: Weight for TV loss (default: 1e-4)
        perc_weight: Weight for perceptual loss (default: 1.0)
        use_perc: Whether to use perceptual loss (default: True)
    """

    def __init__(
        self,
        l1_weight: float = 0.05,
        l2_weight: float = 0.05,
        tv_weight: float = 1e-4,
        perc_weight: float = 1.0,
        use_perc: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
        self.perc_weight = perc_weight
        self.use_perc = use_perc

        # Random perceptual network (frozen)
        if use_perc:
            self.perc_net = TinyRandNet()
            self.perc_net.eval()  # Always in eval mode

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined distillation loss.

        Args:
            pred: Predicted images [B, 3, H, W] in [-1, 1]
            target: Target images [B, 3, H, W] in [-1, 1]

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components
        """
        # Pixel losses
        pixel_dict = pixel_losses(pred, target)
        l1_loss = pixel_dict['l1']
        l2_loss = pixel_dict['l2']

        # TV loss
        tv = tv_loss(pred)

        # Total loss with separate L1 and L2 weights
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss + self.tv_weight * tv

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'l2': l2_loss.item(),
            'tv': tv.item(),
        }

        # Perceptual loss (if enabled)
        if self.use_perc:
            with torch.no_grad():
                # Ensure perceptual network is in eval mode
                self.perc_net.eval()

            perc = rand_perc_loss(pred, target, self.perc_net)
            total_loss = total_loss + self.perc_weight * perc
            loss_dict['perc'] = perc.item()
            loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == '__main__':
    # Quick test
    batch_size = 4
    H, W = 64, 64

    # Test pixel losses
    pred = torch.randn(batch_size, 3, H, W) * 0.5
    target = torch.randn(batch_size, 3, H, W) * 0.5

    pixel_dict = pixel_losses(pred, target)
    print(f"L1 loss: {pixel_dict['l1']:.4f}")
    print(f"L2 loss: {pixel_dict['l2']:.4f}")

    # Test TV loss
    tv = tv_loss(pred)
    print(f"TV loss: {tv:.6f}")

    # Test random perceptual loss
    perc_net = TinyRandNet()
    print(f"TinyRandNet parameters: {sum(p.numel() for p in perc_net.parameters()):,}")
    print(f"All parameters frozen: {all(not p.requires_grad for p in perc_net.parameters())}")

    perc = rand_perc_loss(pred, target, perc_net)
    print(f"Random perceptual loss: {perc:.4f}")

    # Test combined loss
    loss_fn = DistillationLoss(tv_weight=1e-5, perc_weight=1e-3, use_perc=True)
    total_loss, loss_dict = loss_fn(pred, target)

    print("\nCombined loss:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")

    # Test gradient flow
    total_loss.backward()
    print("\n✓ Gradients computed successfully")

    # Test without perceptual loss
    loss_fn_no_perc = DistillationLoss(use_perc=False)
    total_loss_no_perc, loss_dict_no_perc = loss_fn_no_perc(pred, target)
    print(f"\n✓ Loss without perceptual: {total_loss_no_perc:.6f}")
    assert 'perc' not in loss_dict_no_perc
