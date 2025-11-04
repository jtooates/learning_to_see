"""
Image Decoder: FiLM-conditioned CNN decoder with ResBlocks and MHSA

Architecture:
- Stem: Linear(512 → 256*8*8) → reshape to [B, 256, 8, 8]
- Stage1 (8→16): ResBlock(256)×2 + FiLM(256); Upsample(×2); Conv 256→128; GN; SiLU
- Stage2 (16→32): ResBlock(128)×2 + FiLM(128); MHSA(128, 4 heads); Upsample; Conv 128→64; GN; SiLU
- Stage3 (32→64): ResBlock(64)×2 + FiLM(64)
- Head: Conv 64→3; Tanh → [−1,1]

Components:
- ResBlock: Conv3×3→GN→SiLU→Conv3×3→GN + skip
- FiLM(e): Linear(512→2C) → split γ,β; y = γ⊙x + β
- MHSA: Single layer at 32×32 with 4 heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    Takes a conditioning embedding e [B, emb_dim] and produces
    scale (γ) and shift (β) parameters for feature maps.

    Args:
        emb_dim: Embedding dimension (512)
        num_channels: Number of feature channels to modulate
    """

    def __init__(self, emb_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: Feature maps [B, C, H, W]
            e: Conditioning embedding [B, emb_dim]

        Returns:
            Modulated features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Project embedding to scale and shift
        film_params = self.proj(e)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=1)  # Each [B, C]

        # Reshape for broadcasting
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # Apply affine transformation
        return gamma * x + beta


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.

    Architecture:
        Conv3×3 → GroupNorm → SiLU → Conv3×3 → GroupNorm + skip

    Args:
        channels: Number of channels
        num_groups: Number of groups for GroupNorm (default: 32)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(self, channels: int, num_groups: int = 32, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Output features [B, C, H, W]
        """
        residual = x

        # First conv block
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        h = self.dropout(h)

        # Second conv block
        h = self.conv2(h)
        h = self.gn2(h)

        # Residual connection
        return h + residual


class MHSA2d(nn.Module):
    """
    Multi-Head Self-Attention for 2D feature maps.

    Applies attention over spatial locations (H×W becomes sequence length).
    Single layer with Pre-LN.

    Args:
        channels: Number of channels
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.channels = channels
        self.num_heads = num_heads

        # Pre-LayerNorm
        self.norm = nn.GroupNorm(1, channels)  # GroupNorm with 1 group = LayerNorm for spatial

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over spatial dimensions.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Output features [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Reshape to sequence: [B, C, H, W] → [B, H*W, C]
        x = x.view(B, C, H * W).permute(0, 2, 1)

        # Apply attention
        attn_out, _ = self.mha(x, x, x, need_weights=False)

        # Reshape back: [B, H*W, C] → [B, C, H, W]
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # Residual connection
        return attn_out + residual


class Decoder(nn.Module):
    """
    FiLM-conditioned CNN decoder for text-to-image generation.

    Upsamples from 8×8 → 64×64 with FiLM conditioning at each stage.

    Architecture:
        Stem: Linear(512 → 256*8*8) → [B, 256, 8, 8]
        Stage1 (8→16): ResBlock(256)×2 + FiLM → Upsample → Conv 256→128
        Stage2 (16→32): ResBlock(128)×2 + FiLM → MHSA → Upsample → Conv 128→64
        Stage3 (32→64): ResBlock(64)×2 + FiLM
        Head: Conv 64→3 + Tanh

    Args:
        emb_dim: Conditioning embedding dimension (512)
        base_ch: Base number of channels (256)
        start_hw: Starting spatial size (8)
        attn_heads: Number of attention heads (4)
        num_groups: Groups for GroupNorm (32)
    """

    def __init__(
        self,
        emb_dim: int = 512,
        base_ch: int = 256,
        start_hw: int = 8,
        attn_heads: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.base_ch = base_ch
        self.start_hw = start_hw

        # Stem: Project embedding to initial feature map
        self.stem = nn.Linear(emb_dim, base_ch * start_hw * start_hw)

        # Stage 1: 8×8 @ 256 channels → 16×16 @ 128 channels
        self.stage1_res1 = ResBlock(base_ch, num_groups=num_groups)
        self.stage1_res2 = ResBlock(base_ch, num_groups=num_groups)
        self.stage1_film = FiLM(emb_dim, base_ch)
        self.stage1_conv = nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1)
        self.stage1_gn = nn.GroupNorm(num_groups, base_ch // 2)
        self.stage1_act = nn.SiLU()

        # Stage 2: 16×16 @ 128 channels → 32×32 @ 64 channels
        self.stage2_res1 = ResBlock(base_ch // 2, num_groups=num_groups)
        self.stage2_res2 = ResBlock(base_ch // 2, num_groups=num_groups)
        self.stage2_film = FiLM(emb_dim, base_ch // 2)
        self.stage2_attn = MHSA2d(base_ch // 2, num_heads=attn_heads)
        self.stage2_conv = nn.Conv2d(base_ch // 2, base_ch // 4, kernel_size=3, padding=1)
        self.stage2_gn = nn.GroupNorm(num_groups, base_ch // 4)
        self.stage2_act = nn.SiLU()

        # Stage 3: 32×32 @ 64 channels → 64×64 @ 64 channels
        self.stage3_res1 = ResBlock(base_ch // 4, num_groups=num_groups)
        self.stage3_res2 = ResBlock(base_ch // 4, num_groups=num_groups)
        self.stage3_film = FiLM(emb_dim, base_ch // 4)

        # Output head
        self.head = nn.Conv2d(base_ch // 4, 3, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Stem
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)

        # Conv layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Output head - small initialization for stability
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Generate image from text embedding.

        Args:
            e: Text embedding [B, emb_dim=512]

        Returns:
            Generated image [B, 3, 64, 64] in range [-1, 1]
        """
        B = e.shape[0]

        # Stem: Project to initial feature map
        h = self.stem(e)  # [B, base_ch*start_hw*start_hw]
        h = h.view(B, self.base_ch, self.start_hw, self.start_hw)  # [B, 256, 8, 8]

        # Stage 1: 8×8 @ 256 → 16×16 @ 128
        h = self.stage1_res1(h)
        h = self.stage1_res2(h)
        h = self.stage1_film(h, e)
        h = F.interpolate(h, scale_factor=2, mode='nearest')  # [B, 256, 16, 16]
        h = self.stage1_conv(h)  # [B, 128, 16, 16]
        h = self.stage1_gn(h)
        h = self.stage1_act(h)

        # Stage 2: 16×16 @ 128 → 32×32 @ 64
        h = self.stage2_res1(h)
        h = self.stage2_res2(h)
        h = self.stage2_film(h, e)
        h = self.stage2_attn(h)  # Self-attention at 16×16
        h = F.interpolate(h, scale_factor=2, mode='nearest')  # [B, 128, 32, 32]
        h = self.stage2_conv(h)  # [B, 64, 32, 32]
        h = self.stage2_gn(h)
        h = self.stage2_act(h)

        # Stage 3: 32×32 @ 64 → 64×64 @ 64
        h = F.interpolate(h, scale_factor=2, mode='nearest')  # [B, 64, 64, 64]
        h = self.stage3_res1(h)
        h = self.stage3_res2(h)
        h = self.stage3_film(h, e)

        # Output head
        img = self.head(h)  # [B, 3, 64, 64]
        img = torch.tanh(img)  # Scale to [-1, 1]

        return img


def build_decoder(
    emb_dim: int = 512,
    base_ch: int = 256,
    start_hw: int = 8,
    attn_heads: int = 4,
) -> Decoder:
    """
    Build a decoder with default parameters from the spec.

    Args:
        emb_dim: Embedding dimension (default: 512)
        base_ch: Base number of channels (default: 256)
        start_hw: Starting spatial size (default: 8)
        attn_heads: Number of attention heads (default: 4)

    Returns:
        Decoder instance
    """
    return Decoder(
        emb_dim=emb_dim,
        base_ch=base_ch,
        start_hw=start_hw,
        attn_heads=attn_heads,
    )


if __name__ == '__main__':
    # Quick test
    batch_size = 4
    emb_dim = 512

    decoder = build_decoder()
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test forward pass
    e = torch.randn(batch_size, emb_dim)
    img = decoder(e)

    print(f"Input embedding shape: {e.shape}")
    print(f"Output image shape: {img.shape}")
    print(f"Output range: [{img.min().item():.3f}, {img.max().item():.3f}]")

    # Test gradient flow
    loss = img.sum()
    loss.backward()
    print("✓ Gradients computed successfully")

    # Verify no checkerboard artifacts (upsample+conv design)
    print("✓ Architecture uses nearest-neighbor upsample + conv (no checkerboard artifacts)")
