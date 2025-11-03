"""ConvNeXt-Tiny encoder for captioner."""
import torch
import torch.nn as nn
from typing import Tuple


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise convolution and inverted bottleneck.

    Architecture:
    - Depthwise 7x7 conv
    - LayerNorm
    - Inverted bottleneck (dim -> 4*dim -> dim)
    - Residual connection
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        """Initialize ConvNeXt block.

        Args:
            dim: Channel dimension
            drop_path: Drop path rate for stochastic depth
        """
        super().__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm (channels last format)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted bottleneck: dim -> 4*dim -> dim
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        shortcut = x

        # Depthwise conv
        x = self.dwconv(x)

        # Permute to channels-last for LayerNorm: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # LayerNorm + FFN
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Permute back: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual
        x = shortcut + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Downsample(nn.Module):
    """Downsampling layer with stride-2 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, C, H, W)

        Returns:
            Output (B, out_channels, H//2, W//2)
        """
        # Permute to channels-last for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back for conv
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny encoder for 64x64 images.

    Architecture:
    - Patch stem: Conv(3→64, 4x4 stride 4) → 16x16x64
    - Stage1: 2 ConvNeXt blocks → downsample stride-2 → 8x8x128
    - Stage2: 2 blocks → downsample stride-2 → 4x4x256
    - Stage3: 4 blocks → 4x4x256

    Returns:
    - grid_tokens: flatten to 16 tokens × 256 dim
    - pooled: mean over 4x4 → 256
    """

    def __init__(self,
                 in_channels: int = 3,
                 dims: Tuple[int, ...] = (64, 128, 256),
                 depths: Tuple[int, ...] = (2, 2, 4),
                 drop_path_rate: float = 0.1):
        """Initialize encoder.

        Args:
            in_channels: Input channels (3 for RGB)
            dims: Channel dimensions for each stage
            depths: Number of blocks in each stage
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()

        # Patch stem: 64x64 -> 16x16x64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 16, 16], eps=1e-6)
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(len(depths)):
            # Stage blocks
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j]
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

            # Downsample (except for last stage)
            if i < len(depths) - 1:
                downsample = Downsample(dims[i], dims[i + 1])
                self.downsamples.append(downsample)

        self.final_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images (B, 3, 64, 64)

        Returns:
            Tuple of:
            - grid_tokens: (B, 16, 256) - flattened spatial features
            - pooled: (B, 256) - global pooled features
        """
        # Stem: (B, 3, 64, 64) -> (B, 64, 16, 16)
        x = self.stem(x)

        # Stage 1: (B, 64, 16, 16) -> (B, 128, 8, 8)
        x = self.stages[0](x)
        x = self.downsamples[0](x)

        # Stage 2: (B, 128, 8, 8) -> (B, 256, 4, 4)
        x = self.stages[1](x)
        x = self.downsamples[1](x)

        # Stage 3: (B, 256, 4, 4) -> (B, 256, 4, 4)
        x = self.stages[2](x)

        # x is now (B, 256, 4, 4)
        B, C, H, W = x.shape

        # Grid tokens: flatten spatial dimensions
        grid_tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 16, 256)

        # Pooled: mean over spatial dimensions
        pooled = x.mean(dim=[2, 3])  # (B, 256)

        return grid_tokens, pooled


def build_convnext_tiny(drop_path_rate: float = 0.1) -> ConvNeXtEncoder:
    """Build ConvNeXt-Tiny encoder.

    Args:
        drop_path_rate: Stochastic depth rate

    Returns:
        ConvNeXt encoder
    """
    return ConvNeXtEncoder(
        in_channels=3,
        dims=(64, 128, 256),
        depths=(2, 2, 4),
        drop_path_rate=drop_path_rate
    )
