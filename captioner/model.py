"""Complete captioner model combining encoder and decoder."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import ConvNeXtEncoder, build_convnext_tiny
from .decoder import AttentionGRUDecoder, build_decoder


class Captioner(nn.Module):
    """Complete image captioning model.

    Architecture:
    - ConvNeXt-Tiny encoder
    - Attention GRU decoder
    - Label smoothing loss
    """

    def __init__(self,
                 vocab_size: int,
                 encoder: Optional[ConvNeXtEncoder] = None,
                 decoder: Optional[AttentionGRUDecoder] = None,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 encoder_dim: int = 256,
                 attention_dim: int = 256,
                 dropout: float = 0.3,
                 drop_path_rate: float = 0.1,
                 label_smoothing: float = 0.1):
        """Initialize captioner.

        Args:
            vocab_size: Size of vocabulary
            encoder: Optional pre-built encoder
            decoder: Optional pre-built decoder
            embed_dim: Embedding dimension
            hidden_dim: GRU hidden dimension
            encoder_dim: Encoder output dimension
            attention_dim: Attention projection dimension
            dropout: Dropout rate
            drop_path_rate: Stochastic depth rate for encoder
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

        # Build encoder if not provided
        if encoder is None:
            encoder = build_convnext_tiny(drop_path_rate=drop_path_rate)
        self.encoder = encoder

        # Build decoder if not provided
        if decoder is None:
            decoder = build_decoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                encoder_dim=encoder_dim,
                attention_dim=attention_dim,
                dropout=dropout
            )
        self.decoder = decoder

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=2,  # Ignore <PAD> token
            label_smoothing=label_smoothing
        )

    def forward(self,
                images: torch.Tensor,
                targets: torch.Tensor,
                teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            images: Input images (B, 3, 64, 64)
            targets: Target token IDs (B, max_len) including <BOS> and <EOS>
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Tuple of:
            - logits: Output logits (B, max_len-1, vocab_size)
            - loss: Cross-entropy loss with label smoothing
        """
        # Encode images
        grid_tokens, pooled = self.encoder(images)

        # Decode with teacher forcing
        logits = self.decoder(
            encoder_out=grid_tokens,
            pooled=pooled,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # Compute loss
        # logits: (B, max_len-1, vocab_size)
        # targets: (B, max_len) -> shift to get (B, max_len-1)
        target_tokens = targets[:, 1:]  # Remove <BOS>

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)

        loss = self.criterion(logits_flat, targets_flat)

        return logits, loss

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to features.

        Args:
            images: Input images (B, 3, 64, 64)

        Returns:
            Tuple of (grid_tokens, pooled)
        """
        return self.encoder(images)

    def init_decoder_state(self, pooled: torch.Tensor) -> torch.Tensor:
        """Initialize decoder hidden state.

        Args:
            pooled: Encoder pooled features (B, encoder_dim)

        Returns:
            Initial hidden state (1, B, hidden_dim)
        """
        return self.decoder.init_hidden_state(pooled)

    def decode_step(self,
                    input_token: torch.Tensor,
                    hidden: torch.Tensor,
                    encoder_out: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step.

        Args:
            input_token: Input token IDs (B,)
            hidden: Previous hidden state (1, B, hidden_dim)
            encoder_out: Encoder grid tokens (B, seq_len, encoder_dim)
            mask: Optional attention mask (B, seq_len)

        Returns:
            Tuple of (logits, hidden, attention_weights)
        """
        return self.decoder.forward_step(input_token, hidden, encoder_out, mask)


def build_captioner(vocab_size: int,
                    embed_dim: int = 256,
                    hidden_dim: int = 512,
                    encoder_dim: int = 256,
                    attention_dim: int = 256,
                    dropout: float = 0.3,
                    drop_path_rate: float = 0.1,
                    label_smoothing: float = 0.1) -> Captioner:
    """Build complete captioner model.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: GRU hidden dimension
        encoder_dim: Encoder output dimension
        attention_dim: Attention projection dimension
        dropout: Dropout rate
        drop_path_rate: Stochastic depth rate for encoder
        label_smoothing: Label smoothing factor

    Returns:
        Captioner model
    """
    return Captioner(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        encoder_dim=encoder_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        label_smoothing=label_smoothing
    )
