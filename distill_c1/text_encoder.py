"""
Text Encoder: Transformer-based encoder for DSL tokens → 512-d embedding

Architecture:
- Token embeddings (vocab_size → 256) + learned positional encodings
- [CLS] token prepended
- 4-layer Transformer encoder (Pre-LN, d_model=256, nhead=4, dim_ff=512)
- Pooling: Extract [CLS] state → LayerNorm → Linear(256→512) + SiLU
- Output: e ∈ R^{B×512}
"""

import torch
import torch.nn as nn
import math


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for DSL tokens.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension (256)
        nhead: Number of attention heads (4)
        dim_ff: Feedforward dimension (512)
        num_layers: Number of transformer layers (4)
        dropout: Dropout rate (0.1)
        max_len: Maximum sequence length (40)
        emb_dim: Output embedding dimension (512)
        pad_id: Padding token ID
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_ff: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 40,
        emb_dim: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.emb_dim = emb_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learned positional encodings
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # [CLS] token embedding (trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder layers (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection from [CLS] token
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, emb_dim)
        self.output_act = nn.SiLU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following transformer conventions."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.pad_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_id].fill_(0)

        # Positional embeddings
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # CLS token
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, token_ids: torch.LongTensor, pad_id: int = None) -> torch.Tensor:
        """
        Forward pass through text encoder.

        Args:
            token_ids: Token IDs [B, T]
            pad_id: Padding token ID (defaults to self.pad_id)

        Returns:
            Text embeddings [B, emb_dim=512]
        """
        if pad_id is None:
            pad_id = self.pad_id

        B, T = token_ids.shape
        device = token_ids.device

        # Token embeddings [B, T, d_model]
        token_emb = self.token_embedding(token_ids)

        # Positional embeddings [B, T, d_model]
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)

        # Combine token + positional embeddings
        x = self.dropout(token_emb + pos_emb)  # [B, T, d_model]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]

        # Create attention mask for padding (including CLS position)
        # CLS token is never masked, original tokens may be masked
        pad_mask = (token_ids == pad_id)  # [B, T]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # [B, T+1]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [B, T+1, d_model]

        # Extract [CLS] token and project to output dimension
        cls_state = x[:, 0, :]  # [B, d_model]
        cls_state = self.output_norm(cls_state)
        e = self.output_proj(cls_state)  # [B, emb_dim]
        e = self.output_act(e)

        return e


def build_text_encoder(
    vocab_size: int,
    pad_id: int = 0,
    d_model: int = 256,
    nhead: int = 4,
    dim_ff: int = 512,
    num_layers: int = 4,
    dropout: float = 0.1,
    max_len: int = 40,
    emb_dim: int = 512,
) -> TextEncoder:
    """
    Build a text encoder with default parameters from the spec.

    Args:
        vocab_size: Size of vocabulary
        pad_id: Padding token ID
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 4)
        dim_ff: Feedforward dimension (default: 512)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 40)
        emb_dim: Output embedding dimension (default: 512)

    Returns:
        TextEncoder instance
    """
    return TextEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        dim_ff=dim_ff,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
        emb_dim=emb_dim,
        pad_id=pad_id,
    )


if __name__ == '__main__':
    # Quick test
    vocab_size = 32
    pad_id = 0
    batch_size = 4
    seq_len = 20

    encoder = build_text_encoder(vocab_size=vocab_size, pad_id=pad_id)
    print(f"TextEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    token_ids[:, -3:] = pad_id  # Add some padding

    e = encoder(token_ids, pad_id=pad_id)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {e.shape}")
    print(f"Output range: [{e.min().item():.3f}, {e.max().item():.3f}]")

    # Test gradient flow
    loss = e.sum()
    loss.backward()
    print("✓ Gradients computed successfully")
