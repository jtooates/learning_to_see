"""GRU decoder with Bahdanau attention for captioner."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism.

    Computes attention weights over encoder grid tokens and returns
    weighted context vector.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, attention_dim: int = 256):
        """Initialize attention module.

        Args:
            hidden_dim: Decoder hidden state dimension
            encoder_dim: Encoder feature dimension
            attention_dim: Attention projection dimension
        """
        super().__init__()

        # Project encoder features
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)

        # Project decoder hidden state
        self.decoder_proj = nn.Linear(hidden_dim, attention_dim)

        # Attention scoring
        self.score = nn.Linear(attention_dim, 1)

    def forward(self,
                hidden: torch.Tensor,
                encoder_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention.

        Args:
            hidden: Decoder hidden state (B, hidden_dim)
            encoder_out: Encoder grid tokens (B, seq_len, encoder_dim)
            mask: Optional attention mask (B, seq_len)

        Returns:
            Tuple of:
            - context: Weighted context vector (B, encoder_dim)
            - weights: Attention weights (B, seq_len)
        """
        # Project encoder features: (B, seq_len, attention_dim)
        enc_proj = self.encoder_proj(encoder_out)

        # Project decoder hidden: (B, attention_dim)
        dec_proj = self.decoder_proj(hidden)

        # Additive attention: tanh(W_e * e + W_d * d)
        # Broadcast: (B, seq_len, attention_dim) + (B, 1, attention_dim)
        energy = torch.tanh(enc_proj + dec_proj.unsqueeze(1))

        # Score: (B, seq_len, 1) -> (B, seq_len)
        scores = self.score(energy).squeeze(2)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        weights = F.softmax(scores, dim=1)

        # Context: weighted sum of encoder outputs
        # (B, 1, seq_len) @ (B, seq_len, encoder_dim) -> (B, 1, encoder_dim)
        context = torch.bmm(weights.unsqueeze(1), encoder_out).squeeze(1)

        return context, weights


class AttentionGRUDecoder(nn.Module):
    """GRU decoder with Bahdanau attention for constrained captioning.

    Architecture:
    - Embedding layer for input tokens
    - Bahdanau attention over encoder grid tokens
    - Single-layer GRU with (embedding + context) input
    - Output projection to vocabulary
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 encoder_dim: int,
                 attention_dim: int = 256,
                 dropout: float = 0.3):
        """Initialize decoder.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: GRU hidden dimension
            encoder_dim: Encoder feature dimension
            attention_dim: Attention projection dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=2)  # 2 is <PAD>

        # Attention
        self.attention = BahdanauAttention(hidden_dim, encoder_dim, attention_dim)

        # GRU: input is (embedding + context)
        self.gru = nn.GRU(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout for single layer
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Initialize hidden state projection from encoder pooled features
        self.init_hidden = nn.Linear(encoder_dim, hidden_dim)

    def init_hidden_state(self, pooled: torch.Tensor) -> torch.Tensor:
        """Initialize decoder hidden state from encoder pooled features.

        Args:
            pooled: Encoder pooled features (B, encoder_dim)

        Returns:
            Initial hidden state (1, B, hidden_dim)
        """
        h0 = torch.tanh(self.init_hidden(pooled))
        return h0.unsqueeze(0)  # (1, B, hidden_dim)

    def forward_step(self,
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
            Tuple of:
            - logits: Output logits (B, vocab_size)
            - hidden: New hidden state (1, B, hidden_dim)
            - attention_weights: Attention weights (B, seq_len)
        """
        # Embed input: (B,) -> (B, embed_dim)
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        # Compute attention context
        # hidden: (1, B, hidden_dim) -> (B, hidden_dim)
        context, attention_weights = self.attention(hidden.squeeze(0), encoder_out, mask)

        # Concatenate embedding and context: (B, embed_dim + encoder_dim)
        gru_input = torch.cat([embedded, context], dim=1)

        # GRU expects (B, 1, input_size)
        gru_input = gru_input.unsqueeze(1)

        # GRU step: output (B, 1, hidden_dim), hidden (1, B, hidden_dim)
        output, hidden = self.gru(gru_input, hidden)

        # Remove sequence dimension: (B, 1, hidden_dim) -> (B, hidden_dim)
        output = output.squeeze(1)
        output = self.dropout(output)

        # Project to vocabulary
        logits = self.fc_out(output)

        return logits, hidden, attention_weights

    def forward(self,
                encoder_out: torch.Tensor,
                pooled: torch.Tensor,
                targets: torch.Tensor,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """Training forward pass with teacher forcing.

        Args:
            encoder_out: Encoder grid tokens (B, seq_len, encoder_dim)
            pooled: Encoder pooled features (B, encoder_dim)
            targets: Target token IDs (B, max_len) including <BOS> and <EOS>
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            logits: Output logits (B, max_len-1, vocab_size)
        """
        batch_size = encoder_out.size(0)
        max_len = targets.size(1)

        # Initialize hidden state
        hidden = self.init_hidden_state(pooled)

        # Storage for outputs
        outputs = []

        # Start with <BOS> token
        input_token = targets[:, 0]  # (B,)

        # Decode for max_len - 1 steps (predict tokens 1 to max_len-1)
        for t in range(1, max_len):
            # Forward step
            logits, hidden, _ = self.forward_step(input_token, hidden, encoder_out)

            outputs.append(logits.unsqueeze(1))

            # Teacher forcing: use ground truth as next input
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                # Use predicted token
                input_token = logits.argmax(dim=1)

        # Concatenate outputs: (B, max_len-1, vocab_size)
        outputs = torch.cat(outputs, dim=1)

        return outputs


def build_decoder(vocab_size: int,
                  embed_dim: int = 256,
                  hidden_dim: int = 512,
                  encoder_dim: int = 256,
                  attention_dim: int = 256,
                  dropout: float = 0.3) -> AttentionGRUDecoder:
    """Build attention GRU decoder.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: GRU hidden dimension
        encoder_dim: Encoder feature dimension
        attention_dim: Attention projection dimension
        dropout: Dropout rate

    Returns:
        Decoder module
    """
    return AttentionGRUDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        encoder_dim=encoder_dim,
        attention_dim=attention_dim,
        dropout=dropout
    )
