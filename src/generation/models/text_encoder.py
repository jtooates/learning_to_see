"""Text encoder for captions."""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """LSTM-based encoder for caption text."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize the text encoder.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            output_dim: Output embedding dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Project LSTM output to desired dimension
        # Bidirectional LSTM outputs hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token sequences to caption embeddings.

        Args:
            token_ids: Token indices of shape (batch_size, seq_length)

        Returns:
            Caption embeddings of shape (batch_size, output_dim)
        """
        # Embed tokens: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(token_ids)

        # Pass through LSTM: output shape (batch_size, seq_length, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # We want the last layer's forward and backward hidden states
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction

        # Concatenate forward and backward: (batch_size, hidden_dim * 2)
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Project to output dimension: (batch_size, output_dim)
        caption_embedding = self.projection(combined_hidden)

        return caption_embedding


class SimpleTextEncoder(nn.Module):
    """Simpler MLP-based encoder as an alternative."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Initialize a simple text encoder.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            output_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Simple pooling + MLP
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token sequences to caption embeddings.

        Args:
            token_ids: Token indices of shape (batch_size, seq_length)

        Returns:
            Caption embeddings of shape (batch_size, output_dim)
        """
        # Embed tokens: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(token_ids)

        # Mean pooling over sequence: (batch_size, embedding_dim)
        pooled = embedded.mean(dim=1)

        # Pass through MLP: (batch_size, output_dim)
        caption_embedding = self.encoder(pooled)

        return caption_embedding
