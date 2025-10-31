"""Caption tokenizer for text-to-image generation."""

from typing import List, Dict, Optional
import re


class CaptionTokenizer:
    """Simple word-level tokenizer for shape scene captions."""

    def __init__(self, max_length: int = 32):
        """
        Initialize the tokenizer.

        Args:
            max_length: Maximum sequence length (will pad/truncate to this)
        """
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word: Dict[int, str] = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.vocab_size = 4

    def fit(self, captions: List[str]) -> None:
        """
        Build vocabulary from a list of captions.

        Args:
            captions: List of caption strings
        """
        # Collect all unique words
        word_set = set()
        for caption in captions:
            words = self._tokenize(caption)
            word_set.update(words)

        # Add to vocabulary (sorted for consistency)
        for word in sorted(word_set):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

    def encode(self, caption: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a caption to token indices.

        Args:
            caption: The caption string
            add_special_tokens: Whether to add <SOS> and <EOS> tokens

        Returns:
            List of token indices (padded/truncated to max_length)
        """
        words = self._tokenize(caption)

        # Convert words to indices
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

        # Add special tokens
        if add_special_tokens:
            tokens = [self.word2idx['<SOS>']] + tokens + [self.word2idx['<EOS>']]

        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.word2idx['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices back to a caption.

        Args:
            tokens: List of token indices
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            The decoded caption string
        """
        special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
        words = []

        for token in tokens:
            word = self.idx2word.get(token, '<UNK>')
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)

        return ' '.join(words)

    def _tokenize(self, caption: str) -> List[str]:
        """
        Tokenize a caption into words.

        Args:
            caption: The caption string

        Returns:
            List of words
        """
        # Lowercase and split by whitespace
        caption = caption.lower().strip()

        # Handle parentheses and commas
        caption = re.sub(r'[()]', ' ', caption)
        caption = re.sub(r',', ' , ', caption)

        # Split on whitespace and filter empty strings
        words = [w for w in caption.split() if w]

        return words

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size

    def save_vocab(self, path: str) -> None:
        """
        Save vocabulary to a file.

        Args:
            path: File path to save vocabulary
        """
        import json
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocab(self, path: str) -> None:
        """
        Load vocabulary from a file.

        Args:
            path: File path to load vocabulary from
        """
        import json
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.max_length = vocab_data['max_length']
        self.vocab_size = vocab_data['vocab_size']
