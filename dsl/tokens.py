"""Token vocabulary management for the scene DSL."""
import json
from pathlib import Path
from typing import List, Optional, Union


class Vocab:
    """Vocabulary for the scene description DSL."""

    def __init__(self, vocab_path: Optional[Union[str, Path]] = None):
        """Initialize vocabulary from JSON file.

        Args:
            vocab_path: Path to vocab.json. If None, loads from dsl/vocab.json
        """
        if vocab_path is None:
            vocab_path = Path(__file__).parent / "vocab.json"
        else:
            vocab_path = Path(vocab_path)

        with open(vocab_path, 'r') as f:
            data = json.load(f)

        self.tokens = data["tokens"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.special_tokens = data["special_tokens"]

        # Convenience accessors
        self.bos_id = self.special_tokens["<BOS>"]
        self.eos_id = self.special_tokens["<EOS>"]
        self.pad_id = self.special_tokens["<PAD>"]

        self.vocab_size = len(self.tokens)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text string to token IDs.

        Args:
            text: Input text (space-separated tokens)
            add_special: If True, adds <BOS> and <EOS> tokens

        Returns:
            List of token IDs

        Raises:
            ValueError: If text contains unknown tokens
        """
        # Tokenize on whitespace, but handle period separately
        # Split by space, then handle period attachment
        tokens = []
        for word in text.split():
            # If word ends with period, split it off
            if word.endswith('.') and len(word) > 1:
                tokens.append(word[:-1])
                tokens.append('.')
            else:
                tokens.append(word)

        # Encode tokens
        token_ids = []
        if add_special:
            token_ids.append(self.bos_id)

        for token in tokens:
            if token not in self.token_to_id:
                raise ValueError(f"Unknown token: '{token}'. Vocabulary contains only: {sorted(self.token_to_id.keys())}")
            token_ids.append(self.token_to_id[token])

        if add_special:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text string.

        Args:
            token_ids: List of token IDs
            skip_special: If True, skips <BOS>, <EOS>, <PAD> tokens

        Returns:
            Decoded text string

        Raises:
            ValueError: If token_ids contains unknown IDs
        """
        tokens = []
        special_ids = {self.bos_id, self.eos_id, self.pad_id}

        for tid in token_ids:
            if tid not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {tid}")

            if skip_special and tid in special_ids:
                continue

            tokens.append(self.id_to_token[tid])

        # Join tokens, but attach period to previous word
        result = []
        for i, token in enumerate(tokens):
            if token == '.' and result:
                # Attach period to previous token
                result[-1] += '.'
            else:
                result.append(token)

        return " ".join(result)

    def pad_sequence(self, token_ids: List[int], max_length: int,
                     truncate: bool = False) -> List[int]:
        """Pad token sequence to max_length with <PAD>.

        Args:
            token_ids: List of token IDs
            max_length: Target sequence length
            truncate: If True, truncates sequences longer than max_length

        Returns:
            Padded (or truncated) token sequence
        """
        if len(token_ids) > max_length:
            if truncate:
                return token_ids[:max_length]
            else:
                raise ValueError(f"Sequence length {len(token_ids)} exceeds max_length {max_length}")

        padding = [self.pad_id] * (max_length - len(token_ids))
        return token_ids + padding

    def get_token(self, token_id: int) -> str:
        """Get token string for a given ID."""
        return self.id_to_token.get(token_id, "<UNK>")

    def get_id(self, token: str) -> Optional[int]:
        """Get token ID for a given token string."""
        return self.token_to_id.get(token)

    @classmethod
    def save(cls, tokens: List[str], special_tokens_dict: dict,
             save_path: Union[str, Path]):
        """Save vocabulary to JSON file.

        Args:
            tokens: Ordered list of all tokens
            special_tokens_dict: Mapping of special token names to IDs
            save_path: Path to save vocab.json
        """
        token_to_id = {token: i for i, token in enumerate(tokens)}
        id_to_token = {str(i): token for i, token in enumerate(tokens)}

        data = {
            "tokens": tokens,
            "special_tokens": special_tokens_dict,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Vocab(size={self.vocab_size}, special_tokens={list(self.special_tokens.keys())})"
