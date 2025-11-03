"""PyTorch Dataset for loading synthetic scene data."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from dsl.tokens import Vocab


class SceneDataset(Dataset):
    """PyTorch Dataset for scene images and captions.

    Loads data from sharded format:
    - images_{idx:04d}.pt: Tensor of shape (N, C, H, W)
    - texts_{idx:04d}.jsonl: JSONL with canonical text
    - graphs_{idx:04d}.jsonl: JSONL with scene graphs
    - meta_{idx:04d}.pt: Metadata with masks, bboxes, etc.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 vocab: Optional[Vocab] = None,
                 max_seq_len: int = 32,
                 image_transforms: Optional[Callable] = None):
        """Initialize dataset.

        Args:
            data_dir: Directory containing sharded data
            split: Which split to load ('train', 'val', 'test')
            vocab: Vocabulary for tokenization
            max_seq_len: Maximum sequence length (for padding)
            image_transforms: Optional image augmentation transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.image_transforms = image_transforms

        # Load vocabulary
        if vocab is None:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        # Load split indices
        with open(self.data_dir / 'splits.json', 'r') as f:
            split_data = json.load(f)

        split_key = f'{split}_indices'
        if split_key not in split_data:
            raise ValueError(f"Split '{split}' not found in splits.json")

        self.indices = split_data[split_key]
        print(f"Loaded {len(self.indices)} samples for split '{split}'")

        # Load manifest
        with open(self.data_dir / 'manifest.json', 'r') as f:
            self.manifest = json.load(f)

        self.n_shards = self.manifest['n_shards']
        self.shard_size = self.manifest['shard_size']

        # Cache for loaded shards
        self._shard_cache = {}

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        """Load a shard from disk (with caching).

        Args:
            shard_idx: Shard index

        Returns:
            Dictionary with images, texts, graphs, metadata
        """
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]

        # Load shard data
        # Note: weights_only=False is safe here as we're loading our own generated data
        images = torch.load(self.data_dir / f'images_{shard_idx:04d}.pt', weights_only=False)

        texts = []
        with open(self.data_dir / f'texts_{shard_idx:04d}.jsonl', 'r') as f:
            for line in f:
                texts.append(json.loads(line)['text'])

        graphs = []
        with open(self.data_dir / f'graphs_{shard_idx:04d}.jsonl', 'r') as f:
            for line in f:
                graphs.append(json.loads(line))

        metadata = torch.load(self.data_dir / f'meta_{shard_idx:04d}.pt', weights_only=False)

        shard_data = {
            'images': images,
            'texts': texts,
            'graphs': graphs,
            'metadata': metadata
        }

        # Cache (simple LRU: keep last 3 shards)
        if len(self._shard_cache) >= 3:
            # Remove oldest
            oldest_key = next(iter(self._shard_cache))
            del self._shard_cache[oldest_key]

        self._shard_cache[shard_idx] = shard_data

        return shard_data

    def _get_sample_from_global_index(self, global_idx: int) -> Dict[str, Any]:
        """Get sample by global index.

        Args:
            global_idx: Global sample index

        Returns:
            Sample dictionary
        """
        # Determine which shard
        shard_idx = global_idx // self.shard_size
        within_shard_idx = global_idx % self.shard_size

        # Load shard
        shard_data = self._load_shard(shard_idx)

        # Extract sample
        return {
            'image': shard_data['images'][within_shard_idx],
            'text': shard_data['texts'][within_shard_idx],
            'graph': shard_data['graphs'][within_shard_idx],
            'metadata': shard_data['metadata'][within_shard_idx]
        }

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index within the split

        Returns:
            Dictionary with:
                - image: Tensor of shape (C, H, W)
                - input_ids: Tensor of token IDs (max_seq_len,)
                - attention_mask: Tensor of padding mask (max_seq_len,)
                - text: Original text string
        """
        # Map split index to global index
        global_idx = self.indices[idx]

        # Get sample
        sample = self._get_sample_from_global_index(global_idx)

        # Get image
        image = sample['image']

        # Apply transforms if provided
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        # Tokenize text
        text = sample['text']
        token_ids = self.vocab.encode(text, add_special=True)

        # Pad/truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            attention_mask = [1] * self.max_seq_len
        else:
            attention_mask = [1] * len(token_ids) + [0] * (self.max_seq_len - len(token_ids))
            token_ids = token_ids + [self.vocab.pad_id] * (self.max_seq_len - len(token_ids))

        return {
            'image': image,
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text': text  # Keep original text for debugging
        }

    def get_graph(self, idx: int) -> Dict[str, Any]:
        """Get scene graph for a sample (useful for analysis).

        Args:
            idx: Index within the split

        Returns:
            Scene graph dictionary
        """
        global_idx = self.indices[idx]
        sample = self._get_sample_from_global_index(global_idx)
        return sample['graph']

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample.

        Args:
            idx: Index within the split

        Returns:
            Metadata dictionary
        """
        global_idx = self.indices[idx]
        sample = self._get_sample_from_global_index(global_idx)
        return sample['metadata']


def get_default_transforms(mode: str = 'train') -> Callable:
    """Get default image transforms for training or evaluation.

    Args:
        mode: 'train' or 'eval'

    Returns:
        Torchvision transform
    """
    if mode == 'train':
        # Light augmentation for training
        return T.Compose([
            T.RandomHorizontalFlip(p=0.0),  # Disabled for now (breaks left/right relations)
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # Image is already normalized to [0, 1], no need for Normalize
        ])
    else:
        # No augmentation for eval
        return T.Compose([])


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    texts = [item['text'] for item in batch]

    return {
        'image': images,  # Use singular for consistency
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'text': texts  # Use singular for consistency
    }


def create_dataloaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       vocab: Optional[Vocab] = None,
                       augment_train: bool = True,
                       pin_memory: bool = True) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, val, test dataloaders.

    Args:
        data_dir: Directory containing sharded data
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        vocab: Vocabulary instance
        augment_train: Whether to apply augmentation to training data
        pin_memory: Whether to use pinned memory (faster GPU transfer)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if vocab is None:
        vocab = Vocab()

    # Create datasets
    train_transforms = get_default_transforms('train') if augment_train else None
    eval_transforms = get_default_transforms('eval')

    train_dataset = SceneDataset(
        data_dir, split='train', vocab=vocab, image_transforms=train_transforms
    )
    val_dataset = SceneDataset(
        data_dir, split='val', vocab=vocab, image_transforms=eval_transforms
    )
    test_dataset = SceneDataset(
        data_dir, split='test', vocab=vocab, image_transforms=eval_transforms
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
