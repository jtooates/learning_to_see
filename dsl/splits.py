"""Deterministic compositional splits for evaluating generalization."""
from typing import Dict, List, Set, Tuple, Any
import itertools
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """Configuration for a compositional split."""
    name: str
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class SplitStrategy:
    """Base class for split strategies."""

    def __init__(self, seed: int = 42):
        """Initialize split strategy.

        Args:
            seed: Random seed for deterministic splits
        """
        self.seed = seed

    def make_split(self, samples: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, List[int]]:
        """Create train/val/test split indices.

        Args:
            samples: List of sample dictionaries (scene graphs or metadata)
            config: Split configuration

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing indices
        """
        raise NotImplementedError


class RandomSplit(SplitStrategy):
    """Random split (baseline)."""

    def make_split(self, samples: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, List[int]]:
        """Create random train/val/test split."""
        import random
        random.seed(self.seed)

        n = len(samples)
        indices = list(range(n))
        random.shuffle(indices)

        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        return {
            'train': sorted(indices[:n_train]),
            'val': sorted(indices[n_train:n_train + n_val]),
            'test': sorted(indices[n_train + n_val:])
        }


class ColorShapeHoldout(SplitStrategy):
    """Hold out specific color-shape combinations for testing generalization."""

    def __init__(self, holdout_pairs: List[Tuple[str, str]], seed: int = 42):
        """Initialize color-shape holdout strategy.

        Args:
            holdout_pairs: List of (color, shape) tuples to hold out
            seed: Random seed
        """
        super().__init__(seed)
        self.holdout_pairs = set(holdout_pairs)

    def make_split(self, samples: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, List[int]]:
        """Split based on color-shape combinations.

        Test set contains only holdout pairs.
        Train/val contain everything else.
        """
        import random
        random.seed(self.seed)

        holdout_indices = []
        train_val_indices = []

        for idx, sample in enumerate(samples):
            # Check if sample contains any holdout pair
            is_holdout = False
            for obj in sample['objects']:
                color = obj['color']
                shape = obj['shape']
                if (color, shape) in self.holdout_pairs:
                    is_holdout = True
                    break

            if is_holdout:
                holdout_indices.append(idx)
            else:
                train_val_indices.append(idx)

        # Shuffle train_val indices
        random.shuffle(train_val_indices)

        # Split train_val into train and val
        n_train_val = len(train_val_indices)
        # Adjust ratios to account for holdout test set
        train_ratio_adjusted = config.train_ratio / (config.train_ratio + config.val_ratio)
        n_train = int(n_train_val * train_ratio_adjusted)

        return {
            'train': sorted(train_val_indices[:n_train]),
            'val': sorted(train_val_indices[n_train:]),
            'test': sorted(holdout_indices)
        }


class CountShapeHoldout(SplitStrategy):
    """Hold out specific count-shape combinations for testing generalization."""

    def __init__(self, holdout_pairs: List[Tuple[int, str]], seed: int = 42):
        """Initialize count-shape holdout strategy.

        Args:
            holdout_pairs: List of (count, shape) tuples to hold out
            seed: Random seed
        """
        super().__init__(seed)
        self.holdout_pairs = set(holdout_pairs)

    def make_split(self, samples: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, List[int]]:
        """Split based on count-shape combinations.

        Test set contains only holdout pairs.
        Train/val contain everything else.
        """
        import random
        random.seed(self.seed)

        holdout_indices = []
        train_val_indices = []

        for idx, sample in enumerate(samples):
            # Check if sample contains any holdout pair
            is_holdout = False
            for obj in sample['objects']:
                count = obj['count']
                shape = obj['shape']
                if (count, shape) in self.holdout_pairs:
                    is_holdout = True
                    break

            if is_holdout:
                holdout_indices.append(idx)
            else:
                train_val_indices.append(idx)

        # Shuffle train_val indices
        random.shuffle(train_val_indices)

        # Split train_val into train and val
        n_train_val = len(train_val_indices)
        train_ratio_adjusted = config.train_ratio / (config.train_ratio + config.val_ratio)
        n_train = int(n_train_val * train_ratio_adjusted)

        return {
            'train': sorted(train_val_indices[:n_train]),
            'val': sorted(train_val_indices[n_train:]),
            'test': sorted(holdout_indices)
        }


class RelationHoldout(SplitStrategy):
    """Hold out specific relation types for testing generalization."""

    def __init__(self, holdout_relations: List[str], seed: int = 42):
        """Initialize relation holdout strategy.

        Args:
            holdout_relations: List of relation types to hold out
                              (e.g., ['left_of', 'right_of'])
            seed: Random seed
        """
        super().__init__(seed)
        self.holdout_relations = set(holdout_relations)

    def make_split(self, samples: List[Dict[str, Any]], config: SplitConfig) -> Dict[str, List[int]]:
        """Split based on relation types.

        Test set contains only holdout relations.
        Train/val contain everything else.
        """
        import random
        random.seed(self.seed)

        holdout_indices = []
        train_val_indices = []

        for idx, sample in enumerate(samples):
            # Check if sample contains any holdout relation
            is_holdout = False
            for rel in sample.get('relations', []):
                if rel['type'] in self.holdout_relations:
                    is_holdout = True
                    break

            if is_holdout:
                holdout_indices.append(idx)
            else:
                train_val_indices.append(idx)

        # Shuffle train_val indices
        random.shuffle(train_val_indices)

        # Split train_val into train and val
        n_train_val = len(train_val_indices)
        train_ratio_adjusted = config.train_ratio / (config.train_ratio + config.val_ratio)
        n_train = int(n_train_val * train_ratio_adjusted)

        return {
            'train': sorted(train_val_indices[:n_train]),
            'val': sorted(train_val_indices[n_train:]),
            'test': sorted(holdout_indices)
        }


def make_split_indices(samples: List[Dict[str, Any]],
                       strategy: str = 'random',
                       config: SplitConfig = None,
                       seed: int = 42,
                       **kwargs) -> Dict[str, List[int]]:
    """Create compositional splits for the dataset.

    Args:
        samples: List of scene graph samples
        strategy: Split strategy name ('random', 'color_shape', 'count_shape', 'relation')
        config: Split configuration (ratios)
        seed: Random seed for deterministic splits
        **kwargs: Additional arguments for specific strategies

    Returns:
        Dictionary with 'train', 'val', 'test' index lists

    Example:
        >>> samples = [...]  # List of scene graphs
        >>> splits = make_split_indices(
        ...     samples,
        ...     strategy='color_shape',
        ...     holdout_pairs=[('yellow', 'cube'), ('red', 'ball')],
        ...     seed=42
        ... )
    """
    if config is None:
        config = SplitConfig(name=strategy)

    if strategy == 'random':
        splitter = RandomSplit(seed=seed)
    elif strategy == 'color_shape':
        holdout_pairs = kwargs.get('holdout_pairs', [('yellow', 'cube')])
        splitter = ColorShapeHoldout(holdout_pairs=holdout_pairs, seed=seed)
    elif strategy == 'count_shape':
        holdout_pairs = kwargs.get('holdout_pairs', [(5, 'ball')])
        splitter = CountShapeHoldout(holdout_pairs=holdout_pairs, seed=seed)
    elif strategy == 'relation':
        holdout_relations = kwargs.get('holdout_relations', ['in_front_of'])
        splitter = RelationHoldout(holdout_relations=holdout_relations, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return splitter.make_split(samples, config)


def enumerate_all_samples() -> List[Dict[str, Any]]:
    """Enumerate all possible valid samples in the DSL.

    Returns:
        List of all valid scene graphs
    """
    samples = []

    colors = ["red", "green", "blue", "yellow"]
    shapes = ["ball", "cube", "block"]
    counts = [1, 2, 3, 4, 5]
    relations = ["left_of", "right_of", "on", "in_front_of"]

    # COUNT_SENT: all combinations of color × shape × count
    for color, shape, count in itertools.product(colors, shapes, counts):
        sample = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": shape, "color": color, "count": count}
            ],
            "relations": [],
            "constraints": {"no_overlap": True, "min_dist_px": 4, "layout": "pack"}
        }
        samples.append(sample)

    # REL_SENT: all combinations of (color × shape) × relation × (color × shape)
    for color1, shape1, rel, color2, shape2 in itertools.product(colors, shapes, relations, colors, shapes):
        sample = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": shape1, "color": color1, "count": 1},
                {"id": "o2", "shape": shape2, "color": color2, "count": 1}
            ],
            "relations": [
                {"type": rel, "src": "o1", "dst": "o2"}
            ],
            "constraints": {"no_overlap": True, "min_dist_px": 4, "layout": "pack"}
        }
        samples.append(sample)

    return samples
