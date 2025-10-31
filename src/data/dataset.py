"""PyTorch Dataset wrapper for shape scene generation."""

from typing import Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from src.data.generator import SceneGenerator
from src.data.caption import CaptionGenerator
from src.data.renderer import SceneRenderer


class ShapeSceneDataset(Dataset):
    """PyTorch Dataset that generates (image, caption) pairs on-the-fly."""

    def __init__(
        self,
        size: int = 1000,
        canvas_size: int = 256,
        min_objects: int = 1,
        max_objects: int = 3,
        min_quantity: int = 1,
        max_quantity: int = 4,
        relationship_probability: float = 0.8,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            size: Number of samples in the dataset
            canvas_size: Size of the square canvas in pixels
            min_objects: Minimum number of object groups per scene
            max_objects: Maximum number of object groups per scene
            min_quantity: Minimum number of shapes per object group
            max_quantity: Maximum number of shapes per object group
            relationship_probability: Probability of adding relationships between objects
            transform: Optional transform to apply to images
            seed: Random seed for reproducibility
            normalize: Whether to normalize images to [0, 1] range
        """
        self.size = size
        self.canvas_size = canvas_size
        self.normalize = normalize

        # Initialize generators
        self.scene_generator = SceneGenerator(
            canvas_size=canvas_size,
            min_objects=min_objects,
            max_objects=max_objects,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            relationship_probability=relationship_probability,
            seed=seed,
        )
        self.caption_generator = CaptionGenerator(seed=seed)
        self.renderer = SceneRenderer(canvas_size=canvas_size, antialias=True)

        # Image transforms
        self.transform = transform
        if self.transform is None and self.normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to [0, 1] and changes to (C, H, W)
            ])
        elif self.transform is None and not self.normalize:
            # Just convert to tensor without normalization
            self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single (image, caption) pair.

        Args:
            idx: Index (used to seed the random generator for reproducibility)

        Returns:
            Tuple of (image_tensor, caption_string)
            - image_tensor: shape (3, canvas_size, canvas_size), values in [0, 1] if normalized
            - caption_string: natural language description
        """
        # Use index to seed for reproducibility within the dataset
        # This allows the same index to always return the same sample
        scene_seed = hash((self.scene_generator.canvas_size, idx)) % (2**32)
        import random
        random.seed(scene_seed)
        np.random.seed(scene_seed)

        # Generate scene
        scene = self.scene_generator.generate_scene()

        # Generate caption
        caption = self.caption_generator.generate_caption(scene)

        # Render image
        image = self.renderer.render(scene)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)

        return image, caption

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get a sample as a numpy array instead of tensor.

        Args:
            idx: Index

        Returns:
            Tuple of (image_array, caption_string)
            - image_array: shape (canvas_size, canvas_size, 3), values in [0, 255]
            - caption_string: natural language description
        """
        # Generate with same seed
        scene_seed = hash((self.scene_generator.canvas_size, idx)) % (2**32)
        import random
        random.seed(scene_seed)
        np.random.seed(scene_seed)

        scene = self.scene_generator.generate_scene()
        caption = self.caption_generator.generate_caption(scene)
        image_array = self.renderer.render_to_array(scene)

        return image_array, caption


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of (image, caption) tuples

    Returns:
        Tuple of (images_tensor, captions_list)
    """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(captions)
