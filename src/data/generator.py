"""Scene generation logic for creating random 2D shape scenes."""

import random
from typing import List, Tuple, Optional
import numpy as np

from src.data.scene import (
    Scene, SceneObject, Shape, ShapeType, Size, SpatialRelation, COLORS
)


class SceneGenerator:
    """Generates random scenes with shapes and spatial relationships."""

    def __init__(
        self,
        canvas_size: int = 256,
        min_objects: int = 1,
        max_objects: int = 3,
        min_quantity: int = 1,
        max_quantity: int = 4,
        relationship_probability: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize the scene generator.

        Args:
            canvas_size: Size of the square canvas in pixels
            min_objects: Minimum number of object groups per scene
            max_objects: Maximum number of object groups per scene
            min_quantity: Minimum number of shapes per object group
            max_quantity: Maximum number of shapes per object group
            relationship_probability: Probability of adding relationships between objects
            seed: Random seed for reproducibility
        """
        self.canvas_size = canvas_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.relationship_probability = relationship_probability

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.available_colors = list(COLORS.keys())
        self.available_shapes = list(ShapeType)
        self.available_sizes = list(Size)

    def generate_scene(self) -> Scene:
        """
        Generate a random scene.

        Returns:
            A Scene object with objects and relationships
        """
        scene = Scene(canvas_size=self.canvas_size)

        # Determine number of object groups
        num_objects = random.randint(self.min_objects, self.max_objects)

        # Generate each object group
        used_positions = []
        for _ in range(num_objects):
            obj = self._generate_scene_object(scene, used_positions)
            scene.add_object(obj)

            # Track positions to avoid overlaps
            for shape in obj.shapes:
                used_positions.append((shape.position, shape.get_pixel_size()))

        # Generate relationships between objects
        if len(scene.objects) >= 2 and random.random() < self.relationship_probability:
            self._generate_relationships(scene)

        return scene

    def _generate_scene_object(
        self, scene: Scene, used_positions: List[Tuple[Tuple[int, int], int]]
    ) -> SceneObject:
        """
        Generate a single scene object (group of identical shapes).

        Args:
            scene: The scene being built
            used_positions: List of (position, size) tuples for existing shapes

        Returns:
            A SceneObject with positioned shapes
        """
        # Sample attributes
        shape_type = random.choice(self.available_shapes)
        color = random.choice(self.available_colors)
        size = random.choice(self.available_sizes)
        quantity = random.randint(self.min_quantity, self.max_quantity)

        # Create individual shapes with positions
        shapes = []
        for _ in range(quantity):
            position = self._generate_position(used_positions, size)
            shape = Shape(
                shape_type=shape_type,
                color=color,
                size=size,
                position=position,
            )
            shapes.append(shape)
            # Temporarily add to used positions to avoid overlaps within this group
            used_positions.append((position, shape.get_pixel_size()))

        return SceneObject(
            shape_type=shape_type,
            color=color,
            size=size,
            quantity=quantity,
            shapes=shapes,
        )

    def _generate_position(
        self,
        used_positions: List[Tuple[Tuple[int, int], int]],
        size: Size,
    ) -> Tuple[int, int]:
        """
        Generate a position that doesn't overlap with existing shapes.

        Args:
            used_positions: List of (position, size) tuples for existing shapes
            size: Size of the new shape

        Returns:
            (x, y) position tuple
        """
        # Get pixel size for this shape
        pixel_size = {Size.SMALL: 30, Size.MEDIUM: 50, Size.LARGE: 80}[size]

        # Margins to keep shapes on canvas
        margin = pixel_size // 2 + 10

        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(margin, self.canvas_size - margin)
            y = random.randint(margin, self.canvas_size - margin)

            # Check for overlaps
            if self._is_valid_position((x, y), pixel_size, used_positions):
                return (x, y)

        # If we can't find a good position, return a random one anyway
        return (
            random.randint(margin, self.canvas_size - margin),
            random.randint(margin, self.canvas_size - margin),
        )

    def _is_valid_position(
        self,
        position: Tuple[int, int],
        size: int,
        used_positions: List[Tuple[Tuple[int, int], int]],
    ) -> bool:
        """
        Check if a position is valid (no overlaps with existing shapes).

        Args:
            position: Candidate (x, y) position
            size: Pixel size of the shape
            used_positions: List of (position, size) tuples for existing shapes

        Returns:
            True if position is valid
        """
        x, y = position

        for (used_x, used_y), used_size in used_positions:
            # Calculate minimum distance needed to avoid overlap
            min_distance = (size + used_size) // 2 + 15  # 15px buffer

            # Check Euclidean distance
            distance = np.sqrt((x - used_x) ** 2 + (y - used_y) ** 2)
            if distance < min_distance:
                return False

        return True

    def _generate_relationships(self, scene: Scene):
        """
        Generate spatial relationships between objects in the scene.

        Args:
            scene: The scene to add relationships to
        """
        # Pick two different objects
        if len(scene.objects) < 2:
            return

        indices = random.sample(range(len(scene.objects)), 2)
        subject_idx, object_idx = indices

        # Determine the actual spatial relationship based on positions
        subject_obj = scene.objects[subject_idx]
        object_obj = scene.objects[object_idx]

        # Use the centroid of each object group
        subject_centroid = self._get_centroid(subject_obj)
        object_centroid = self._get_centroid(object_obj)

        relation = self._determine_relation(subject_centroid, object_centroid)
        scene.add_relationship(subject_idx, relation, object_idx)

    def _get_centroid(self, obj: SceneObject) -> Tuple[float, float]:
        """
        Get the centroid position of a scene object.

        Args:
            obj: The scene object

        Returns:
            (x, y) centroid position
        """
        positions = [shape.position for shape in obj.shapes]
        x_mean = sum(p[0] for p in positions) / len(positions)
        y_mean = sum(p[1] for p in positions) / len(positions)
        return (x_mean, y_mean)

    def _determine_relation(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> SpatialRelation:
        """
        Determine the spatial relationship from pos1 to pos2.

        Args:
            pos1: (x, y) position of subject
            pos2: (x, y) position of object

        Returns:
            The spatial relationship
        """
        x1, y1 = pos1
        x2, y2 = pos2

        dx = x2 - x1
        dy = y2 - y1

        # Determine primary direction (use larger difference)
        if abs(dx) > abs(dy):
            # Horizontal relationship
            return SpatialRelation.LEFT_OF if dx > 0 else SpatialRelation.RIGHT_OF
        else:
            # Vertical relationship (note: y increases downward)
            return SpatialRelation.BELOW if dy > 0 else SpatialRelation.ABOVE
