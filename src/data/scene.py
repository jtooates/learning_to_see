"""Scene representation for 2D shape scenes."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class ShapeType(Enum):
    """Supported shape types."""
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"


class Size(Enum):
    """Size categories for shapes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SpatialRelation(Enum):
    """Spatial relationships between scene objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left of"
    RIGHT_OF = "right of"
    NEAR = "near"


# Color name to RGB mapping
COLORS = {
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (50, 205, 50),
    "yellow": (255, 215, 0),
    "purple": (138, 43, 226),
    "orange": (255, 140, 0),
    "pink": (255, 105, 180),
    "cyan": (0, 206, 209),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
}


@dataclass
class Shape:
    """A single shape instance."""
    shape_type: ShapeType
    color: str  # Color name (e.g., "red")
    size: Size
    position: Tuple[int, int]  # (x, y) center position in pixels

    def get_rgb(self) -> Tuple[int, int, int]:
        """Get RGB values for this shape's color."""
        return COLORS.get(self.color, (128, 128, 128))

    def get_pixel_size(self) -> int:
        """Get the pixel dimension for this shape's size."""
        size_mapping = {
            Size.SMALL: 30,
            Size.MEDIUM: 50,
            Size.LARGE: 80,
        }
        return size_mapping[self.size]


@dataclass
class SceneObject:
    """A group of identical shapes in a scene."""
    shape_type: ShapeType
    color: str
    size: Size
    quantity: int
    shapes: List[Shape]  # Individual shape instances with positions

    def __post_init__(self):
        """Validate that quantity matches number of shapes."""
        if len(self.shapes) != self.quantity:
            raise ValueError(
                f"Quantity {self.quantity} doesn't match number of shapes {len(self.shapes)}"
            )


@dataclass
class Relationship:
    """A spatial relationship between two scene objects."""
    subject: int  # Index of subject object in scene.objects list
    relation: SpatialRelation
    object: int  # Index of object in scene.objects list


class Scene:
    """A complete 2D scene with multiple objects and relationships."""

    def __init__(self, canvas_size: int = 256, background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize a scene.

        Args:
            canvas_size: Size of the square canvas in pixels
            background_color: RGB tuple for background
        """
        self.canvas_size = canvas_size
        self.background_color = background_color
        self.objects: List[SceneObject] = []
        self.relationships: List[Relationship] = []

    def add_object(self, obj: SceneObject) -> int:
        """
        Add a scene object and return its index.

        Args:
            obj: The scene object to add

        Returns:
            Index of the added object
        """
        self.objects.append(obj)
        return len(self.objects) - 1

    def add_relationship(self, subject_idx: int, relation: SpatialRelation, object_idx: int):
        """
        Add a spatial relationship between two objects.

        Args:
            subject_idx: Index of the subject object
            relation: The spatial relationship
            object_idx: Index of the object
        """
        if subject_idx >= len(self.objects) or object_idx >= len(self.objects):
            raise ValueError("Invalid object index in relationship")

        self.relationships.append(Relationship(subject_idx, relation, object_idx))

    def get_all_shapes(self) -> List[Shape]:
        """Get a flat list of all shapes in the scene."""
        all_shapes = []
        for obj in self.objects:
            all_shapes.extend(obj.shapes)
        return all_shapes

    def __repr__(self) -> str:
        """String representation of the scene."""
        obj_strs = []
        for i, obj in enumerate(self.objects):
            count = "" if obj.quantity == 1 else f"{obj.quantity} "
            obj_strs.append(f"[{i}] {count}{obj.size.value} {obj.color} {obj.shape_type.value}(s)")

        rel_strs = []
        for rel in self.relationships:
            rel_strs.append(f"  {rel.subject} {rel.relation.value} {rel.object}")

        return "Scene:\n  Objects:\n    " + "\n    ".join(obj_strs) + \
               "\n  Relationships:\n" + "\n".join(rel_strs) if rel_strs else ""
