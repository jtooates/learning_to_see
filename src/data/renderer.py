"""Image rendering for 2D shape scenes."""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple

from src.data.scene import Scene, Shape, ShapeType


class SceneRenderer:
    """Renders scenes as PIL Images."""

    def __init__(self, canvas_size: int = 256, antialias: bool = True):
        """
        Initialize the renderer.

        Args:
            canvas_size: Size of the square canvas in pixels
            antialias: Whether to use antialiasing (renders at higher resolution then downsamples)
        """
        self.canvas_size = canvas_size
        self.antialias = antialias
        self.aa_scale = 4 if antialias else 1  # Antialiasing scale factor

    def render(self, scene: Scene) -> Image.Image:
        """
        Render a scene to a PIL Image.

        Args:
            scene: The scene to render

        Returns:
            PIL Image of the scene
        """
        # Create canvas (possibly at higher resolution for antialiasing)
        render_size = self.canvas_size * self.aa_scale
        image = Image.new("RGB", (render_size, render_size), scene.background_color)
        draw = ImageDraw.Draw(image)

        # Draw all shapes
        for shape in scene.get_all_shapes():
            self._draw_shape(draw, shape)

        # Downsample if antialiasing
        if self.antialias:
            image = image.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)

        return image

    def render_to_array(self, scene: Scene) -> np.ndarray:
        """
        Render a scene to a numpy array.

        Args:
            scene: The scene to render

        Returns:
            Numpy array of shape (height, width, 3) with values in [0, 255]
        """
        image = self.render(scene)
        return np.array(image)

    def _draw_shape(self, draw: ImageDraw.Draw, shape: Shape):
        """
        Draw a single shape.

        Args:
            draw: PIL ImageDraw object
            shape: The shape to draw
        """
        # Scale position and size for antialiasing
        x, y = shape.position
        x *= self.aa_scale
        y *= self.aa_scale
        size = shape.get_pixel_size() * self.aa_scale

        color = shape.get_rgb()

        if shape.shape_type == ShapeType.CIRCLE:
            self._draw_circle(draw, x, y, size, color)
        elif shape.shape_type == ShapeType.SQUARE:
            self._draw_square(draw, x, y, size, color)
        elif shape.shape_type == ShapeType.TRIANGLE:
            self._draw_triangle(draw, x, y, size, color)
        elif shape.shape_type == ShapeType.RECTANGLE:
            self._draw_rectangle(draw, x, y, size, color)

    def _draw_circle(
        self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]
    ):
        """Draw a circle."""
        radius = size // 2
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=color, outline=color)

    def _draw_square(
        self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]
    ):
        """Draw a square."""
        half = size // 2
        bbox = [x - half, y - half, x + half, y + half]
        draw.rectangle(bbox, fill=color, outline=color)

    def _draw_triangle(
        self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]
    ):
        """Draw an equilateral triangle pointing upward."""
        # Height of equilateral triangle
        height = int(size * 0.866)  # sqrt(3)/2
        half_base = size // 2

        # Define triangle points (pointing up)
        points = [
            (x, y - height // 2),  # Top vertex
            (x - half_base, y + height // 2),  # Bottom left
            (x + half_base, y + height // 2),  # Bottom right
        ]

        draw.polygon(points, fill=color, outline=color)

    def _draw_rectangle(
        self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]
    ):
        """Draw a rectangle (2:1 aspect ratio)."""
        width = size
        height = size // 2

        bbox = [x - width // 2, y - height // 2, x + width // 2, y + height // 2]
        draw.rectangle(bbox, fill=color, outline=color)
