"""Tests for scene renderer."""

import pytest
import numpy as np
from PIL import Image
from src.data.renderer import SceneRenderer
from src.data.scene import Scene, SceneObject, Shape, ShapeType, Size


def test_renderer_creation():
    """Test creating a renderer."""
    renderer = SceneRenderer(canvas_size=256)
    assert renderer.canvas_size == 256
    assert renderer.antialias is True


def test_render_empty_scene():
    """Test rendering an empty scene."""
    renderer = SceneRenderer(canvas_size=256, antialias=False)
    scene = Scene(canvas_size=256)

    image = renderer.render(scene)

    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_render_single_shape():
    """Test rendering a scene with a single shape."""
    renderer = SceneRenderer(canvas_size=256, antialias=False)
    scene = Scene(canvas_size=256)

    obj = SceneObject(
        ShapeType.CIRCLE, "red", Size.MEDIUM, 1,
        [Shape(ShapeType.CIRCLE, "red", Size.MEDIUM, (128, 128))]
    )
    scene.add_object(obj)

    image = renderer.render(scene)

    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)


def test_render_to_array():
    """Test rendering to numpy array."""
    renderer = SceneRenderer(canvas_size=256, antialias=False)
    scene = Scene(canvas_size=256)

    obj = SceneObject(
        ShapeType.SQUARE, "blue", Size.LARGE, 1,
        [Shape(ShapeType.SQUARE, "blue", Size.LARGE, (128, 128))]
    )
    scene.add_object(obj)

    array = renderer.render_to_array(scene)

    assert isinstance(array, np.ndarray)
    assert array.shape == (256, 256, 3)
    assert array.dtype == np.uint8
    assert array.min() >= 0
    assert array.max() <= 255


def test_render_all_shapes():
    """Test rendering all shape types."""
    renderer = SceneRenderer(canvas_size=256, antialias=False)

    shape_types = [ShapeType.CIRCLE, ShapeType.SQUARE, ShapeType.TRIANGLE, ShapeType.RECTANGLE]

    for shape_type in shape_types:
        scene = Scene(canvas_size=256)
        obj = SceneObject(
            shape_type, "green", Size.MEDIUM, 1,
            [Shape(shape_type, "green", Size.MEDIUM, (128, 128))]
        )
        scene.add_object(obj)

        image = renderer.render(scene)
        assert isinstance(image, Image.Image)


def test_antialiasing():
    """Test that antialiasing produces different output."""
    scene = Scene(canvas_size=64)
    obj = SceneObject(
        ShapeType.CIRCLE, "red", Size.LARGE, 1,
        [Shape(ShapeType.CIRCLE, "red", Size.LARGE, (32, 32))]
    )
    scene.add_object(obj)

    renderer_aa = SceneRenderer(canvas_size=64, antialias=True)
    renderer_no_aa = SceneRenderer(canvas_size=64, antialias=False)

    array_aa = renderer_aa.render_to_array(scene)
    array_no_aa = renderer_no_aa.render_to_array(scene)

    # With antialiasing, there should be more color variation (smoother edges)
    # Without antialiasing, colors should be more binary
    assert not np.array_equal(array_aa, array_no_aa)
