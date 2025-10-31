"""Tests for scene generator."""

import pytest
from src.data.generator import SceneGenerator
from src.data.scene import ShapeType, Size


def test_generator_creation():
    """Test creating a scene generator."""
    gen = SceneGenerator(
        canvas_size=256,
        min_objects=1,
        max_objects=3,
        seed=42,
    )

    assert gen.canvas_size == 256
    assert gen.min_objects == 1
    assert gen.max_objects == 3


def test_generate_scene():
    """Test generating a scene."""
    gen = SceneGenerator(canvas_size=256, seed=42)
    scene = gen.generate_scene()

    assert scene is not None
    assert scene.canvas_size == 256
    assert len(scene.objects) >= gen.min_objects
    assert len(scene.objects) <= gen.max_objects


def test_scene_reproducibility():
    """Test that scenes are reproducible with same seed."""
    gen1 = SceneGenerator(canvas_size=256, seed=42)
    gen2 = SceneGenerator(canvas_size=256, seed=42)

    scene1 = gen1.generate_scene()
    scene2 = gen2.generate_scene()

    # Check that scenes have same structure
    assert len(scene1.objects) == len(scene2.objects)

    for obj1, obj2 in zip(scene1.objects, scene2.objects):
        assert obj1.shape_type == obj2.shape_type
        assert obj1.color == obj2.color
        assert obj1.size == obj2.size
        assert obj1.quantity == obj2.quantity


def test_object_attributes():
    """Test that generated objects have valid attributes."""
    gen = SceneGenerator(canvas_size=256, seed=42)
    scene = gen.generate_scene()

    for obj in scene.objects:
        # Check that shape type is valid
        assert obj.shape_type in ShapeType

        # Check that size is valid
        assert obj.size in Size

        # Check that quantity matches shapes
        assert obj.quantity == len(obj.shapes)
        assert obj.quantity >= gen.min_quantity
        assert obj.quantity <= gen.max_quantity


def test_position_validity():
    """Test that generated positions are within canvas bounds."""
    canvas_size = 256
    gen = SceneGenerator(canvas_size=canvas_size, seed=42)

    # Generate multiple scenes to test
    for _ in range(10):
        scene = gen.generate_scene()

        for shape in scene.get_all_shapes():
            x, y = shape.position
            size = shape.get_pixel_size()
            margin = size // 2 + 10

            # Positions should be within reasonable bounds
            assert x >= margin
            assert x <= canvas_size - margin
            assert y >= margin
            assert y <= canvas_size - margin
