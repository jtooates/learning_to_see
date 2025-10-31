"""Tests for scene representation."""

import pytest
from src.data.scene import (
    Scene, SceneObject, Shape, ShapeType, Size, SpatialRelation, COLORS
)


def test_shape_creation():
    """Test creating a shape."""
    shape = Shape(
        shape_type=ShapeType.CIRCLE,
        color="red",
        size=Size.MEDIUM,
        position=(100, 100),
    )

    assert shape.shape_type == ShapeType.CIRCLE
    assert shape.color == "red"
    assert shape.size == Size.MEDIUM
    assert shape.position == (100, 100)
    assert shape.get_rgb() == COLORS["red"]
    assert shape.get_pixel_size() == 50  # Medium size


def test_scene_object_creation():
    """Test creating a scene object."""
    shapes = [
        Shape(ShapeType.SQUARE, "blue", Size.SMALL, (50, 50)),
        Shape(ShapeType.SQUARE, "blue", Size.SMALL, (70, 50)),
    ]

    obj = SceneObject(
        shape_type=ShapeType.SQUARE,
        color="blue",
        size=Size.SMALL,
        quantity=2,
        shapes=shapes,
    )

    assert obj.quantity == 2
    assert len(obj.shapes) == 2
    assert obj.color == "blue"


def test_scene_object_validation():
    """Test that scene object validates quantity."""
    shapes = [Shape(ShapeType.CIRCLE, "red", Size.LARGE, (100, 100))]

    with pytest.raises(ValueError):
        SceneObject(
            shape_type=ShapeType.CIRCLE,
            color="red",
            size=Size.LARGE,
            quantity=2,  # Mismatch!
            shapes=shapes,
        )


def test_scene_creation():
    """Test creating a scene."""
    scene = Scene(canvas_size=256)

    assert scene.canvas_size == 256
    assert len(scene.objects) == 0
    assert len(scene.relationships) == 0


def test_scene_add_object():
    """Test adding objects to a scene."""
    scene = Scene(canvas_size=256)

    shapes = [Shape(ShapeType.TRIANGLE, "green", Size.MEDIUM, (128, 128))]
    obj = SceneObject(
        shape_type=ShapeType.TRIANGLE,
        color="green",
        size=Size.MEDIUM,
        quantity=1,
        shapes=shapes,
    )

    idx = scene.add_object(obj)
    assert idx == 0
    assert len(scene.objects) == 1


def test_scene_add_relationship():
    """Test adding relationships to a scene."""
    scene = Scene(canvas_size=256)

    # Add two objects
    obj1 = SceneObject(
        ShapeType.SQUARE, "red", Size.SMALL, 1,
        [Shape(ShapeType.SQUARE, "red", Size.SMALL, (50, 50))]
    )
    obj2 = SceneObject(
        ShapeType.CIRCLE, "blue", Size.LARGE, 1,
        [Shape(ShapeType.CIRCLE, "blue", Size.LARGE, (200, 200))]
    )

    idx1 = scene.add_object(obj1)
    idx2 = scene.add_object(obj2)

    scene.add_relationship(idx1, SpatialRelation.LEFT_OF, idx2)

    assert len(scene.relationships) == 1
    assert scene.relationships[0].subject == idx1
    assert scene.relationships[0].relation == SpatialRelation.LEFT_OF
    assert scene.relationships[0].object == idx2


def test_scene_get_all_shapes():
    """Test getting all shapes from a scene."""
    scene = Scene(canvas_size=256)

    obj1 = SceneObject(
        ShapeType.SQUARE, "red", Size.SMALL, 2,
        [
            Shape(ShapeType.SQUARE, "red", Size.SMALL, (50, 50)),
            Shape(ShapeType.SQUARE, "red", Size.SMALL, (70, 50)),
        ]
    )
    obj2 = SceneObject(
        ShapeType.CIRCLE, "blue", Size.LARGE, 1,
        [Shape(ShapeType.CIRCLE, "blue", Size.LARGE, (200, 200))]
    )

    scene.add_object(obj1)
    scene.add_object(obj2)

    all_shapes = scene.get_all_shapes()
    assert len(all_shapes) == 3
