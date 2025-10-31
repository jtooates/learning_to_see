"""Tests for caption generator."""

import pytest
from src.data.caption import CaptionGenerator
from src.data.scene import Scene, SceneObject, Shape, ShapeType, Size, SpatialRelation


def test_caption_generator_creation():
    """Test creating a caption generator."""
    gen = CaptionGenerator(seed=42)
    assert gen is not None


def test_single_object_caption():
    """Test captioning a scene with a single object."""
    gen = CaptionGenerator(seed=42)

    scene = Scene(canvas_size=256)
    obj = SceneObject(
        ShapeType.CIRCLE, "red", Size.LARGE, 1,
        [Shape(ShapeType.CIRCLE, "red", Size.LARGE, (128, 128))]
    )
    scene.add_object(obj)

    caption = gen.generate_caption(scene)

    assert isinstance(caption, str)
    assert len(caption) > 0
    assert "red" in caption
    assert "circle" in caption
    assert "large" in caption


def test_multiple_objects_caption():
    """Test captioning a scene with multiple objects."""
    gen = CaptionGenerator(seed=42)

    scene = Scene(canvas_size=256)

    obj1 = SceneObject(
        ShapeType.SQUARE, "blue", Size.SMALL, 1,
        [Shape(ShapeType.SQUARE, "blue", Size.SMALL, (50, 50))]
    )
    obj2 = SceneObject(
        ShapeType.TRIANGLE, "green", Size.MEDIUM, 2,
        [
            Shape(ShapeType.TRIANGLE, "green", Size.MEDIUM, (150, 50)),
            Shape(ShapeType.TRIANGLE, "green", Size.MEDIUM, (180, 50)),
        ]
    )

    scene.add_object(obj1)
    scene.add_object(obj2)

    caption = gen.generate_caption(scene)

    assert isinstance(caption, str)
    assert len(caption) > 0


def test_caption_with_relationship():
    """Test captioning a scene with relationships."""
    gen = CaptionGenerator(seed=42)

    scene = Scene(canvas_size=256)

    obj1 = SceneObject(
        ShapeType.SQUARE, "blue", Size.LARGE, 1,
        [Shape(ShapeType.SQUARE, "blue", Size.LARGE, (100, 50))]
    )
    obj2 = SceneObject(
        ShapeType.CIRCLE, "red", Size.SMALL, 3,
        [
            Shape(ShapeType.CIRCLE, "red", Size.SMALL, (100, 150)),
            Shape(ShapeType.CIRCLE, "red", Size.SMALL, (120, 150)),
            Shape(ShapeType.CIRCLE, "red", Size.SMALL, (140, 150)),
        ]
    )

    idx1 = scene.add_object(obj1)
    idx2 = scene.add_object(obj2)
    scene.add_relationship(idx1, SpatialRelation.ABOVE, idx2)

    caption = gen.generate_caption(scene)

    assert isinstance(caption, str)
    assert len(caption) > 0


def test_empty_scene():
    """Test captioning an empty scene."""
    gen = CaptionGenerator(seed=42)
    scene = Scene(canvas_size=256)

    caption = gen.generate_caption(scene)
    assert caption == "an empty scene"


def test_article_selection():
    """Test article selection (a vs an)."""
    gen = CaptionGenerator(seed=42)

    # Test 'an' for vowel-starting words
    assert gen._get_article("orange") == "an"
    assert gen._get_article("apple") == "an"

    # Test 'a' for consonant-starting words
    assert gen._get_article("blue") == "a"
    assert gen._get_article("red") == "a"


def test_pluralization():
    """Test pluralization."""
    gen = CaptionGenerator(seed=42)

    assert gen._pluralize("circle") == "circles"
    assert gen._pluralize("square") == "squares"
    assert gen._pluralize("triangle") == "triangles"
