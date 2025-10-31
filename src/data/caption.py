"""Caption generation with natural language variation."""

import random
from typing import List, Optional

from src.data.scene import Scene, SceneObject, SpatialRelation


class CaptionGenerator:
    """Generates natural language captions for scenes with variation."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the caption generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate_caption(self, scene: Scene) -> str:
        """
        Generate a caption for a scene.

        Args:
            scene: The scene to caption

        Returns:
            A natural language caption
        """
        if len(scene.objects) == 0:
            return "an empty scene"

        if len(scene.objects) == 1:
            return self._describe_single_object(scene.objects[0])

        # Multiple objects - use relationships if available
        if scene.relationships:
            return self._describe_with_relationships(scene)
        else:
            return self._describe_multiple_objects(scene)

    def _describe_single_object(self, obj: SceneObject) -> str:
        """
        Describe a single object.

        Args:
            obj: The scene object

        Returns:
            Description string
        """
        templates = [
            "{article} {size} {color} {shape}",
            "{article} {color} {shape} ({size})",
            "{article} {size} {color} {shape_plural}",
        ]

        template = random.choice(templates)

        # Handle plurality
        if obj.quantity == 1:
            article = self._get_article(obj.color)
            shape = obj.shape_type.value
            shape_plural = obj.shape_type.value
        else:
            article = str(obj.quantity)
            shape = obj.shape_type.value
            shape_plural = self._pluralize(obj.shape_type.value)

        return template.format(
            article=article,
            size=obj.size.value,
            color=obj.color,
            shape=shape,
            shape_plural=shape_plural,
        )

    def _describe_multiple_objects(self, scene: Scene) -> str:
        """
        Describe multiple objects without relationships.

        Args:
            scene: The scene

        Returns:
            Description string
        """
        descriptions = [self._describe_single_object(obj) for obj in scene.objects]

        if len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

    def _describe_with_relationships(self, scene: Scene) -> str:
        """
        Describe objects with spatial relationships.

        Args:
            scene: The scene with relationships

        Returns:
            Description string
        """
        # Use the first relationship
        rel = scene.relationships[0]
        subject_obj = scene.objects[rel.subject]
        object_obj = scene.objects[rel.object]

        # Choose a template format
        templates = [
            "{subject} {relation} {object}",
            "{object}, with {subject} {relation} it",
            "{subject} positioned {relation} {object}",
        ]

        # Randomly decide whether to flip the relationship
        if random.random() < 0.3:
            # Flip: describe from object's perspective
            subject_obj, object_obj = object_obj, subject_obj
            relation = self._flip_relation(rel.relation)
        else:
            relation = rel.relation

        template = random.choice(templates)

        subject_desc = self._describe_single_object(subject_obj)
        object_desc = self._describe_single_object(object_obj)

        return template.format(
            subject=subject_desc,
            relation=relation.value,
            object=object_desc,
        )

    def _flip_relation(self, relation: SpatialRelation) -> SpatialRelation:
        """
        Flip a spatial relationship to its opposite.

        Args:
            relation: The original relation

        Returns:
            The flipped relation
        """
        flips = {
            SpatialRelation.ABOVE: SpatialRelation.BELOW,
            SpatialRelation.BELOW: SpatialRelation.ABOVE,
            SpatialRelation.LEFT_OF: SpatialRelation.RIGHT_OF,
            SpatialRelation.RIGHT_OF: SpatialRelation.LEFT_OF,
            SpatialRelation.NEAR: SpatialRelation.NEAR,
        }
        return flips[relation]

    def _get_article(self, word: str) -> str:
        """
        Get the appropriate article (a/an) for a word.

        Args:
            word: The word to get article for

        Returns:
            "a" or "an"
        """
        vowels = "aeiou"
        return "an" if word[0].lower() in vowels else "a"

    def _pluralize(self, word: str) -> str:
        """
        Simple pluralization of shape names.

        Args:
            word: The singular word

        Returns:
            Plural form
        """
        if word.endswith("s"):
            return word + "es"
        elif word.endswith("y"):
            return word[:-1] + "ies"
        else:
            return word + "s"


def generate_varied_captions(scene: Scene, num_variations: int = 5) -> List[str]:
    """
    Generate multiple varied captions for the same scene.

    Args:
        scene: The scene to caption
        num_variations: Number of caption variations to generate

    Returns:
        List of caption strings
    """
    generator = CaptionGenerator()
    captions = []

    for _ in range(num_variations):
        caption = generator.generate_caption(scene)
        if caption not in captions:  # Avoid exact duplicates
            captions.append(caption)

    return captions
