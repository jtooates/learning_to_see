"""Convert scene graph to canonical text representation."""
from typing import Dict, Any


class CanonicalizeError(Exception):
    """Raised when canonicalization fails."""
    pass


def to_canonical(scene_graph: Dict[str, Any]) -> str:
    """Convert scene graph to canonical text following the DSL grammar.

    Args:
        scene_graph: Scene graph dictionary

    Returns:
        Canonical text string

    Raises:
        CanonicalizeError: If scene graph is malformed
    """
    # Validate scene graph structure
    if "objects" not in scene_graph:
        raise CanonicalizeError("Scene graph missing 'objects' field")
    if "relations" not in scene_graph:
        raise CanonicalizeError("Scene graph missing 'relations' field")

    objects = scene_graph["objects"]
    relations = scene_graph["relations"]

    # Determine sentence type
    if len(relations) == 0:
        # COUNT_SENT
        if len(objects) != 1:
            raise CanonicalizeError(
                f"COUNT_SENT requires exactly 1 object, got {len(objects)}"
            )
        return _to_count_sentence(objects[0])

    elif len(relations) == 1:
        # REL_SENT
        if len(objects) != 2:
            raise CanonicalizeError(
                f"REL_SENT requires exactly 2 objects, got {len(objects)}"
            )
        return _to_rel_sentence(objects, relations[0])

    else:
        raise CanonicalizeError(
            f"Scene graph has {len(relations)} relations; only 0 or 1 supported in v1"
        )


def _to_count_sentence(obj: Dict[str, Any]) -> str:
    """Generate COUNT_SENT: 'There (is|are) NUMBER COLOR SHAPE(s)?'"""
    # Validate object fields
    required = ["shape", "color", "count"]
    for field in required:
        if field not in obj:
            raise CanonicalizeError(f"Object missing required field: {field}")

    shape = obj["shape"]
    color = obj["color"]
    count = obj["count"]

    # Map count to number word
    count_to_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    if count not in count_to_word:
        raise CanonicalizeError(f"Invalid count: {count}. Must be 1-5.")

    number_word = count_to_word[count]

    # Determine verb and plural suffix
    if count == 1:
        verb = "is"
        shape_form = shape
    else:
        verb = "are"
        shape_form = shape + "s"

    return f"There {verb} {number_word} {color} {shape_form}."


def _to_rel_sentence(objects: list, relation: Dict[str, Any]) -> str:
    """Generate REL_SENT: 'The COLOR SHAPE (is|are) REL the COLOR SHAPE'"""
    # Validate relation fields
    required = ["type", "src", "dst"]
    for field in required:
        if field not in relation:
            raise CanonicalizeError(f"Relation missing required field: {field}")

    rel_type = relation["type"]
    src_id = relation["src"]
    dst_id = relation["dst"]

    # Find source and destination objects
    src_obj = None
    dst_obj = None
    for obj in objects:
        if obj["id"] == src_id:
            src_obj = obj
        if obj["id"] == dst_id:
            dst_obj = obj

    if src_obj is None:
        raise CanonicalizeError(f"Source object not found: {src_id}")
    if dst_obj is None:
        raise CanonicalizeError(f"Destination object not found: {dst_id}")

    # Validate objects have count=1 (v1 constraint for relations)
    if src_obj.get("count", 1) != 1:
        raise CanonicalizeError(
            f"Relational objects must have count=1, src has count={src_obj.get('count')}"
        )
    if dst_obj.get("count", 1) != 1:
        raise CanonicalizeError(
            f"Relational objects must have count=1, dst has count={dst_obj.get('count')}"
        )

    # Map relation type to text
    rel_to_text = {
        "left_of": "left of",
        "right_of": "right of",
        "on": "on",
        "in_front_of": "in front of"
    }
    if rel_type not in rel_to_text:
        raise CanonicalizeError(f"Invalid relation type: {rel_type}")

    rel_text = rel_to_text[rel_type]

    # Extract colors and shapes
    src_color = src_obj["color"]
    src_shape = src_obj["shape"]
    dst_color = dst_obj["color"]
    dst_shape = dst_obj["shape"]

    # For singular objects, use "is"
    verb = "is"

    return f"The {src_color} {src_shape} {verb} {rel_text} the {dst_color} {dst_shape}."
