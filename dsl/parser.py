"""Deterministic parser for scene DSL: text -> scene graph JSON."""
import re
from typing import Dict, List, Any, Optional


class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class SceneParser:
    """Deterministic finite-state parser for the scene DSL."""

    # Grammar constants
    COLORS = ["red", "green", "blue", "yellow"]
    SHAPES = ["ball", "cube", "block"]
    NUMBERS = ["one", "two", "three", "four", "five"]
    NUMBER_TO_INT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    RELATIONS = {
        "left of": "left_of",
        "right of": "right_of",
        "on": "on",
        "in front of": "in_front_of"
    }

    def __init__(self):
        """Initialize parser with regex patterns."""
        # Build regex patterns for the two sentence types
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for COUNT_SENT and REL_SENT."""
        # COUNT_SENT: "There (is|are) NUMBER COLOR SHAPE(s)?"
        number_pat = "|".join(self.NUMBERS)
        color_pat = "|".join(self.COLORS)
        shape_pat = "|".join(self.SHAPES)

        self.count_pattern = re.compile(
            rf"^There (is|are) ({number_pat}) ({color_pat}) ({shape_pat})(s)?\.$"
        )

        # REL_SENT: "The COLOR SHAPE (is|are) REL the COLOR SHAPE."
        # Relations need special handling for multi-word phrases
        self.rel_pattern_left = re.compile(
            rf"^The ({color_pat}) ({shape_pat}) (is|are) left of the ({color_pat}) ({shape_pat})\.$"
        )
        self.rel_pattern_right = re.compile(
            rf"^The ({color_pat}) ({shape_pat}) (is|are) right of the ({color_pat}) ({shape_pat})\.$"
        )
        self.rel_pattern_on = re.compile(
            rf"^The ({color_pat}) ({shape_pat}) (is|are) on the ({color_pat}) ({shape_pat})\.$"
        )
        self.rel_pattern_in_front = re.compile(
            rf"^The ({color_pat}) ({shape_pat}) (is|are) in front of the ({color_pat}) ({shape_pat})\.$"
        )

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text to scene graph JSON.

        Args:
            text: Input text following the DSL grammar

        Returns:
            Scene graph dictionary

        Raises:
            ParseError: If text doesn't match grammar
        """
        # Validate text ends with period
        if not text.endswith("."):
            raise ParseError(f"Text must end with '.': {text}")

        # Try COUNT_SENT pattern
        match = self.count_pattern.match(text)
        if match:
            return self._parse_count_sentence(match)

        # Try REL_SENT patterns
        for rel_name, pattern in [
            ("left_of", self.rel_pattern_left),
            ("right_of", self.rel_pattern_right),
            ("on", self.rel_pattern_on),
            ("in_front_of", self.rel_pattern_in_front)
        ]:
            match = pattern.match(text)
            if match:
                return self._parse_rel_sentence(match, rel_name)

        # No pattern matched
        raise ParseError(f"Text doesn't match any valid grammar pattern: {text}")

    def _parse_count_sentence(self, match: re.Match) -> Dict[str, Any]:
        """Parse COUNT_SENT: 'There (is|are) NUMBER COLOR SHAPE(s)?'"""
        verb, number_word, color, shape, plural_suffix = match.groups()
        count = self.NUMBER_TO_INT[number_word]

        # Validate plurality agreement
        if count == 1:
            if verb != "is":
                raise ParseError(f"Singular number '{number_word}' requires 'is', got '{verb}'")
            if plural_suffix:
                raise ParseError(f"Singular number '{number_word}' cannot have plural 's'")
        else:  # count >= 2
            if verb != "are":
                raise ParseError(f"Plural number '{number_word}' requires 'are', got '{verb}'")
            if not plural_suffix:
                raise ParseError(f"Plural number '{number_word}' requires plural 's'")

        # Build scene graph
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {
                    "id": "o1",
                    "shape": shape,
                    "color": color,
                    "count": count
                }
            ],
            "relations": [],
            "constraints": {
                "no_overlap": True,
                "min_dist_px": 4,
                "layout": "pack"
            }
        }

        return scene_graph

    def _parse_rel_sentence(self, match: re.Match, rel_type: str) -> Dict[str, Any]:
        """Parse REL_SENT: 'The COLOR SHAPE (is|are) REL the COLOR SHAPE'"""
        color1, shape1, verb, color2, shape2 = match.groups()

        # For relations, we expect singular objects (count=1)
        # Verb should be "is" for singular
        if verb != "is":
            raise ParseError(f"Relational sentence with singular objects requires 'is', got '{verb}'")

        # Build scene graph with two objects and a relation
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {
                    "id": "o1",
                    "shape": shape1,
                    "color": color1,
                    "count": 1
                },
                {
                    "id": "o2",
                    "shape": shape2,
                    "color": color2,
                    "count": 1
                }
            ],
            "relations": [
                {
                    "type": rel_type,
                    "src": "o1",
                    "dst": "o2"
                }
            ],
            "constraints": {
                "no_overlap": True,
                "min_dist_px": 4,
                "layout": "pack"
            }
        }

        return scene_graph

    def validate(self, text: str) -> bool:
        """Check if text matches the grammar without parsing.

        Args:
            text: Input text

        Returns:
            True if text is valid, False otherwise
        """
        try:
            self.parse(text)
            return True
        except ParseError:
            return False
