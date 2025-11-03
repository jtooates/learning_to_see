"""Scene renderer: converts scene graphs to 64x64 RGB images with metadata."""
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import random


@dataclass
class InstanceMetadata:
    """Metadata for a single rendered instance."""
    object_id: str
    shape: str
    color: str
    center: Tuple[int, int]  # (x, y)
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    mask: np.ndarray  # Binary mask (H, W)


@dataclass
class RenderMetadata:
    """Complete metadata for a rendered scene."""
    instances: List[InstanceMetadata] = field(default_factory=list)
    relations_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


class SceneRenderer:
    """Renders scene graphs to images with procedural drawing."""

    # Color mappings (RGB)
    COLORS = {
        'red': (220, 50, 50),
        'green': (50, 200, 50),
        'blue': (50, 100, 220),
        'yellow': (230, 220, 50),
        'white': (255, 255, 255),
        'gray': (180, 180, 180),
    }

    # Shape sizes (radius or half-width)
    BALL_RADIUS = 6
    CUBE_SIZE = 8  # half-width
    BLOCK_WIDTH = 12
    BLOCK_HEIGHT = 7

    def __init__(self, width: int = 64, height: int = 64,
                 min_dist_px: int = 2, max_tries: int = 100, seed: Optional[int] = None):
        """Initialize renderer.

        Args:
            width: Image width
            height: Image height
            min_dist_px: Minimum distance between instance centers
            max_tries: Maximum rejection sampling attempts
            seed: Random seed for deterministic rendering
        """
        self.width = width
        self.height = height
        self.min_dist_px = min_dist_px
        self.max_tries = max_tries
        self.rng = random.Random(seed)

    def render(self, scene_graph: Dict[str, Any]) -> Tuple[Image.Image, RenderMetadata]:
        """Render scene graph to image.

        Args:
            scene_graph: Scene graph dictionary

        Returns:
            Tuple of (image, metadata)

        Raises:
            RuntimeError: If layout cannot be satisfied after max_tries
        """
        # Extract scene parameters
        canvas = scene_graph.get('canvas', {})
        objects = scene_graph.get('objects', [])
        relations = scene_graph.get('relations', [])
        constraints = scene_graph.get('constraints', {})

        # Create blank canvas
        bg_color = self.COLORS.get(canvas.get('bg', 'white'), (255, 255, 255))
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image, 'RGBA')

        # Render instances with layout constraints
        metadata = RenderMetadata()

        for attempt in range(self.max_tries):
            try:
                # Generate layout
                layout = self._generate_layout(objects, relations, metadata.instances)

                # Clear and redraw
                image = Image.new('RGB', (self.width, self.height), bg_color)
                draw = ImageDraw.Draw(image, 'RGBA')
                metadata.instances.clear()

                # Draw all instances
                for obj_id, obj_data, positions in layout:
                    for pos in positions:
                        instance = self._draw_shape(
                            draw, obj_data['shape'], obj_data['color'],
                            pos, obj_id, image
                        )
                        metadata.instances.append(instance)

                # Validate layout
                if self._validate_layout(metadata.instances, relations, constraints):
                    metadata.relations_valid = True
                    return image, metadata

            except LayoutError:
                continue

        raise RuntimeError(f"Could not generate valid layout after {self.max_tries} attempts")

    def _generate_layout(self, objects: List[Dict], relations: List[Dict],
                         existing: List[InstanceMetadata]) -> List[Tuple[str, Dict, List[Tuple[int, int]]]]:
        """Generate positions for all instances.

        Returns:
            List of (object_id, object_data, positions) tuples
        """
        layout = []
        used_positions = []

        # Handle relational constraints first
        rel_map = {rel['src']: rel for rel in relations}

        # Build dependency order (topological sort)
        ordered_objs = self._order_objects_by_relations(objects, relations)

        for obj in ordered_objs:
            obj_id = obj['id']
            count = obj.get('count', 1)
            positions = []

            # Check if this object has a relation constraint
            if obj_id in rel_map:
                rel = rel_map[obj_id]
                # Find destination object instances (should be laid out already)
                dst_instances = [
                    inst for inst in existing
                    if inst.object_id == rel['dst']
                ]

                if dst_instances:
                    # Get destination info
                    dst_inst = dst_instances[0]
                    dst_pos = dst_inst.center
                    dst_shape = dst_inst.shape

                    # Position relative to destination
                    base_pos = self._compute_relative_position(
                        dst_pos, rel['type'], obj['shape'], dst_shape
                    )
                else:
                    # Destination not placed yet, use random
                    base_pos = self._random_position(obj['shape'])
            else:
                # No constraint, random position
                base_pos = self._random_position(obj['shape'])

            # Generate positions for all instances of this object
            for i in range(count):
                # Try to find a valid position with retries
                pos_found = False
                for retry in range(20):  # Per-instance retry
                    if i == 0 and retry == 0:
                        pos = base_pos
                    else:
                        # Add jitter for multiple instances
                        pos = self._jitter_position(base_pos, i, obj['shape'])

                    # Check constraints
                    if self._check_position_valid(pos, used_positions, obj['shape']):
                        positions.append(pos)
                        used_positions.append(pos)
                        pos_found = True
                        break

                if not pos_found:
                    raise LayoutError(f"Could not place instance {i} of {obj_id}")

            layout.append((obj_id, obj, positions))

        return layout

    def _order_objects_by_relations(self, objects: List[Dict],
                                     relations: List[Dict]) -> List[Dict]:
        """Order objects so that relation destinations are placed first."""
        # Build dependency graph
        obj_dict = {obj['id']: obj for obj in objects}
        depends_on = {obj['id']: [] for obj in objects}

        for rel in relations:
            # src depends on dst being placed first
            depends_on[rel['src']].append(rel['dst'])

        # Topological sort (simple: dst first, then src)
        ordered = []
        placed = set()

        while len(ordered) < len(objects):
            for obj_id in obj_dict:
                if obj_id in placed:
                    continue

                # Check if dependencies are placed
                deps = depends_on[obj_id]
                if all(dep in placed for dep in deps):
                    ordered.append(obj_dict[obj_id])
                    placed.add(obj_id)
                    break

        return ordered

    def _random_position(self, shape: str) -> Tuple[int, int]:
        """Generate a random valid position for a shape."""
        margin = 12  # Stay away from edges
        x = self.rng.randint(margin, self.width - margin)
        y = self.rng.randint(margin, self.height - margin)
        return (x, y)

    def _jitter_position(self, base_pos: Tuple[int, int], index: int,
                         shape: str) -> Tuple[int, int]:
        """Add jitter to position for multiple instances."""
        x, y = base_pos
        jitter_range = 20
        dx = self.rng.randint(-jitter_range, jitter_range)
        dy = self.rng.randint(-jitter_range, jitter_range)

        # Ensure within bounds
        x = max(12, min(self.width - 12, x + dx))
        y = max(12, min(self.height - 12, y + dy))

        return (x, y)

    def _compute_relative_position(self, dst_pos: Tuple[int, int],
                                    rel_type: str, shape: str, dst_shape: str = None) -> Tuple[int, int]:
        """Compute position for src relative to dst based on relation type."""
        x_dst, y_dst = dst_pos
        margin = 20

        if rel_type == 'left_of':
            x = x_dst - margin
            y = y_dst + self.rng.randint(-5, 5)
        elif rel_type == 'right_of':
            x = x_dst + margin
            y = y_dst + self.rng.randint(-5, 5)
        elif rel_type == 'on':
            # Stack vertically: src sits on top of dst with no gap
            # Align centers horizontally for realistic "resting on" appearance
            x = x_dst  # Centers aligned (no jitter for stability)

            # Get shape heights
            src_height = self._get_shape_height(shape)
            dst_height = self._get_shape_height(dst_shape) if dst_shape else 8

            # Position src so bottom of src touches top of dst
            # y_dst is center of dst, top of dst is at y_dst - dst_height
            # src center should be at (top of dst) - src_height
            y = y_dst - dst_height - src_height
        elif rel_type == 'in_front_of':
            # Place nearly at same position to create occlusion
            # Src should overlap dst (drawn later = in front)
            x = x_dst + self.rng.randint(-3, 3)
            y = y_dst + self.rng.randint(-3, 3)  # Very small offset for occlusion
        else:
            x, y = dst_pos

        # Clamp to valid range
        x = max(15, min(self.width - 15, x))
        y = max(15, min(self.height - 15, y))

        return (x, y)

    def _get_shape_height(self, shape: str) -> int:
        """Get the height of a shape from its center to bottom edge."""
        if shape == 'ball':
            return self.BALL_RADIUS
        elif shape == 'cube':
            return self.CUBE_SIZE
        elif shape == 'block':
            return self.BLOCK_HEIGHT
        return 8  # default

    def _check_position_valid(self, pos: Tuple[int, int],
                              used_positions: List[Tuple[int, int]],
                              shape: str,
                              min_dist: Optional[int] = None) -> bool:
        """Check if position satisfies distance constraints."""
        if min_dist is None:
            min_dist = self.min_dist_px

        x, y = pos

        # Check not too close to existing positions
        for used_x, used_y in used_positions:
            dist = np.sqrt((x - used_x)**2 + (y - used_y)**2)
            if dist < min_dist:
                return False

        return True

    def _draw_shape(self, draw: ImageDraw.ImageDraw, shape: str, color: str,
                    center: Tuple[int, int], obj_id: str,
                    image: Image.Image) -> InstanceMetadata:
        """Draw a shape and return its metadata."""
        x, y = center
        rgb = self.COLORS.get(color, (128, 128, 128))

        if shape == 'ball':
            r = self.BALL_RADIUS
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=rgb, outline=None)

        elif shape == 'cube':
            s = self.CUBE_SIZE
            bbox = (x - s, y - s, x + s, y + s)
            draw.rectangle(bbox, fill=rgb, outline=None)

        elif shape == 'block':
            w, h = self.BLOCK_WIDTH, self.BLOCK_HEIGHT
            bbox = (x - w, y - h, x + w, y + h)
            draw.rectangle(bbox, fill=rgb, outline=None)
        else:
            bbox = (x - 5, y - 5, x + 5, y + 5)

        # Generate mask
        mask = self._create_mask(image, bbox, shape, center)

        return InstanceMetadata(
            object_id=obj_id,
            shape=shape,
            color=color,
            center=center,
            bbox=bbox,
            mask=mask
        )

    def _create_mask(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                     shape: str, center: Tuple[int, int]) -> np.ndarray:
        """Create binary mask for a shape."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        x0, y0, x1, y1 = bbox

        # Clamp to image bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.width, x1)
        y1 = min(self.height, y1)

        if shape == 'ball':
            cx, cy = center
            r = self.BALL_RADIUS
            for yi in range(y0, y1):
                for xi in range(x0, x1):
                    if (xi - cx)**2 + (yi - cy)**2 <= r**2:
                        mask[yi, xi] = 1
        else:
            # Rectangle shapes
            mask[y0:y1, x0:x1] = 1

        return mask

    def _validate_layout(self, instances: List[InstanceMetadata],
                         relations: List[Dict],
                         constraints: Dict) -> bool:
        """Validate that layout satisfies all constraints."""
        # Import here to avoid circular dependency
        from .validate import validate_relations, check_no_overlap_except_relations

        # Check overlap constraint (but allow overlap for in_front_of relations)
        if constraints.get('no_overlap', True):
            if not check_no_overlap_except_relations(instances, relations):
                return False

        # Check relation constraints
        try:
            validate_relations(instances, relations)
            return True
        except ValueError:
            return False


class LayoutError(Exception):
    """Raised when layout constraints cannot be satisfied."""
    pass
