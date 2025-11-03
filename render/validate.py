"""Validation functions for rendered scenes."""
import numpy as np
from typing import List, Dict
from .renderer import InstanceMetadata


def validate_relations(instances: List[InstanceMetadata], relations: List[Dict]) -> None:
    """Validate that all relations hold in pixel coordinates.

    Args:
        instances: List of rendered instance metadata
        relations: List of relation specifications

    Raises:
        ValueError: If any relation does not hold
    """
    # Build instance lookup by object_id
    inst_by_id = {}
    for inst in instances:
        if inst.object_id not in inst_by_id:
            inst_by_id[inst.object_id] = []
        inst_by_id[inst.object_id].append(inst)

    for rel in relations:
        rel_type = rel['type']
        src_id = rel['src']
        dst_id = rel['dst']

        # Get instances (use first instance if multiple)
        src_insts = inst_by_id.get(src_id, [])
        dst_insts = inst_by_id.get(dst_id, [])

        if not src_insts or not dst_insts:
            raise ValueError(f"Missing instances for relation {src_id} -> {dst_id}")

        # Validate relation (use first instance of each)
        src = src_insts[0]
        dst = dst_insts[0]

        if rel_type == 'left_of':
            if not _check_left_of(src, dst):
                raise ValueError(
                    f"Relation 'left_of' violated: {src.center[0]} should be < {dst.center[0]}"
                )
        elif rel_type == 'right_of':
            if not _check_right_of(src, dst):
                raise ValueError(
                    f"Relation 'right_of' violated: {src.center[0]} should be > {dst.center[0]}"
                )
        elif rel_type == 'on':
            if not _check_on(src, dst):
                raise ValueError(
                    f"Relation 'on' violated: {src.bbox} should be on top of {dst.bbox}"
                )
        elif rel_type == 'in_front_of':
            if not _check_in_front_of(src, dst):
                raise ValueError(
                    f"Relation 'in_front_of' violated: {src.center} should be in front of {dst.center}"
                )


def _check_left_of(src: InstanceMetadata, dst: InstanceMetadata, margin: int = 5) -> bool:
    """Check if src is to the left of dst."""
    src_x = src.center[0]
    dst_x = dst.center[0]
    return src_x + margin < dst_x


def _check_right_of(src: InstanceMetadata, dst: InstanceMetadata, margin: int = 5) -> bool:
    """Check if src is to the right of dst."""
    src_x = src.center[0]
    dst_x = dst.center[0]
    return src_x > dst_x + margin


def _check_on(src: InstanceMetadata, dst: InstanceMetadata, tolerance: int = 3) -> bool:
    """Check if src is on top of dst (vertical stacking with contact, no overlap).

    For 'on', we expect:
    1. Bottom of src touches top of dst (vertical contact)
    2. Some horizontal overlap (objects aligned)
    3. Very little or no mask IoU (objects touching but not overlapping)
    """
    # Bottom of src should be near top of dst
    src_bottom = src.bbox[3]  # y1
    dst_top = dst.bbox[1]  # y0

    # Check vertical alignment - should be very close (touching)
    vertical_distance = abs(src_bottom - dst_top)
    vertical_ok = vertical_distance <= tolerance

    # Check horizontal overlap (x ranges should overlap for stacking)
    src_x0, src_x1 = src.bbox[0], src.bbox[2]
    dst_x0, dst_x1 = dst.bbox[0], dst.bbox[2]

    horizontal_overlap = not (src_x1 < dst_x0 or src_x0 > dst_x1)

    # Check mask IoU - should be very small (objects touching, not overlapping)
    iou = compute_iou(src.mask, dst.mask)
    no_overlap = iou < 0.05  # Very small overlap allowed for contact

    return vertical_ok and horizontal_overlap and no_overlap


def _check_in_front_of(src: InstanceMetadata, dst: InstanceMetadata) -> bool:
    """Check if src is in front of dst (requires overlap/occlusion).

    For 'in front of', we expect the objects to overlap significantly,
    indicating that src occludes dst.
    """
    # Check that centers are close (objects should be nearly at same position)
    cx1, cy1 = src.center
    cx2, cy2 = dst.center

    center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # Centers should be very close for occlusion (within ~10 pixels)
    if center_dist > 10:
        return False

    # Check for overlap using masks
    iou = compute_iou(src.mask, dst.mask)

    # Should have significant overlap (at least 0.2 IoU)
    return iou >= 0.2


def check_no_overlap(instances: List[InstanceMetadata], threshold: float = 0.1) -> bool:
    """Check that instances have minimal overlap.

    Args:
        instances: List of rendered instances
        threshold: Maximum allowed IoU between any pair

    Returns:
        True if no significant overlap detected
    """
    n = len(instances)

    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(instances[i].mask, instances[j].mask)
            if iou > threshold:
                return False

    return True


def check_no_overlap_except_relations(instances: List[InstanceMetadata],
                                       relations: List[Dict],
                                       threshold: float = 0.1) -> bool:
    """Check that instances have minimal overlap, except for in_front_of and on relations.

    Args:
        instances: List of rendered instances
        relations: List of relations (to identify allowed overlaps)
        threshold: Maximum allowed IoU between any pair

    Returns:
        True if no unexpected overlap detected
    """
    # Build set of (src_id, dst_id) pairs that are allowed to overlap/touch
    allowed_overlaps = set()
    for rel in relations:
        if rel['type'] in ('in_front_of', 'on'):
            allowed_overlaps.add((rel['src'], rel['dst']))
            allowed_overlaps.add((rel['dst'], rel['src']))  # Order doesn't matter

    n = len(instances)

    for i in range(n):
        for j in range(i + 1, n):
            # Check if this pair is allowed to overlap
            id1 = instances[i].object_id
            id2 = instances[j].object_id

            if (id1, id2) in allowed_overlaps or (id2, id1) in allowed_overlaps:
                continue  # Skip overlap check for in_front_of and on relations

            iou = compute_iou(instances[i].mask, instances[j].mask)
            if iou > threshold:
                return False

    return True


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)

    Returns:
        IoU score [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def check_min_distance(instances: List[InstanceMetadata], min_dist_px: int) -> bool:
    """Check that all instances maintain minimum distance.

    Args:
        instances: List of rendered instances
        min_dist_px: Minimum allowed distance between centers

    Returns:
        True if all pairs satisfy minimum distance
    """
    n = len(instances)

    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1 = instances[i].center
            cx2, cy2 = instances[j].center

            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            if dist < min_dist_px:
                return False

    return True
