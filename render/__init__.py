"""Scene renderer package for generating synthetic images from scene graphs."""

from .renderer import SceneRenderer, RenderMetadata
from .validate import validate_relations, check_no_overlap, check_no_overlap_except_relations

__all__ = [
    'SceneRenderer',
    'RenderMetadata',
    'validate_relations',
    'check_no_overlap',
    'check_no_overlap_except_relations',
]
