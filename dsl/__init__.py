"""Scene DSL package for deterministic text ↔ scene graph conversion."""

from .tokens import Vocab
from .parser import SceneParser, ParseError
from .canonicalize import to_canonical, CanonicalizeError
from .fsm import ConstrainedPolicy, DecodingState, State
from .splits import (
    make_split_indices,
    enumerate_all_samples,
    SplitConfig,
    ColorShapeHoldout,
    CountShapeHoldout,
    RelationHoldout,
    RandomSplit
)

__all__ = [
    'Vocab',
    'SceneParser',
    'ParseError',
    'to_canonical',
    'CanonicalizeError',
    'ConstrainedPolicy',
    'DecodingState',
    'State',
    'make_split_indices',
    'enumerate_all_samples',
    'SplitConfig',
    'ColorShapeHoldout',
    'CountShapeHoldout',
    'RelationHoldout',
    'RandomSplit',
]
