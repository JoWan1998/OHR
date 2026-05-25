"""Compatibility re-exports for historical OHR model block imports."""

from ohr.models.adapter import FusionModule, TabularAdapter
from ohr.models.classifier import CooperativeOutputAggregator, ExpertBank, LeafExpert
from ohr.models.projection import OrthogonalProjection

__all__ = [
    "CooperativeOutputAggregator",
    "ExpertBank",
    "FusionModule",
    "LeafExpert",
    "OrthogonalProjection",
    "TabularAdapter",
]
