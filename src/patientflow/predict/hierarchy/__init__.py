"""Hierarchical demand prediction module.

This module provides tools for predicting hospital bed demand across different
organisational levels (subspecialty, division, hospital, etc.) using a
hierarchical structure.
"""

from .types import (
    DemandPrediction,
    PredictionBundle,
    FlowSelection,
    DEFAULT_PERCENTILES,
    DEFAULT_PRECISION,
    DEFAULT_MAX_PROBS,
)
from .structure import (
    Hierarchy,
    EntityType,
    HierarchyLevel,
    populate_hierarchy_from_dataframe,
)
from .calculation import DemandPredictor
from .orchestrator import HierarchicalPredictor, create_hierarchical_predictor

__all__ = [
    "DEFAULT_PERCENTILES",
    "DEFAULT_PRECISION",
    "DEFAULT_MAX_PROBS",
    "DemandPrediction",
    "PredictionBundle",
    "FlowSelection",
    "Hierarchy",
    "EntityType",
    "HierarchyLevel",
    "populate_hierarchy_from_dataframe",
    "DemandPredictor",
    "HierarchicalPredictor",
    "create_hierarchical_predictor",
]
