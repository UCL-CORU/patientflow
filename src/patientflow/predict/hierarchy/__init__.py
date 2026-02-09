"""Hierarchical demand prediction module.

This module provides tools for predicting hospital bed demand across different
organisational levels (e.g. subspecialty, division, hospital) using a
hierarchical structure. Predictions are generated at the bottom level
(services) and aggregated upward through convolution of probability
distributions, with statistical capping at each level to control
distribution size.

Classes
-------
Hierarchy
    Defines the organisational structure (levels and parent-child
    relationships). Can be loaded from YAML or populated from a DataFrame.
HierarchicalPredictor
    Orchestrates the 3-phase prediction algorithm across all hierarchy
    levels. This is the main entry point for hierarchical predictions.
PredictionResults
    Dictionary-like container returned by ``HierarchicalPredictor``, with
    flexible key access by entity name or prefixed ID.
FlowSelection
    Configures which patient flows (ED current, yet-to-arrive, transfers,
    departures) to include in predictions, with cohort filtering
    (all, elective, emergency).
DemandPrediction
    Individual prediction result with PMF, expected value, mode, and
    percentiles for a single flow type.
PredictionBundle
    Complete prediction results for an entity, containing arrivals,
    departures, and net flow ``DemandPrediction`` objects.

Notes
-----
The prediction process follows a 3-phase algorithm managed by
``HierarchicalPredictor``:

1. **Phase 1 -- Bottom-up stats and top-down capping**: recursively
   calculates statistical properties (sum of means, combined variance)
   for each node and derives ``max_support`` caps to bound distribution
   sizes.
2. **Phase 2 -- Bottom-level prediction**: generates full PMF predictions
   for leaf-level entities (services), bounded by the Phase 1 caps.
3. **Phase 3 -- Aggregation**: aggregates predictions upward through the
   hierarchy using convolution, applying caps at each level and computing
   net flow (arrivals minus departures).

Examples
--------
Create a predictor using the factory function:

>>> predictor = create_hierarchical_predictor(
...     hierarchy_df=hierarchy_df,
...     column_mapping=column_mapping,
...     top_level_id="Hospital",
...     k_sigma=8.0,
... )

Or construct one manually for more control:

>>> from patientflow.predict.demand import DemandPredictor
>>> demand_predictor = DemandPredictor(k_sigma=10.0)
>>> predictor = HierarchicalPredictor(hierarchy, demand_predictor)

Run predictions and access results:

>>> results = predictor.predict_all_levels(
...     prediction_inputs,
...     flow_selection=FlowSelection.default(),
... )
>>> bundle = results["medical"]
>>> print(f"Expected arrivals: {bundle.arrivals.expectation:.1f}")
>>> print(f"Percentiles: {bundle.arrivals.percentiles}")
>>> print(f"90% probability of needing at least "
...       f"{bundle.arrivals.min_beds_with_probability(0.9)} beds")
"""

from patientflow.predict.types import (
    DemandPrediction,
    FlowSelection,
    PredictionBundle,
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
from .calculate import calculate_hierarchical_stats
from .orchestrate import (
    HierarchicalPredictor,
    create_hierarchical_predictor,
    PredictionResults,
)

__all__ = [
    "DEFAULT_PERCENTILES",
    "DEFAULT_PRECISION",
    "DEFAULT_MAX_PROBS",
    "DemandPrediction",
    "FlowSelection",
    "PredictionBundle",
    "Hierarchy",
    "EntityType",
    "HierarchyLevel",
    "populate_hierarchy_from_dataframe",
    "calculate_hierarchical_stats",
    "HierarchicalPredictor",
    "create_hierarchical_predictor",
    "PredictionResults",
]
