"""Hierarchical demand prediction module.

This module provides tools for predicting hospital bed demand across different
organisational levels (subspecialty, division, hospital, etc.) using a
hierarchical structure.

Usage
-----
The typical workflow is:

1. Create or load a hierarchy structure:
   >>> hierarchy = Hierarchy.create_default_hospital()
   >>> # or load from YAML configuration
   >>> hierarchy = Hierarchy.from_yaml("config.yaml")

2. Populate the hierarchy with organizational data:
   >>> import pandas as pd
   >>> # The DataFrame must represent a proper tree structure where each child
   >>> # entity has exactly one parent. Each row represents one path through
   >>> # the hierarchy from bottom to top.
   >>> hierarchy_df = pd.DataFrame({
   ...     'subspecialty': ['Cardiology', 'Cardiology', 'Neurology', 'Neurology'],
   ...     'reporting_unit': ['Cardiac Unit', 'Cardiac Unit', 'Neuro Unit', 'Neuro Unit'],
   ...     'division': ['Medical Division', 'Medical Division', 'Medical Division', 'Medical Division'],
   ... })
   >>> # Important: The hierarchy must form a tree structure. Each child entity
   >>> # can only have one parent. If the same child appears with different
   >>> # parents in different rows, the last relationship processed will
   >>> # overwrite previous ones, silently creating an invalid hierarchy.
   >>> # For example, this would be INVALID:
   >>> #   Row 1: subspecialty='Cardiology', reporting_unit='Unit X', division='Division A'
   >>> #   Row 2: subspecialty='Cardiology', reporting_unit='Unit X', division='Division B'
   >>> # This would result in 'Unit X' being assigned only to 'Division B'
   >>> # (the last one processed), overwriting the relationship to 'Division A'.
   >>> # This violates the tree structure requirement and should be avoided.
   >>> # Note: Only entity types with a column_mapping entry will have entities
   >>> # created from the DataFrame. The top_level_id ensures a single top-level
   >>> # entity exists (created if not in DataFrame, consolidated if present).
   >>> column_mapping = {
   ...     'subspecialty': 'subspecialty',
   ...     'reporting_unit': 'reporting_unit',
   ...     'division': 'division',
   ...     # 'hospital' not included, so top-level entity will be created separately
   ... }
   >>> populate_hierarchy_from_dataframe(
   ...     hierarchy, hierarchy_df, column_mapping, top_level_id="uclh"
   ... )

3. Prepare prediction inputs (using functions from predict.service):
   >>> from patientflow.predict.service import build_service_data
   >>> bottom_level_data = build_service_data(
   ...     models=(ed_classifier, inpatient_classifier, spec_model, ...),
   ...     prediction_time=(12, 0),
   ...     ed_snapshots=ed_snapshots_df,
   ...     inpatient_snapshots=inpatient_snapshots_df,
   ...     specialties=['Cardiology', 'Neurology', ...],
   ...     prediction_window=timedelta(hours=8),
   ...     x1=4.0, y1=0.95, x2=8.0, y2=0.99
   ... )

4. Create predictor and run predictions:
   >>> predictor = create_hierarchical_predictor(
   ...     hierarchy_df=hierarchy_df,
   ...     column_mapping=column_mapping,
   ...     top_level_id="Hospital",
   ...     k_sigma=8.0
   ... )
   >>> # Or create manually:
   >>> from patientflow.predict.hierarchy import HierarchyPredictor
   >>> from patientflow.predict.demand import DemandPredictor
   >>> demand_predictor = DemandPredictor(k_sigma=8.0)
   >>> predictor = HierarchyPredictor(hierarchy, demand_predictor)
   >>> 
   >>> # Run predictions
   >>> results = predictor.predict_all_levels(
   ...     bottom_level_data,
   ...     flow_selection=FlowSelection.default()
   ... )

5. Access results:
   >>> bundle = results["Cardiology"]
   >>> print(f"Expected arrivals: {bundle.arrivals.expected_value:.1f}")
   >>> print(f"95th percentile: {bundle.arrivals.percentiles[95]} beds")
   >>> 
   >>> # Access hospital-level aggregated prediction
   >>> hospital_bundle = results["Hospital"]
   >>> print(f"Hospital expected arrivals: {hospital_bundle.arrivals.expected_value:.1f}")

Components
----------
- **Hierarchy**: Defines the organizational structure (levels, relationships).
  Use this to represent your hospital's organizational hierarchy. Can be created
  from YAML configuration or populated from a DataFrame.

- **DemandPredictor**: Core calculation engine for probability distributions.
  Performs convolution of distributions and statistical capping. Typically used
  indirectly through HierarchyPredictor, but can be used directly for
  single-level predictions.

- **HierarchyPredictor**: Orchestrates predictions across all hierarchy levels.
  This is the main entry point for hierarchical predictions. Manages the 3-phase
  algorithm (capping, bottom-level prediction, aggregation).

- **FlowSelection**: Configures which patient flows to include in predictions.
  Allows you to specify which inflows (ED current, ED yet-to-arrive, transfers,
  etc.) and outflows (departures) to include, and filter by cohort (all,
  elective, emergency).

- **PredictionBundle**: Complete prediction results for an entity, containing
  arrivals, departures, and net flow predictions with probability distributions.

- **DemandPrediction**: Individual prediction result with PMF, expected value,
  and percentiles for a single flow type (arrivals, departures, or net flow).

Architecture and Component Relationships
----------------------------------------
Understanding when to use each component and how they relate:

**Component Hierarchy:**
::
    
    Hierarchy (structure)
        ↓
    HierarchyPredictor (orchestration)
        ↓
    DemandPredictor (calculations)
        ↓
    PredictionBundle (results)

**When to Use Each Component:**

1. **`Hierarchy`**: Always needed first. Defines your organizational structure.
   - Use `Hierarchy.create_default_hospital()` for standard hospital hierarchy
   - Use `Hierarchy.from_yaml()` for custom hierarchy structures
   - Populate with `populate_hierarchy_from_dataframe()` to add your organizational data

2. **`create_hierarchical_predictor()` vs Manual Construction:**
   - **Use `create_hierarchical_predictor()`** (factory function) when:
     - You have a DataFrame with your hierarchy structure
     - You want a quick, one-step setup
     - You're using standard configurations
   - **Use manual construction** when:
     - You need fine-grained control over `DemandPredictor` parameters
     - You're building the hierarchy programmatically
     - You want to reuse a `DemandPredictor` instance across multiple hierarchies
     - Example manual construction:
       ::
       
           from patientflow.predict.hierarchy import HierarchyPredictor
           from patientflow.predict.demand import DemandPredictor
           demand_predictor = DemandPredictor(k_sigma=10.0)  # Custom k_sigma
           predictor = HierarchyPredictor(hierarchy, demand_predictor)

3. **`DemandPredictor` vs `HierarchyPredictor`:**
   - **Use `HierarchyPredictor`** (recommended) when:
     - You need predictions across multiple organizational levels
     - You want automatic aggregation from bottom to top
     - You're working with a complete hierarchy structure
   - **Use `DemandPredictor` directly** when:
     - You only need predictions for a single level (no aggregation)
     - You're doing custom prediction workflows
     - You're testing or debugging individual prediction calculations

**Prediction Flow (3-Phase Algorithm):**

The prediction process follows a strict 3-phase algorithm managed by `HierarchyPredictor`:

1. **Phase 1: Bottom-up Stats & Top-down Capping**
   - Recursively traverses the hierarchy from bottom to top
   - Calculates statistical properties (sum of means, combined variance) for each entity
   - Calculates max_support caps for each node based on statistical properties
   - Ensures bounded distribution sizes while maintaining accuracy

2. **Phase 2: Bottom-level Prediction**
   - Generates full PMF predictions for all bottom-level entities (services)
   - Uses the caps calculated in Phase 1 to bound distribution sizes
   - Only processes entities that have data in `bottom_level_data`

3. **Phase 3: Aggregation**
   - Aggregates predictions upward through the hierarchy using convolution
   - Applies caps at each level during aggregation
   - Computes net flow (arrivals - departures) at each level
   - Results are stored in `prediction_results` dictionary keyed by entity ID

**Data Flow:**
::
    
    build_service_data() → ServicePredictionInputs
        ↓
    predict_all_levels(bottom_level_data) → Dict[str, PredictionBundle]
        ↓
    Access results by entity ID → PredictionBundle → DemandPrediction

Notes
-----
The prediction process follows a 3-phase algorithm:

1. **Phase 1: Bottom-up Stats & Top-down Capping**
   - Recursively calculate statistical properties (sum of means, combined variance)
   - Calculate max_support caps for each node (statistical cap)

2. **Phase 2: Bottom-level Prediction**
   - Compute full PMF predictions for all bottom-level entities (services)
   - Uses caps from Phase 1 to bound distribution sizes

3. **Phase 3: Aggregation**
   - Aggregate predictions upwards through the hierarchy using convolution
   - Apply caps at each level during aggregation
   - Compute net flow at each level

The module uses discrete convolution to combine probability distributions.
Supports are clamped using top-down statistical caps calculated before
convolution, ensuring bounded array sizes while maintaining statistical accuracy.
"""

from patientflow.predict.types import (
    DemandPrediction,
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
from .orchestrate import HierarchicalPredictor, create_hierarchical_predictor

__all__ = [
    "DEFAULT_PERCENTILES",
    "DEFAULT_PRECISION",
    "DEFAULT_MAX_PROBS",
    "DemandPrediction",
    "PredictionBundle",
    "Hierarchy",
    "EntityType",
    "HierarchyLevel",
    "populate_hierarchy_from_dataframe",
    "calculate_hierarchical_stats",
    "HierarchicalPredictor",
    "create_hierarchical_predictor",
]
