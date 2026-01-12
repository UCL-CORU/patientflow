"""Hierarchical prediction orchestrator.

This module provides HierarchicalPredictor, which orchestrates the 3-phase
prediction algorithm across all levels of the organizational hierarchy.
"""

from typing import Dict, Optional, Tuple, Any

from patientflow.predict.subspecialty import SubspecialtyPredictionInputs
from .structure import Hierarchy
from patientflow.predict.types import PredictionBundle, FlowSelection
from .calculation import DemandPredictor


class HierarchicalPredictor:
    """Orchestrates the hierarchical prediction process.

    This class manages the interaction between the hierarchy structure and the demand predictor,
    providing a method running predictions across all levels of the organization.

    The prediction process follows a 3-phase algorithm:

    Phase 1: Bottom-up Stats & Top-down Capping
    - Recursively calculate statistical properties (sum of means, combined variance) from bottom up
    - Calculate max_support caps for each node (statistical cap)
    - The top-level cap is driven by its total statistical properties
    - Intermediate caps respect both their own statistical properties and are sufficient to support children

    Phase 2: Bottom-level Prediction
    - Compute full PMF predictions for all bottom-level entities (subspecialties) using the caps from Phase 1

    Phase 3: Aggregation
    - Aggregate predictions upwards through the hierarchy using convolution
    - Apply caps at each level during aggregation to keep distribution sizes manageable
    - Compute net flow at each level

    Attributes
    ----------
    hierarchy : Hierarchy
        The organisational hierarchy structure
    predictor : DemandPredictor
        The calculation engine
    """

    def __init__(self, hierarchy: Hierarchy, predictor: DemandPredictor):
        self.hierarchy = hierarchy
        self.predictor = predictor
        # Store computed bundles: entity_id -> PredictionBundle
        self.prediction_results: Dict[str, PredictionBundle] = {}

    def predict_all_levels(
        self,
        bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
        top_level_id: Optional[str] = None,
        flow_selection: Optional[FlowSelection] = None,
    ) -> Dict[str, PredictionBundle]:
        """Run predictions for the entire hierarchy.

        This method executes the full 3-phase prediction algorithm (capping,
        bottom-level prediction, aggregation) for the specified subtree.

        Parameters
        ----------
        bottom_level_data : Dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping bottom-level entity IDs to their prediction inputs.
            Each value is a `SubspecialtyPredictionInputs` object containing:
            
            - ``subspecialty_id``: Identifier for the subspecialty (must match the key)
            - ``prediction_window``: Time window for predictions (typically a timedelta)
            - ``inflows``: Dictionary of arrival flows (e.g., 'ed_current', 'ed_yta', 
              'non_ed_yta', 'elective_yta', 'elective_transfers', 'emergency_transfers')
            - ``outflows``: Dictionary of departure flows (e.g., 'elective_departures', 
              'emergency_departures')
            
            **Key Requirements:**
            - Keys must exactly match the entity IDs in the hierarchy at the bottom level.
              For example, if your bottom level is 'subspecialty', the keys should be
              subspecialty names like 'Gsurg LowGI', 'Gsurg UppGI', 'Older Acute', etc., 
              exactly as they appear in the hierarchy (case-sensitive).
            - Typically created using `build_subspecialty_data()` from the 
              `patientflow.predict.subspecialty` module.
            - Only entities that exist in the hierarchy and are reachable from the
              specified `top_level_id` will have predictions generated.
        top_level_id : str, optional
            Root entity to start prediction from. If None, predicts for all
            top-level entities in the hierarchy. The entity ID must exist in the
            hierarchy and match exactly (case-sensitive).
        flow_selection : FlowSelection, optional
            Selection specifying which flows to include. If None, uses
            FlowSelection.default() which includes all flows. Use this to restrict
            predictions to specific patient flows (e.g., ED inflows only).

        Returns
        -------
        Dict[str, PredictionBundle]
            Dictionary containing prediction bundles for all entities in the
            hierarchy (bottom-level and aggregated), keyed by entity ID.
            
            **Dictionary Structure:**
            - Keys are entity IDs (original names, not prefixed IDs) for all entities
              in the hierarchy that were processed, including:
              - Bottom-level entities (subspecialties) that had data in `bottom_level_data`
              - All intermediate-level entities (reporting units, divisions, etc.)
              - Top-level entities (hospital, board, etc.)
            - Values are `PredictionBundle` objects containing:
              - ``arrivals``: `DemandPrediction` with PMF, expected value, percentiles
              - ``departures``: `DemandPrediction` with PMF, expected value, percentiles
              - ``net_flow``: `DemandPrediction` with PMF, expected value, percentiles
            
            **Example structure:**
            ::
            
                {
                    'Gsurg LowGI': PredictionBundle(...),              # Bottom-level
                    'Gsurg UppGI': PredictionBundle(...),              # Bottom-level
                    'Gastrointestinal Surgery': PredictionBundle(...), # Aggregated
                    'Surgery Division': PredictionBundle(...),          # Aggregated
                    'uclh': PredictionBundle(...)                       # Top-level aggregated
                }
            
            **Note:** Entities in the hierarchy that don't have corresponding entries
            in `bottom_level_data` will not appear in the results dictionary, even if
            they are part of the hierarchy structure.

        Examples
        --------
        Run predictions for entire hierarchy with default flow selection:

        >>> from patientflow.predict.subspecialty import build_subspecialty_data
        >>> bottom_level_data = build_subspecialty_data(
        ...     models=(...),
        ...     prediction_time=(12, 0),
        ...     ed_snapshots=ed_snapshots_df,
        ...     inpatient_snapshots=inpatient_snapshots_df,
        ...     specialties=['Gsurg LowGI', 'Gsurg UppGI', 'Older Acute', 'Older Gen'],
        ...     prediction_window=timedelta(hours=8),
        ...     x1=4.0, y1=0.95, x2=8.0, y2=0.99
        ... )
        >>> results = predictor.predict_all_levels(bottom_level_data)
        >>> # Access subspecialty-level prediction
        >>> gsurg_bundle = results['Gsurg LowGI']
        >>> print(f"Expected arrivals: {gsurg_bundle.arrivals.expected_value:.1f}")
        >>> # Access hospital-level aggregated prediction
        >>> hospital_bundle = results['uclh']
        >>> print(f"Hospital expected arrivals: {hospital_bundle.arrivals.expected_value:.1f}")

        Run predictions with custom flow selection (ED inflows only):

        >>> from patientflow.predict.hierarchy import FlowSelection
        >>> flow_selection = FlowSelection.custom(
        ...     include_ed_current=True,
        ...     include_ed_yta=True,
        ...     include_non_ed_yta=False,
        ...     include_elective_yta=False,
        ...     include_transfers_in=False,
        ...     include_departures=False,
        ...     cohort="emergency"
        ... )
        >>> results = predictor.predict_all_levels(
        ...     bottom_level_data,
        ...     flow_selection=flow_selection
        ... )

        Run predictions for a specific subtree:

        >>> # Predict only for entities under 'Surgery Division'
        >>> results = predictor.predict_all_levels(
        ...     bottom_level_data,
        ...     top_level_id="Surgery Division"
        ... )
        >>> # Results will only contain entities under 'Surgery Division'
        >>> print(list(results.keys()))
        ['Gsurg LowGI', 'Gsurg UppGI', 'Gastrointestinal Surgery', 'Surgery Division']

        Notes
        -----
        **Data Format Requirements:**

        The `bottom_level_data` dictionary should be created using `build_subspecialty_data()`
        from `patientflow.predict.subspecialty`. Each `SubspecialtyPredictionInputs` object
        contains probability distributions (PMFs) for current patients and Poisson parameters
        for yet-to-arrive patients, organized by flow type (inflows/outflows).

        **Entity ID Matching:**

        - Subspecialty IDs in `bottom_level_data` keys **must exactly match** (case-sensitive)
          the entity IDs at the bottom level of the hierarchy.
        - If a subspecialty ID in `bottom_level_data` doesn't exist in the hierarchy:
          - It will be silently ignored during prediction (no error raised)
          - It won't appear in the results dictionary
          - This is typically not an error condition, as you may have data for subspecialties
            that aren't part of the current hierarchy structure
        - If a subspecialty exists in the hierarchy but not in `bottom_level_data`:
          - No prediction will be generated for that subspecialty
          - It won't appear in the results dictionary
          - Aggregated predictions at higher levels will exclude this subspecialty
          - This is the expected behavior when you don't have data for all subspecialties

        **Error Handling:**

        - ``ValueError``: Raised if `top_level_id` is provided but doesn't exist in the hierarchy
        - ``ValueError``: Raised if `flow_selection` is invalid (e.g., conflicting settings)
        - Missing subspecialties in `bottom_level_data` are handled gracefully (no error)
        - Extra subspecialties in `bottom_level_data` (not in hierarchy) are ignored (no error)

        **Prediction Process:**

        The method follows a 3-phase algorithm:

        1. **Phase 1 (Capping)**: Calculate statistical caps for each entity based on
           the sum of means and combined variance of all distributions in its subtree
        2. **Phase 2 (Bottom-level Prediction)**: Generate full PMF predictions for
           all bottom-level entities that have data
        3. **Phase 3 (Aggregation)**: Aggregate predictions upward through the hierarchy
           using convolution, applying caps at each level

        **Performance Considerations:**

        - Predictions are computed recursively from the specified `top_level_id` downward
        - Only entities in the subtree rooted at `top_level_id` are processed
        - Intermediate results are cached in `self.prediction_results` for efficient access
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()
        
        # Validate flow selection configuration
        flow_selection.validate()

        self.prediction_results.clear()
        self.predictor.clear_truncated_mass()

        if top_level_id is None:
            # Predict for all top-level entities
            top_level_type = self.hierarchy.get_top_level_type()
            top_entities = self.hierarchy.get_entities_by_type(top_level_type)
            for entity_id in top_entities:
                self._predict_subtree(entity_id, bottom_level_data, flow_selection)
        else:
            self._predict_subtree(top_level_id, bottom_level_data, flow_selection)

        return self.prediction_results

    def _predict_subtree(
        self,
        entity_id: str,
        bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
        flow_selection: FlowSelection,
    ):
        entity_type = self.hierarchy.get_entity_type(entity_id)
        if entity_type is None:
            raise ValueError(f"Entity type not found for id: {entity_id}")

        # PHASE 1: Calculate caps top-down
        # We calculate max_support for this node based on the statistics of its subtree
        _, _, arrivals_max_support = self.predictor.calculate_hierarchical_stats(
            entity_id,
            entity_type,
            bottom_level_data,
            self.hierarchy,
            "arrivals",
            flow_selection,
        )
        _, _, departures_max_support = self.predictor.calculate_hierarchical_stats(
            entity_id,
            entity_type,
            bottom_level_data,
            self.hierarchy,
            "departures",
            flow_selection,
        )

        # Iterate children
        children = self.hierarchy.get_children(entity_id, entity_type)

        if not children:
            # PHASE 2: Bottom-level prediction
            # This is a leaf node (subspecialty)
            if entity_id in bottom_level_data:
                # Prediction logic for base level
                # Pass explicit caps to prevent large arrays from Poisson tails
                # But wait, predict_subspecialty doesn't adhere to external caps directly
                # It computes its own based on its flows.
                # The caps calculated in Phase 1 are for THIS level's aggregation.
                # For leaf nodes, we use the standard prediction logic.
                inputs = bottom_level_data[entity_id]
                bundle = self.predictor.predict_subspecialty(
                    entity_id, inputs, flow_selection
                )
                self.prediction_results[entity_id] = bundle
            return

        # Recursive step: predict all children
        # Note: In a true top-down capping implementation, we would pass down
        # constraints. Here we calculate caps independently at each level which
        # is sufficient for correctness and simpler.
        child_bundles = []
        for child_id in children:
            # Verify child is in hierarchy (should be guaranteed by get_children)
            self._predict_subtree(child_id, bottom_level_data, flow_selection)
            if child_id in self.prediction_results:
                child_bundles.append(self.prediction_results[child_id])

        # PHASE 3: Aggregation
        # Aggregate child results to create this level's prediction
        if child_bundles:
            bundle = self.predictor._create_bundle_from_children(
                entity_id,
                entity_type.name,
                child_bundles,
                arrivals_max_support,
                departures_max_support,
            )
            self.prediction_results[entity_id] = bundle

    def get_prediction(self, entity_id: str) -> Optional[PredictionBundle]:
        """Get the prediction bundle for a specific entity.

        Parameters
        ----------
        entity_id : str
            Entity ID to retrieve

        Returns
        -------
        Optional[PredictionBundle]
            Prediction bundle if available, None otherwise
        """
        return self.prediction_results.get(entity_id)

    def get_truncated_mass_stats(self) -> Dict[str, Any]:
        """Get statistics about truncated probability mass from the predictor.

        Returns
        -------
        dict
            Truncation statistics from the DemandPredictor
        """
        return self.predictor.get_truncated_mass_stats()


import pandas as pd
from .structure import populate_hierarchy_from_dataframe

def create_hierarchical_predictor(
    hierarchy_df: Optional[pd.DataFrame] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    top_level_id: Optional[str] = None,
    config_path: Optional[str] = None,
    k_sigma: float = 8.0,
) -> HierarchicalPredictor:
    """Factory function to create a configured HierarchicalPredictor.

    This helper function simplifies the initialization of the predictor by
    creating calculation components and populating the hierarchy structure.

    Parameters
    ----------
    hierarchy_df : pd.DataFrame, optional
        DataFrame containing organizational structure. Each row represents
        one path through the hierarchy from bottom to top. The DataFrame
        must form a proper tree structure (each child has exactly one parent).
    column_mapping : Dict[str, str], optional
        Mapping from DataFrame column names to entity type names. Entity type names
        are the names of the levels defined in the hierarchy structure (e.g., 'subspecialty',
        'reporting_unit', 'division', 'board', 'hospital'). These must match the entity
        type names used when creating the hierarchy (from create_default_hospital() or
        from_yaml()). Example: {'sub_specialty': 'subspecialty', 'reporting_unit': 'reporting_unit'}
    top_level_id : str, optional
        ID of the top-level entity (e.g., "Hospital" or "uclh")
    config_path : str, optional
        Path to the hierarchy configuration YAML file.
        If provided, loads hierarchy structure from file.
        If None, uses default hospital hierarchy.
    k_sigma : float, default=8.0
        Cap width in standard deviations. Controls the maximum support size
        for probability distributions. Higher values allow larger distributions
        but use more memory.

    Returns
    -------
    HierarchicalPredictor
        Ready-to-use predictor instance

    Examples
    --------
    Create predictor with default hospital hierarchy:

    >>> predictor = create_hierarchical_predictor()

    Create predictor from DataFrame:

    >>> import pandas as pd
    >>> hierarchy_df = pd.DataFrame({
    ...     'subspecialty': ['Gsurg LowGI', 'Gsurg UppGI', 'Older Acute', 'Older Gen'],
    ...     'reporting_unit': ['Gastrointestinal Surgery', 'Gastrointestinal Surgery', 'Care Of the Elderly', 'Care Of the Elderly'],
    ...     'division': ['Surgery Division', 'Surgery Division', 'Medicine Division', 'Medicine Division'],
    ... })
    >>> column_mapping = {
    ...     'subspecialty': 'subspecialty',
    ...     'reporting_unit': 'reporting_unit',
    ...     'division': 'division',
    ... }
    >>> predictor = create_hierarchical_predictor(
    ...     hierarchy_df=hierarchy_df,
    ...     column_mapping=column_mapping,
    ...     top_level_id="uclh"
    ... )

    Create predictor from YAML configuration:

    >>> predictor = create_hierarchical_predictor(
    ...     config_path="hierarchy_config.yaml",
    ...     hierarchy_df=hierarchy_df,
    ...     column_mapping=column_mapping,
    ...     top_level_id="Hospital",
    ...     k_sigma=10.0
    ... )
    """
    if config_path:
        hierarchy = Hierarchy.from_yaml(config_path)
    else:
        hierarchy = Hierarchy.create_default_hospital()

    if (
        hierarchy_df is not None
        and column_mapping is not None
        and top_level_id is not None
    ):
        populate_hierarchy_from_dataframe(
            hierarchy, hierarchy_df, column_mapping, top_level_id
        )

    predictor = DemandPredictor(k_sigma=k_sigma)
    return HierarchicalPredictor(hierarchy, predictor)
