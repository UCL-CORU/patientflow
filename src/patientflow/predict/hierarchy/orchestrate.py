"""Hierarchical prediction orchestrator.

This module provides HierarchicalPredictor, which orchestrates the 3-phase
prediction algorithm across all levels of the organisational hierarchy.
"""

from typing import Dict, Optional, Any, List

from patientflow.predict.service import ServicePredictionInputs
from patientflow.predict.hierarchy.structure import (
    Hierarchy,
    populate_hierarchy_from_dataframe,
    EntityType,
)
from patientflow.predict.types import PredictionBundle, FlowSelection
from patientflow.predict.demand import DemandPredictor
from patientflow.predict.hierarchy.calculate import calculate_hierarchical_stats

import pandas as pd


class PredictionResults(dict):
    """Dictionary-like container for prediction results with flexible key access.

    Supports two key formats:
    1. Prefixed ID: "hospital:UCLH" (always works)
    2. Original name: "UCLH" (works if unique, raises KeyError if ambiguous)

    This class prevents data loss from name collisions while allowing convenient
    access using original entity names when they are unambiguous.

    Examples
    --------
    >>> results = predictor.predict_all_levels(bottom_level_data)
    >>> # Prefixed ID (always works)
    >>> bundle = results["hospital:UCLH"]
    >>>
    >>> # Original name (only if unique)
    >>> bundle = results["UCLH"]  # Works if "UCLH" appears only once
    >>> bundle = results["Acute Medicine"]  # Raises KeyError if ambiguous

    Notes
    -----
    All dict access methods support flexible key lookup using either prefixed
    IDs or original names. For ambiguous original names (appearing at multiple
    levels), bracket access raises ``KeyError`` while ``get()`` returns the
    default value.
    """

    def __init__(self, data: Dict[str, PredictionBundle]):
        super().__init__(data)
        # Build cache: original_name -> [list of prefixed_ids]
        self._name_to_prefixed: Dict[str, List[str]] = {}
        for prefixed_id in data.keys():
            if ":" in prefixed_id:
                _, original_name = prefixed_id.split(":", 1)
                if original_name not in self._name_to_prefixed:
                    self._name_to_prefixed[original_name] = []
                self._name_to_prefixed[original_name].append(prefixed_id)

    def __getitem__(self, key: str) -> PredictionBundle:
        # Prefixed ID (contains ':')
        if ":" in key:
            return super().__getitem__(key)

        # Original name - check if unambiguous
        if key in self._name_to_prefixed:
            matching_ids = self._name_to_prefixed[key]
            if len(matching_ids) == 1:
                # Unique match
                return super().__getitem__(matching_ids[0])
            else:
                # Ambiguous - raise helpful error
                raise KeyError(
                    f"Ambiguous entity name '{key}'. Found at multiple levels: "
                    f"{', '.join(matching_ids)}. Use prefixed ID instead."
                )

        # Not found
        raise KeyError(f"Entity '{key}' not found in results")

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False

        # Prefixed ID (contains ':')
        if ":" in key:
            return super().__contains__(key)

        # Original name - check if unambiguous
        if key in self._name_to_prefixed:
            matching_ids = self._name_to_prefixed[key]
            # Return True only if unique (ambiguous keys are not considered "in")
            return len(matching_ids) == 1

        return False

    def get(  # type: ignore[override]
        self, key: str, default: Optional[PredictionBundle] = None
    ) -> Optional[PredictionBundle]:
        if not isinstance(key, str):
            return default

        # Prefixed ID (contains ':')
        if ":" in key:
            return super().get(key, default)

        # Original name - check if unambiguous
        if key in self._name_to_prefixed:
            matching_ids = self._name_to_prefixed[key]
            if len(matching_ids) == 1:
                # Unique match
                return super().get(matching_ids[0], default)
            # Ambiguous - return default (don't raise error for get())
            return default

        # Not found
        return default


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
    - Compute full PMF predictions for all bottom-level entities (services) using the caps from Phase 1

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
        # Store computed bundles: prefixed_id -> PredictionBundle
        # Uses prefixed IDs (e.g., "subspecialty:Acute Medicine") to prevent name collisions
        self.prediction_results: Dict[str, PredictionBundle] = {}

    def predict_all_levels(
        self,
        bottom_level_data: Dict[str, ServicePredictionInputs],
        top_level_id: Optional[str] = None,
        flow_selection: Optional[FlowSelection] = None,
    ) -> "PredictionResults":
        """Run predictions for the entire hierarchy.

        This method executes the full 3-phase prediction algorithm (capping,
        bottom-level prediction, aggregation) for the specified subtree.

        Parameters
        ----------
        bottom_level_data : Dict[str, ServicePredictionInputs]
            Dictionary mapping bottom-level entity IDs to their prediction inputs.
            Each value is a `ServicePredictionInputs` object containing:

            - ``service_id``: Identifier for the service (must match the key)
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
            - Typically created using `build_service_data()` from the
              `patientflow.predict.service` module.
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
        PredictionResults
            Dictionary-like container keyed by prefixed entity ID
            (e.g. ``"specialty:medical"``, ``"division:Medical Division"``).
            Supports access by original name when unambiguous. Values are
            ``PredictionBundle`` objects with ``arrivals``, ``departures``,
            and ``net_flow`` predictions. Entities without data in
            ``bottom_level_data`` are excluded.

        Examples
        --------
        >>> results = predictor.predict_all_levels(prediction_inputs)
        >>> bundle = results["medical"]
        >>> print(f"Expected arrivals: {bundle.arrivals.expectation:.1f}")

        With custom flow selection (ED inflows only, no departures):

        >>> results = predictor.predict_all_levels(
        ...     prediction_inputs,
        ...     flow_selection=FlowSelection.custom(
        ...         include_ed_current=True,
        ...         include_ed_yta=True,
        ...         include_non_ed_yta=False,
        ...         include_elective_yta=False,
        ...         include_transfers_in=False,
        ...         include_departures=False,
        ...         cohort="emergency",
        ...     ),
        ... )

        Notes
        -----
        ``bottom_level_data`` is typically created using
        ``build_service_data()`` from ``patientflow.predict.service``.

        Keys in ``bottom_level_data`` must exactly match (case-sensitive) the
        entity IDs at the bottom level of the hierarchy. Unmatched keys on
        either side are silently skipped.
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

        # Return a copy wrapped in PredictionResults for flexible key access
        return PredictionResults(self.prediction_results.copy())

    def _predict_subtree(
        self,
        entity_id: str,
        bottom_level_data: Dict[str, ServicePredictionInputs],
        flow_selection: FlowSelection,
    ):
        entity_type = self.hierarchy.get_entity_type(entity_id)
        if entity_type is None:
            raise ValueError(f"Entity type not found for id: {entity_id}")

        # PHASE 1: Calculate caps top-down
        # We calculate max_support for this node based on the statistics of its subtree
        _, _, arrivals_max_support = calculate_hierarchical_stats(
            entity_id,
            entity_type,
            bottom_level_data,
            self.hierarchy,
            "arrivals",
            flow_selection,
            self.predictor.k_sigma,
        )
        _, _, departures_max_support = calculate_hierarchical_stats(
            entity_id,
            entity_type,
            bottom_level_data,
            self.hierarchy,
            "departures",
            flow_selection,
            self.predictor.k_sigma,
        )

        # Iterate children
        children = self.hierarchy.get_children(entity_id, entity_type)

        # Create prefixed ID for storing results (prevents name collisions)
        prefixed_id = f"{entity_type.name}:{entity_id}"

        if not children:
            # PHASE 2: Bottom-level prediction
            # This is a leaf node (service)
            if entity_id in bottom_level_data:
                # Prediction logic for base level (leaf nodes)
                inputs = bottom_level_data[entity_id]
                bundle = self.predictor.predict_service(inputs, flow_selection)
                self.prediction_results[prefixed_id] = bundle
            return

        # Determine child entity type from hierarchy structure
        # Children of entity_type are all of the type that has entity_type as parent_type
        child_entity_type = None
        for level in self.hierarchy.levels.values():
            if level.parent_type == entity_type:
                child_entity_type = level.entity_type
                break

        if child_entity_type is None:
            # This should not happen in a valid hierarchy
            raise ValueError(f"No child type found for entity type {entity_type.name}")

        # Recursive step: predict all children
        # Note: In a true top-down capping implementation, we would pass down
        # constraints. Here we calculate caps independently at each level which
        # is sufficient for correctness and simpler.
        child_bundles = []
        for child_id in children:
            child_prefixed_id = f"{child_entity_type.name}:{child_id}"

            # Verify child is in hierarchy (should be guaranteed by get_children)
            self._predict_subtree(child_id, bottom_level_data, flow_selection)
            if child_prefixed_id in self.prediction_results:
                child_bundles.append(self.prediction_results[child_prefixed_id])

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
            self.prediction_results[prefixed_id] = bundle

    def get_prediction(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[PredictionBundle]:
        """Get the prediction bundle for a specific entity.

        Parameters
        ----------
        entity_id : str
            Entity ID to retrieve (prefixed ID like "subspecialty:Acute Medicine"
            or original name if entity_type is provided)
        entity_type : EntityType, optional
            Entity type of the entity. If provided, entity_id is treated as an
            original name and converted to prefixed ID. If not provided, entity_id
            should be a prefixed ID.

        Returns
        -------
        Optional[PredictionBundle]
            Prediction bundle if available, None otherwise
        """
        if entity_type is not None:
            # Convert original name to prefixed ID
            prefixed_id = f"{entity_type.name}:{entity_id}"
            return self.prediction_results.get(prefixed_id)
        else:
            # Assume entity_id is already a prefixed ID
            return self.prediction_results.get(entity_id)

    def get_truncated_mass_stats(self) -> Dict[str, Any]:
        """Get statistics about truncated probability mass from the predictor.

        Returns
        -------
        dict
            Truncation statistics from the DemandPredictor
        """
        return self.predictor.get_truncated_mass_stats()


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
