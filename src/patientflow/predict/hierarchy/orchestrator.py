"""Hierarchical prediction orchestrator."""

from typing import Dict, Optional, Tuple, Any

from patientflow.predict.subspecialty import SubspecialtyPredictionInputs
from .structure import Hierarchy
from .types import PredictionBundle, FlowSelection
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
            Dictionary mapping bottom-level entity IDs to their prediction inputs
        top_level_id : str, optional
            Root entity to start prediction from. If None, predicts for all
            top-level entities in the hierarchy.
        flow_selection : FlowSelection, optional
            Selection specifying which flows to include. If None, uses
            FlowSelection.default() which includes all flows.

        Returns
        -------
        Dict[str, PredictionBundle]
            Dictionary containing prediction bundles for all entities in the
            hierarchy (bottom-level and aggregated), keyed by entity ID.
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
        """Recursively predict for a subtree.

        This internal method implements the 3-phase prediction algorithm for a
        specific subtree rooted at entity_id.
        """
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
        DataFrame containing organizational structure
    column_mapping : Dict[str, str], optional
        Mapping from DataFrame columns to entity types
    top_level_id : str, optional
        ID of the top-level entity
    config_path : str, optional
        Path to the hierarchy configuration YAML file.
        If provided, loads hierarchy structure from file.
        If None, uses default hospital hierarchy.
    k_sigma : float, default=8.0
        Cap width in standard deviations

    Returns
    -------
    HierarchicalPredictor
        Ready-to-use predictor instance
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
