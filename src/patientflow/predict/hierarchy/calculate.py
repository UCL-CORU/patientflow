from typing import Dict, List, Optional, Tuple
import numpy as np

from patientflow.predict.service import ServicePredictionInputs
from patientflow.predict.types import FlowSelection
from patientflow.predict.hierarchy.structure import Hierarchy, EntityType


def calculate_hierarchical_stats(
    entity_id: str,
    entity_type: EntityType,
    bottom_level_data: Dict[str, ServicePredictionInputs],
    hierarchy: "Hierarchy",
    flow_type: str,
    flow_selection: Optional[FlowSelection] = None,
    k_sigma: float = 8.0,
) -> Tuple[float, float, int]:
    """Calculate sum of means, combined SD, and maximum support for an entity.

    This method recursively gathers all bottom-level entities in the subtree
    and calculates statistical caps based on the sum of means and combined
    variance of all distributions in the subtree.

    Parameters
    ----------
    entity_id : str
        Unique identifier for the entity
    entity_type : EntityType
        Type of entity being analyzed
    bottom_level_data : Dict[str, ServicePredictionInputs]
        Dictionary mapping bottom-level entity IDs to their prediction inputs
    hierarchy : Hierarchy
        Hierarchy structure for traversing the tree
    flow_type : str
        Type of flow to analyze: 'arrivals' or 'departures' only
    flow_selection : FlowSelection, optional
        Selection for which flows to include. If None, includes all flows.
    k_sigma : float, default=8.0
        Number of standard deviations used to cap supports

    Returns
    -------
    Tuple[float, float, int]
        Tuple of (sum_of_means, combined_sd, max_support)
        - sum_of_means: Sum of all means from distributions in subtree
        - combined_sd: Combined standard deviation (sqrt of sum of variances)
        - max_support: Maximum support value (min of statistical and physical caps)

    Notes
    -----
    For Poisson distributions: variance = mean (lambda)
    For PMF distributions: variance = E[X²] - E[X]² calculated from PMF array
    Physical max for PMF = len(pmf_array) - 1 (maximum index value)
    Physical max for Poisson = infinity (unbounded)
    """
    if flow_type not in {"arrivals", "departures"}:
        raise ValueError(
            f"flow_type must be 'arrivals' or 'departures', got '{flow_type}'"
        )

    if flow_selection is None:
        flow_selection = FlowSelection.default()

    # Get bottom level type
    bottom_type = hierarchy.get_bottom_level_type()

    # Gather all bottom-level entities in the subtree
    def gather_bottom_level_entities(
        current_id: str, current_type: EntityType
    ) -> List[str]:
        if current_type == bottom_type:
            # This is a bottom-level entity
            return [current_id] if current_id in bottom_level_data else []

        # Get children and recurse
        children = hierarchy.get_children(current_id, current_type)
        result = []
        for child_id in children:
            child_type = hierarchy.get_entity_type(child_id)
            if child_type is not None:
                result.extend(gather_bottom_level_entities(child_id, child_type))
        return result

    bottom_level_ids = gather_bottom_level_entities(entity_id, entity_type)

    # Extract relevant distributions based on flow_type
    means: List[float] = []
    variances: List[float] = []
    physical_maxes: List[float] = []

    for bottom_id in bottom_level_ids:
        if bottom_id not in bottom_level_data:
            continue

        inputs = bottom_level_data[bottom_id]

        # Get flows based on flow_type
        if flow_type == "arrivals":
            # Get inflows
            flow_keys: List[str] = []
            if flow_selection.include_ed_current:
                flow_keys.append("ed_current")
            if flow_selection.include_ed_yta:
                flow_keys.append("ed_yta")
            if flow_selection.include_non_ed_yta:
                flow_keys.append("non_ed_yta")
            if flow_selection.include_elective_yta:
                flow_keys.append("elective_yta")
            if flow_selection.include_transfers_in:
                flow_keys.extend(["elective_transfers", "emergency_transfers"])

            def flow_allowed(key: str) -> bool:
                if flow_selection.cohort == "all":
                    return True
                if flow_selection.cohort == "elective":
                    return key.startswith("elective_") or key == "elective_yta"
                if flow_selection.cohort == "emergency":
                    return key in {
                        "ed_current",
                        "ed_yta",
                        "non_ed_yta",
                    } or key.startswith("emergency_")
                return True

            flows = [
                inputs.inflows[k]
                for k in flow_keys
                if k in inputs.inflows and flow_allowed(k)
            ]
        else:  # flow_type == "departures"
            # Get outflows
            flow_keys = []
            if flow_selection.include_departures:
                flow_keys.extend(["elective_departures", "emergency_departures"])

            def flow_allowed(key: str) -> bool:
                if flow_selection.cohort == "all":
                    return True
                if flow_selection.cohort == "elective":
                    return key.startswith("elective_")
                if flow_selection.cohort == "emergency":
                    return key.startswith("emergency_")
                return True

            flows = [
                inputs.outflows[k]
                for k in flow_keys
                if k in inputs.outflows and flow_allowed(k)
            ]

        # Process each flow
        for flow in flows:
            if flow.flow_type == "poisson":
                # Departures should never be Poisson (they're always PMF-based from current patients)
                if flow_type == "departures":
                    raise ValueError(
                        f"Unexpected Poisson flow in departures: {flow.flow_id}. "
                        f"Departures must be PMF-based (physically bounded by current patients)."
                    )
                lam = float(flow.distribution)
                means.append(lam)
                variances.append(lam)  # Var(Poisson(λ)) = λ
                physical_maxes.append(float("inf"))  # Unbounded
            elif flow.flow_type == "pmf":
                pmf_array = flow.distribution
                if isinstance(pmf_array, np.ndarray):
                    mean = _expected_value(pmf_array)
                    variance = _variance(pmf_array)
                    physical_max = len(pmf_array) - 1  # Maximum index value
                    means.append(mean)
                    variances.append(variance)
                    physical_maxes.append(physical_max)

    # Calculate combined statistics
    sum_of_means = float(np.sum(means)) if means else 0.0
    combined_variance = float(np.sum(variances)) if variances else 0.0
    combined_sd = float(np.sqrt(combined_variance)) if combined_variance > 0 else 0.0

    # Statistical cap: mean + k_sigma * SD
    statistical_cap = sum_of_means + k_sigma * combined_sd

    # Physical cap: sum of all physical maxes (for convolution)
    finite_physical_maxes = [pm for pm in physical_maxes if pm != float("inf")]
    if finite_physical_maxes:
        physical_cap = float(np.sum(finite_physical_maxes))  # Sum for convolution
    else:
        physical_cap = float("inf")

    # For departures: always physically bounded (all flows are PMF-based)
    # The only way physical_cap could be infinite is if flows is empty
    if flow_type == "departures":
        if physical_cap == float("inf"):
            # No departure flows (empty flows list) - return zero distribution cap
            max_support = 0
        else:
            # Departures are always physically bounded - use physical cap only
            max_support = int(np.ceil(physical_cap))
    elif physical_cap == float("inf"):
        # For arrivals: no physical constraint (has Poisson flows), use statistical cap
        max_support = int(np.ceil(statistical_cap))
    else:
        # For arrivals: physical cap exists (all flows are PMF-based), use physical cap
        # The physical cap is the hard limit - we should not use statistical cap
        max_support = int(np.ceil(physical_cap))

    # Ensure non-negative
    max_support = max(0, max_support)

    return (sum_of_means, combined_sd, max_support)


def _variance(p: np.ndarray, offset: int = 0) -> float:
    if len(p) == 0:
        return 0.0
    indices = np.arange(len(p)) + offset
    mean = float(np.sum(indices * p))
    mean_sq = float(np.sum((indices**2) * p))
    return max(0.0, mean_sq - mean * mean)


def _expected_value(p: np.ndarray, offset: int = 0) -> float:
    return np.sum((np.arange(len(p)) + offset) * p)
