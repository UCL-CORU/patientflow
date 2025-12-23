"""Core calculation engine for demand prediction."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from patientflow.predict.subspecialty import SubspecialtyPredictionInputs, FlowInputs
from patientflow.predict.distribution import Distribution
from .types import DemandPrediction, PredictionBundle, FlowSelection, DEFAULT_PERCENTILES
from .structure import Hierarchy, EntityType


class DemandPredictor:
    """Hierarchical demand prediction for hospital bed capacity.

    This class provides the functionality for computing bed demand at
    different organisational levels. It uses convolution of probability distributions
    to combine predictions from lower levels into higher levels.

    Parameters
    ----------
    k_sigma : float, default=8.0
        Cap width measured in standard deviations. Maximum support for each
        organisational level is calculated top-down using statistical caps:
        mean + k_sigma * sqrt(sum of variances). This ensures bounded array
        sizes while maintaining statistical accuracy. Net flow is computed
        naively from already-capped arrivals and departures, with no additional
        truncation.

    Attributes
    ----------
    k_sigma : float
        Number of standard deviations used to cap supports
    cache : dict
        Cache for storing computed predictions (currently unused)

    Notes
    -----
    The prediction process involves:

    1. Generating Poisson distributions for yet-to-arrive patients
    2. Combining with current patient distributions using convolution
    3. Aggregating across organisational levels using multiple convolutions
    4. Computing statistics (expected value, percentiles) for each level

    The class uses discrete convolution to combine probability distributions.
    Supports are clamped using top-down statistical caps calculated before
    convolution, ensuring bounded array sizes while maintaining statistical
    accuracy. Truncation uses renormalisation to preserve probability mass.

    Flow Selection
    --------------
    The predictor supports flexible flow selection via FlowSelection objects,
    allowing users to specify which patient flows (inflows/outflows) and
    cohorts (elective/emergency/all) to include in predictions.
    """

    def __init__(self, k_sigma: float = 8.0):
        self.k_sigma = k_sigma
        self.cache: Dict[str, DemandPrediction] = {}
        # Track truncated mass: {(entity_id, flow_type): truncated_mass}
        # flow_type is 'arrivals' or 'departures'
        self.truncated_mass: Dict[Tuple[str, str], float] = {}

    def calculate_hierarchical_stats(
        self,
        entity_id: str,
        entity_type: EntityType,
        bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
        hierarchy: "Hierarchy",
        flow_type: str,
        flow_selection: Optional[FlowSelection] = None,
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
        bottom_level_data : Dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping bottom-level entity IDs to their prediction inputs
        hierarchy : Hierarchy
            Hierarchy structure for traversing the tree
        flow_type : str
            Type of flow to analyze: 'arrivals' or 'departures' only
        flow_selection : FlowSelection, optional
            Selection for which flows to include. If None, includes all flows.

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
                        mean = self._expected_value(pmf_array)
                        variance = self._variance(pmf_array)
                        physical_max = len(pmf_array) - 1  # Maximum index value
                        means.append(mean)
                        variances.append(variance)
                        physical_maxes.append(physical_max)

        # Calculate combined statistics
        sum_of_means = float(np.sum(means)) if means else 0.0
        combined_variance = float(np.sum(variances)) if variances else 0.0
        combined_sd = (
            float(np.sqrt(combined_variance)) if combined_variance > 0 else 0.0
        )

        # Statistical cap: mean + k_sigma * SD
        statistical_cap = sum_of_means + self.k_sigma * combined_sd

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

    def predict_flow_total(
        self,
        flow_inputs: List[FlowInputs],
        entity_id: str,
        entity_type: str,
        max_support: Optional[int] = None,
    ) -> DemandPrediction:
        """Combine multiple flows into a single distribution.

        This method is flow-agnostic and works for any combination of PMF and
        Poisson flows. It convolves all flows together to produce a single
        probability distribution for the total.

        Parameters
        ----------
        flow_inputs : List[FlowInputs]
            List of flow inputs to combine. Each FlowInputs specifies whether
            it's a PMF or Poisson distribution.
        entity_id : str
            Unique identifier for the entity
        entity_type : str
            Type of entity (for labeling purposes)
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization. If None, no truncation
            is applied (for backward compatibility during transition).

        Returns
        -------
        DemandPrediction
            Combined prediction for all flows

        Notes
        -----
        Flows are combined through convolution, which represents the distribution
        of the sum of independent random variables. If max_support is provided,
        the result is truncated and renormalized to ensure probability conservation.
        """
        # Materialize flows via Distribution
        dist_total = Distribution.from_pmf(np.array([1.0]))
        for flow in flow_inputs:
            if flow.flow_type == "poisson":
                lam = float(flow.distribution)
                # Use max_support if provided, otherwise use a reasonable default
                if max_support is not None:
                    poisson_cap = max_support
                else:
                    # Fallback: use k-sigma cap for Poisson
                    poisson_cap = Distribution.poisson_cap_from_k_sigma(
                        lam, self.k_sigma
                    )
                flow_dist = Distribution.from_poisson_with_cap(lam, poisson_cap)
            elif flow.flow_type == "pmf":
                pmf_array = flow.distribution
                if isinstance(pmf_array, np.ndarray):
                    flow_dist = Distribution.from_pmf(pmf_array)
                else:
                    raise ValueError(
                        f"PMF distribution must be numpy array, got {type(pmf_array)}"
                    )
            else:
                raise ValueError(
                    f"Unknown flow type: {flow.flow_type}. Expected 'pmf' or 'poisson'."
                )
            dist_total = dist_total.convolve(flow_dist)

        # Apply cap with renormalization if max_support is provided
        if max_support is not None:
            truncated_pmf, truncated_mass = self.apply_cap_with_renormalization(
                dist_total.probabilities, max_support, return_truncated_mass=True
            )
            # Track truncated mass for this entity and flow type
            key = (entity_id, entity_type)
            self.truncated_mass[key] = (
                self.truncated_mass.get(key, 0.0) + truncated_mass
            )
            return self._create_prediction(entity_id, entity_type, truncated_pmf)
        else:
            return self._create_prediction(
                entity_id, entity_type, dist_total.probabilities
            )

    def predict_subspecialty(
        self,
        subspecialty_id: str,
        inputs: SubspecialtyPredictionInputs,
        flow_selection: Optional[FlowSelection] = None,
    ) -> PredictionBundle:
        """Predict subspecialty demand with flexible flow selection.

        This method computes predictions for arrivals, departures, and net flow
        for a single subspecialty. Users can customize which flows to include
        via the flow_selection parameter.

        Parameters
        ----------
        subspecialty_id : str
            Unique identifier for the subspecialty
        inputs : SubspecialtyPredictionInputs
            Dataclass containing all prediction inputs for this subspecialty.
            See SubspecialtyPredictionInputs for field details.
        flow_selection : FlowSelection, optional
            Selection specifying which flows to include. If None, uses
            FlowSelection.default() which includes all flows.

        Returns
        -------
        PredictionBundle
            Bundle containing arrivals, departures, and net flow predictions

        Notes
        -----
        The method separately computes:

        1. Arrivals: Convolution of all selected inflow distributions
        2. Departures: Convolution of all selected outflow distributions
        3. Net flow: Difference of expected values (arrivals - departures)

        Examples
        --------
        >>> # Default: all flows included
        >>> bundle = predictor.predict_subspecialty(spec_id, inputs)

        >>> # Only incoming flows (arrivals, no departures)
        >>> bundle = predictor.predict_subspecialty(
        ...     spec_id, inputs, flow_selection=FlowSelection.incoming_only()
        ... )

        >>> # Custom selection
        >>> bundle = predictor.predict_subspecialty(
        ...     spec_id, inputs,
        ...     flow_selection=FlowSelection.custom(
        ...         include_ed_current=True,
        ...         include_ed_yta=True,
        ...         include_non_ed_yta=False,
        ...         include_elective_yta=False,
        ...         include_transfers_in=False,
        ...         include_departures=True,
        ...         cohort="emergency",
        ...     )
        ... )
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()

        # Validate flow selection configuration
        flow_selection.validate()

        # Build inflows from families and cohort
        inflow_keys: List[str] = []
        if flow_selection.include_ed_current:
            inflow_keys.append("ed_current")
        if flow_selection.include_ed_yta:
            inflow_keys.append("ed_yta")
        if flow_selection.include_non_ed_yta:
            inflow_keys.append("non_ed_yta")
        if flow_selection.include_elective_yta:
            inflow_keys.append("elective_yta")
        if flow_selection.include_transfers_in:
            # include both then cohort-filter
            inflow_keys.extend(["elective_transfers", "emergency_transfers"])

        def inflow_allowed(key: str) -> bool:
            if flow_selection.cohort == "all":
                return True
            if flow_selection.cohort == "elective":
                return key.startswith("elective_") or key == "elective_yta"
            if flow_selection.cohort == "emergency":
                return key in {"ed_current", "ed_yta", "non_ed_yta"} or key.startswith(
                    "emergency_"
                )
            return True

        # Validate that required inflow keys exist
        missing_inflow_keys = [k for k in inflow_keys if k not in inputs.inflows]
        if missing_inflow_keys:
            raise KeyError(
                f"Missing inflow keys in SubspecialtyPredictionInputs: {missing_inflow_keys}"
            )

        selected_inflows = [inputs.inflows[k] for k in inflow_keys if inflow_allowed(k)]
        arrivals = self.predict_flow_total(
            selected_inflows, subspecialty_id, "arrivals"
        )

        # Build outflows from families and cohort
        outflow_keys: List[str] = []
        if flow_selection.include_departures:
            outflow_keys.extend(["elective_departures", "emergency_departures"])

        def outflow_allowed(key: str) -> bool:
            if flow_selection.cohort == "all":
                return True
            if flow_selection.cohort == "elective":
                return key.startswith("elective_")
            if flow_selection.cohort == "emergency":
                return key.startswith("emergency_")
            return True

        # Validate that required outflow keys exist
        missing_outflow_keys = [k for k in outflow_keys if k not in inputs.outflows]
        if missing_outflow_keys:
            raise KeyError(
                f"Missing outflow keys in SubspecialtyPredictionInputs: {missing_outflow_keys}"
            )

        selected_outflows = [
            inputs.outflows[k] for k in outflow_keys if outflow_allowed(k)
        ]

        # Calculate physical cap for departures (sum of physical maxes from PMF-based flows)
        # Departures are always PMF-based (physically bounded by current patients)
        departures_physical_cap = 0
        for flow in selected_outflows:
            if flow.flow_type == "pmf":
                pmf_array = flow.distribution
                if isinstance(pmf_array, np.ndarray):
                    # Physical max is the maximum index value (len - 1)
                    departures_physical_cap += len(pmf_array) - 1
            elif flow.flow_type == "poisson":
                # Departures should never be Poisson (enforced elsewhere)
                # But if somehow one exists, it's unbounded, so we skip it
                pass

        # Apply physical cap to departures prediction
        departures = self.predict_flow_total(
            selected_outflows,
            subspecialty_id,
            "departures",
            max_support=departures_physical_cap,
        )

        # Renormalize arrivals and departures before net-flow
        arrivals_p = self._renormalize(arrivals.probabilities)
        departures_p = self._renormalize(departures.probabilities)

        # Compute net flow distribution naively from already-capped arrivals/departures
        # Since arrivals and departures are already capped, no additional truncation is needed
        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p)
        )
        # Ensure expected value matches arrivals - departures exactly for tests
        expected_diff = float(arrivals.expected_value - departures.expected_value)
        net_flow = self._create_prediction(
            subspecialty_id,
            "net_flow",
            net_dist.probabilities,
            net_dist.offset,
            expected_override=expected_diff,
        )

        return PredictionBundle(
            entity_id=subspecialty_id,
            entity_type="subspecialty",
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

    def predict_hierarchical_level(
        self,
        entity_id: str,
        entity_type: EntityType,
        child_predictions: List[DemandPrediction],
        max_support: Optional[int] = None,
        flow_type: Optional[str] = None,
    ) -> DemandPrediction:
        """Generic method for hierarchical prediction at any level.

        This method aggregates demand predictions from child entities by convolving
        their probability distributions. It works for any hierarchical level.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : EntityType
            Type of entity being predicted
        child_predictions : list[DemandPrediction]
            List of demand predictions from child entities
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization.
        flow_type : str, optional
            Flow type ('arrivals' or 'departures') for truncated mass tracking.
            If None, truncated mass will not be tracked for this aggregation.

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the entity

        Notes
        -----
        The method convolves multiple probability distributions efficiently by
        sorting them by expected value. If max_support is provided, truncation
        uses renormalization to ensure probability conservation.
        """
        distributions = [p.probabilities for p in child_predictions]
        p_total = self.convolve_multiple(
            distributions, max_support, entity_id=entity_id, flow_type=flow_type
        )
        return self._create_prediction(entity_id, entity_type.name, p_total)

    def _create_bundle_from_children(
        self,
        entity_id: str,
        entity_type: str,
        child_bundles: List[PredictionBundle],
        arrivals_max_support: Optional[int] = None,
        departures_max_support: Optional[int] = None,
    ) -> PredictionBundle:
        arrivals_preds = [b.arrivals for b in child_bundles]
        departures_preds = [b.departures for b in child_bundles]

        arrivals = self.predict_hierarchical_level(
            entity_id,
            EntityType(entity_type),
            arrivals_preds,
            arrivals_max_support,
            flow_type="arrivals",
        )

        # For departures: always calculate physical cap from child predictions' PMF lengths
        # This ensures correctness when aggregating, especially for single-child cases.
        # The physical cap is the sum of physical maxes (len(pmf) - 1) from each child.
        # We use child predictions rather than pre-calculated caps because child predictions
        # have already been computed with correct physical caps applied.
        departures_physical_cap = 0
        for pred in departures_preds:
            if len(pred.probabilities) > 0:
                # Physical max is the maximum index value (len - 1)
                departures_physical_cap += len(pred.probabilities) - 1

        # Use calculated cap from child predictions (more accurate than pre-calculated)
        departures_max_support_from_children = (
            int(departures_physical_cap) if departures_physical_cap > 0 else None
        )

        departures = self.predict_hierarchical_level(
            entity_id,
            EntityType(entity_type),
            departures_preds,
            departures_max_support_from_children,
            flow_type="departures",
        )
        net_flow = self._compute_net_flow(arrivals, departures, entity_id)

        # Use flow_selection from first child if available, otherwise use default
        flow_selection = (
            child_bundles[0].flow_selection
            if child_bundles
            else FlowSelection.default()
        )

        return PredictionBundle(
            entity_id=entity_id,
            entity_type=entity_type,
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

    def convolve_multiple(
        self,
        distributions: List[np.ndarray],
        max_support: Optional[int] = None,
        entity_id: Optional[str] = None,
        flow_type: Optional[str] = None,
    ) -> np.ndarray:
        """Convolve multiple distributions with optional truncation and renormalization.

        This method efficiently convolves multiple probability distributions after
        sorting them by expected value. If max_support is provided, the result
        is truncated and renormalized to ensure probability conservation.

        Parameters
        ----------
        distributions : list[numpy.ndarray]
            List of probability mass functions to convolve
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization. If None, no truncation
            is applied.
        entity_id : str, optional
            Entity identifier for truncated mass tracking. If provided along with
            flow_type, truncated mass will be tracked.
        flow_type : str, optional
            Flow type ('arrivals' or 'departures') for truncated mass tracking.
            If provided along with entity_id, truncated mass will be tracked.

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Distributions are sorted by expected value for computational efficiency.
        If max_support is provided, truncation uses _apply_cap_with_renormalization()
        to ensure the PMF sums to 1.0 exactly.
        """
        if not distributions:
            return np.array([1.0])
        if len(distributions) == 1:
            result = distributions[0]
            if max_support is not None:
                if entity_id is not None and flow_type is not None:
                    result, truncated_mass = self.apply_cap_with_renormalization(
                        result, max_support, return_truncated_mass=True
                    )
                    key = (entity_id, flow_type)
                    self.truncated_mass[key] = (
                        self.truncated_mass.get(key, 0.0) + truncated_mass
                    )
                else:
                    result = self.apply_cap_with_renormalization(result, max_support)
            return result

        # Sort by expected value for efficiency
        distributions = sorted(distributions, key=lambda p: self._expected_value(p))

        # Convolve all distributions
        result = distributions[0]
        for dist in distributions[1:]:
            result = self._convolve(result, dist)

        # Apply cap with renormalization if max_support is provided
        if max_support is not None:
            if entity_id is not None and flow_type is not None:
                result, truncated_mass = self.apply_cap_with_renormalization(
                    result, max_support, return_truncated_mass=True
                )
                key = (entity_id, flow_type)
                self.truncated_mass[key] = (
                    self.truncated_mass.get(key, 0.0) + truncated_mass
                )
            else:
                result = self.apply_cap_with_renormalization(result, max_support)

        return result

    def _convolve(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.convolve(p, q)

    def apply_cap_with_renormalization(
        self, pmf: np.ndarray, max_support: int, return_truncated_mass: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Truncate PMF to max_support and assign remaining probability to the last bin.

        This ensures probability conservation: the PMF sums to 1.0 exactly.
        The last bin (at index max_support) represents "at least this many".

        Parameters
        ----------
        pmf : np.ndarray
            Original probability mass function
        max_support : int
            Maximum support value (corresponds to array index max_support)
        return_truncated_mass : bool, default=False
            If True, return a tuple (truncated_pmf, truncated_mass) instead of just truncated_pmf.
            truncated_mass is the sum of probability mass beyond max_support.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, float]
            If return_truncated_mass=False: Truncated PMF with length (max_support + 1)
            If return_truncated_mass=True: Tuple of (truncated_pmf, truncated_mass)
        """
        if max_support < 0:
            # Return a single bin with all probability mass
            result = np.array([np.sum(pmf)])
            if return_truncated_mass:
                # No truncation occurred (all mass fits in one bin)
                return result, 0.0
            return result

        max_len = max_support + 1
        if len(pmf) <= max_len:
            # No truncation needed
            result = pmf.copy()
            if return_truncated_mass:
                return result, 0.0
            return result

        # Truncate to max_support + 1 elements (indices 0 to max_support)
        truncated = pmf[:max_len].copy()

        # Calculate remaining probability mass beyond max_support
        remaining_mass = np.sum(pmf[max_len:])
        truncated[max_support] += remaining_mass

        if return_truncated_mass:
            return truncated, float(remaining_mass)
        return truncated

    def _variance(self, p: np.ndarray, offset: int = 0) -> float:
        if len(p) == 0:
            return 0.0
        indices = np.arange(len(p)) + offset
        mean = float(np.sum(indices * p))
        mean_sq = float(np.sum((indices**2) * p))
        return max(0.0, mean_sq - mean * mean)

    def _renormalize(self, p: np.ndarray) -> np.ndarray:
        if p.size == 0:
            return p
        total = float(np.sum(p))
        if total <= 0.0:
            return p
        return p / total

    def get_truncated_mass_stats(self) -> Dict[str, Any]:
        """Get statistics about truncated mass across all predictions.

        Returns
        -------
        dict
            Dictionary containing:
            - 'total_truncated_mass': Total truncated mass across all entities/flows
            - 'max_truncated_mass': Maximum single truncation value
            - 'num_truncations': Number of distinct (entity_id, flow_type) pairs that were truncated
            - 'by_entity': Dictionary mapping entity_id to dict of {flow_type: truncated_mass}
            - 'by_flow_type': Dictionary with 'arrivals' and 'departures' totals
        """
        if not self.truncated_mass:
            return {
                "total_truncated_mass": 0.0,
                "max_truncated_mass": 0.0,
                "num_truncations": 0,
                "by_entity": {},
                "by_flow_type": {
                    "arrivals": 0.0,
                    "departures": 0.0,
                },
            }

        total = sum(self.truncated_mass.values())
        max_mass = max(self.truncated_mass.values())

        # Group by entity_id
        by_entity: Dict[str, Dict[str, float]] = {}
        for (entity_id, flow_type), mass in self.truncated_mass.items():
            if entity_id not in by_entity:
                by_entity[entity_id] = {}
            by_entity[entity_id][flow_type] = mass

        # Group by flow_type
        by_flow_type = {
            "arrivals": sum(
                m for (_, ft), m in self.truncated_mass.items() if ft == "arrivals"
            ),
            "departures": sum(
                m for (_, ft), m in self.truncated_mass.items() if ft == "departures"
            ),
        }

        return {
            "total_truncated_mass": total,
            "max_truncated_mass": max_mass,
            "num_truncations": len(self.truncated_mass),
            "by_entity": by_entity,
            "by_flow_type": by_flow_type,
        }

    def clear_truncated_mass(self) -> None:
        """Clear the truncated mass tracking dictionary.

        This is useful when starting a new prediction iteration to get fresh measurements.
        """
        self.truncated_mass.clear()

    def _expected_value(self, p: np.ndarray, offset: int = 0) -> float:
        return np.sum((np.arange(len(p)) + offset) * p)

    def _percentiles(
        self, p: np.ndarray, percentile_list: List[int], offset: int = 0
    ) -> Dict[int, int]:
        cumsum = np.cumsum(p)
        result = {}
        for pct in percentile_list:
            idx = np.searchsorted(cumsum, pct / 100.0)
            result[pct] = int(idx + offset)
        return result

    def _compute_net_flow(
        self, arrivals: DemandPrediction, departures: DemandPrediction, entity_id: str
    ) -> DemandPrediction:
        # Renormalize arrivals and departures before net-flow
        arrivals_p = self._renormalize(arrivals.probabilities)
        departures_p = self._renormalize(departures.probabilities)

        # Compute net flow distribution naively from already-capped arrivals/departures
        # Since arrivals and departures are already capped using top-down statistical caps,
        # the net flow is naturally bounded by physical limits [-max_departures, +max_arrivals].
        # No additional truncation is needed.
        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p)
        )
        return self._create_prediction(
            entity_id, "net_flow", net_dist.probabilities, net_dist.offset
        )

    def _create_prediction(
        self,
        entity_id: str,
        entity_type: str,
        probabilities: np.ndarray,
        offset: int = 0,
        expected_override: Optional[float] = None,
    ) -> DemandPrediction:
        expected_value = (
            float(expected_override)
            if expected_override is not None
            else float(self._expected_value(probabilities, offset))
        )
        return DemandPrediction(
            entity_id=entity_id,
            entity_type=entity_type,
            probabilities=probabilities,
            expected_value=expected_value,
            percentiles=self._percentiles(probabilities, DEFAULT_PERCENTILES, offset),
            offset=offset,
        )
