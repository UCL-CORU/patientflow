"""Demand prediction engine.

This module provides the core mathematical engine for demand prediction,
handling convolution of probability distributions and statistical processing."""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from patientflow.predict.service import ServicePredictionInputs, FlowInputs
from patientflow.predict.distribution import Distribution
from patientflow.predict.types import (
    DemandPrediction,
    PredictionBundle,
    FlowSelection,
    DEFAULT_PERCENTILES,
)


class DemandPredictor:
    """Core engine for demand prediction math.

    This class provides the functionality for computing bed demand predictions
    via convolution of probability distributions. It performs the mathematical
    heavy lifting (convolution, capping, statistical calculations).

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
    truncated_mass : dict
        Track truncated mass: {(entity_id, flow_type): truncated_mass}
        flow_type is 'arrivals' or 'departures'

    Notes
    -----
    The prediction process involves:

    1. Generating Poisson distributions for yet-to-arrive patients
    2. Combining with current patient distributions using convolution
    3. Aggregating distributions using multiple convolutions
    4. Computing statistics (expected value, percentiles)

    The class uses discrete convolution to combine probability distributions.
    Supports are clamped using statistical caps ensuring bounded array sizes.
    Truncation uses renormalisation to preserve probability mass.
    """

    def __init__(self, k_sigma: float = 8.0):
        self.k_sigma = k_sigma
        self.cache: Dict[str, DemandPrediction] = {}
        self.truncated_mass: Dict[Tuple[str, str], float] = {}

    def predict_service(
        self,
        inputs: ServicePredictionInputs,
        flow_selection: Optional[FlowSelection] = None,
    ) -> PredictionBundle:
        """Predict service demand with flexible flow selection.

        This method computes predictions for arrivals, departures, and net flow
        for a single service. Users can customise which flows to include
        via the flow_selection parameter.

        Parameters
        ----------
        inputs : ServicePredictionInputs
            Dataclass containing all prediction inputs for this service.
            See ServicePredictionInputs for field details.
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
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()

        # Validate flow selection configuration
        flow_selection.validate()

        service_id = inputs.service_id

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
                f"Missing inflow keys in ServicePredictionInputs: {missing_inflow_keys}"
            )

        selected_inflows = [inputs.inflows[k] for k in inflow_keys if inflow_allowed(k)]
        arrivals = self.predict_flow_total(selected_inflows, service_id, "arrivals")

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
                f"Missing outflow keys in ServicePredictionInputs: {missing_outflow_keys}"
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
            service_id,
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
        expected_diff = float(arrivals.expectation - departures.expectation)
        net_flow = self._create_prediction(
            service_id,
            "net_flow",
            net_dist.probabilities,
            net_dist.offset,
            expected_override=expected_diff,
        )

        return PredictionBundle(
            entity_id=service_id,
            entity_type="service",
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

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
            is applied.

        Returns
        -------
        DemandPrediction
            Combined prediction for all flows
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

    def aggregate_predictions(
        self,
        entity_id: str,
        entity_type: str,
        child_predictions: List[DemandPrediction],
        max_support: Optional[int] = None,
        flow_type: Optional[str] = None,
    ) -> DemandPrediction:
        """Aggregate multiple demand predictions into a single prediction.

        This method sums demand predictions (e.g. from child entities) by convolving
        their probability distributions.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : str
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
        """
        distributions = [p.probabilities for p in child_predictions]
        p_total = self.convolve_multiple(
            distributions, max_support, entity_id=entity_id, flow_type=flow_type
        )
        return self._create_prediction(entity_id, entity_type, p_total)

    def _create_bundle_from_children(
        self,
        entity_id: str,
        entity_type: str,
        child_bundles: List[PredictionBundle],
        arrivals_max_support: Optional[int] = None,
        departures_max_support: Optional[int] = None,
    ) -> PredictionBundle:
        """Create a prediction bundle by aggregating child bundles."""
        arrivals_preds = [b.arrivals for b in child_bundles]
        departures_preds = [b.departures for b in child_bundles]

        arrivals = self.aggregate_predictions(
            entity_id,
            entity_type,
            arrivals_preds,
            arrivals_max_support,
            flow_type="arrivals",
        )

        # For departures: always calculate physical cap from child predictions' PMF lengths
        departures_physical_cap = 0
        for pred in departures_preds:
            if len(pred.probabilities) > 0:
                # Physical max is the maximum index value (len - 1)
                departures_physical_cap += len(pred.probabilities) - 1

        # Use calculated cap from child predictions
        departures_max_support_from_children = (
            int(departures_physical_cap) if departures_physical_cap > 0 else None
        )

        departures = self.aggregate_predictions(
            entity_id,
            entity_type,
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
        """Convolve multiple distributions with optional truncation and renormalization."""
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
        """Truncate PMF to max_support and assign remaining probability to the last bin."""
        if max_support < 0:
            # Return a single bin with all probability mass
            result = np.array([np.sum(pmf)])
            if return_truncated_mass:
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
        """Get statistics about truncated mass across all predictions."""
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
        """Clear the truncated mass tracking dictionary."""
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
        expectation = (
            float(expected_override)
            if expected_override is not None
            else float(self._expected_value(probabilities, offset))
        )
        
        # Calculate mode
        if len(probabilities) > 0:
            mode_idx = int(np.argmax(probabilities))
            mode = mode_idx + offset
        else:
            mode = offset

        return DemandPrediction(
            entity_id=entity_id,
            entity_type=entity_type,
            probabilities=probabilities,
            expectation=expectation,
            mode=mode,
            percentiles=self._percentiles(probabilities, DEFAULT_PERCENTILES, offset),
            offset=offset,
        )
