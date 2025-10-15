"""Hierarchical demand prediction for hospital bed capacity management.

This module provides classes and functions for predicting hospital bed demand at multiple
hierarchical levels (subspecialty, reporting unit, division, board, hospital) using
probability distributions and convolution operations.

The main components are:

- DemandPrediction: A dataclass representing prediction results with probabilities,
  expected values, and percentiles
- DemandPredictor: Core prediction engine using convolution of probability distributions
- HospitalHierarchy: Represents organizational structure of a hospital
- HierarchicalPredictor: High-level interface for making predictions across all levels
- Utility functions for populating hierarchy from DataFrames

Functions
---------
populate_hierarchy_from_dataframe
    Create HospitalHierarchy from pandas DataFrame with organizational structure
create_hierarchical_predictor
    Create complete HierarchicalPredictor from DataFrame and parameters
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from scipy.stats import poisson

from patientflow.predict.subspecialty import SubspecialtyPredictionInputs, FlowInputs


@dataclass
class DemandPrediction:
    """Result of a demand prediction at any hierarchical level.

    This dataclass encapsulates the complete prediction results for a single
    organizational entity (subspecialty, reporting unit, division, board, or hospital).
    It contains the probability mass function, expected value, and key percentiles.

    Parameters
    ----------
    entity_id : str
        Unique identifier for the organizational entity (e.g., subspecialty name)
    entity_type : str
        Type of entity ('subspecialty', 'reporting_unit', 'division', 'board', 'hospital')
    probabilities : numpy.ndarray
        Probability mass function for bed demand (probability of 0, 1, 2, ... beds)
    expected_value : float
        Expected number of beds needed (mean of the distribution)
    percentiles : dict[int, int]
        Dictionary mapping percentile values (50, 75, 90, 95, 99) to bed counts
    offset : int, optional
        Offset for the support of the distribution. Index i corresponds to value (i + offset).
        Default is 0 for non-negative distributions. For net flow, offset is typically negative.

    Attributes
    ----------
    entity_id : str
        Unique identifier for the organizational entity
    entity_type : str
        Type of entity in the hierarchy
    probabilities : numpy.ndarray
        Probability mass function for bed demand
    expected_value : float
        Expected number of beds needed
    percentiles : dict[int, int]
        Percentile values for bed demand
    offset : int
        Offset for the support (index 0 corresponds to value = offset)
    max_beds : int
        Maximum number of beds in the probability distribution

    Notes
    -----
    The probabilities array represents P(X=k) where X is the number of beds needed
    and k ranges from offset to (offset + max_beds). The sum of all probabilities should equal 1.0.
    For non-negative distributions (arrivals, departures), offset=0.
    For net flow distributions, offset can be negative (e.g., -10 means support starts at -10).
    """

    entity_id: str
    entity_type: str
    probabilities: np.ndarray
    expected_value: float
    percentiles: Dict[int, int]
    offset: int = 0

    @property
    def max_beds(self) -> int:
        """Maximum number of beds in the probability distribution.

        Returns
        -------
        int
            Maximum bed count (length of probabilities array minus 1)
        """
        return len(self.probabilities) - 1

    def to_pretty(self, max_probs: int = 10, precision: int = 3) -> str:
        """Return a concise, human-friendly string representation.

        Parameters
        ----------
        max_probs : int, default 10
            Maximum number of head probabilities to display from the PMF
        precision : int, default 3
            Number of decimal places for numeric values

        Returns
        -------
        str
            Formatted summary string
        """
        probs = self.probabilities

        # Find where the probability mass is concentrated
        if len(probs) <= max_probs:
            start_idx = 0
            end_idx = len(probs)
        else:
            mode_idx = int(np.argmax(probs))
            half_window = max_probs // 2
            start_idx = max(0, mode_idx - half_window)
            end_idx = min(len(probs), start_idx + max_probs)
            if end_idx - start_idx < max_probs:
                start_idx = max(0, end_idx - max_probs)

        head = ", ".join([f"{p:.{precision}g}" for p in probs[start_idx:end_idx]])
        remaining = len(probs) - end_idx
        tail_note = f" … +{remaining} more" if remaining > 0 else ""

        pct_items = ", ".join(
            [f"P{p}={self.percentiles.get(p)}" for p in sorted(self.percentiles.keys())]
        )

        # Use fixed-width formatting for proper alignment
        pmf_label = f"PMF[{start_idx}:{end_idx}]:"
        return (
            f"{self.entity_type}: {self.entity_id}\n"
            f"  {'Expectation:':<16} {self.expected_value:.{precision}f}\n"
            f"  {'Percentiles:':<16} {pct_items}\n"
            f"  {pmf_label:<16} [{head}]{tail_note}"
        )

    def __str__(self) -> str:
        return self.to_pretty()


@dataclass
class FlowSelection:
    """Configuration for which flows to include in predictions.

    This class controls which patient flows (arrivals and departures) are included
    when making predictions. It allows users to customize predictions based on their
    specific use cases, such as excluding transfers or focusing only on certain types
    of admissions.

    Attributes
    ----------
    inflow_keys : List[str]
        List of inflow identifiers to include in predictions.
        Standard values: ["ed_current", "ed_yta", "non_ed_yta", "elective_yta", "transfers_in"]
    outflow_keys : List[str]
        List of outflow identifiers to include in predictions.
        Standard values: ["departures"]
        Future extensions may include: ["transfers_out", "deaths"]

    Examples
    --------
    >>> # Include all flows (arrivals and departures)
    >>> selection = FlowSelection.default()

    >>> # Only incoming flows (all arrivals, no departures)
    >>> selection = FlowSelection.incoming_flow()

    >>> # Only outgoing flows (all departures, no arrivals)
    >>> selection = FlowSelection.outgoing_flow()

    >>> # Custom selection
    >>> selection = FlowSelection.custom(
    ...     include_inflows=["ed_current", "ed_yta"],
    ...     include_outflows=["departures"]
    ... )
    """

    inflow_keys: List[str]
    outflow_keys: List[str]

    @classmethod
    def default(cls) -> "FlowSelection":
        """All flows included (default behavior).

        Returns
        -------
        FlowSelection
            Configuration including all standard inflows and outflows
        """
        return cls(
            inflow_keys=[
                "ed_current",
                "ed_yta",
                "non_ed_yta",
                "elective_yta",
                "transfers_in",
            ],
            outflow_keys=["departures"],
        )

    @classmethod
    def incoming_flow(cls) -> "FlowSelection":
        """Select incoming patient flows only (all arrivals, no departures).

        Includes all inflow types (ed_current, ed_yta, non_ed_yta,
        elective_yta, transfers_in) but excludes all outflows.
        Useful for analyzing arrival patterns independently of departures.

        Returns
        -------
        FlowSelection
            Configuration with all inflows and no outflows
        """
        return cls(
            inflow_keys=[
                "ed_current",
                "ed_yta",
                "non_ed_yta",
                "elective_yta",
                "transfers_in",
            ],
            outflow_keys=[],
        )

    @classmethod
    def outgoing_flow(cls) -> "FlowSelection":
        """Select outgoing patient flows only (all departures, no arrivals).

        Includes all outflow types but excludes all inflows.
        Useful for analysing discharge patterns independently of arrivals.

        Returns
        -------
        FlowSelection
            Configuration with no inflows and all outflows
        """
        return cls(inflow_keys=[], outflow_keys=["departures"])

    @classmethod
    def custom(
        cls, include_inflows: List[str], include_outflows: List[str]
    ) -> "FlowSelection":
        """Custom flow selection.

        Parameters
        ----------
        include_inflows : List[str]
            List of inflow identifiers to include
        include_outflows : List[str]
            List of outflow identifiers to include

        Returns
        -------
        FlowSelection
            Custom configuration with specified flows
        """
        return cls(inflow_keys=include_inflows, outflow_keys=include_outflows)


@dataclass
class PredictionBundle:
    """Complete prediction results for arrivals, departures, and net flow.

    This dataclass bundles together predictions for patient arrivals, departures,
    and the net change in bed occupancy for a single organizational entity.
    It provides a comprehensive view of demand dynamics.

    Attributes
    ----------
    entity_id : str
        Unique identifier for the entity (subspecialty, reporting unit, etc.)
    entity_type : str
        Type of entity in the hierarchy
    arrivals : DemandPrediction
        Prediction for total patient arrivals
    departures : DemandPrediction
        Prediction for total patient departures
    net_flow : DemandPrediction
        Prediction for net change in bed occupancy (arrivals - departures).
        This is the full probability distribution of the difference.
    flow_selection : FlowSelection
        Configuration specifying which flows were included in this prediction.
        Tracks which inflows and outflows were aggregated into the arrivals
        and departures distributions.

    Notes
    -----
    Net flow is computed as the distribution of the difference between arrivals
    and departures. The PMF may include negative values, representing net decrease
    in bed demand.

    The flow_selection attribute allows you to determine which flows contributed
    to the aggregated predictions. To see individual flow contributions, access
    the original SubspecialtyPredictionInputs.
    """

    entity_id: str
    entity_type: str
    arrivals: DemandPrediction
    departures: DemandPrediction
    net_flow: DemandPrediction
    flow_selection: FlowSelection

    def to_summary(self) -> Dict[str, Any]:
        """Return human-readable summary of predictions.

        Returns
        -------
        dict
            Dictionary containing key summary statistics:
            - entity: Entity identifier with type
            - arrivals_pmf: PMF representation of arrivals
            - departures_pmf: PMF representation of departures
            - net_flow_pmf: PMF representation of net flow
            - flows_included: Number of inflows and outflows included
        """

        def format_pmf(pred: DemandPrediction, max_display: int = 10) -> str:
            """Format PMF similar to SubspecialtyPredictionInputs."""
            arr = pred.probabilities
            offset = pred.offset
            expectation = pred.expected_value

            if len(arr) <= max_display:
                values = ", ".join(f"{v:.3f}" for v in arr)
                start_val = offset
                end_val = len(arr) + offset
                return f"PMF[{start_val}:{end_val}]: [{values}] (E={expectation:.1f})"

            # Determine display window centered on expectation
            center_idx = int(np.round(expectation - offset))
            half_window = max_display // 2
            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(arr), start_idx + max_display)

            # Adjust if we're near the end
            if end_idx - start_idx < max_display:
                start_idx = max(0, end_idx - max_display)

            # Format the displayed portion
            display_values = ", ".join(f"{v:.3f}" for v in arr[start_idx:end_idx])

            # Show with value range (accounting for offset)
            start_val = start_idx + offset
            end_val = end_idx + offset
            return (
                f"PMF[{start_val}:{end_val}]: [{display_values}] (E={expectation:.1f})"
            )

        return {
            "entity": f"{self.entity_type}: {self.entity_id}",
            "arrivals_pmf": format_pmf(self.arrivals),
            "departures_pmf": format_pmf(self.departures),
            "net_flow_pmf": format_pmf(self.net_flow),
            "flows_included": f"{len(self.flow_selection.inflow_keys)} inflows, {len(self.flow_selection.outflow_keys)} outflows",
        }

    def __str__(self) -> str:
        summary = self.to_summary()
        return (
            f"PredictionBundle({summary['entity']})\n"
            f"  {'Arrivals:':<12} {summary['arrivals_pmf']}\n"
            f"  {'Departures:':<12} {summary['departures_pmf']}\n"
            f"  {'Net flow:':<12} {summary['net_flow_pmf']}\n"
            f"  {'Flows:':<12} {summary['flows_included']}"
        )


class DemandPredictor:
    """Hierarchical demand prediction for hospital bed capacity.

    This class provides the core prediction engine for computing bed demand at
    different hierarchical levels. It uses convolution of probability distributions
    to combine predictions from lower levels into higher levels.

    Parameters
    ----------
    epsilon : float, default=1e-7
        Truncation threshold for probability distribution tails. Distributions
        are truncated when the cumulative probability exceeds (1 - epsilon).

    Attributes
    ----------
    epsilon : float
        Truncation threshold for probability distributions
    cache : dict
        Cache for storing computed predictions (currently unused)

    Notes
    -----
    The prediction process involves:

    1. Generating Poisson distributions for yet-to-arrive patients
    2. Combining with current patient distributions using convolution
    3. Aggregating across organizational levels using multiple convolutions
    4. Computing statistics (expected value, percentiles) for each level

    The class uses discrete convolution to combine probability distributions.
    For computational efficiency, distributions are periodically truncated during
    multiple convolutions to prevent exponential growth in array sizes.
    """

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        self.cache: Dict[str, DemandPrediction] = {}

    def predict_flow_total(
        self,
        flow_inputs: List[FlowInputs],
        entity_id: str,
        entity_type: str,
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

        Returns
        -------
        DemandPrediction
            Combined prediction for all flows

        Notes
        -----
        Flows are combined through convolution, which represents the distribution
        of the sum of independent random variables. Periodic truncation is applied
        to maintain computational efficiency.
        """
        # Start with degenerate distribution at 0
        p_total = np.array([1.0])

        for flow in flow_inputs:
            if flow.flow_type == "poisson":
                # Generate Poisson PMF
                max_k = self._calculate_poisson_max(flow.distribution)
                p_flow = self._poisson_pmf(flow.distribution, max_k)
            elif flow.flow_type == "pmf":
                # Use PMF directly
                p_flow = flow.distribution
            else:
                raise ValueError(
                    f"Unknown flow type: {flow.flow_type}. Expected 'pmf' or 'poisson'."
                )

            # Convolve this flow with running total
            p_total = self._convolve(p_total, p_flow)
            p_total = self._truncate(p_total)

        return self._create_prediction(entity_id, entity_type, p_total)

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
            Configuration specifying which flows to include. If None, uses
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
        ...     spec_id, inputs, flow_selection=FlowSelection.incoming_flow()
        ... )

        >>> # Custom selection
        >>> bundle = predictor.predict_subspecialty(
        ...     spec_id, inputs,
        ...     flow_selection=FlowSelection.custom(
        ...         include_inflows=["ed_current", "ed_yta"],
        ...         include_outflows=["departures"]
        ...     )
        ... )
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()

        # Compute arrivals from selected inflows
        selected_inflows = [
            inputs.inflows[key]
            for key in flow_selection.inflow_keys
            if key in inputs.inflows
        ]
        arrivals = self.predict_flow_total(
            selected_inflows, subspecialty_id, "arrivals"
        )

        # Compute departures from selected outflows
        selected_outflows = [
            inputs.outflows[key]
            for key in flow_selection.outflow_keys
            if key in inputs.outflows
        ]
        departures = self.predict_flow_total(
            selected_outflows, subspecialty_id, "departures"
        )

        # Compute net flow distribution
        p_net_flow, net_offset = self._compute_net_flow_pmf(
            arrivals.probabilities, departures.probabilities
        )
        net_flow = self._create_prediction(
            subspecialty_id, "net_flow", p_net_flow, net_offset
        )

        return PredictionBundle(
            entity_id=subspecialty_id,
            entity_type="subspecialty",
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

    def predict_reporting_unit(
        self, reporting_unit_id: str, subspecialty_predictions: List[DemandPrediction]
    ) -> DemandPrediction:
        """Predict demand for a reporting unit from its subspecialties.

        This method aggregates demand predictions from all subspecialties within
        a reporting unit by convolving their probability distributions.

        Parameters
        ----------
        reporting_unit_id : str
            Unique identifier for the reporting unit
        subspecialty_predictions : list[DemandPrediction]
            List of demand predictions from subspecialties belonging to this reporting unit

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the reporting unit

        Notes
        -----
        The method convolves multiple probability distributions efficiently by
        sorting them by expected value and applying periodic truncation to prevent
        computational overflow.
        """
        distributions = [p.probabilities for p in subspecialty_predictions]
        p_total = self._convolve_multiple(distributions)
        return self._create_prediction(reporting_unit_id, "reporting_unit", p_total)

    def predict_division(
        self, division_id: str, reporting_unit_predictions: List[DemandPrediction]
    ) -> DemandPrediction:
        """Predict demand for a division from its reporting units.

        This method aggregates demand predictions from all reporting units within
        a division by convolving their probability distributions.

        Parameters
        ----------
        division_id : str
            Unique identifier for the division
        reporting_unit_predictions : list[DemandPrediction]
            List of demand predictions from reporting units belonging to this division

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the division
        """
        distributions = [p.probabilities for p in reporting_unit_predictions]
        p_total = self._convolve_multiple(distributions)
        return self._create_prediction(division_id, "division", p_total)

    def predict_board(
        self, board_id: str, division_predictions: List[DemandPrediction]
    ) -> DemandPrediction:
        """Predict demand for a board from its divisions.

        This method aggregates demand predictions from all divisions within
        a board by convolving their probability distributions.

        Parameters
        ----------
        board_id : str
            Unique identifier for the board
        division_predictions : list[DemandPrediction]
            List of demand predictions from divisions belonging to this board

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the board
        """
        distributions = [p.probabilities for p in division_predictions]
        p_total = self._convolve_multiple(distributions)
        return self._create_prediction(board_id, "board", p_total)

    def predict_hospital(
        self, hospital_id: str, board_predictions: List[DemandPrediction]
    ) -> DemandPrediction:
        """Predict demand for entire hospital from its boards.

        This method aggregates demand predictions from all boards within
        a hospital by convolving their probability distributions.

        Parameters
        ----------
        hospital_id : str
            Unique identifier for the hospital
        board_predictions : list[DemandPrediction]
            List of demand predictions from boards belonging to this hospital

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the entire hospital
        """
        distributions = [p.probabilities for p in board_predictions]
        p_total = self._convolve_multiple(distributions)
        return self._create_prediction(hospital_id, "hospital", p_total)

    def _convolve_multiple(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Convolve multiple distributions with periodic truncation.

        This method efficiently convolves multiple probability distributions by
        sorting them by expected value and applying periodic truncation to prevent
        computational overflow.

        Parameters
        ----------
        distributions : list[numpy.ndarray]
            List of probability mass functions to convolve

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Distributions are sorted by expected value for computational efficiency.
        Truncation is applied every 5 convolutions or when the result exceeds 500 elements.
        """
        if not distributions:
            return np.array([1.0])
        if len(distributions) == 1:
            return distributions[0]

        # Sort by expected value for efficiency
        distributions = sorted(distributions, key=lambda p: self._expected_value(p))

        result = distributions[0]
        for i, dist in enumerate(distributions[1:], 1):
            result = self._convolve(result, dist)
            # Periodic truncation
            if i % 5 == 0 or len(result) > 500:
                result = self._truncate(result)

        return self._truncate(result)

    def _convolve(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Discrete convolution of two probability distributions.

        This method computes the convolution of two probability mass functions,
        which represents the distribution of the sum of two independent random variables.

        Parameters
        ----------
        p : numpy.ndarray
            First probability mass function
        q : numpy.ndarray
            Second probability mass function

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Uses numpy.convolve for efficient computation of discrete convolution.
        """
        return np.convolve(p, q)

    def _truncate(self, p: np.ndarray) -> np.ndarray:
        """Truncate distribution to maintain tail probability < epsilon.

        This method truncates a probability distribution by removing the tail
        such that the remaining probability mass is at least (1 - epsilon).

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function to truncate

        Returns
        -------
        numpy.ndarray
            Truncated probability mass function

        Notes
        -----
        The truncation point is found using binary search on the cumulative sum.
        This prevents computational overflow while maintaining high accuracy.
        """
        cumsum = np.cumsum(p)
        cutoff_idx = np.searchsorted(cumsum, 1 - self.epsilon) + 1
        return p[:cutoff_idx]

    def _calculate_poisson_max(self, lambda_param: float) -> int:
        """Find maximum k where P(X > k) < epsilon.

        This method determines the upper bound for a Poisson distribution such
        that the tail probability is less than epsilon.

        Parameters
        ----------
        lambda_param : float
            Poisson parameter (mean and variance)

        Returns
        -------
        int
            Maximum value k such that P(X > k) < epsilon

        Notes
        -----
        Uses the inverse cumulative distribution function (percent point function)
        to find the cutoff point efficiently.
        """
        if lambda_param == 0:
            return 0
        return poisson.ppf(1 - self.epsilon, lambda_param).astype(int)

    def _poisson_pmf(self, lambda_param: float, max_k: int) -> np.ndarray:
        """Generate Poisson PMF from 0 to max_k.

        This method generates the probability mass function for a Poisson
        distribution truncated at max_k.

        Parameters
        ----------
        lambda_param : float
            Poisson parameter (mean and variance)
        max_k : int
            Maximum value to include in the PMF

        Returns
        -------
        numpy.ndarray
            Probability mass function from 0 to max_k

        Notes
        -----
        Returns [1.0] for lambda_param = 0 (degenerate distribution at 0).
        """
        if lambda_param == 0:
            return np.array([1.0])
        k = np.arange(max_k + 1)
        return poisson.pmf(k, lambda_param)

    def _expected_value(self, p: np.ndarray, offset: int = 0) -> float:
        """Calculate expected value of distribution.

        This method computes the expected value (mean) of a discrete probability
        distribution.

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        float
            Expected value E[X] = sum(k * P(X=k))
        """
        return np.sum((np.arange(len(p)) + offset) * p)

    def _percentiles(
        self, p: np.ndarray, percentile_list: List[int], offset: int = 0
    ) -> Dict[int, int]:
        """Calculate percentiles from probability distribution.

        This method computes specified percentiles from a discrete probability
        distribution using the cumulative distribution function.

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function
        percentile_list : list[int]
            List of percentiles to compute (e.g., [50, 75, 90, 95, 99])
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        dict[int, int]
            Dictionary mapping percentiles to corresponding values

        Notes
        -----
        Uses binary search on the cumulative sum for efficient percentile computation.
        """
        cumsum = np.cumsum(p)
        result = {}
        for pct in percentile_list:
            idx = np.searchsorted(cumsum, pct / 100.0)
            result[pct] = int(idx + offset)
        return result

    def _compute_net_flow_pmf(
        self, p_arrivals: np.ndarray, p_departures: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Compute PMF for net flow (arrivals - departures).

        This method computes the probability distribution for the difference
        between two independent random variables (arrivals and departures).

        Parameters
        ----------
        p_arrivals : numpy.ndarray
            Probability mass function for arrivals (support: 0 to max_arrivals)
        p_departures : numpy.ndarray
            Probability mass function for departures (support: 0 to max_departures)

        Returns
        -------
        tuple[numpy.ndarray, int]
            Tuple of (PMF, offset) where:
            - PMF is the truncated probability mass function for net flow
            - offset is the value corresponding to index 0 (typically negative)

        Notes
        -----
        For each net flow value n in [-max_departures, max_arrivals]:
            P(Net = n) = Σ P(Arrivals = a) × P(Departures = d) where a - d = n

        The resulting PMF is truncated using the same epsilon threshold
        to remove negligible tail probabilities from both ends.
        The offset tracks where the support starts after truncation.
        """
        max_arrivals = len(p_arrivals) - 1
        max_departures = len(p_departures) - 1

        # Net flow ranges from -max_departures to +max_arrivals
        # Array index k corresponds to net flow value (k - max_departures)
        net_flow_size = max_arrivals + max_departures + 1
        p_net = np.zeros(net_flow_size)

        # Compute probability for each possible net flow value
        for a in range(len(p_arrivals)):
            for d in range(len(p_departures)):
                net = a - d  # Net flow value
                idx = net + max_departures  # Array index (shifted to handle negatives)
                p_net[idx] += p_arrivals[a] * p_departures[d]

        # Truncate from both ends to remove negligible probabilities
        # Track the initial offset and how much we remove from the left
        initial_offset = -max_departures
        p_truncated, left_cutoff = self._truncate_symmetric_with_offset(p_net)
        final_offset = initial_offset + left_cutoff

        return p_truncated, final_offset

    def _truncate_symmetric_with_offset(self, p: np.ndarray) -> tuple[np.ndarray, int]:
        """Truncate a distribution from both ends and return offset.

        This method truncates a probability distribution that may have negative
        support (like net flow) by removing negligible probabilities from both tails.

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function to truncate

        Returns
        -------
        tuple[numpy.ndarray, int]
            Tuple of (truncated PMF, left_cutoff index)
            The left_cutoff indicates how many elements were removed from the left.

        Notes
        -----
        Truncation is applied to keep total removed probability < epsilon from each tail.
        """
        if len(p) == 0:
            return p, 0

        cumsum = np.cumsum(p)
        total_prob = cumsum[-1]

        if total_prob == 0:
            return np.array([1.0]), 0  # Degenerate case

        # Find where cumulative probability exceeds epsilon (from left)
        left_cutoff = np.searchsorted(cumsum, self.epsilon)

        # Find where remaining probability is less than epsilon (from right)
        cumsum_reversed = np.cumsum(p[::-1])
        right_cutoff = len(p) - np.searchsorted(cumsum_reversed, self.epsilon)

        # Ensure we keep at least one element
        left_cutoff = max(0, left_cutoff)
        right_cutoff = min(len(p), max(right_cutoff, left_cutoff + 1))

        return p[left_cutoff:right_cutoff], left_cutoff

    def _create_prediction(
        self,
        entity_id: str,
        entity_type: str,
        probabilities: np.ndarray,
        offset: int = 0,
    ) -> DemandPrediction:
        """Create a DemandPrediction object with computed statistics.

        This helper method creates a complete DemandPrediction object by computing
        the expected value and percentiles from the probability mass function.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : str
            Type of entity in the hierarchy
        probabilities : numpy.ndarray
            Probability mass function
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        DemandPrediction
            Complete prediction object with all computed statistics

        Notes
        -----
        Computes percentiles for 50th, 75th, 90th, 95th, and 99th percentiles.
        """
        return DemandPrediction(
            entity_id=entity_id,
            entity_type=entity_type,
            probabilities=probabilities,
            expected_value=self._expected_value(probabilities, offset),
            percentiles=self._percentiles(probabilities, [50, 75, 90, 95, 99], offset),
            offset=offset,
        )


class HospitalHierarchy:
    """Represents the organizational hierarchy of a hospital.

    This class maintains the hierarchical structure of a hospital organization,
    mapping entities at each level to their parent entities. The hierarchy typically
    follows the structure: Hospital -> Board -> Division -> Reporting Unit -> Subspecialty.

    The class provides methods to build the hierarchy and query relationships
    between entities at different levels.

    Attributes
    ----------
    subspecialties : dict[str, str]
        Mapping from subspecialty_id to parent reporting_unit_id
    reporting_units : dict[str, str]
        Mapping from reporting_unit_id to parent division_id
    divisions : dict[str, str]
        Mapping from division_id to parent board_id
    boards : dict[str, str]
        Mapping from board_id to parent hospital_id

    Notes
    -----
    The hierarchy is built bottom-up, starting with subspecialties and working
    up to the hospital level. Each entity is mapped to its immediate parent.
    """

    def __init__(self):
        self.subspecialties = {}  # subspecialty_id -> parent reporting_unit_id
        self.reporting_units = {}  # reporting_unit_id -> parent division_id
        self.divisions = {}  # division_id -> parent board_id
        self.boards = {}  # board_id -> parent hospital_id

    def __repr__(self) -> str:
        lines = []
        lines.append("HospitalHierarchy:")
        lines.append(f"  Subspecialties: {len(self.subspecialties)}")
        lines.append(f"  Reporting Units: {len(set(self.subspecialties.values()))}")
        lines.append(f"  Divisions: {len(set(self.reporting_units.values()))}")
        lines.append(f"  Boards: {len(set(self.divisions.values()))}")
        lines.append(f"  Hospitals: {len(set(self.boards.values()))}")
        return "\n".join(lines)

    def add_subspecialty(self, subspecialty_id: str, reporting_unit_id: str):
        """Add a subspecialty to the hierarchy.

        Parameters
        ----------
        subspecialty_id : str
            Unique identifier for the subspecialty
        reporting_unit_id : str
            Parent reporting unit identifier
        """
        self.subspecialties[subspecialty_id] = reporting_unit_id

    def add_reporting_unit(self, reporting_unit_id: str, division_id: str):
        """Add a reporting unit to the hierarchy.

        Parameters
        ----------
        reporting_unit_id : str
            Unique identifier for the reporting unit
        division_id : str
            Parent division identifier
        """
        self.reporting_units[reporting_unit_id] = division_id

    def add_division(self, division_id: str, board_id: str):
        """Add a division to the hierarchy.

        Parameters
        ----------
        division_id : str
            Unique identifier for the division
        board_id : str
            Parent board identifier
        """
        self.divisions[division_id] = board_id

    def add_board(self, board_id: str, hospital_id: str):
        """Add a board to the hierarchy.

        Parameters
        ----------
        board_id : str
            Unique identifier for the board
        hospital_id : str
            Parent hospital identifier
        """
        self.boards[board_id] = hospital_id

    def get_subspecialties_for_reporting_unit(
        self, reporting_unit_id: str
    ) -> List[str]:
        """Get all subspecialties belonging to a reporting unit.

        Parameters
        ----------
        reporting_unit_id : str
            Reporting unit identifier

        Returns
        -------
        list[str]
            List of subspecialty identifiers belonging to the reporting unit
        """
        return [
            sid for sid, rid in self.subspecialties.items() if rid == reporting_unit_id
        ]

    def get_reporting_units_for_division(self, division_id: str) -> List[str]:
        """Get all reporting units belonging to a division.

        Parameters
        ----------
        division_id : str
            Division identifier

        Returns
        -------
        list[str]
            List of reporting unit identifiers belonging to the division
        """
        return [rid for rid, did in self.reporting_units.items() if did == division_id]

    def get_divisions_for_board(self, board_id: str) -> List[str]:
        """Get all divisions belonging to a board.

        Parameters
        ----------
        board_id : str
            Board identifier

        Returns
        -------
        list[str]
            List of division identifiers belonging to the board
        """
        return [did for did, bid in self.divisions.items() if bid == board_id]

    def get_boards_for_hospital(self, hospital_id: str) -> List[str]:
        """Get all boards belonging to a hospital.

        Parameters
        ----------
        hospital_id : str
            Hospital identifier

        Returns
        -------
        list[str]
            List of board identifiers belonging to the hospital
        """
        return [bid for bid, hid in self.boards.items() if hid == hospital_id]

    def get_all_subspecialties(self) -> List[str]:
        """Get all subspecialty identifiers in the hierarchy.

        Returns
        -------
        list[str]
            List of all subspecialty identifiers
        """
        return list(self.subspecialties.keys())

    def get_all_hospitals(self) -> List[str]:
        """Get all hospital identifiers in the hierarchy.

        Returns
        -------
        list[str]
            List of all hospital identifiers
        """
        return list(set(self.boards.values()))


class HierarchicalPredictor:
    """High-level interface for hierarchical predictions with caching.

    This class provides a convenient interface for making predictions across all
    levels of the hospital hierarchy. It orchestrates the bottom-up prediction
    process, starting from subspecialties and aggregating up to the hospital level.

    The class maintains a cache of prediction bundles for efficient retrieval and
    provides methods to compute predictions for all levels at once. Each bundle
    tracks arrivals, departures, and net flow separately.

    Parameters
    ----------
    hierarchy : HospitalHierarchy
        Organizational structure of the hospital
    predictor : DemandPredictor
        Core prediction engine for demand calculations

    Attributes
    ----------
    hierarchy : HospitalHierarchy
        Hospital organizational structure
    predictor : DemandPredictor
        Core prediction engine
    cache : dict
        Cache for storing computed prediction bundles (PredictionBundle objects)

    Notes
    -----
    Predictions are computed bottom-up: subspecialties -> reporting units ->
    divisions -> boards -> hospital. At each level, arrivals and departures
    are aggregated separately using convolution, and net flows are computed.
    """

    def __init__(self, hierarchy: HospitalHierarchy, predictor: DemandPredictor):
        self.hierarchy = hierarchy
        self.predictor = predictor
        self.cache: Dict[str, PredictionBundle] = {}

    def predict_all_levels(
        self,
        subspecialty_data: Dict[str, SubspecialtyPredictionInputs],
        hospital_id: Optional[str] = None,
        flow_selection: Optional[FlowSelection] = None,
    ) -> Dict[str, PredictionBundle]:
        """Compute predictions for all levels bottom-up.

        This method orchestrates the complete hierarchical prediction process,
        starting from subspecialties and aggregating predictions up through
        reporting units, divisions, boards, and finally to the hospital level.

        For each level, tracks arrivals, departures, and net flow separately.

        Parameters
        ----------
        subspecialty_data : dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping subspecialty_id to SubspecialtyPredictionInputs dataclass
            containing all prediction parameters for that subspecialty
        hospital_id : str, optional
            Unique identifier for the hospital. If not provided and the hierarchy
            contains exactly one hospital, that hospital will be used automatically.
            Required if the hierarchy contains multiple hospitals.
        flow_selection : FlowSelection, optional
            Configuration for which flows to include. If None, uses FlowSelection.default()
            which includes all flows.

        Returns
        -------
        dict[str, PredictionBundle]
            Dictionary mapping entity_id to PredictionBundle for all levels:
            subspecialties, reporting units, divisions, boards, and hospital.
            Each bundle contains arrivals, departures, and net flow predictions.

        Raises
        ------
        ValueError
            If hospital_id is not provided and the hierarchy contains multiple hospitals

        Notes
        -----
        The prediction process follows this sequence:
        1. Predict subspecialties using provided parameters (returns bundles)
        2. Aggregate subspecialty arrivals/departures into reporting unit bundles
        3. Aggregate reporting unit arrivals/departures into division bundles
        4. Aggregate division arrivals/departures into board bundles
        5. Aggregate board arrivals/departures into hospital bundle

        All predictions are cached for efficient retrieval.
        """
        # Determine hospital_id if not provided
        if hospital_id is None:
            hospitals = self.hierarchy.get_all_hospitals()
            if len(hospitals) == 0:
                raise ValueError("No hospitals found in hierarchy")
            elif len(hospitals) == 1:
                hospital_id = hospitals[0]
            else:
                raise ValueError(
                    f"Multiple hospitals found in hierarchy: {hospitals}. "
                    "Please specify hospital_id parameter."
                )

        results = {}

        # Level 1: Subspecialties
        for subspecialty_id, inputs in subspecialty_data.items():
            bundle = self.predictor.predict_subspecialty(
                subspecialty_id, inputs, flow_selection
            )
            results[subspecialty_id] = bundle
            self.cache[subspecialty_id] = bundle

        # Level 2: Reporting units
        reporting_units_set = set(self.hierarchy.subspecialties.values())
        for reporting_unit_id in reporting_units_set:
            subspecialties = self.hierarchy.get_subspecialties_for_reporting_unit(
                reporting_unit_id
            )
            subspecialty_bundles = [results[sid] for sid in subspecialties]

            # Aggregate arrivals and departures separately
            arrivals_preds = [b.arrivals for b in subspecialty_bundles]
            departures_preds = [b.departures for b in subspecialty_bundles]

            arrivals = self.predictor.predict_reporting_unit(
                reporting_unit_id, arrivals_preds
            )
            departures = self.predictor.predict_reporting_unit(
                reporting_unit_id, departures_preds
            )

            # Compute net flow distribution
            p_net_flow, net_offset = self.predictor._compute_net_flow_pmf(
                arrivals.probabilities, departures.probabilities
            )
            net_flow = self.predictor._create_prediction(
                reporting_unit_id, "net_flow", p_net_flow, net_offset
            )

            bundle = PredictionBundle(
                entity_id=reporting_unit_id,
                entity_type="reporting_unit",
                arrivals=arrivals,
                departures=departures,
                net_flow=net_flow,
                flow_selection=subspecialty_bundles[
                    0
                ].flow_selection,  # Inherit from children
            )
            results[reporting_unit_id] = bundle
            self.cache[reporting_unit_id] = bundle

        # Level 3: Divisions
        divisions_set = set(self.hierarchy.reporting_units.values())
        for division_id in divisions_set:
            reporting_units_list = self.hierarchy.get_reporting_units_for_division(
                division_id
            )
            ru_bundles = [results[rid] for rid in reporting_units_list]

            arrivals_preds = [b.arrivals for b in ru_bundles]
            departures_preds = [b.departures for b in ru_bundles]

            arrivals = self.predictor.predict_division(division_id, arrivals_preds)
            departures = self.predictor.predict_division(division_id, departures_preds)

            # Compute net flow distribution
            p_net_flow, net_offset = self.predictor._compute_net_flow_pmf(
                arrivals.probabilities, departures.probabilities
            )
            net_flow = self.predictor._create_prediction(
                division_id, "net_flow", p_net_flow, net_offset
            )

            bundle = PredictionBundle(
                entity_id=division_id,
                entity_type="division",
                arrivals=arrivals,
                departures=departures,
                net_flow=net_flow,
                flow_selection=ru_bundles[0].flow_selection,  # Inherit from children
            )
            results[division_id] = bundle
            self.cache[division_id] = bundle

        # Level 4: Boards
        boards_set = set(self.hierarchy.divisions.values())
        for board_id in boards_set:
            divisions_list = self.hierarchy.get_divisions_for_board(board_id)
            div_bundles = [results[did] for did in divisions_list]

            arrivals_preds = [b.arrivals for b in div_bundles]
            departures_preds = [b.departures for b in div_bundles]

            arrivals = self.predictor.predict_board(board_id, arrivals_preds)
            departures = self.predictor.predict_board(board_id, departures_preds)

            # Compute net flow distribution
            p_net_flow, net_offset = self.predictor._compute_net_flow_pmf(
                arrivals.probabilities, departures.probabilities
            )
            net_flow = self.predictor._create_prediction(
                board_id, "net_flow", p_net_flow, net_offset
            )

            bundle = PredictionBundle(
                entity_id=board_id,
                entity_type="board",
                arrivals=arrivals,
                departures=departures,
                net_flow=net_flow,
                flow_selection=div_bundles[0].flow_selection,  # Inherit from children
            )
            results[board_id] = bundle
            self.cache[board_id] = bundle

        # Level 5: Hospital
        boards_list = self.hierarchy.get_boards_for_hospital(hospital_id)
        board_bundles = [results[bid] for bid in boards_list]

        arrivals_preds = [b.arrivals for b in board_bundles]
        departures_preds = [b.departures for b in board_bundles]

        arrivals = self.predictor.predict_hospital(hospital_id, arrivals_preds)
        departures = self.predictor.predict_hospital(hospital_id, departures_preds)

        # Compute net flow distribution
        p_net_flow, net_offset = self.predictor._compute_net_flow_pmf(
            arrivals.probabilities, departures.probabilities
        )
        net_flow = self.predictor._create_prediction(
            hospital_id, "net_flow", p_net_flow, net_offset
        )

        bundle = PredictionBundle(
            entity_id=hospital_id,
            entity_type="hospital",
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=board_bundles[0].flow_selection,  # Inherit from children
        )
        results[hospital_id] = bundle
        self.cache[hospital_id] = bundle

        return results

    def get_prediction(self, entity_id: str) -> Optional[PredictionBundle]:
        """Retrieve cached prediction for any entity.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity

        Returns
        -------
        PredictionBundle or None
            Cached prediction bundle if available, None otherwise.
            Bundle contains arrivals, departures, and net flow predictions.
        """
        return self.cache.get(entity_id)


def populate_hierarchy_from_dataframe(
    df: pd.DataFrame, hospital_id: Optional[str] = None
) -> HospitalHierarchy:
    """Populate HospitalHierarchy from a pandas DataFrame.

    This function extracts the organizational hierarchy from a DataFrame containing
    the hospital structure. It builds the hierarchy by establishing parent-child
    relationships between entities at different levels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing hospital organizational structure. Must have columns:
        - 'board': Board identifier
        - 'division': Division identifier
        - 'reporting_unit': Reporting unit identifier
        - 'sub_specialty': Subspecialty identifier
        Additional columns are ignored.
    hospital_id : str, optional
        Hospital identifier to link all boards to a single hospital.
        If None, boards are not linked to a hospital.

    Returns
    -------
    HospitalHierarchy
        Populated hierarchy with all relationships established

    Notes
    -----
    The function:

    1. Removes duplicate rows and rows with missing values
    2. Establishes subspecialty -> reporting_unit relationships
    3. Establishes reporting_unit -> division relationships
    4. Establishes division -> board relationships
    5. Optionally establishes board -> hospital relationships

    Duplicate relationships are automatically handled.
    """
    hierarchy = HospitalHierarchy()

    # Remove duplicates and any rows with missing values
    df = df.dropna().drop_duplicates()

    # Add all subspecialties to reporting units
    for _, row in df[["sub_specialty", "reporting_unit"]].drop_duplicates().iterrows():
        hierarchy.add_subspecialty(row["sub_specialty"], row["reporting_unit"])

    # Add all reporting units to divisions
    for _, row in df[["reporting_unit", "division"]].drop_duplicates().iterrows():
        hierarchy.add_reporting_unit(row["reporting_unit"], row["division"])

    # Add all divisions to boards
    for _, row in df[["division", "board"]].drop_duplicates().iterrows():
        hierarchy.add_division(row["division"], row["board"])

    # Add boards to hospital if hospital_id provided
    if hospital_id:
        unique_boards = df["board"].dropna().unique()
        for board in unique_boards:
            hierarchy.add_board(board, hospital_id)

    return hierarchy


def create_hierarchical_predictor(
    specs_df: pd.DataFrame,
    hospital_id: str,
    epsilon: float = 1e-7,
) -> HierarchicalPredictor:
    """Create a HierarchicalPredictor from a hospital structure DataFrame.

    This convenience function creates a fully configured HierarchicalPredictor
    by extracting the hospital organizational structure from a DataFrame and
    setting up the prediction engine with the specified truncation threshold.

    Parameters
    ----------
    specs_df : pandas.DataFrame
        DataFrame containing hospital organizational structure. Must include
        at least these columns:
        - 'board': Board identifier
        - 'division': Division identifier
        - 'reporting_unit': Reporting unit identifier
        - 'sub_specialty': Subspecialty identifier
        Additional columns are ignored.
    hospital_id : str
        Hospital identifier to link all boards to a single hospital
    epsilon : float, default=1e-7
        Truncation threshold for probability distribution tails during
        convolution operations. Smaller values provide higher accuracy but
        require more computation.

    Returns
    -------
    HierarchicalPredictor
        Fully configured predictor with:
        - HospitalHierarchy populated from specs_df
        - DemandPredictor configured with specified epsilon
        - Ready to use for making predictions

    Notes
    -----
    This function is typically used in a workflow like:

    1. Use create_hierarchical_predictor() to set up the predictor with hospital structure
    2. Use build_subspecialty_data() to prepare prediction inputs from patient data
    3. Use predictor.predict_all_levels(subspecialty_data) to compute predictions
       (hospital_id is automatically inferred if there's only one hospital)

    The function automatically handles duplicate relationships and missing
    values in the DataFrame by removing duplicates and dropping rows with
    missing values.
    """
    hierarchy = populate_hierarchy_from_dataframe(specs_df, hospital_id=hospital_id)
    predictor = DemandPredictor(epsilon=epsilon)
    return HierarchicalPredictor(hierarchy, predictor)
