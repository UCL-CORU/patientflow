"""Type definitions for hierarchical demand prediction.

This module defines the core data structures used for hierarchical predictions:
DemandPrediction, PredictionBundle, and FlowSelection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

# Constants for magic numbers
DEFAULT_PERCENTILES = [50, 75, 90, 95, 99]
DEFAULT_PRECISION = 3
DEFAULT_MAX_PROBS = 10


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

    def to_pretty(
        self, max_probs: int = DEFAULT_MAX_PROBS, precision: int = DEFAULT_PRECISION
    ) -> str:
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

    This defines booleans to include each flow family and a
    single cohort selector ("all", "elective", "emergency").

    Families
    --------
    Inflows families:
    - include_ed_current: current ED cohort admissions (emergency only)
    - include_ed_yta: yet-to-arrive ED admissions (emergency only)
    - include_non_ed_yta: yet-to-arrive non-ED emergency admissions (emergency only)
    - include_elective_yta: yet-to-arrive elective admissions (elective only)
    - include_transfers_in: internal transfers into subspecialty (both cohorts)

    Outflows families:
    - include_departures: inpatient departures (both cohorts)

    Cohort
    ------
    - cohort: one of {"all", "elective", "emergency"}
    """

    include_ed_current: bool = True
    include_ed_yta: bool = True
    include_non_ed_yta: bool = True
    include_elective_yta: bool = True
    include_transfers_in: bool = True
    include_departures: bool = True
    cohort: str = "all"

    @classmethod
    def default(cls) -> "FlowSelection":
        return cls()

    @classmethod
    def incoming_only(cls) -> "FlowSelection":
        return cls(include_departures=False)

    @classmethod
    def outgoing_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=False,
            include_ed_yta=False,
            include_non_ed_yta=False,
            include_elective_yta=False,
            include_transfers_in=False,
            include_departures=True,
        )

    @classmethod
    def elective_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=False,
            include_ed_yta=False,
            include_non_ed_yta=False,
            include_elective_yta=True,
            include_transfers_in=True,  # Will be filtered by cohort
            include_departures=True,  # Will be filtered by cohort
            cohort="elective",
        )

    @classmethod
    def emergency_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=True,
            include_ed_yta=True,
            include_non_ed_yta=True,
            include_elective_yta=False,
            include_transfers_in=True,  # Will be filtered by cohort
            include_departures=True,  # Will be filtered by cohort
            cohort="emergency",
        )

    @classmethod
    def custom(
        cls,
        *,
        include_ed_current: bool = True,
        include_ed_yta: bool = True,
        include_non_ed_yta: bool = True,
        include_elective_yta: bool = True,
        include_transfers_in: bool = True,
        include_departures: bool = True,
        cohort: str = "all",
    ) -> "FlowSelection":
        return cls(
            include_ed_current=include_ed_current,
            include_ed_yta=include_ed_yta,
            include_non_ed_yta=include_non_ed_yta,
            include_elective_yta=include_elective_yta,
            include_transfers_in=include_transfers_in,
            include_departures=include_departures,
            cohort=cohort,
        )

    def validate(self) -> None:
        """Validate the flow selection configuration.

        Raises
        ------
        ValueError
            If cohort is not one of the expected values
        """
        if self.cohort not in {"all", "elective", "emergency"}:
            raise ValueError(
                f"Invalid cohort '{self.cohort}'. Must be one of: 'all', 'elective', 'emergency'"
            )


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
        Selection specifying which flow families and cohort were included in this
        prediction. Tracks which inflows and outflows were aggregated into the
        arrivals and departures distributions.

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
            - flows_included: Number of inflow families and outflow families included
        """

        def format_pmf(
            pred: DemandPrediction, max_display: int = DEFAULT_MAX_PROBS
        ) -> str:
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
            "flows_included": (
                f"selection cohort={self.flow_selection.cohort} "
                f"inflows(ed_current={self.flow_selection.include_ed_current}, "
                f"ed_yta={self.flow_selection.include_ed_yta}, "
                f"non_ed_yta={self.flow_selection.include_non_ed_yta}, "
                f"elective_yta={self.flow_selection.include_elective_yta}, "
                f"transfers_in={self.flow_selection.include_transfers_in}) "
                f"outflows(departures={self.flow_selection.include_departures})"
            ),
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
