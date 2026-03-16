"""Core typed data structures for evaluation workflows.

The classes in this module define the canonical in-memory contracts used by
the typed evaluation APIs:

- snapshot-level predicted-vs-observed distribution records
- classifier evaluation input bundles
- payloads for arrival-delta and survival-curve diagnostics
- target definitions and top-level run inputs

These types are intentionally explicit to keep orchestration, plotting, and
scalar generation logic readable and testable.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from patientflow.predict.types import FlowSelection


@dataclass
class SnapshotResult:
    """Observed count paired with a predicted PMF for one snapshot date.

    Parameters
    ----------
    predicted_pmf
        Probability mass function for the target count at one snapshot date.
        The array is expected to sum to approximately 1.0.
    observed
        Observed count for the same snapshot date.
    offset
        Integer support offset for the PMF. Index ``i`` in ``predicted_pmf``
        corresponds to value ``i + offset``.
    """

    predicted_pmf: np.ndarray
    observed: int
    offset: int = 0


@dataclass
class ClassifierInput:
    """Input payload required for classifier diagnostics.

    Parameters
    ----------
    trained_models
        Collection of trained classifier artifacts for one classifier target,
        typically one model per prediction time.
    visits_df
        Evaluation dataframe used by classifier diagnostics such as MADCAP,
        discrimination, and calibration plots.
    label_col
        Name of the binary outcome column in ``visits_df``.
    """

    trained_models: List[Any]
    visits_df: Optional[pd.DataFrame] = None
    label_col: str = "is_admitted"


@dataclass
class ArrivalDeltaPayload:
    """Input payload for arrival delta plots.

    Parameters
    ----------
    df
        Dataframe of arrivals used to compare observed arrivals against
        historical arrival-rate expectations.
    snapshot_dates
        Snapshot dates to evaluate.
    prediction_window
        Forecast window associated with each prediction time.
    yta_time_interval
        Yet-to-arrive time interval used in cumulative delta calculations.
    """

    df: pd.DataFrame
    snapshot_dates: List[date]
    prediction_window: timedelta
    yta_time_interval: timedelta = field(default_factory=lambda: timedelta(minutes=15))


@dataclass
class SurvivalCurvePayload:
    """Input payload for survival-curve comparison.

    Parameters
    ----------
    train_df
        Training-period dataframe used to estimate the reference survival curve.
    test_df
        Test-period dataframe used for comparison against training behaviour.
    start_time_col
        Column name for event start timestamps.
    end_time_col
        Column name for event end timestamps.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    start_time_col: str = "arrival_datetime"
    end_time_col: str = "departure_datetime"


@dataclass(frozen=True)
class EvaluationTarget:
    """Configuration for one evaluation target.

    Parameters
    ----------
    name
        Stable target name used in output paths and scalar rows.
    flow_type
        Semantic flow category (for example ``"classifier"``, ``"pmf"``,
        ``"poisson"``, or ``"special"``).
    evaluation_mode
        Evaluation strategy for this target. Supported values currently include
        ``"classifier"``, ``"distribution"``, ``"arrival_deltas"``,
        ``"survival_curve"``, and ``"aspirational_skip"``.
    component
        Single evaluated component (for example ``"arrivals"``,
        ``"departures"``, ``"net_flow"``, or ``"classifier"``).
    flow_selection
        Optional flow-selection metadata that documents how the flow is
        assembled upstream.
    """

    name: str
    flow_type: str
    evaluation_mode: str
    component: str
    flow_selection: Optional[FlowSelection] = None

    @property
    def aspirational(self) -> bool:
        """Whether this target is aspirational and should skip diagnostics."""
        return self.evaluation_mode == "aspirational_skip"


FlowInputPayload = Union[
    Dict[date, SnapshotResult],
    ArrivalDeltaPayload,
    SurvivalCurvePayload,
]


@dataclass
class EvaluationInputs:
    """Top-level input container for typed evaluation runs.

    Parameters
    ----------
    prediction_times
        Prediction times to evaluate.
    evaluation_targets
        Mapping from target name to evaluation target configuration.
    classifier_inputs
        Classifier payloads keyed by classifier target name.
    flow_inputs_by_service
        Nested mapping:
        ``service -> flow -> prediction_time -> payload``.
        Payload type depends on target mode:
        distribution snapshots, arrival-delta payloads, or survival payloads.
    """

    prediction_times: List[Tuple[int, int]]
    evaluation_targets: Dict[str, EvaluationTarget]
    classifier_inputs: Dict[str, ClassifierInput]
    flow_inputs_by_service: Dict[str, Dict[str, Dict[Tuple[int, int], FlowInputPayload]]]
