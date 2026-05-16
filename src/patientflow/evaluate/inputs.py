"""Typed evaluation inputs and builder.

`EvaluationTarget` rows describe what the runner evaluates; `EvaluationInputs`
is the immutable snapshot passed to `run_evaluation`. Build it with
`EvaluationInputsBuilder`.

Submodules should be imported explicitly, for example::

    from patientflow.evaluate.inputs import (
        EvaluationInputs,
        EvaluationInputsBuilder,
        EvaluationTarget,
    )

See Also
--------
patientflow.evaluate.runner.run_evaluation
patientflow.evaluate.targets.get_default_evaluation_targets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from patientflow.evaluate.observations import OBSERVATION_MODES
from patientflow.model_artifacts import TrainedClassifier
from patientflow.predict.types import FlowSelection

EvaluationModeLiteral = Literal[
    "classifier_model_diagnostics",
    "classifier_probability_quality",
    "distribution",
    "arrival_deltas",
    "survival_curve",
]

EVALUATION_MODES: Tuple[str, ...] = (
    "classifier_model_diagnostics",
    "classifier_probability_quality",
    "distribution",
    "arrival_deltas",
    "survival_curve",
)


@dataclass(frozen=True)
class EvaluationTarget:
    """Describe one evaluation slice (one scalar row family in `scalars.json`).

    `flow_name` links the target to data registered on the builder (for
    example `"ed"`). It is not a `FlowSelection` from `patientflow.predict.types`;
    scenario selection lives only on `EvaluationInputs`.

    Parameters
    ----------
    flow_name : str
        Key passed to `add_*` methods on `EvaluationInputsBuilder`.
    name : str
        Stable identifier for this target within the flow.
    flow_type : str
        Logical pathway type (for example `"admissions"` or `"departures"`).
    evaluation_mode : str
        Runner branch; one of the strings in `EVALUATION_MODES`.
    component : str
        Distinguishes chart or scalar families at the same
        `(flow_name, service, prediction_time)`.
    observation_mode : str
        One of the strings in `patientflow.evaluate.observations.OBSERVATION_MODES`;
        used when distribution evaluation recomputes observed counts.

    Raises
    ------
    ValueError
        If `observation_mode` or `evaluation_mode` is not recognised.
    """

    flow_name: str
    name: str
    flow_type: str
    evaluation_mode: EvaluationModeLiteral
    component: str
    observation_mode: str

    def __post_init__(self) -> None:
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"Unknown observation_mode {self.observation_mode!r}; "
                f"expected one of {OBSERVATION_MODES}"
            )
        if self.evaluation_mode not in EVALUATION_MODES:
            raise ValueError(
                f"Unknown evaluation_mode {self.evaluation_mode!r}; "
                f"expected one of {EVALUATION_MODES}"
            )


def _normalize_trained_models(
    trained_models: Union[
        Sequence[TrainedClassifier],
        Mapping[Tuple[int, int], TrainedClassifier],
    ],
) -> List[TrainedClassifier]:
    if isinstance(trained_models, Mapping):
        return [trained_models[k] for k in sorted(trained_models.keys())]
    return list(trained_models)


@dataclass
class EvaluationInputs:
    """Immutable bundle of everything needed for one evaluation run.

    Consumed by `patientflow.evaluate.runner.run_evaluation`.

    Attributes
    ----------
    flow_selection : FlowSelection
        Single scenario for the run (required once per evaluation).
    prediction_times : list of tuple of int
        Global `(hour, minute)` list shared by classifiers, distributions, and
        arrival-delta targets.
    evaluation_targets : list of EvaluationTarget
        Targets the runner dispatches over.
    classifier_by_flow : dict
        Nested `flow_name` → `{"trained_models", "visits_df", "label_col",
        "model_name"}` (``model_name`` is the prefix for :func:`get_model_key`).
    distribution_by_flow : dict
        Nested `flow_name` → distribution block (`prob_dist_by_service`,
        `model_name`, `prediction_window`, etc.).
    arrival_by_flow : dict
        Nested `flow_name` → arrival-delta block (dataframes, snapshot dates,
        predictors, optional filter keys).
    survival : dict or None
        When set, keys include `train_df`, `test_df`, column names, `labels`.
    observation_contexts : dict
        `flow_name` → `service` → visit frames for observation counting.
    """

    flow_selection: FlowSelection
    prediction_times: List[Tuple[int, int]]
    evaluation_targets: List[EvaluationTarget]
    classifier_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    distribution_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arrival_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    survival: Optional[Dict[str, Any]] = None
    observation_contexts: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )


class EvaluationInputsBuilder:
    """Construct `EvaluationInputs` with a fluent `add_*` API.

    `flow_selection` and `prediction_times` must be set (via the
    constructor or setters) before any `add_*` method is called.

    Parameters
    ----------
    flow_selection : FlowSelection, optional
        Scenario for the run; may be set later via `set_flow_selection`.
    prediction_times : list of tuple of int, optional
        Global prediction clock times; may be set via `set_prediction_times`.
    """

    def __init__(
        self,
        flow_selection: Optional[FlowSelection] = None,
        prediction_times: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Create builder state; see class docstring for required fields before `add_*`."""
        self._flow_selection: Optional[FlowSelection] = flow_selection
        self._prediction_times: Optional[List[Tuple[int, int]]] = prediction_times
        self._targets: List[EvaluationTarget] = []
        self._classifier_by_flow: Dict[str, Dict[str, Any]] = {}
        self._distribution_by_flow: Dict[str, Dict[str, Any]] = {}
        self._arrival_by_flow: Dict[str, Dict[str, Any]] = {}
        self._survival: Optional[Dict[str, Any]] = None
        self._observation_contexts: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def set_flow_selection(
        self, flow_selection: FlowSelection
    ) -> EvaluationInputsBuilder:
        """Set `flow_selection` for the run.

        Parameters
        ----------
        flow_selection : FlowSelection
            Validated selection passed through to `EvaluationInputs`.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.
        """
        self._flow_selection = flow_selection
        return self

    def set_prediction_times(
        self, prediction_times: List[Tuple[int, int]]
    ) -> EvaluationInputsBuilder:
        """Set the ordered global prediction times.

        Parameters
        ----------
        prediction_times : list of tuple of int
            Each tuple is `(hour, minute)`.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.
        """
        self._prediction_times = list(prediction_times)
        return self

    def with_evaluation_targets(
        self, targets: Sequence[EvaluationTarget]
    ) -> EvaluationInputsBuilder:
        """Replace the list of evaluation targets.

        Parameters
        ----------
        targets : sequence of EvaluationTarget
            Targets processed in order by the runner.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.
        """
        self._targets = list(targets)
        return self

    def _require_basics(self) -> None:
        if self._flow_selection is None:
            raise ValueError("flow_selection must be set before adding inputs.")
        if not self._prediction_times:
            raise ValueError("prediction_times must be set before adding inputs.")

    def add_classifier(
        self,
        flow_name: str,
        trained_models: Union[
            Sequence[TrainedClassifier],
            Mapping[Tuple[int, int], TrainedClassifier],
        ],
        visits_df: pd.DataFrame,
        label_col: str,
        *,
        model_name: str = "admissions",
    ) -> EvaluationInputsBuilder:
        """Register classifiers and visit data for one flow.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        trained_models : sequence or mapping of TrainedClassifier
            Models sorted by prediction time if a sequence; mapping keys are
            `(hour, minute)` tuples.
        visits_df : pandas.DataFrame
            Visit-level frame for MADCAP / calibration / SHAP.
        label_col : str
            Binary outcome column on `visits_df`.
        model_name : str, optional
            Base model name passed to :func:`patientflow.load.get_model_key` when
            recording scalar rows (default ``"admissions"``).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` has not been set.
        """
        self._require_basics()
        models = _normalize_trained_models(trained_models)
        self._classifier_by_flow[flow_name] = {
            "trained_models": models,
            "visits_df": visits_df,
            "label_col": label_col,
            "model_name": model_name,
        }
        return self

    def add_distributions_from_service_dict(
        self,
        flow_name: str,
        prob_dist_by_service: Mapping[str, Any],
        model_name: str,
    ) -> EvaluationInputsBuilder:
        """Attach per-service probability dictionaries for distribution evaluation.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        prob_dist_by_service : mapping
            `service` → `date` → payload with `agg_predicted` and typically
            `agg_observed` (see `patientflow.viz.epudd.plot_epudd`).
        model_name : str
            Base model name used with `patientflow.load.get_model_key`.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` has not been set.
        """
        self._require_basics()
        block = self._distribution_by_flow.setdefault(
            flow_name,
            {
                "prob_dist_by_service": {},
                "model_name": model_name,
                "prediction_window": None,
            },
        )
        block["model_name"] = model_name
        cast_prob: MutableMapping[str, Any] = block["prob_dist_by_service"]
        for svc, dist in prob_dist_by_service.items():
            cast_prob[str(svc)] = dist
        return self

    def add_distribution_observations(
        self,
        flow_name: str,
        observations_by_service: Mapping[str, pd.DataFrame],
        prediction_window: timedelta,
        *,
        ed_visits_key: str = "ed_visits",
        inpatient_visits_key: str = "inpatient_visits",
        visits_key: str = "visits",
    ) -> EvaluationInputsBuilder:
        """Attach per-service visit frames for future `count_observed` wiring.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        observations_by_service : mapping
            `service` → dataframe stored under `ed_visits_key`,
            `inpatient_visits_key`, and `visits_key` in the observation context.
        prediction_window : datetime.timedelta
            Horizon stored on the distribution block.
        ed_visits_key : str, optional
            Context key for ED frames (default `"ed_visits"`).
        inpatient_visits_key : str, optional
            Context key for inpatient frames (default `"inpatient_visits"`).
        visits_key : str, optional
            Context key for generic visits (default `"visits"`).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` has not been set.

        Notes
        -----
        Handlers may still use caller-supplied `agg_observed` inside
        `prob_dist_by_service` until observation recomputation is implemented.
        """
        self._require_basics()
        block = self._distribution_by_flow.setdefault(
            flow_name,
            {
                "prob_dist_by_service": {},
                "model_name": "admissions",
                "prediction_window": prediction_window,
            },
        )
        block["prediction_window"] = prediction_window
        obs_ctx = self._observation_contexts.setdefault(flow_name, {})
        for svc, df in observations_by_service.items():
            ctx = obs_ctx.setdefault(str(svc), {})
            ctx[ed_visits_key] = df
            ctx[inpatient_visits_key] = df
            ctx[visits_key] = df
        return self

    def add_arrival_deltas(
        self,
        flow_name: str,
        arrivals_by_service: Mapping[str, pd.DataFrame],
        snapshot_dates: Sequence[date],
        prediction_window: timedelta,
        *,
        predictors_by_service: Optional[Mapping[str, Any]] = None,
        filter_keys_by_service: Optional[Mapping[str, Optional[str]]] = None,
        strict_prediction_date_by_service: Optional[Mapping[str, bool]] = None,
        yta_time_interval: timedelta = timedelta(minutes=15),
    ) -> EvaluationInputsBuilder:
        """Register arrival data and optional predictors for delta plots.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        arrivals_by_service : mapping
            `service` → arrivals dataframe (must include `arrival_datetime` for
            inactive-service detection).
        snapshot_dates : sequence of datetime.date
            Dates passed to `patientflow.viz.observed_against_expected.plot_arrival_deltas`.
        prediction_window : datetime.timedelta
            Horizon for delta charts.
        predictors_by_service : mapping, optional
            Fitted incoming-admission predictors keyed by service (optional).
        filter_keys_by_service : mapping, optional
            `predictor.weights` keys when predictors expose multiple profiles.
        strict_prediction_date_by_service : mapping, optional
            Per-service strict weekday flag for predictors.
        yta_time_interval : datetime.timedelta, optional
            Grid spacing; must match `predictor.yta_time_interval` when a
            predictor is supplied (default 15 minutes).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` has not been set.
        """
        self._require_basics()
        self._arrival_by_flow[flow_name] = {
            "arrivals_by_service": {str(k): v for k, v in arrivals_by_service.items()},
            "snapshot_dates": list(snapshot_dates),
            "prediction_window": prediction_window,
            "predictors_by_service": dict(predictors_by_service or {}),
            "filter_keys_by_service": dict(filter_keys_by_service or {}),
            "strict_prediction_date_by_service": dict(
                strict_prediction_date_by_service or {}
            ),
            "yta_time_interval": yta_time_interval,
        }
        return self

    def add_survival_curve(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        start_time_col: str = "arrival_datetime",
        end_time_col: str = "departure_datetime",
        labels: Optional[Tuple[str, str]] = None,
    ) -> EvaluationInputsBuilder:
        """Attach train and test frames for a single global survival comparison.

        Parameters
        ----------
        train_df, test_df : pandas.DataFrame
            Cohort tables passed to
            `patientflow.viz.survival_curve.plot_admission_time_survival_curve`.
        start_time_col : str, optional
            Start-time column (default `"arrival_datetime"`).
        end_time_col : str, optional
            End-time column (default `"departure_datetime"`).
        labels : tuple of str, optional
            Curve labels (default `("train", "test")`).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` has not been set.
        """
        self._require_basics()
        self._survival = {
            "train_df": train_df,
            "test_df": test_df,
            "start_time_col": start_time_col,
            "end_time_col": end_time_col,
            "labels": labels or ("train", "test"),
        }
        return self

    def build(self) -> EvaluationInputs:
        """Materialise `EvaluationInputs`.

        Returns
        -------
        EvaluationInputs
            Immutable inputs for `patientflow.evaluate.runner.run_evaluation`.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_times` is missing.
        """
        if self._flow_selection is None:
            raise ValueError("flow_selection is required to build EvaluationInputs.")
        if not self._prediction_times:
            raise ValueError("prediction_times is required to build EvaluationInputs.")
        if not self._targets:
            self._targets = []
        return EvaluationInputs(
            flow_selection=self._flow_selection,
            prediction_times=list(self._prediction_times),
            evaluation_targets=list(self._targets),
            classifier_by_flow=dict(self._classifier_by_flow),
            distribution_by_flow=dict(self._distribution_by_flow),
            arrival_by_flow=dict(self._arrival_by_flow),
            survival=self._survival,
            observation_contexts={
                fn: {svc: dict(ctx) for svc, ctx in per.items()}
                for fn, per in self._observation_contexts.items()
            },
        )
