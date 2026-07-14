"""Typed evaluation inputs and builder.

`EvaluationTarget` rows describe what the runner evaluates; `EvaluationInputs`
is the immutable snapshot passed to `run_evaluation`. Build it with
`EvaluationInputsBuilder`.

Submodules should be imported explicitly, for example::

    from patientflow.evaluate.inputs import (
        EvaluationInputs,
        EvaluationInputsBuilder,
        EvaluationTarget,
        eval_split_label,
        standard_ed_targets,
    )

See Also
--------
patientflow.evaluate.runner.run_evaluation
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

from patientflow.evaluate.observations import (
    DEFAULT_ADMISSION_LABEL_COL,
    DEFAULT_DEPARTURE_OUTCOME_COLUMN,
    OBSERVATION_MODES,
)
from patientflow.model_artifacts import TrainedClassifier
from patientflow.predict.types import FlowSelection

EvaluationModeLiteral = Literal[
    "classifier_model_diagnostics",
    "classifier_probability_quality",
    "distribution",
    "arrival_deltas",
    "survival_curve",
    "transition_matrix",
]

EVALUATION_MODES: Tuple[str, ...] = (
    "classifier_model_diagnostics",
    "classifier_probability_quality",
    "distribution",
    "arrival_deltas",
    "survival_curve",
    "transition_matrix",
)

EvalSplitLiteral = Literal["valid", "test"]

EVAL_SPLITS: Tuple[str, ...] = ("valid", "test")


def eval_split_label(split: Optional[str]) -> str:
    """Return a human-readable cohort label for plot titles and reporting.

    Parameters
    ----------
    split : str or None
        Run-level evaluation holdout: ``"valid"``, ``"test"``, or ``None``.

    Returns
    -------
    str
        For example ``"validation set"`` or ``"evaluation cohort"`` when
        ``split`` is unrecognised.
    """
    if split == "test":
        return "test set"
    if split == "valid":
        return "validation set"
    return "evaluation cohort"


def _validate_eval_split(eval_split: str) -> EvalSplitLiteral:
    if eval_split not in EVAL_SPLITS:
        raise ValueError(
            f"Unknown eval_split {eval_split!r}; expected one of {EVAL_SPLITS}"
        )
    return eval_split  # type: ignore[return-value]


def normalize_prediction_dict(
    prediction_dict: Mapping[Tuple[int, int], timedelta],
) -> Dict[Tuple[int, int], timedelta]:
    """Validate and copy a prediction schedule (clock time → horizon).

    Parameters
    ----------
    prediction_dict : mapping
        Keys are ``(hour, minute)``; values are prediction horizons.

    Returns
    -------
    dict
        Normalised copy with integer hour/minute keys.

    Raises
    ------
    ValueError
        If the mapping is empty or keys/values have wrong types.
    """
    if not prediction_dict:
        raise ValueError("prediction_dict must not be empty")
    normalized: Dict[Tuple[int, int], timedelta] = {}
    for key, window in prediction_dict.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"prediction_dict keys must be (hour, minute) tuples; got {key!r}"
            )
        if not isinstance(window, timedelta):
            raise ValueError(
                f"prediction_dict values must be timedelta; got {type(window).__name__}"
            )
        normalized[(int(key[0]), int(key[1]))] = window
    return normalized


def prediction_times_from_dict(
    prediction_dict: Mapping[Tuple[int, int], timedelta],
) -> List[Tuple[int, int]]:
    """Return prediction clock times in ascending order."""
    return sorted(prediction_dict.keys())


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
    flow_type : str
        Logical pathway type (for example `"admissions"` or `"departures"`).
    evaluation_mode : str
        Runner branch; one of the strings in `EVALUATION_MODES`.
    component : str
        Distinguishes chart or scalar families at the same
        `(flow_name, service, prediction_time)`.
    observation_mode : str or None, optional
        One of the strings in `patientflow.evaluate.observations.OBSERVATION_MODES`;
        used when distribution evaluation recomputes observed counts. May be
        omitted when `evaluation_mode` is "transition_matrix".

    Raises
    ------
    ValueError
        If `observation_mode` or `evaluation_mode` is not recognised.
    """

    flow_name: str
    flow_type: str
    evaluation_mode: EvaluationModeLiteral
    component: str
    observation_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evaluation_mode not in EVALUATION_MODES:
            raise ValueError(
                f"Unknown evaluation_mode {self.evaluation_mode!r}; "
                f"expected one of {EVALUATION_MODES}"
            )
        if self.evaluation_mode == "transition_matrix":
            if (
                self.observation_mode is not None
                and self.observation_mode not in OBSERVATION_MODES
            ):
                raise ValueError(
                    f"Unknown observation_mode {self.observation_mode!r}; "
                    f"expected one of {OBSERVATION_MODES}"
                )
            return
        if self.observation_mode is None:
            raise ValueError(
                f"observation_mode is required when evaluation_mode is "
                f"{self.evaluation_mode!r}"
            )
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"Unknown observation_mode {self.observation_mode!r}; "
                f"expected one of {OBSERVATION_MODES}"
            )


def standard_ed_targets(
    *,
    include_classifier_diagnostics: bool = True,
    include_classifier_probability_quality: bool = True,
    include_ed_current_distribution: bool = True,
    include_ed_yta_distribution: bool = False,
    include_ed_yta_arrival_deltas: bool = True,
    ed_current_observation_mode: str = "admitted_at_some_point",
    ed_yta_observation_mode: str = "arrived_in_window",
) -> List[EvaluationTarget]:
    """Return the common ED evaluation target list used in notebook 4d.

    Parameters
    ----------
    include_classifier_diagnostics : bool, optional
        Add model-level classifier diagnostics (SHAP, headline metrics).
    include_classifier_probability_quality : bool, optional
        Add discrimination, MADCAP, and calibration on the full visit frame.
    include_ed_current_distribution : bool, optional
        Add ED-current bed-demand distribution evaluation.
    include_ed_yta_distribution : bool, optional
        Add ED yet-to-arrive bed-demand distribution evaluation (requires
        meaningful ward-admission timestamps for most sites).
    include_ed_yta_arrival_deltas : bool, optional
        Add arrival-rate delta plots per service.
    ed_current_observation_mode : str, optional
        Observation strategy for ED-current distribution targets.
    ed_yta_observation_mode : str, optional
        Observation strategy for yet-to-arrive distribution targets.

    Returns
    -------
    list of EvaluationTarget
        Targets whose ``flow_name`` values match the usual ``add_*`` registrations
        in notebook 4d.
    """
    targets: List[EvaluationTarget] = []
    if include_classifier_diagnostics:
        targets.append(
            EvaluationTarget(
                flow_name="ed_admissions_cls",
                flow_type="admissions",
                evaluation_mode="classifier_model_diagnostics",
                component="classifier_model_diagnostics",
                observation_mode=ed_current_observation_mode,
            )
        )
    if include_classifier_probability_quality:
        targets.append(
            EvaluationTarget(
                flow_name="ed_admissions_cls",
                flow_type="admissions",
                evaluation_mode="classifier_probability_quality",
                component="classifier_discrimination_madcap_calibration",
                observation_mode=ed_current_observation_mode,
            )
        )
    if include_ed_current_distribution:
        targets.append(
            EvaluationTarget(
                flow_name="ed_current_beds",
                flow_type="admissions",
                evaluation_mode="distribution",
                component="bed_demand_ed_current",
                observation_mode=ed_current_observation_mode,
            )
        )
    if include_ed_yta_distribution:
        targets.append(
            EvaluationTarget(
                flow_name="ed_yta_beds",
                flow_type="admissions",
                evaluation_mode="distribution",
                component="bed_demand_ed_yta",
                observation_mode=ed_yta_observation_mode,
            )
        )
    if include_ed_yta_arrival_deltas:
        targets.append(
            EvaluationTarget(
                flow_name="ed_yta_arrival_rates",
                flow_type="admissions",
                evaluation_mode="arrival_deltas",
                component="arrival_delta",
                observation_mode=ed_yta_observation_mode,
            )
        )
    return targets


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
    prediction_dict : dict
        Maps each ``(hour, minute)`` clock time to its prediction horizon.
    prediction_times : list of tuple of int
        Sorted keys of ``prediction_dict`` (ascending by hour, then minute).
    evaluation_targets : list of EvaluationTarget
        Targets the runner dispatches over.
    classifier_by_flow : dict
        Nested `flow_name` → `{"trained_models", "visits_df", "label_col",
        "model_name"}` (``model_name`` is the prefix for :func:`get_model_key`).
    distribution_by_flow : dict
        Nested `flow_name` → distribution block (`prob_dist_by_service`,
        `model_name`, etc.).
    arrival_by_flow : dict
        Nested `flow_name` → arrival-delta block (dataframes, snapshot dates,
        predictors, optional filter keys).
    survival : dict or None
        When set, keys include `train_df`, `test_df`, column names, `labels`.
    transition_matrix_by_flow : dict
        `flow_name` → block with `estimator`, `departure_events`, cohort and
        column metadata for transition-matrix evaluation.
    observation_contexts : dict
        `flow_name` → `service` → visit frames for observation counting.
    distribution_benchmark_cohorts : dict
        Optional ``"admissions"`` / ``"departures"`` cohorts for global p̄
        (see `add_distribution_benchmark_cohort`).
    distribution_benchmark_pmfs : dict
        Optional ``flow_name`` → ``benchmark_kind`` → specialty PMF dicts for
        alternative-distribution benchmarks (see
        `add_distribution_benchmark_from_service_dict`).
    eval_split : str
        Holdout assessed by this run: ``"valid"`` (default) or ``"test"``.
        Drives plot cohort labels; visit frames and snapshot dates must match.
    """

    flow_selection: FlowSelection
    prediction_dict: Dict[Tuple[int, int], timedelta]
    evaluation_targets: List[EvaluationTarget]
    prediction_times: List[Tuple[int, int]] = field(default_factory=list)
    eval_split: EvalSplitLiteral = "valid"
    classifier_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    distribution_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arrival_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    survival: Optional[Dict[str, Any]] = None
    transition_matrix_by_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    observation_contexts: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )
    distribution_benchmark_cohorts: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    distribution_benchmark_pmfs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction_times:
            self.prediction_times = prediction_times_from_dict(self.prediction_dict)


class EvaluationInputsBuilder:
    """Construct `EvaluationInputs` with a fluent `add_*` API.

    `flow_selection`, `prediction_dict`, and `eval_split` must be set
    (via the constructor or setters) before any `add_*` method is called.
    Register visit frames and snapshot dates for the same holdout as
    ``eval_split``.

    Parameters
    ----------
    flow_selection : FlowSelection, optional
        Scenario for the run; may be set later via `set_flow_selection`.
    prediction_dict : mapping, optional
        Maps each ``(hour, minute)`` to a ``timedelta`` horizon (uclhflow
        ``prediction.prediction_dict`` shape). May be set via `set_prediction_dict`.
    eval_split : str, optional
        Holdout for this run: ``"valid"`` (default) or ``"test"``. May be set
        later via `set_eval_split`.
    """

    def __init__(
        self,
        flow_selection: Optional[FlowSelection] = None,
        prediction_dict: Optional[Mapping[Tuple[int, int], timedelta]] = None,
        *,
        eval_split: EvalSplitLiteral = "valid",
    ) -> None:
        self._flow_selection: Optional[FlowSelection] = flow_selection
        self._prediction_dict: Optional[Dict[Tuple[int, int], timedelta]] = (
            normalize_prediction_dict(prediction_dict) if prediction_dict else None
        )
        self._eval_split: EvalSplitLiteral = _validate_eval_split(eval_split)
        self._targets: List[EvaluationTarget] = []
        self._classifier_by_flow: Dict[str, Dict[str, Any]] = {}
        self._distribution_by_flow: Dict[str, Dict[str, Any]] = {}
        self._arrival_by_flow: Dict[str, Dict[str, Any]] = {}
        self._survival: Optional[Dict[str, Any]] = None
        self._transition_matrix_by_flow: Dict[str, Dict[str, Any]] = {}
        self._observation_contexts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._distribution_benchmark_cohorts: Dict[str, Dict[str, Any]] = {}
        self._distribution_benchmark_pmfs: Dict[str, Dict[str, Any]] = {}

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

    def set_prediction_dict(
        self,
        prediction_dict: Mapping[Tuple[int, int], timedelta],
    ) -> EvaluationInputsBuilder:
        """Set the run-level prediction schedule (time → horizon).

        Parameters
        ----------
        prediction_dict : mapping
            Each key is ``(hour, minute)``; each value is the horizon for that
            clock time.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.
        """
        self._prediction_dict = normalize_prediction_dict(prediction_dict)
        return self

    def set_eval_split(self, eval_split: EvalSplitLiteral) -> EvaluationInputsBuilder:
        """Set which temporal holdout this evaluation run assesses.

        Parameters
        ----------
        eval_split : str
            ``"valid"`` or ``"test"``. Register visit frames and snapshot dates
            for the same cohort when calling `add_classifier`, `add_arrival_deltas`,
            and distribution observation helpers.

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If ``eval_split`` is not ``"valid"`` or ``"test"``.
        """
        self._eval_split = _validate_eval_split(eval_split)
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
        if not self._prediction_dict:
            raise ValueError("prediction_dict must be set before adding inputs.")

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
            If `flow_selection` or `prediction_dict` has not been set.
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
            If `flow_selection` or `prediction_dict` has not been set.
        """
        self._require_basics()
        block = self._distribution_by_flow.setdefault(
            flow_name,
            {
                "prob_dist_by_service": {},
                "model_name": model_name,
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
        *,
        ed_visits_by_service: Optional[Mapping[str, pd.DataFrame]] = None,
        inpatient_arrivals_by_service: Optional[Mapping[str, pd.DataFrame]] = None,
        inpatient_visits_by_service: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> EvaluationInputsBuilder:
        """Attach per-service observation frames for distribution evaluation.

        Observation horizons come from the builder's ``prediction_dict`` (per
        clock time). Before plotting, :func:`evaluate_distribution` recomputes
        ``agg_observed`` via :func:`count_observed` for each leaf.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        ed_visits_by_service : mapping, optional
            `service` → ED snapshot dataframe (`ed_visits` context key).
        inpatient_arrivals_by_service : mapping, optional
            `service` → inpatient arrival-time rows (`inpatient_arrivals` key).
        inpatient_visits_by_service : mapping, optional
            `service` → inpatient snapshot dataframe (`inpatient_visits` key).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_dict` has not been set, or if
            no observation mapping is provided.
        """
        if not any(
            (
                ed_visits_by_service,
                inpatient_arrivals_by_service,
                inpatient_visits_by_service,
            )
        ):
            raise ValueError(
                "add_distribution_observations requires at least one of "
                "ed_visits_by_service, inpatient_arrivals_by_service, or "
                "inpatient_visits_by_service"
            )
        self._require_basics()
        self._distribution_by_flow.setdefault(
            flow_name,
            {
                "prob_dist_by_service": {},
                "model_name": "admissions",
            },
        )
        obs_ctx = self._observation_contexts.setdefault(flow_name, {})

        def _register(
            mapping: Optional[Mapping[str, pd.DataFrame]], frame_key: str
        ) -> None:
            if not mapping:
                return
            for svc, df in mapping.items():
                ctx = obs_ctx.setdefault(str(svc), {})
                ctx[frame_key] = df

        _register(ed_visits_by_service, "ed_visits")
        _register(inpatient_arrivals_by_service, "inpatient_arrivals")
        _register(inpatient_visits_by_service, "inpatient_visits")
        return self

    def add_distribution_benchmark_cohort(
        self,
        *,
        admissions_ed_visits: Optional[pd.DataFrame] = None,
        admissions_label_col: str = DEFAULT_ADMISSION_LABEL_COL,
        departures_inpatient_visits: Optional[pd.DataFrame] = None,
        departures_label_col: str = DEFAULT_DEPARTURE_OUTCOME_COLUMN,
    ) -> EvaluationInputsBuilder:
        """Register eval-split cohorts for global binomial-benchmark p̄.

        The same label columns are used when distribution evaluation recomputes
        ``agg_observed`` (via :func:`count_observed`) and for benchmark p̄.

        Parameters
        ----------
        admissions_ed_visits : pandas.DataFrame, optional
            Full eval-split ED visits with *admissions_label_col* and
            ``prediction_time``. Used when ``observation_mode`` is
            ``admitted_at_some_point`` or ``admitted_in_window``.
        admissions_label_col : str, optional
            Boolean admission label on *admissions_ed_visits*. Default
            ``"is_admitted"``.
        departures_inpatient_visits : pandas.DataFrame, optional
            Full eval-split inpatient snapshots with *departures_label_col* and
            ``prediction_time``. Used when ``observation_mode`` is
            ``departed_in_window``.
        departures_label_col : str, optional
            Boolean departure label on *departures_inpatient_visits*. Default
            ``"left_subspecialty_in_window"``.

        Returns
        -------
        EvaluationInputsBuilder
            ``self`` for method chaining.
        """
        if admissions_ed_visits is not None:
            self._distribution_benchmark_cohorts["admissions"] = {
                "visits_df": admissions_ed_visits,
                "label_col": admissions_label_col,
            }
        if departures_inpatient_visits is not None:
            self._distribution_benchmark_cohorts["departures"] = {
                "visits_df": departures_inpatient_visits,
                "label_col": departures_label_col,
            }
        return self

    def add_distribution_benchmark_from_service_dict(
        self,
        flow_name: str,
        prob_dist_by_service: Mapping[str, Any],
        *,
        benchmark_kind: str = "specialty_proportions",
    ) -> EvaluationInputsBuilder:
        """Register alternative PMFs for distribution benchmark comparison.

        Use this for admission-scoped baselines that require a full re-prediction
        pass (for example average specialty proportions instead of a sequence
        predictor). The handler emits ``rpit_cvm_{benchmark_kind}_*`` scalars and
        ``rpit_cvm_{benchmark_kind}_w2_reduction`` when at least two snapshots
        have observations.

        Parameters
        ----------
        flow_name : str
            Must match the ``flow_name`` on the primary distribution target and
            ``add_distributions_from_service_dict``.
        prob_dist_by_service : mapping
            Same nested layout as the primary PMF dict:
            ``service`` → ``model_key`` → ``snapshot`` → leaf.
        benchmark_kind : str, optional
            Short label used as the scalar key prefix. Default
            ``"specialty_proportions"``.

        Returns
        -------
        EvaluationInputsBuilder
            ``self`` for method chaining.
        """
        self._require_basics()
        per_flow = self._distribution_benchmark_pmfs.setdefault(flow_name, {})
        per_flow[str(benchmark_kind)] = dict(prob_dist_by_service)
        return self

    def add_arrival_deltas(
        self,
        flow_name: str,
        arrivals_by_service: Mapping[str, pd.DataFrame],
        snapshot_dates: Sequence[date],
        *,
        predictors_by_service: Optional[Mapping[str, Any]] = None,
        filter_keys_by_service: Optional[Mapping[str, Optional[str]]] = None,
        strict_prediction_date_by_service: Optional[Mapping[str, bool]] = None,
        yta_time_interval: timedelta = timedelta(minutes=15),
    ) -> EvaluationInputsBuilder:
        """Register arrival data and optional predictors for delta plots.

        Horizons for each prediction clock come from the builder's
        ``prediction_dict``.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for targets that use this block.
        arrivals_by_service : mapping
            `service` → arrivals dataframe (must include `arrival_datetime` for
            inactive-service detection).
        snapshot_dates : sequence of datetime.date
            Dates passed to `patientflow.viz.observed_against_expected.plot_arrival_deltas`.
        predictors_by_service : mapping, optional
            Fitted incoming-admission predictors keyed by service (optional).
        filter_keys_by_service : mapping, optional
            `arrival_rate_model.weights` keys when models expose multiple profiles.
        strict_prediction_date_by_service : mapping, optional
            Per-service strict weekday flag for arrival-rate models.
        yta_time_interval : datetime.timedelta, optional
            Grid spacing; must match `arrival_rate_model.yta_time_interval` when a
            model is supplied to `plot_arrival_deltas` (default 15 minutes).

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Raises
        ------
        ValueError
            If `flow_selection` or `prediction_dict` has not been set.
        """
        self._require_basics()
        self._arrival_by_flow[flow_name] = {
            "arrivals_by_service": {str(k): v for k, v in arrivals_by_service.items()},
            "snapshot_dates": list(snapshot_dates),
            "predictors_by_service": dict(predictors_by_service or {}),
            "filter_keys_by_service": dict(filter_keys_by_service or {}),
            "strict_prediction_date_by_service": dict(
                strict_prediction_date_by_service or {}
            ),
            "yta_time_interval": yta_time_interval,
        }
        return self

    def add_transition_matrix(
        self,
        flow_name: str,
        estimator: Any,
        departure_events: pd.DataFrame,
        *,
        cohort: Optional[str] = None,
        source_col: Optional[str] = None,
        destination_col: Optional[str] = None,
        discharge_label: str = "Discharge",
        n_simulations: int = 10_000,
        seed: Optional[int] = None,
        model_name: str = "transfers",
    ) -> EvaluationInputsBuilder:
        """Register a fitted transfer estimator and departure events for evaluation.

        Pairs a fitted `TransferProbabilityEstimator` with the observed departure
        events the handler will score. Expected and observed destination counts,
        per-patient routing vectors, and the Pearson test are derived inside
        `evaluate_transition_matrix`; this method only registers inputs.

        Parameters
        ----------
        flow_name : str
            Must match `EvaluationTarget.flow_name` for transition-matrix targets.
        estimator : TransferProbabilityEstimator
            Fitted model with subgroup routing tables.
        departure_events : pandas.DataFrame
            One row per observed departure in the evaluation window.
        cohort : str or None, optional
            Cohort to evaluate; must be one of the estimator's cohorts when
            `estimator.cohort_col` is set.
        source_col : str or None, optional
            Source subspecialty column (default `estimator.source_col`).
        destination_col : str or None, optional
            Destination column (default `estimator.destination_col`).
        discharge_label : str, optional
            Label for discharge in the transition matrix (default "Discharge").
        n_simulations : int, optional
            Monte Carlo draws per source (default 10_000).
        seed : int or None, optional
            Slice-level RNG seed; per-source offsets are derived deterministically.
        model_name : str, optional
            Base model name stored on scalar rows (default "transfers").

        Returns
        -------
        EvaluationInputsBuilder
            `self` for method chaining.

        Notes
        -----
        The caller must pre-scope `departure_events` to the evaluation window;
        the builder does not apply a date filter. When `estimator.cohort_col` is
        set, the handler filters events to the registered `cohort` at evaluation
        time.

        `departure_events` must include the source and destination columns plus
        the columns required by `estimator.subgroup_functions` (typically age
        and sex) so each row can be routed like production. NaN/None in the
        destination column denotes discharge.

        Register one block per `(flow_name, cohort)` slice. When `cohort` is
        set, stored scalar rows use `model_name="{base}_{cohort}"` (for example
        `transfers_elective`) so multiple cohort slices in one run stay distinct.
        """
        self._require_basics()
        resolved_source_col = source_col or estimator.source_col
        resolved_destination_col = destination_col or estimator.destination_col
        stored_model_name = (
            f"{model_name}_{cohort}" if cohort is not None else model_name
        )
        self._transition_matrix_by_flow[flow_name] = {
            "estimator": estimator,
            "departure_events": departure_events,
            "cohort": cohort,
            "source_col": resolved_source_col,
            "destination_col": resolved_destination_col,
            "discharge_label": discharge_label,
            "n_simulations": n_simulations,
            "seed": seed,
            "model_name": stored_model_name,
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
            If `flow_selection` or `prediction_dict` has not been set.
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
            If `flow_selection` or `prediction_dict` is missing.
        """
        if self._flow_selection is None:
            raise ValueError("flow_selection is required to build EvaluationInputs.")
        if not self._prediction_dict:
            raise ValueError("prediction_dict is required to build EvaluationInputs.")
        if not self._targets:
            self._targets = []
        for target in self._targets:
            if target.evaluation_mode == "transition_matrix":
                if target.flow_name not in self._transition_matrix_by_flow:
                    raise ValueError(
                        f"transition_matrix target for flow {target.flow_name!r} "
                        "requires add_transition_matrix() on the builder."
                    )
        prediction_dict = dict(self._prediction_dict)
        return EvaluationInputs(
            flow_selection=self._flow_selection,
            prediction_dict=prediction_dict,
            prediction_times=prediction_times_from_dict(prediction_dict),
            evaluation_targets=list(self._targets),
            eval_split=self._eval_split,
            classifier_by_flow=dict(self._classifier_by_flow),
            distribution_by_flow=dict(self._distribution_by_flow),
            arrival_by_flow=dict(self._arrival_by_flow),
            survival=self._survival,
            transition_matrix_by_flow=dict(self._transition_matrix_by_flow),
            observation_contexts={
                fn: {svc: dict(ctx) for svc, ctx in per.items()}
                for fn, per in self._observation_contexts.items()
            },
            distribution_benchmark_cohorts=dict(self._distribution_benchmark_cohorts),
            distribution_benchmark_pmfs={
                fn: {kind: dict(pmfs) for kind, pmfs in per.items()}
                for fn, per in self._distribution_benchmark_pmfs.items()
            },
        )
