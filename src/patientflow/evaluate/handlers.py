"""Per-mode evaluation handlers (explicit entrypoints, no registry).

Functions here are called from `patientflow.evaluate.runner.run_evaluation`
according to each `EvaluationTarget.evaluation_mode`. They write chart PNGs
under flow-specific subdirectories and append scalar rows to a shared
`ScalarsCollector`.

See Also
--------
patientflow.evaluate.runner.run_evaluation
patientflow.evaluate.inputs.EvaluationInputs
patientflow.evaluate.scalars.ScalarsCollector
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patientflow.evaluate.calibration import (
    MIN_DISTRIBUTION_SNAPSHOTS,
    benchmark_cohort_key_for_observation_mode,
    build_benchmark_prob_dist_dict,
    global_p_bar_by_prediction_time,
    rpit_cvm_calibration_score,
    rpit_cvm_result_to_scalar_fields,
)
from patientflow.evaluate.inputs import (
    EvaluationInputs,
    EvaluationTarget,
    eval_split_label,
    prediction_times_from_dict,
)
from patientflow.evaluate.observations import (
    admission_type_filter_for_distribution_component,
    count_observed,
    count_observed_applies_specialty_filter,
    count_observed_label_kwargs,
    observation_context_frame_key,
)
from patientflow.evaluate.scalars import (
    RELIABILITY_MIN_OBSERVATIONS_TRANSITION,
    SERVICE_SENTINEL_ALL,
    ScalarsCollector,
    classifier_reliable,
    scalar_target_fields,
)
from patientflow.evaluate.goodness_of_fit import (
    derive_seed_offset,
    multinomial_gof_montecarlo,
)
from patientflow.predict.transfers import build_per_patient_probabilities
from patientflow.load import get_model_key
from patientflow.model_artifacts import TrainedClassifier
from patientflow.viz.epudd import plot_epudd
from patientflow.viz.features import plot_features
from patientflow.viz.observed_against_expected import plot_arrival_deltas
from patientflow.viz.calibration import plot_calibration
from patientflow.viz.estimated_probabilities import plot_estimated_probabilities
from patientflow.viz.madcap import plot_madcap, plot_madcap_by_group
from patientflow.viz.survival_curve import plot_admission_time_survival_curve


def _safe_fs_segment(name: str) -> str:
    """Return a filesystem-safe directory or filename token (no path separators)."""
    return str(name).replace("/", "_").replace("\\", "_")


_FLOW_TYPE_LABELS: Dict[str, str] = {
    "admissions": "ED admission classifier",
    "departures": "Inpatient departure classifier",
}

_DISTRIBUTION_FLOW_LABELS: Dict[str, str] = {
    "ed_current_beds": "ED current bed demand",
    "ed_yta_beds": "ED yet-to-arrive bed demand",
}

_DISTRIBUTION_COMPONENT_LABELS: Dict[str, str] = {
    "bed_demand_ed_current": "ED current bed demand",
    "bed_demand_ed_yta": "ED yet-to-arrive bed demand",
    "departures_elective": "Elective departures",
    "departures_emergency": "Emergency departures",
    "departures_all_inpatient": "All inpatient departures",
}

_ARRIVAL_FLOW_LABELS: Dict[str, str] = {
    "ed_yta_arrival_rates": "ED yet-to-arrive arrival deltas",
}


def _flow_label(target: EvaluationTarget) -> str:
    """Return a display label for ``target.flow_type`` or ``target.flow_name``."""
    return _FLOW_TYPE_LABELS.get(
        target.flow_type,
        target.flow_type.replace("_", " ").title(),
    )


def _classifier_diagnostics_suptitle(
    target: EvaluationTarget,
    chart: str,
    *,
    eval_split: Optional[str],
    prediction_time: Optional[Tuple[int, int]] = None,
) -> str:
    """Return a figure suptitle for classifier model-diagnostics charts.

    Parameters
    ----------
    target : EvaluationTarget
        Classifier model-diagnostics target.
    chart : str
        Chart description (for example ``"feature importances"``).
    eval_split : str or None
        Run-level holdout from `EvaluationInputs.eval_split`.
    prediction_time : tuple of int or None, optional
        ``(hour, minute)`` when the chart is for one prediction clock.

    Returns
    -------
    str
        Title including cohort label.
    """
    flow_label = _flow_label(target)
    cohort = eval_split_label(eval_split)
    if prediction_time is None:
        return f"{flow_label}: {chart} ({cohort})"
    hour, minute = prediction_time
    return f"{flow_label}: {chart} at {hour:02d}:{minute:02d} ({cohort})"


def _classifier_quality_suptitle(
    target: EvaluationTarget, chart: str, *, eval_split: Optional[str] = None
) -> str:
    """Return a human-readable figure title for classifier probability-quality plots.

    Parameters
    ----------
    target : EvaluationTarget
        Classifier probability-quality target.
    chart : str
        Short chart name (for example ``"discrimination"``).
    eval_split : str or None, optional
        Run-level holdout from `EvaluationInputs.eval_split`.

    Returns
    -------
    str
        Title including the cohort label.
    """
    flow_label = _flow_label(target)
    cohort = eval_split_label(eval_split)
    return f"{flow_label}: {chart} ({cohort})"


def _arrival_delta_suptitle(
    target: EvaluationTarget,
    service: str,
    prediction_time: Tuple[int, int],
    *,
    eval_split: Optional[str],
) -> str:
    """Return a figure suptitle for cumulative arrival-delta charts.

    Parameters
    ----------
    target : EvaluationTarget
        Arrival-deltas evaluation target.
    service : str
        Hospital service name.
    prediction_time : tuple of int
        ``(hour, minute)`` for the prediction clock.
    eval_split : str or None
        Run-level holdout from `EvaluationInputs.eval_split`.

    Returns
    -------
    str
        Title including service, clock, and cohort label.
    """
    subject = _ARRIVAL_FLOW_LABELS.get(
        target.flow_name,
        target.flow_name.replace("_", " ").title(),
    )
    hour, minute = prediction_time
    cohort = eval_split_label(eval_split)
    return f"{subject}: {service} at {hour:02d}:{minute:02d} ({cohort})"


def _distribution_comparison_suptitle(
    target: EvaluationTarget, service: str, *, eval_split: Optional[str] = None
) -> str:
    """Return a human-readable figure title for distribution comparison charts.

    Parameters
    ----------
    target : EvaluationTarget
        Distribution evaluation target.
    service : str
        Hospital service name.
    eval_split : str or None, optional
        Run-level holdout from `EvaluationInputs.eval_split`.

    Returns
    -------
    str
        Title including service and cohort label.
    """
    subject = _DISTRIBUTION_FLOW_LABELS.get(target.flow_name)
    if subject is None:
        subject = _DISTRIBUTION_COMPONENT_LABELS.get(
            target.component,
            target.flow_name.replace("_", " ").title(),
        )
    cohort = eval_split_label(eval_split)
    return f"{subject}: {service} ({cohort})"


try:
    from patientflow.viz.shap import SHAP_AVAILABLE, plot_shap
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False
    plot_shap = None  # type: ignore[misc, assignment]


def _require_classifier_eval_artifacts(model: TrainedClassifier) -> None:
    """Validate that `model` carries metrics and dataset metadata for evaluation.

    Parameters
    ----------
    model : TrainedClassifier
        Classifier produced by `patientflow.train.classifiers` (or compatible).

    Raises
    ------
    ValueError
        If `selected_eval_metrics` is empty, missing `auroc`, `auprc`, or
        `log_loss`, or if `training_results.training_info['dataset_info']`
        lacks a non-empty `train_valid_test_positive_cases` mapping.
    """
    metrics = model.selected_eval_metrics
    if not metrics:
        raise ValueError(
            "TrainedClassifier.selected_eval_metrics is empty; "
            "evaluation requires trained artefacts from patientflow.train.classifiers."
        )
    for key in ("auroc", "auprc", "log_loss"):
        if key not in metrics:
            raise ValueError(
                f"TrainedClassifier.selected_eval_metrics missing required key {key!r}."
            )
    info = (model.training_results.training_info or {}).get("dataset_info") or {}
    pos = info.get("train_valid_test_positive_cases")
    if not isinstance(pos, dict) or not pos:
        raise ValueError(
            "training_results.training_info['dataset_info']['train_valid_test_positive_cases'] "
            "is required for evaluation (breaking change — retrain or populate metadata)."
        )


def _sort_models(models: Sequence[TrainedClassifier]) -> List[TrainedClassifier]:
    """Return `models` sorted by `prediction_time` (hour, then minute)."""
    return sorted(
        models,
        key=lambda m: m.training_results.prediction_time[0] * 60
        + m.training_results.prediction_time[1],
    )


def _classifier_base_name(block: Mapping[str, Any]) -> str:
    """Return the model name prefix used with :func:`get_model_key`."""
    return str(block.get("model_name") or "admissions")


def _classifier_model_key(model: TrainedClassifier, base_name: str) -> str:
    return get_model_key(base_name, model.training_results.prediction_time)


def _disambiguate_classifier_plot_filename(
    file_name: str,
    prediction_time: Tuple[int, int],
    *,
    multi_clock: bool,
) -> str:
    """Return ``file_name`` or a per-clock variant when several models are plotted."""
    if not multi_clock:
        return file_name
    hour, minutes = prediction_time
    fn = Path(file_name)
    clock = f"{hour:02d}{minutes:02d}"
    return f"{fn.stem}_{clock}{fn.suffix}"


def _classifier_diagnostics_scalar_row(
    target: EvaluationTarget,
    model: TrainedClassifier,
    *,
    base_name: str,
) -> Dict[str, Any]:
    """Build one model-diagnostics scalar row (headline metrics from training)."""
    metrics = model.selected_eval_metrics
    pt = model.training_results.prediction_time
    info = (model.training_results.training_info or {}).get("dataset_info") or {}
    pos_cases = info.get("train_valid_test_positive_cases") or {}
    reliable = classifier_reliable(metrics, pos_cases)
    hour, minute = pt
    return {
        **scalar_target_fields(target),
        "service": SERVICE_SENTINEL_ALL,
        "component": target.component,
        "prediction_time": [hour, minute],
        "model_name": _classifier_model_key(model, base_name),
        "metrics_split": metrics.get("split"),
        "charts_generated": True,
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "log_loss": metrics.get("log_loss"),
        "n_samples": metrics.get("n_samples"),
        "n_positive_cases": metrics.get("n_positive_cases"),
        "reliable": reliable,
    }


def _ensure_distribution_leaf(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise one snapshot payload for `patientflow.viz.epudd.plot_epudd`.

    Parameters
    ----------
    data : mapping
        Must include `agg_predicted` as either a `dict` with `agg_proba` or
        a `pandas.Series` of probabilities indexed by support.

    Returns
    -------
    dict
        Mapping with `agg_predicted` shaped as `{"agg_proba": ndarray}` and
        `agg_observed` preserved or defaulted to `0`.

    Raises
    ------
    TypeError
        If `agg_predicted` is neither a supported dict nor a Series.
    """
    out = dict(data)
    ap = out.get("agg_predicted")
    if isinstance(ap, dict) and "agg_proba" in ap:
        return out
    if isinstance(ap, pd.DataFrame) and "agg_proba" in ap.columns:
        arr = np.asarray(ap["agg_proba"].to_numpy(), dtype=float).flatten()
        return {
            "agg_predicted": {"agg_proba": arr},
            "agg_observed": out.get("agg_observed", 0),
        }
    if isinstance(ap, pd.Series):
        arr = np.asarray(ap.values, dtype=float).flatten()
        return {
            "agg_predicted": {"agg_proba": arr},
            "agg_observed": out.get("agg_observed", 0),
        }
    raise TypeError(
        "Each snapshot value must have agg_predicted as a Series, a DataFrame "
        "with an 'agg_proba' column, or as a dict with key 'agg_proba'."
    )


def _distribution_per_date_is_model_key_indexed(
    per_date: Mapping[Any, Any],
    model_name: str,
    prediction_times: Sequence[Tuple[int, int]],
) -> bool:
    """Return True when *per_date* is ``model_key -> snapshot -> leaf`` (not date-first)."""
    if not per_date:
        return False
    mk_set = {get_model_key(model_name, pt) for pt in prediction_times}
    if not set(per_date.keys()) <= mk_set:
        return False
    inner = next(iter(per_date.values()))
    return isinstance(inner, Mapping)


def _flatten_distribution_leaves(
    per_date: Mapping[Any, Any],
    model_name: str,
    prediction_times: Sequence[Tuple[int, int]],
) -> List[Mapping[str, Any]]:
    """Collect leaf dicts for inactive-service checks for either per-date encoding."""
    if _distribution_per_date_is_model_key_indexed(
        per_date, model_name, prediction_times
    ):
        leaves: List[Mapping[str, Any]] = []
        for by_snap in per_date.values():
            if isinstance(by_snap, Mapping):
                for leaf in by_snap.values():
                    if isinstance(leaf, Mapping):
                        leaves.append(leaf)
        return leaves
    return [v for v in per_date.values() if isinstance(v, Mapping)]


def _build_prob_dist_dict_all_for_service(
    per_date: Mapping[Any, Any],
    model_name: str,
    prediction_times: Sequence[Tuple[int, int]],
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """Build `get_model_key`-indexed prediction dicts for every `prediction_time`.

    Parameters
    ----------
    per_date : mapping
        Either ``snapshot`` → raw leaf (legacy: same PMF copied under every
        model key), or ``model_key`` → ``snapshot`` → leaf when each clock has
        its own predicted PMFs (preferred for multi-panel EPUDD).
    model_name : str
        Base name for `patientflow.load.get_model_key`.
    prediction_times : sequence of tuple of int
        Each tuple is `(hour, minute)`.

    Returns
    -------
    dict
        `model_key` → `{snapshot: normalised_leaf, ...}`.
    """
    if _distribution_per_date_is_model_key_indexed(
        per_date, model_name, prediction_times
    ):
        out_mk: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        for pt in prediction_times:
            mk = get_model_key(model_name, pt)
            src = per_date.get(mk)
            inner: Dict[Any, Dict[str, Any]] = {}
            if isinstance(src, Mapping):
                for snap, payload in src.items():
                    inner[snap] = _ensure_distribution_leaf(payload)
            out_mk[mk] = inner
        return out_mk

    out: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for pt in prediction_times:
        mk = get_model_key(model_name, pt)
        inner_legacy: Dict[Any, Dict[str, Any]] = {}
        for snap, payload in per_date.items():
            inner_legacy[snap] = _ensure_distribution_leaf(payload)
        out[mk] = inner_legacy
    return out


def _coerce_snapshot_date(snap: Any) -> date:
    """Parse a snapshot key from a distribution tree as `datetime.date`."""
    if isinstance(snap, datetime):
        return snap.date()
    if isinstance(snap, date):
        return snap
    if isinstance(snap, str):
        return date.fromisoformat(snap)
    raise TypeError(
        f"Cannot interpret distribution snapshot key {snap!r} as a calendar date"
    )


def _observation_frame_for_distribution(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    service: str,
) -> pd.DataFrame:
    """Return the observation dataframe for *target* and *service*."""
    frame_key = observation_context_frame_key(target.observation_mode)
    flow_ctx = inputs.observation_contexts.get(target.flow_name) or {}
    svc_ctx = flow_ctx.get(str(service))
    if svc_ctx is None:
        raise ValueError(
            f"Distribution evaluation for flow {target.flow_name!r}, service "
            f"{service!r}, observation_mode={target.observation_mode!r} requires "
            f"observation context {frame_key!r}; register "
            f"add_distribution_observations on this flow with "
            f"{frame_key}_by_service=..."
        )
    frame = svc_ctx.get(frame_key)
    if frame is None:
        raise ValueError(
            f"Distribution evaluation for flow {target.flow_name!r}, service "
            f"{service!r}, observation_mode={target.observation_mode!r} requires "
            f"observation context {frame_key!r}; register "
            f"add_distribution_observations with {frame_key}_by_service=..."
        )
    return frame


def _recompute_leaf_agg_observed(
    leaf: Dict[str, Any],
    *,
    target: EvaluationTarget,
    service: str,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    observation_frame: pd.DataFrame,
    benchmark_cohorts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Recompute and set `agg_observed` on one distribution leaf."""
    frame_key = observation_context_frame_key(target.observation_mode)
    count_kwargs: Dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "prediction_time": prediction_time,
        "prediction_window": prediction_window,
        **count_observed_label_kwargs(target.observation_mode, benchmark_cohorts),
    }
    if count_observed_applies_specialty_filter(target.observation_mode):
        count_kwargs["specialty"] = str(service)
    if frame_key == "ed_visits":
        count_kwargs["ed_visits"] = observation_frame
    elif frame_key == "inpatient_arrivals":
        count_kwargs["inpatient_arrivals"] = observation_frame
    else:
        count_kwargs["inpatient_visits"] = observation_frame
        route = admission_type_filter_for_distribution_component(target.component)
        if route is not None:
            count_kwargs["admission_type"] = route

    recomputed = count_observed(target.observation_mode, **count_kwargs)
    prior = leaf.get("agg_observed")
    if prior is not None and int(prior) != int(recomputed):
        raise AssertionError(
            f"agg_observed on leaf ({prior!r}) does not match recomputed count "
            f"({recomputed!r}) for flow {target.flow_name!r}, service {service!r}, "
            f"snapshot_date={snapshot_date!r}, prediction_time={prediction_time!r}, "
            f"observation_mode={target.observation_mode!r}"
        )
    leaf["agg_observed"] = recomputed


def _recompute_distribution_observed_counts(
    per_date: Mapping[Any, Any],
    *,
    target: EvaluationTarget,
    service: str,
    model_name: str,
    prediction_dict: Mapping[Tuple[int, int], timedelta],
    observation_frame: pd.DataFrame,
    benchmark_cohorts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Update `agg_observed` on every leaf in a per-service distribution tree."""
    prediction_times = prediction_times_from_dict(prediction_dict)
    if _distribution_per_date_is_model_key_indexed(
        per_date, model_name, prediction_times
    ):
        for pt in prediction_times:
            mk = get_model_key(model_name, pt)
            by_snap = per_date.get(mk)
            if not isinstance(by_snap, Mapping):
                continue
            for snap, leaf in by_snap.items():
                if not isinstance(leaf, dict):
                    continue
                _recompute_leaf_agg_observed(
                    leaf,
                    target=target,
                    service=service,
                    snapshot_date=_coerce_snapshot_date(snap),
                    prediction_time=pt,
                    prediction_window=prediction_dict[pt],
                    observation_frame=observation_frame,
                    benchmark_cohorts=benchmark_cohorts,
                )
        return

    mk_set = {get_model_key(model_name, pt) for pt in prediction_times}
    for snap, leaf in per_date.items():
        if snap in mk_set:
            continue
        if not isinstance(leaf, dict):
            continue
        snap_date = _coerce_snapshot_date(snap)
        for pt in prediction_times:
            _recompute_leaf_agg_observed(
                leaf,
                target=target,
                service=service,
                snapshot_date=snap_date,
                prediction_time=pt,
                prediction_window=prediction_dict[pt],
                observation_frame=observation_frame,
                benchmark_cohorts=benchmark_cohorts,
            )


def _distribution_expectation(leaf: Mapping[str, Any]) -> float:
    """Discrete expectation `E[k]` under `agg_predicted` for inactive checks.

    Parameters
    ----------
    leaf : mapping
        Normalised leaf from `_ensure_distribution_leaf` (or compatible).

    Returns
    -------
    float
        `sum_k k * p_k` using probability mass on `0..K-1`; `0.0` if the
        payload cannot be interpreted.
    """
    ap = leaf.get("agg_predicted")
    if isinstance(ap, dict) and "agg_proba" in ap:
        p = np.asarray(ap["agg_proba"], dtype=float).flatten()
        return float(np.dot(np.arange(len(p)), p))
    if isinstance(ap, pd.DataFrame) and "agg_proba" in ap.columns:
        p = np.asarray(ap["agg_proba"].values, dtype=float).flatten()
        return float(np.dot(np.arange(len(p)), p))
    if isinstance(ap, pd.Series):
        return float(np.dot(ap.index.to_numpy(), ap.values.flatten()))
    return 0.0


def _service_is_inactive_distribution(
    per_date: Mapping[Any, Any],
    *,
    model_name: str,
    prediction_times: Sequence[Tuple[int, int]],
) -> bool:
    """Return whether a service has no observed mass and negligible predicted mass.

    Parameters
    ----------
    per_date : mapping
        `date` → leaf, or `model_key` → `date` → leaf (see
        `_distribution_per_date_is_model_key_indexed`).
    model_name : str
        Base model name for resolving `get_model_key` when *per_date* is
        model-key indexed.
    prediction_times : sequence of tuple of int
        Prediction clocks for the run (used only for the model-key form).

    Returns
    -------
    bool
        `True` if total observed count is zero and the mean absolute predicted
        expectation across snapshots is below `1e-9`.
    """
    leaves = _flatten_distribution_leaves(per_date, model_name, prediction_times)
    total_obs = 0
    expectations: List[float] = []
    for leaf in leaves:
        total_obs += int(leaf.get("agg_observed", 0) or 0)
        try:
            expectations.append(abs(_distribution_expectation(leaf)))
        except Exception:
            expectations.append(0.0)
    mean_pred = float(np.mean(expectations)) if expectations else 0.0
    return total_obs == 0 and mean_pred < 1e-9


def _service_is_inactive_arrival(
    df: pd.DataFrame, snapshot_dates: Sequence[date]
) -> bool:
    """Return whether arrivals contain no rows on any `snapshot_dates`.

    Parameters
    ----------
    df : pandas.DataFrame
        Arrivals table; may be empty.
    snapshot_dates : sequence of datetime.date
        Evaluation dates passed to arrival-delta plots.

    Returns
    -------
    bool
        `True` if `df` is empty, `snapshot_dates` is empty, or no row has
        `arrival_datetime` on a listed date. If `arrival_datetime` is missing,
        returns `False` (treat as active / unknown).
    """
    if df is None or len(df) == 0:
        return True
    if not snapshot_dates:
        return True
    if "arrival_datetime" not in df.columns:
        return False
    sub = df[df["arrival_datetime"].dt.date.isin(set(snapshot_dates))]
    return bool(len(sub) == 0)


def evaluate_classifier_model_diagnostics(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    classifiers_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot global feature importance and optional SHAP; emit model-level scalar rows.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include a classifier block for `target.flow_name`.
    target : EvaluationTarget
        `evaluation_mode` must be `"classifier_model_diagnostics"`.
    classifiers_dir : pathlib.Path
        Output directory for `target.flow_name` (created as needed).
    collector : ScalarsCollector
        Receives one row per trained model and prediction time.

    Notes
    -----
    Skips quietly when no classifier block is registered for the flow. Uses
    `patientflow.viz.features.plot_features` and, when available,
    `patientflow.viz.shap.plot_shap`. Closes matplotlib figures after each plot.
    """
    block = inputs.classifier_by_flow.get(target.flow_name)
    if not block:
        return
    models = _sort_models(block["trained_models"])
    visits: pd.DataFrame = block["visits_df"]
    label_col: str = block["label_col"]
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    eval_split = inputs.eval_split

    for m in models:
        _require_classifier_eval_artifacts(m)
    plot_features(
        models,
        media_file_path=classifiers_dir,
        file_name="features.png",
        suptitle=_classifier_diagnostics_suptitle(
            target, "feature importances", eval_split=eval_split
        ),
        return_figure=False,
    )
    plt.close("all")

    if SHAP_AVAILABLE and plot_shap is not None:
        multi_clock = len(models) > 1
        for m in models:
            plot_shap(
                [m],
                visits,
                media_file_path=classifiers_dir,
                file_name=_disambiguate_classifier_plot_filename(
                    "shap.png",
                    m.training_results.prediction_time,
                    multi_clock=multi_clock,
                ),
                suptitle=_classifier_diagnostics_suptitle(
                    target,
                    "SHAP summary",
                    eval_split=eval_split,
                    prediction_time=m.training_results.prediction_time,
                ),
                return_figure=False,
                label_col=label_col,
                show=False,
            )
            plt.close("all")

    base_name = _classifier_base_name(block)
    for m in models:
        _require_classifier_eval_artifacts(m)
        collector.add_row(
            _classifier_diagnostics_scalar_row(target, m, base_name=base_name)
        )


def evaluate_classifier_probability_quality(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    classifiers_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot discrimination, MADCAP, and calibration on all visits; emit one scalar row.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include a classifier block for `target.flow_name`.
    target : EvaluationTarget
        `evaluation_mode` must be `"classifier_probability_quality"`.
    classifiers_dir : pathlib.Path
        Output directory for this flow (same tree as model diagnostics:
        `classifiers_dir / target.flow_name`).
    collector : ScalarsCollector
        Receives one flow-level row (``charts_generated`` only; no headline metrics).

    Notes
    -----
    Plots use the full registered `visits_df` (not split by specialty). Writes
    ``discrimination.png``, ``madcap.png``, ``madcap_by_age.png`` (or per-clock
    variants when several models are registered), and ``calibration.png``. Skips
    quietly when no classifier block exists. Headline metrics are recorded under
    ``classifier_model_diagnostics``.
    """
    block = inputs.classifier_by_flow.get(target.flow_name)
    if not block:
        return
    models = _sort_models(block["trained_models"])
    visits: pd.DataFrame = block["visits_df"]
    label_col: str = block["label_col"]
    for m in models:
        _require_classifier_eval_artifacts(m)

    classifiers_dir.mkdir(parents=True, exist_ok=True)

    if len(visits) == 0:
        return

    eval_split = inputs.eval_split

    plot_estimated_probabilities(
        models,
        visits,
        media_file_path=classifiers_dir,
        file_name="discrimination.png",
        suptitle=_classifier_quality_suptitle(
            target, "discrimination", eval_split=eval_split
        ),
        return_figure=False,
        label_col=label_col,
        show=False,
    )
    plt.close("all")
    plot_madcap(
        models,
        visits,
        media_file_path=classifiers_dir,
        file_name="madcap.png",
        suptitle=_classifier_quality_suptitle(target, "MADCAP", eval_split=eval_split),
        return_figure=False,
        label_col=label_col,
        show=False,
    )
    plt.close("all")
    multi_clock = len(models) > 1
    for m in models:
        plot_madcap_by_group(
            [m],
            visits,
            grouping_var="age_group",
            grouping_var_name="Age group",
            media_file_path=classifiers_dir,
            file_name=_disambiguate_classifier_plot_filename(
                "madcap_by_age.png",
                m.training_results.prediction_time,
                multi_clock=multi_clock,
            ),
            suptitle=_classifier_diagnostics_suptitle(
                target,
                "MADCAP by age group",
                eval_split=eval_split,
                prediction_time=m.training_results.prediction_time,
            ),
            plot_difference=False,
            return_figure=False,
            label_col=label_col,
            show=False,
        )
    plt.close("all")
    plot_calibration(
        models,
        visits,
        media_file_path=classifiers_dir,
        file_name="calibration.png",
        suptitle=_classifier_quality_suptitle(
            target, "calibration", eval_split=eval_split
        ),
        return_figure=False,
        label_col=label_col,
        show=False,
    )
    plt.close("all")

    collector.add_row(
        {
            **scalar_target_fields(target),
            "service": SERVICE_SENTINEL_ALL,
            "component": target.component,
            "prediction_time": None,
            "model_name": "",
            "charts_generated": True,
        }
    )


def _benchmark_p_bar_by_prediction_time(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
) -> Optional[Dict[Tuple[int, int], float]]:
    """Global p̄ per clock when observation mode supports a binomial benchmark."""
    cohort_key = benchmark_cohort_key_for_observation_mode(target.observation_mode)
    if cohort_key is None:
        return None
    spec = inputs.distribution_benchmark_cohorts.get(cohort_key)
    if not spec:
        return None
    visits_df = spec.get("visits_df")
    label_col = spec.get("label_col")
    if visits_df is None or label_col is None:
        return None
    return global_p_bar_by_prediction_time(
        visits_df,
        inputs.prediction_times,
        label_col=str(label_col),
    )


def evaluate_distribution(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    distributions_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot EPUDD and rPIT+CvM calibration per active service and prediction time.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include `distribution_by_flow[target.flow_name]`.
    target : EvaluationTarget
        `evaluation_mode` must be `"distribution"`.
    distributions_dir : pathlib.Path
        Root for `distributions_dir / flow / service` chart paths.
    collector : ScalarsCollector
        Receives rows per service and prediction time; merges a
        `merge_service_summary_slice` aggregate for the slice.

    Notes
    -----
    Inactive services skip evaluation (`skip_reason: inactive_service`).
    Active services with fewer than ``MIN_DISTRIBUTION_SNAPSHOTS`` snapshot
    leaves per clock skip EPUDD and calibration
    (`skip_reason: insufficient_observations`). Binomial benchmark scalars are
    emitted only when ``observation_mode`` supports a benchmark cohort and
    ``add_distribution_benchmark_cohort`` registered the matching eval-split
    frame (not for yet-to-arrive ``arrived_in_window`` modes).
    """
    block = inputs.distribution_by_flow.get(target.flow_name)
    if not block:
        return
    prob_by_svc: Mapping[str, Any] = block.get("prob_dist_by_service") or {}
    model_name: str = str(block.get("model_name") or "admissions")
    prediction_dict = inputs.prediction_dict
    p_bar_by_pt = _benchmark_p_bar_by_prediction_time(inputs, target)
    if not prob_by_svc:
        collector.merge_service_summary_slice(
            f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
            {
                "mode": "distribution",
                "flow": target.flow_name,
                "component": target.component,
                "observation_mode": target.observation_mode,
                "n_active_services": 0,
                "n_inactive_services": 0,
                "inactive_service_names": [],
            },
        )
        return

    inactive_names: List[str] = []
    active_count = 0

    for service, per_date_raw in prob_by_svc.items():
        per_date = per_date_raw  # type: ignore[assignment]
        if not isinstance(per_date, Mapping):
            continue
        observation_frame = _observation_frame_for_distribution(
            inputs, target, str(service)
        )
        _recompute_distribution_observed_counts(
            per_date,
            target=target,
            service=str(service),
            model_name=model_name,
            prediction_dict=prediction_dict,
            observation_frame=observation_frame,
            benchmark_cohorts=inputs.distribution_benchmark_cohorts,
        )
        inactive = _service_is_inactive_distribution(
            per_date,
            model_name=model_name,
            prediction_times=inputs.prediction_times,
        )
        if inactive:
            inactive_names.append(str(service))
            flat_leaves = _flatten_distribution_leaves(
                per_date, model_name, inputs.prediction_times
            )
            n_snapshots_inactive = len(flat_leaves)
            for pt in inputs.prediction_times:
                h, mi = pt
                collector.add_row(
                    {
                        **scalar_target_fields(target),
                        "service": str(service),
                        "component": target.component,
                        "prediction_time": [h, mi],
                        "model_name": model_name,
                        "charts_generated": False,
                        "skip_reason": "inactive_service",
                        "n_snapshots": n_snapshots_inactive,
                        "reliable": False,
                    }
                )
            continue

        active_count += 1
        prob_all = _build_prob_dist_dict_all_for_service(
            per_date, model_name, inputs.prediction_times
        )
        epudd_times = [
            pt
            for pt in inputs.prediction_times
            if len(prob_all.get(get_model_key(model_name, pt)) or {})
            >= MIN_DISTRIBUTION_SNAPSHOTS
        ]
        svc_dir = distributions_dir / target.flow_name / _safe_fs_segment(str(service))
        svc_dir.mkdir(parents=True, exist_ok=True)
        if epudd_times:
            fig = plot_epudd(
                epudd_times,
                prob_all,
                model_name=model_name,
                return_figure=True,
                media_file_path=svc_dir,
                file_name=f"{target.component}.png",
                suptitle=_distribution_comparison_suptitle(
                    target, str(service), eval_split=inputs.eval_split
                ),
            )
            if fig is not None:
                plt.close(fig)
            else:
                plt.close("all")

        for pt in inputs.prediction_times:
            h, mi = pt
            mk = get_model_key(model_name, pt)
            series_dict = prob_all.get(mk) or {}
            n_snap = len(series_dict)
            base_row: Dict[str, Any] = {
                **scalar_target_fields(target),
                "service": str(service),
                "component": target.component,
                "prediction_time": [h, mi],
                "model_name": model_name,
                "n_snapshots": n_snap,
            }
            if n_snap < MIN_DISTRIBUTION_SNAPSHOTS:
                collector.add_row(
                    {
                        **base_row,
                        "charts_generated": False,
                        "skip_reason": "insufficient_observations",
                        "reliable": False,
                    }
                )
                continue

            pf_result = rpit_cvm_calibration_score(series_dict)
            row: Dict[str, Any] = {
                **base_row,
                "charts_generated": True,
                "reliable": True,
            }
            if pf_result is not None:
                row.update(rpit_cvm_result_to_scalar_fields(pf_result, seed=None))

            if p_bar_by_pt is not None:
                p_bar = p_bar_by_pt.get(pt)
                if p_bar is not None:
                    bench_dict = build_benchmark_prob_dist_dict(
                        series_dict, p_bar=p_bar
                    )
                    bm_result = rpit_cvm_calibration_score(bench_dict)
                    if bm_result is not None and pf_result is not None:
                        row.update(
                            rpit_cvm_result_to_scalar_fields(
                                bm_result, prefix="rpit_cvm_benchmark", seed=None
                            )
                        )
                        row.update(
                            {
                                "rpit_cvm_w2_reduction": (
                                    bm_result.mean_w2 - pf_result.mean_w2
                                ),
                                "rpit_cvm_benchmark_p_bar": p_bar,
                            }
                        )

            alt_by_kind = inputs.distribution_benchmark_pmfs.get(target.flow_name, {})
            for benchmark_kind, alt_prob_by_svc in alt_by_kind.items():
                alt_per_date = alt_prob_by_svc.get(str(service))
                if not isinstance(alt_per_date, Mapping):
                    continue
                alt_prob_all = _build_prob_dist_dict_all_for_service(
                    alt_per_date, model_name, inputs.prediction_times
                )
                alt_series_dict = alt_prob_all.get(mk) or {}
                alt_result = rpit_cvm_calibration_score(alt_series_dict)
                if alt_result is None or pf_result is None:
                    continue
                prefix = f"rpit_cvm_{benchmark_kind}"
                row.update(
                    rpit_cvm_result_to_scalar_fields(
                        alt_result, prefix=prefix, seed=None
                    )
                )
                row.update(
                    {
                        f"{prefix}_w2_reduction": (
                            alt_result.mean_w2 - pf_result.mean_w2
                        ),
                    }
                )

            collector.add_row(row)

    collector.merge_service_summary_slice(
        f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
        {
            "mode": "distribution",
            "flow": target.flow_name,
            "component": target.component,
            "observation_mode": target.observation_mode,
            "n_active_services": active_count,
            "n_inactive_services": len(inactive_names),
            "inactive_service_names": inactive_names,
        },
    )


def evaluate_arrival_deltas(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    arrivals_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot observed-vs-expected arrival deltas per service and prediction time.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include `arrival_by_flow[target.flow_name]`.
    target : EvaluationTarget
        `evaluation_mode` must be `"arrival_deltas"`.
    arrivals_dir : pathlib.Path
        Root for `arrivals_dir / flow / service` chart paths.
    collector : ScalarsCollector
        Receives rows per service and prediction time; merges a service summary
        slice for the handler.

    Notes
    -----
    Optional per-service predictors and filter keys are taken from the arrival
    block. Inactive services (no arrivals on snapshot dates) skip plots. Uses
    `patientflow.viz.observed_against_expected.plot_arrival_deltas`.
    """
    block = inputs.arrival_by_flow.get(target.flow_name)
    if not block:
        return
    by_svc: Mapping[str, pd.DataFrame] = block.get("arrivals_by_service") or {}
    snap_dates: Sequence[date] = block.get("snapshot_dates") or []
    prediction_dict = inputs.prediction_dict
    predictors = block.get("predictors_by_service") or {}
    filter_keys = block.get("filter_keys_by_service") or {}
    strict_map = block.get("strict_prediction_date_by_service") or {}
    yta_iv = block.get("yta_time_interval", timedelta(minutes=15))

    inactive_names: List[str] = []
    active = 0

    for svc, df in by_svc.items():
        if _service_is_inactive_arrival(df, snap_dates):
            inactive_names.append(str(svc))
            for pt in inputs.prediction_times:
                h, mi = pt
                collector.add_row(
                    {
                        **scalar_target_fields(target),
                        "service": str(svc),
                        "component": target.component,
                        "prediction_time": [h, mi],
                        "model_name": "",
                        "charts_generated": False,
                        "skip_reason": "inactive_service",
                        "reliable": False,
                    }
                )
            continue

        active += 1
        pred = predictors.get(svc)
        fk = filter_keys.get(svc)
        strict = bool(strict_map.get(svc, False))
        for pt in inputs.prediction_times:
            h, mi = pt
            out_dir = arrivals_dir / target.flow_name / _safe_fs_segment(str(svc))
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{target.component}_{h:02d}{mi:02d}.png"
            plot_arrival_deltas(
                df,
                pt,
                list(snap_dates),
                prediction_dict[pt],
                yta_time_interval=yta_iv,
                media_file_path=out_dir,
                file_name=fname,
                return_figure=False,
                arrival_rate_model=pred,
                filter_key=fk,
                strict_prediction_date=strict,
                suptitle=_arrival_delta_suptitle(
                    target,
                    str(svc),
                    (h, mi),
                    eval_split=inputs.eval_split,
                ),
            )
            plt.close("all")
            collector.add_row(
                {
                    **scalar_target_fields(target),
                    "service": str(svc),
                    "component": target.component,
                    "prediction_time": [h, mi],
                    "model_name": "",
                    "charts_generated": True,
                    "reliable": True,
                }
            )

    collector.merge_service_summary_slice(
        f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
        {
            "mode": "arrival_deltas",
            "flow": target.flow_name,
            "component": target.component,
            "observation_mode": target.observation_mode,
            "n_active_services": active,
            "n_inactive_services": len(inactive_names),
            "inactive_service_names": inactive_names,
        },
    )


def evaluate_survival_curve(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    survival_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot train vs test survival curves and emit one aggregate scalar row.

    Parameters
    ----------
    inputs : EvaluationInputs
        Global `survival` block with train/test frames and column names.
    target : EvaluationTarget
        `evaluation_mode` must be `"survival_curve"`.
    survival_dir : pathlib.Path
        Directory for the survival PNG.
    collector : ScalarsCollector
        Receives a single row with `service: _all_` and `prediction_time: null`.

    Notes
    -----
    Skips when `inputs.survival` is unset. Uses
    `patientflow.viz.survival_curve.plot_admission_time_survival_curve`.
    """
    if not inputs.survival:
        return
    s = inputs.survival
    train_df: pd.DataFrame = s["train_df"]
    test_df: pd.DataFrame = s["test_df"]
    survival_dir.mkdir(parents=True, exist_ok=True)
    labels = s.get("labels") or ("train", "test")
    plot_admission_time_survival_curve(
        [train_df, test_df],
        start_time_col=s.get("start_time_col", "arrival_datetime"),
        end_time_col=s.get("end_time_col", "departure_datetime"),
        labels=list(labels),
        media_file_path=survival_dir,
        file_name=f"{target.component}.png",
        return_figure=False,
    )
    plt.close("all")
    collector.add_row(
        {
            **scalar_target_fields(target),
            "service": SERVICE_SENTINEL_ALL,
            "component": target.component,
            "prediction_time": None,
            "model_name": "",
            "charts_generated": True,
            "reliable": True,
        }
    )


def _observed_destination_counts(
    events: pd.DataFrame,
    destinations: Sequence[str],
    *,
    destination_col: str,
    discharge_label: str = "Discharge",
) -> np.ndarray:
    """Aggregate observed destination counts for departure events."""
    dest_list = list(destinations)
    dest_to_idx = {dest: idx for idx, dest in enumerate(dest_list)}
    counts = np.zeros(len(dest_list), dtype=int)

    for dest_value in events[destination_col]:
        if dest_value is None or (isinstance(dest_value, float) and pd.isna(dest_value)):
            key = discharge_label
        else:
            key = str(dest_value)
        idx = dest_to_idx.get(key)
        if idx is not None:
            counts[idx] += 1
    return counts


def _filter_transition_departure_events(
    block: Mapping[str, Any],
    estimator: Any,
) -> pd.DataFrame:
    """Return departure events, optionally filtered to the registered cohort."""
    events = block["departure_events"]
    cohort = block.get("cohort")
    if estimator.cohort_col and cohort is not None:
        return events.loc[events[estimator.cohort_col] == cohort].copy()
    return events.copy()


def _transition_row_base(
    target: EvaluationTarget,
    source: str,
    block: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "evaluation_mode": target.evaluation_mode,
        "flow": target.flow_name,
        "flow_type": target.flow_type,
        "service": source,
        "component": target.component,
        "prediction_time": None,
        "model_name": str(block.get("model_name") or "transfers"),
        "cohort": block.get("cohort"),
        "charts_generated": False,
    }


def evaluate_transition_matrix(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    transitions_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Score row-wise calibration of subgroup routing tables per source subspecialty.

    Each scalar row answers a conditional question: given that patients departed
    from source `s` during the registered window, with the subgroup mix that
    actually left, do their observed destinations match the routing the fitted
    model would have applied patient-by-patient? The test uses subgroup tables
    from `TransferProbabilityEstimator`, not the pooled diagnostic matrix from
    `get_transition_matrix`.

    For every source in the union of the estimator index and observed departure
    sources, the handler filters departure events to that source, builds a
    per-event routing matrix via `build_per_patient_probabilities`, aggregates
    patient-level expected counts `E`, compares them to observed destination
    counts, and runs a Monte Carlo Pearson goodness-of-fit test when the source
    is active. Active rows store aligned `destinations`, `expected_counts`, and
    `observed_counts`.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include `transition_matrix_by_flow[target.flow_name]`.
    target : EvaluationTarget
        `evaluation_mode` must be "transition_matrix".
    transitions_dir : pathlib.Path
        Run subdirectory for this evaluation mode.
    collector : ScalarsCollector
        Receives one row per source and a `_service_summary` slice fragment.

    Notes
    -----
    Skips quietly when the flow block is absent. Uses the same subgroup
    resolution contract as `transfer_weight_to_target` (unmatched patients are
    treated as all-discharge, with no fallback to the pooled ["services"] row).

    One row is emitted per estimator source even when `n_departures == 0`.
    Skipped rows use `skip_reason`:

    - `no_observed_departures` — no departures from `s` in the window.
    - `all_discharge_expected` — departures exist but every event's routing is
      all-discharge (`sum of non-Discharge E_d == 0`), so the Pearson test is
      degenerate.

    On active rows, `n_excluded_unmatched` counts departures excluded from
    subgroup routing (typically adults with missing or invalid sex). Non-zero
    values flag case-mix the model does not route. `reliable` is true when
    `n_departures >= RELIABILITY_MIN_OBSERVATIONS_TRANSITION` (30).

    This mode does not test departure volume from `s`, column-level inflow
    calibration across sources, intra-window routing drift, or subgroup
    classifier accuracy. See `transition_matrix_evaluation.md` for the full
    algorithm and interpretation notes.
    """
    _ = transitions_dir
    block = inputs.transition_matrix_by_flow.get(target.flow_name)
    if not block:
        return

    estimator = block["estimator"]
    events = _filter_transition_departure_events(block, estimator)
    cohort = block.get("cohort")
    matrix = estimator.get_transition_matrix(cohort)
    destinations = list(matrix.columns)
    source_col = str(block["source_col"])
    destination_col = str(block["destination_col"])
    discharge_label = str(block.get("discharge_label") or "Discharge")

    discharge_idx = destinations.index(discharge_label)
    skipped: Dict[str, int] = {
        "no_observed_departures": 0,
        "all_discharge_expected": 0,
    }
    active = 0
    total_excluded_unmatched = 0

    event_sources = set(events[source_col].dropna().astype(str).unique())
    sources = sorted(set(matrix.index.astype(str)) | event_sources)

    for source in sources:
        source_events = events.loc[events[source_col].astype(str) == str(source)]
        n_departures = len(source_events)
        base_row = _transition_row_base(target, str(source), block)

        if n_departures == 0:
            skipped["no_observed_departures"] += 1
            collector.add_row(
                {
                    **base_row,
                    "skip_reason": "no_observed_departures",
                    "n_departures": 0,
                    "reliable": False,
                }
            )
            continue

        per_patient = build_per_patient_probabilities(
            source_events,
            str(source),
            cohort,
            destinations,
            estimator,
            discharge_label=discharge_label,
        )
        expected = per_patient.routing_matrix.sum(axis=0)
        expected_transfers = float(expected.sum() - expected[discharge_idx])
        total_excluded_unmatched += per_patient.n_excluded_unmatched

        if expected_transfers == 0.0:
            skipped["all_discharge_expected"] += 1
            collector.add_row(
                {
                    **base_row,
                    "skip_reason": "all_discharge_expected",
                    "n_departures": n_departures,
                    "reliable": False,
                }
            )
            continue

        n_obs = _observed_destination_counts(
            source_events,
            destinations,
            destination_col=destination_col,
            discharge_label=discharge_label,
        )
        result = multinomial_gof_montecarlo(
            per_patient.routing_matrix,
            n_obs,
            destinations,
            n_simulations=int(block.get("n_simulations") or 10_000),
            seed=derive_seed_offset(block.get("seed"), str(source)),
        )
        active += 1
        collector.add_row(
            {
                **base_row,
                "reliable": n_departures >= RELIABILITY_MIN_OBSERVATIONS_TRANSITION,
                "n_departures": n_departures,
                "n_destinations": len(destinations),
                "n_subgroups_used": per_patient.n_subgroups_used,
                "n_excluded_unmatched": per_patient.n_excluded_unmatched,
                "pearson_x2": result.pearson_x2,
                "p_value": result.p_value,
                "n_simulations": result.n_simulations,
                "seed": block.get("seed"),
                "n_structural_violations": result.n_structural_violations,
                "destinations": result.destinations,
                "expected_counts": result.expected_counts,
                "observed_counts": result.observed_counts,
            }
        )

    collector.merge_service_summary_slice(
        f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
        {
            "mode": "transition_matrix",
            "flow": target.flow_name,
            "component": target.component,
            "cohort": block.get("cohort"),
            "n_active_sources": active,
            "n_skipped_sources": sum(skipped.values()),
            "skipped_by_reason": skipped,
            "n_excluded_unmatched_departures": total_excluded_unmatched,
        },
    )
