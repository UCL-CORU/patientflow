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

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patientflow.evaluate.inputs import EvaluationInputs, EvaluationTarget
from patientflow.evaluate.scalars import (
    SERVICE_SENTINEL_ALL,
    ScalarsCollector,
    classifier_reliable,
)
from patientflow.load import get_model_key
from patientflow.model_artifacts import TrainedClassifier
from patientflow.viz.epudd import plot_epudd
from patientflow.viz.features import plot_features
from patientflow.viz.observed_against_expected import plot_arrival_deltas
from patientflow.viz.calibration import plot_calibration
from patientflow.viz.madcap import plot_madcap
from patientflow.viz.survival_curve import plot_admission_time_survival_curve

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


def _ensure_epudd_leaf(data: Mapping[str, Any]) -> Dict[str, Any]:
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
    if isinstance(ap, pd.Series):
        arr = np.asarray(ap.values, dtype=float).flatten()
        return {
            "agg_predicted": {"agg_proba": arr},
            "agg_observed": out.get("agg_observed", 0),
        }
    raise TypeError(
        "Each snapshot value must have agg_predicted as a Series or "
        "as a dict with key 'agg_proba'."
    )


def _build_prob_dist_dict_all_for_service(
    per_date: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    prediction_times: Sequence[Tuple[int, int]],
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """Build `get_model_key`-indexed prediction dicts for every `prediction_time`.

    Parameters
    ----------
    per_date : mapping
        `snapshot` → raw leaf payload (passed through `_ensure_epudd_leaf`).
    model_name : str
        Base name for `patientflow.load.get_model_key`.
    prediction_times : sequence of tuple of int
        Each tuple is `(hour, minute)`; the same `per_date` is attached under
        every derived model key so EPUDD can overlay all clocks.

    Returns
    -------
    dict
        `model_key` → `{snapshot: normalised_leaf, ...}`.
    """
    out: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for pt in prediction_times:
        mk = get_model_key(model_name, pt)
        inner: Dict[Any, Dict[str, Any]] = {}
        for snap, payload in per_date.items():
            inner[snap] = _ensure_epudd_leaf(payload)
        out[mk] = inner
    return out


def _distribution_expectation(leaf: Mapping[str, Any]) -> float:
    """Discrete expectation `E[k]` under `agg_predicted` for inactive checks.

    Parameters
    ----------
    leaf : mapping
        Normalised leaf from `_ensure_epudd_leaf` (or compatible).

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
    if isinstance(ap, pd.Series):
        return float(np.dot(ap.index.to_numpy(), ap.values.flatten()))
    return 0.0


def _service_is_inactive_distribution(
    per_date: Mapping[Any, Mapping[str, Any]],
) -> bool:
    """Return whether a service has no observed mass and negligible predicted mass.

    Parameters
    ----------
    per_date : mapping
        `date` → leaf with `agg_observed` and `agg_predicted`.

    Returns
    -------
    bool
        `True` if total observed count is zero and the mean absolute predicted
        expectation across snapshots is below `1e-9`.
    """
    total_obs = 0
    expectations: List[float] = []
    for leaf in per_date.values():
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

    for m in models:
        _require_classifier_eval_artifacts(m)
    plot_features(
        models,
        media_file_path=classifiers_dir,
        file_name=f"{target.component}_features.png",
        suptitle=f"{target.flow_name} {target.name}",
        return_figure=False,
    )
    plt.close("all")

    if SHAP_AVAILABLE and plot_shap is not None:
        plot_shap(
            models,
            visits,
            media_file_path=classifiers_dir,
            file_name=f"{target.component}_shap.png",
            return_figure=False,
            label_col=label_col,
            show=False,
        )
        plt.close("all")

    for m in models:
        _require_classifier_eval_artifacts(m)
        metrics = m.selected_eval_metrics
        pt = m.training_results.prediction_time
        info = (m.training_results.training_info or {}).get("dataset_info") or {}
        pos_cases = info.get("train_valid_test_positive_cases") or {}
        reliable = classifier_reliable(metrics, pos_cases)
        hour, minute = pt
        collector.add_row(
            {
                "evaluation_mode": target.evaluation_mode,
                "flow": target.flow_name,
                "flow_type": target.flow_type,
                "service": SERVICE_SENTINEL_ALL,
                "component": target.component,
                "prediction_time": [hour, minute],
                "model_name": metrics.get("split", "model"),
                "charts_generated": True,
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "log_loss": metrics.get("log_loss"),
                "n_samples": metrics.get("n_samples"),
                "n_positive_cases": metrics.get("n_positive_cases"),
                "reliable": reliable,
            }
        )


def evaluate_classifier_probability_quality(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    services_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot MADCAP and calibration per service; emit service-level scalar rows.

    Parameters
    ----------
    inputs : EvaluationInputs
        Must include a classifier block for `target.flow_name`.
    target : EvaluationTarget
        `evaluation_mode` must be `"classifier_probability_quality"`.
    services_dir : pathlib.Path
        Base directory for per-flow outputs (`services_dir / target.flow_name`).
    collector : ScalarsCollector
        Receives one row per (service, model, prediction time).

    Notes
    -----
    Services are taken from the `specialty` column when present; otherwise a
    single aggregate pseudo-service is used. Skips quietly when no classifier
    block exists. Uses `patientflow.viz.madcap.plot_madcap` and
    `patientflow.viz.calibration.plot_calibration`.
    """
    block = inputs.classifier_by_flow.get(target.flow_name)
    if not block:
        return
    models = _sort_models(block["trained_models"])
    visits: pd.DataFrame = block["visits_df"]
    label_col: str = block["label_col"]
    for m in models:
        _require_classifier_eval_artifacts(m)

    services_dir.mkdir(parents=True, exist_ok=True)

    if "specialty" in visits.columns:
        services = sorted(str(s) for s in visits["specialty"].dropna().unique())
    else:
        services = [SERVICE_SENTINEL_ALL]

    for svc in services:
        if svc == SERVICE_SENTINEL_ALL:
            sub = visits
        else:
            sub = visits[visits["specialty"] == svc]
        if len(sub) == 0:
            continue
        plot_madcap(
            models,
            sub,
            media_file_path=services_dir,
            file_name=f"{target.component}_{svc}_madcap.png",
            suptitle=f"{target.flow_name} {svc}",
            return_figure=False,
            label_col=label_col,
        )
        plt.close("all")
        plot_calibration(
            models,
            sub,
            media_file_path=services_dir,
            file_name=f"{target.component}_{svc}_calibration.png",
            suptitle=f"{target.flow_name} {svc}",
            return_figure=False,
            label_col=label_col,
        )
        plt.close("all")

        for m in models:
            metrics = m.selected_eval_metrics
            pt = m.training_results.prediction_time
            info = (m.training_results.training_info or {}).get("dataset_info") or {}
            pos_cases = info.get("train_valid_test_positive_cases") or {}
            reliable = classifier_reliable(metrics, pos_cases)
            hour, minute = pt
            collector.add_row(
                {
                    "evaluation_mode": target.evaluation_mode,
                    "flow": target.flow_name,
                    "flow_type": target.flow_type,
                    "service": svc,
                    "component": target.component,
                    "prediction_time": [hour, minute],
                    "model_name": "",
                    "charts_generated": True,
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "log_loss": metrics.get("log_loss"),
                    "n_samples": metrics.get("n_samples"),
                    "n_positive_cases": metrics.get("n_positive_cases"),
                    "reliable": reliable,
                }
            )


def evaluate_distribution(
    inputs: EvaluationInputs,
    target: EvaluationTarget,
    *,
    distributions_dir: Path,
    collector: ScalarsCollector,
) -> None:
    """Plot EPUDD per active service and record per-time snapshot counts.

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
    Inactive services (no observed mass and negligible predicted mass) skip charts
    and emit `skip_reason: inactive_service` rows. Uses
    `patientflow.viz.epudd.plot_epudd`.
    """
    block = inputs.distribution_by_flow.get(target.flow_name)
    if not block:
        return
    prob_by_svc: Mapping[str, Any] = block.get("prob_dist_by_service") or {}
    model_name: str = str(block.get("model_name") or "admissions")
    if not prob_by_svc:
        collector.merge_service_summary_slice(
            f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
            {
                "mode": "distribution",
                "flow": target.flow_name,
                "component": target.component,
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
        inactive = _service_is_inactive_distribution(per_date)
        if inactive:
            inactive_names.append(str(service))
            for pt in inputs.prediction_times:
                h, mi = pt
                collector.add_row(
                    {
                        "evaluation_mode": target.evaluation_mode,
                        "flow": target.flow_name,
                        "flow_type": target.flow_type,
                        "service": str(service),
                        "component": target.component,
                        "prediction_time": [h, mi],
                        "model_name": model_name,
                        "charts_generated": False,
                        "skip_reason": "inactive_service",
                        "n_snapshots": len(per_date),
                        "reliable": False,
                    }
                )
            continue

        active_count += 1
        prob_all = _build_prob_dist_dict_all_for_service(
            per_date, model_name, inputs.prediction_times
        )
        svc_dir = distributions_dir / target.flow_name / str(service)
        svc_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_epudd(
            list(inputs.prediction_times),
            prob_all,
            model_name=model_name,
            return_figure=True,
            media_file_path=svc_dir,
            file_name=f"{target.component}_epudd.png",
            suptitle=f"{target.flow_name} {service} {target.component}",
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
            collector.add_row(
                {
                    "evaluation_mode": target.evaluation_mode,
                    "flow": target.flow_name,
                    "flow_type": target.flow_type,
                    "service": str(service),
                    "component": target.component,
                    "prediction_time": [h, mi],
                    "model_name": model_name,
                    "charts_generated": True,
                    "n_snapshots": n_snap,
                    "reliable": n_snap > 0,
                }
            )

    collector.merge_service_summary_slice(
        f"{target.evaluation_mode}/{target.flow_name}/{target.component}",
        {
            "mode": "distribution",
            "flow": target.flow_name,
            "component": target.component,
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
    pred_window = block["prediction_window"]
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
                        "evaluation_mode": target.evaluation_mode,
                        "flow": target.flow_name,
                        "flow_type": target.flow_type,
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
            out_dir = arrivals_dir / target.flow_name / str(svc)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{target.component}_{h:02d}{mi:02d}.png"
            plot_arrival_deltas(
                df,
                pt,
                list(snap_dates),
                pred_window,
                yta_time_interval=yta_iv,
                media_file_path=out_dir,
                file_name=fname,
                return_figure=False,
                predictor=pred,
                filter_key=fk,
                strict_prediction_date=strict,
                suptitle=f"{target.flow_name} {svc} {h:02d}:{mi:02d}",
            )
            plt.close("all")
            collector.add_row(
                {
                    "evaluation_mode": target.evaluation_mode,
                    "flow": target.flow_name,
                    "flow_type": target.flow_type,
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
            "evaluation_mode": target.evaluation_mode,
            "flow": target.flow_name,
            "flow_type": target.flow_type,
            "service": SERVICE_SENTINEL_ALL,
            "component": target.component,
            "prediction_time": None,
            "model_name": "",
            "charts_generated": True,
            "reliable": True,
        }
    )
