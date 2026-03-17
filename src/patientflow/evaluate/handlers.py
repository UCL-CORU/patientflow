"""Per-mode evaluation handlers.

This module contains the concrete evaluation implementations for each
supported mode (classifier, distribution, arrival deltas, survival curve,
and aspirational skip), plus the mode-dispatch registry.

Handlers are designed to be composed by the runner and to emit flat scalar
rows through ``ScalarsCollector``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from patientflow.evaluate.adapters import to_legacy_prob_dist_dict_all
from patientflow.evaluate.constants import RELIABILITY_THRESHOLDS
from patientflow.evaluate.legacy_api import calc_mae_mpe
from patientflow.evaluate.scalars import ScalarsCollector
from patientflow.evaluate.types import (
    ArrivalDeltaPayload,
    ClassifierInput,
    EvaluationTarget,
    SnapshotResult,
    SurvivalCurvePayload,
)
from patientflow.load import get_model_key
from patientflow.viz.calibration import plot_calibration
from patientflow.viz.epudd import plot_epudd
from patientflow.viz.estimated_probabilities import plot_estimated_probabilities
from patientflow.viz.madcap import plot_madcap
from patientflow.viz.observed_against_expected import plot_arrival_deltas, plot_deltas
from patientflow.viz.survival_curve import plot_admission_time_survival_curve


def _format_prediction_time(prediction_time: Tuple[int, int]) -> str:
    hour, minute = prediction_time
    return f"{hour:02d}{minute:02d}"


def _close_fig(fig: Any) -> None:
    if fig is not None:
        plt.close(fig)


def _flatten_reliability(metrics: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(metrics)
    reliability = output.pop("reliability", None)
    if isinstance(reliability, dict):
        output["reliable"] = reliability.get("is_reliable")
        output["reliability_threshold"] = reliability.get("threshold")
        output["reliability_basis"] = reliability.get("basis")
    return output


def _record_not_evaluated(
    collector: ScalarsCollector,
    *,
    flow_name: str,
    target: EvaluationTarget,
    prediction_time: Tuple[int, int],
    reason: str,
    service_name: Optional[str],
) -> None:
    collector.record(
        flow=flow_name,
        service=service_name,
        component=target.component,
        prediction_time=_format_prediction_time(prediction_time),
        flow_type=target.flow_type,
        aspirational=target.aspirational,
        evaluated=False,
        reason=reason,
    )


def _is_inactive_distribution(
    snapshots_by_time: Mapping[Tuple[int, int], Mapping[Any, SnapshotResult]],
    *,
    min_total_observed: int = 1,
    max_mean_predicted: float = 0.5,
) -> bool:
    """Return True when all distribution snapshots show negligible activity.

    A service is inactive when both conditions hold:

    - total observed count across every snapshot and prediction time is below
      *min_total_observed* (default 1, i.e. all zeros)
    - the peak predicted mean across every snapshot is below
      *max_mean_predicted* (default 0.5)

    This catches Poisson(0) services and near-zero services where charts
    would be uninformative.
    """
    total_observed = 0
    peak_mean = 0.0
    for by_date in snapshots_by_time.values():
        for snap in by_date.values():
            total_observed += snap.observed
            support = np.arange(
                snap.offset, snap.offset + len(snap.predicted_pmf)
            )
            mean = float(np.dot(support, snap.predicted_pmf))
            if mean > peak_mean:
                peak_mean = mean
    return total_observed < min_total_observed and peak_mean < max_mean_predicted


def evaluate_aspirational_skip(
    *,
    service_name: str,
    flow_name: str,
    target: EvaluationTarget,
    prediction_times: List[Tuple[int, int]],
    collector: ScalarsCollector,
) -> None:
    """Record non-evaluated rows for aspirational targets.

    Parameters
    ----------
    service_name
        Service associated with the target row.
    flow_name
        Flow target name.
    target
        Evaluation target metadata.
    prediction_times
        Prediction times to mark as skipped.
    collector
        Scalar collector updated in place.
    """
    for prediction_time in prediction_times:
        _record_not_evaluated(
            collector,
            flow_name=flow_name,
            target=target,
            prediction_time=prediction_time,
            reason="Aspirational flow: observed-vs-predicted diagnostics skipped",
            service_name=service_name,
        )


def evaluate_distribution(
    *,
    service_name: str,
    flow_name: str,
    target: EvaluationTarget,
    prediction_times: List[Tuple[int, int]],
    collector: ScalarsCollector,
    output_root: Path,
    snapshots_by_time: Optional[
        Mapping[Tuple[int, int], Mapping[Any, SnapshotResult]]
    ] = None,
    skip_inactive: bool = False,
) -> None:
    """Evaluate one service-level distribution target.

    This handler converts typed snapshot payloads into plot input format,
    renders EPUDD and observed-minus-expected diagnostics, and records flat
    scalar rows.

    Parameters
    ----------
    service_name
        Service being evaluated.
    flow_name
        Flow target name.
    target
        Evaluation target metadata.
    prediction_times
        Prediction times requested for evaluation.
    collector
        Scalar collector updated in place.
    output_root
        Root directory for plot outputs.
    snapshots_by_time
        Optional mapping ``prediction_time -> snapshot_date -> SnapshotResult``.
        Missing times are recorded as non-evaluated rows.
    skip_inactive
        When True, services whose snapshots show negligible activity (zero
        observed counts and near-zero predicted means) are recorded in
        scalars with ``charts_generated: false`` and no chart files are
        written.  Defaults to False for backward compatibility.
    """
    by_time = snapshots_by_time or {}
    available = {pt: by_time[pt] for pt in prediction_times if pt in by_time}

    missing_times = [pt for pt in prediction_times if pt not in available]
    for prediction_time in missing_times:
        _record_not_evaluated(
            collector,
            flow_name=flow_name,
            target=target,
            prediction_time=prediction_time,
            reason="No flow input provided for this service/time",
            service_name=service_name,
        )

    if not available:
        return

    if skip_inactive and _is_inactive_distribution(available):
        for prediction_time in available:
            n_snapshots = len(available[prediction_time])
            collector.record(
                flow=flow_name,
                service=service_name,
                component=target.component,
                prediction_time=_format_prediction_time(prediction_time),
                flow_type=target.flow_type,
                aspirational=target.aspirational,
                evaluated=True,
                metrics={
                    "mae": 0.0,
                    "mpe": 0.0,
                    "n_snapshots": n_snapshots,
                    "reliable": n_snapshots
                    >= RELIABILITY_THRESHOLDS["distribution_snapshots"],
                    "reliability_threshold": RELIABILITY_THRESHOLDS[
                        "distribution_snapshots"
                    ],
                    "reliability_basis": "snapshots",
                    "charts_generated": False,
                    "skip_reason": "inactive_service",
                },
            )
        return

    service_dir = output_root / "services" / service_name.replace("/", "_")
    service_dir.mkdir(parents=True, exist_ok=True)

    legacy_prob_dist_dict_all = to_legacy_prob_dist_dict_all(
        snapshots_by_time=available,
        model_name=flow_name,
    )

    fig = plot_epudd(
        prediction_times=list(available.keys()),
        prob_dist_dict_all=legacy_prob_dist_dict_all,
        model_name=flow_name,
        media_file_path=service_dir,
        file_name=f"{flow_name}_{target.component}_epudd.png",
        return_figure=True,
    )
    _close_fig(fig)

    results = calc_mae_mpe(legacy_prob_dist_dict_all)
    fig = plot_deltas(
        results1=results,
        media_file_path=service_dir,
        file_name=f"{flow_name}_{target.component}_obs_exp.png",
        return_figure=True,
    )
    _close_fig(fig)

    for prediction_time in available:
        prediction_time_key = _format_prediction_time(prediction_time)
        model_key = get_model_key(flow_name, prediction_time)
        if model_key not in results:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No distribution metrics produced for this prediction time",
                service_name=service_name,
            )
            continue

        result_for_time = results[model_key]
        n_snapshots = int(len(legacy_prob_dist_dict_all[model_key]))
        metrics = {
            "mae": float(result_for_time["mae"]),
            "mpe": float(result_for_time["mpe"]),
            "n_snapshots": n_snapshots,
            "reliable": n_snapshots >= RELIABILITY_THRESHOLDS["distribution_snapshots"],
            "reliability_threshold": RELIABILITY_THRESHOLDS["distribution_snapshots"],
            "reliability_basis": "snapshots",
        }
        collector.record(
            flow=flow_name,
            service=service_name,
            component=target.component,
            prediction_time=prediction_time_key,
            flow_type=target.flow_type,
            aspirational=target.aspirational,
            evaluated=True,
            metrics=metrics,
        )


def evaluate_arrival_deltas(
    *,
    service_name: str,
    flow_name: str,
    target: EvaluationTarget,
    prediction_times: List[Tuple[int, int]],
    collector: ScalarsCollector,
    output_root: Path,
    payloads_by_time: Optional[Mapping[Tuple[int, int], ArrivalDeltaPayload]] = None,
) -> None:
    """Evaluate arrival-rate deltas per prediction time.

    Parameters
    ----------
    service_name
        Service being evaluated.
    flow_name
        Flow target name.
    target
        Evaluation target metadata.
    prediction_times
        Prediction times requested for evaluation.
    collector
        Scalar collector updated in place.
    output_root
        Root directory for plot outputs.
    payloads_by_time
        Optional mapping ``prediction_time -> ArrivalDeltaPayload``.
        Missing times are recorded as non-evaluated rows.
    """
    service_dir = output_root / "services" / service_name.replace("/", "_")
    service_dir.mkdir(parents=True, exist_ok=True)
    by_time = payloads_by_time or {}

    for prediction_time in prediction_times:
        payload = by_time.get(prediction_time)
        if payload is None:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No flow input provided for this service/time",
                service_name=service_name,
            )
            continue

        fig = plot_arrival_deltas(
            df=payload.df,
            prediction_time=prediction_time,
            snapshot_dates=payload.snapshot_dates,
            prediction_window=payload.prediction_window,
            yta_time_interval=payload.yta_time_interval,
            suptitle=f"{flow_name} — {service_name}",
            media_file_path=service_dir,
            file_name=(
                f"{flow_name}_{target.component}_{_format_prediction_time(prediction_time)}"
                "_deltas.png"
            ),
            return_figure=True,
        )
        _close_fig(fig)

        n_snapshots = int(len(payload.snapshot_dates))
        collector.record(
            flow=flow_name,
            service=service_name,
            component=target.component,
            prediction_time=_format_prediction_time(prediction_time),
            flow_type=target.flow_type,
            aspirational=target.aspirational,
            evaluated=True,
            metrics={
                "n_snapshots": n_snapshots,
                "reliable": n_snapshots
                >= RELIABILITY_THRESHOLDS["distribution_snapshots"],
                "reliability_threshold": RELIABILITY_THRESHOLDS[
                    "distribution_snapshots"
                ],
                "reliability_basis": "snapshots",
            },
        )


def evaluate_survival_curve(
    *,
    service_name: str,
    flow_name: str,
    target: EvaluationTarget,
    prediction_times: List[Tuple[int, int]],
    collector: ScalarsCollector,
    output_root: Path,
    payloads_by_time: Optional[Mapping[Tuple[int, int], SurvivalCurvePayload]] = None,
) -> None:
    """Evaluate train-vs-test survival curve consistency.

    Parameters
    ----------
    service_name
        Service being evaluated.
    flow_name
        Flow target name.
    target
        Evaluation target metadata.
    prediction_times
        Prediction times requested for evaluation.
    collector
        Scalar collector updated in place.
    output_root
        Root directory for plot outputs.
    payloads_by_time
        Optional mapping ``prediction_time -> SurvivalCurvePayload``.
        Missing times are recorded as non-evaluated rows.
    """
    service_dir = output_root / "services" / service_name.replace("/", "_")
    service_dir.mkdir(parents=True, exist_ok=True)
    by_time = payloads_by_time or {}

    for prediction_time in prediction_times:
        payload = by_time.get(prediction_time)
        if payload is None:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No flow input provided for this service/time",
                service_name=service_name,
            )
            continue

        fig = plot_admission_time_survival_curve(
            df=[payload.train_df, payload.test_df],
            labels=["train", "test"],
            start_time_col=payload.start_time_col,
            end_time_col=payload.end_time_col,
            media_file_path=service_dir,
            file_name=f"{flow_name}_{target.component}_survival.png",
            return_figure=True,
        )
        _close_fig(fig)
        collector.record(
            flow=flow_name,
            service=service_name,
            component=target.component,
            prediction_time=_format_prediction_time(prediction_time),
            flow_type=target.flow_type,
            aspirational=target.aspirational,
            evaluated=True,
        )


def _extract_models(models: Any) -> List[Any]:
    if models is None:
        return []
    if isinstance(models, dict):
        return list(models.values())
    return list(models)


def _models_by_time(models: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for model in models:
        pred_time = model.training_results.prediction_time
        out[_format_prediction_time(pred_time)] = model
    return out


def _metrics_from_training_artifacts(trained_model: Any) -> Optional[Dict[str, Any]]:
    """Extract classifier metrics from training artifacts and flatten reliability."""
    training_results = getattr(trained_model, "training_results", None)
    if training_results is None:
        return None

    selected = getattr(trained_model, "selected_eval_metrics", {}) or {}
    if not selected:
        selected = getattr(training_results, "selected_eval_metrics", {}) or {}
    logloss = selected.get("log_loss")
    auroc = selected.get("auroc")
    auprc = selected.get("auprc")
    n_samples = selected.get("n_samples")
    n_positive = selected.get("n_positive_cases")

    if logloss is None or auroc is None or auprc is None:
        test_results = getattr(training_results, "test_results", None)
        if test_results:
            logloss = test_results.get("test_logloss", logloss)
            auroc = test_results.get("test_auc", auroc)
            auprc = test_results.get("test_auprc", auprc)

    if logloss is None or auroc is None or auprc is None:
        training_info = getattr(training_results, "training_info", {}) or {}
        cv_trials = training_info.get("cv_trials", [])
        if cv_trials:

            def _trial_cv_results(trial: Any) -> Dict[str, Any]:
                if hasattr(trial, "cv_results"):
                    return trial.cv_results
                if isinstance(trial, dict):
                    return trial.get("cv_results", {})
                return {}

            best_trial = min(
                cv_trials,
                key=lambda t: _trial_cv_results(t).get(
                    "valid_logloss", float("inf")
                ),
            )
            cv = _trial_cv_results(best_trial)
            logloss = cv.get("valid_logloss", logloss)
            auroc = cv.get("valid_auc", auroc)
            auprc = cv.get("valid_auprc", auprc)

            dataset_info = training_info.get("dataset_info", {})
            split_sizes = dataset_info.get("train_valid_test_set_no", {})
            split_positives = dataset_info.get(
                "train_valid_test_positive_cases", {}
            )
            if n_samples is None:
                n_samples = split_sizes.get("valid_set_no")
            if n_positive is None:
                n_positive = split_positives.get("valid_positive_cases")
            if n_positive is None:
                split_balances = dataset_info.get(
                    "train_valid_test_class_balance", {}
                )
                valid_balance = split_balances.get("y_valid_class_balance")
                if valid_balance is not None and n_samples is not None:
                    positive_rate = valid_balance.get(1, valid_balance.get("1"))
                    if positive_rate is not None:
                        n_positive = int(
                            round(float(n_samples) * float(positive_rate))
                        )

    if logloss is None or auroc is None or auprc is None:
        return None

    threshold = RELIABILITY_THRESHOLDS["classifier_positive_cases"]
    reliable = None
    if n_positive is not None:
        reliable = int(n_positive) >= threshold

    return {
        "log_loss": float(logloss),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_samples": int(n_samples) if n_samples is not None else None,
        "n_positive_cases": int(n_positive) if n_positive is not None else None,
        "reliable": reliable,
        "reliability_threshold": threshold,
        "reliability_basis": "positive_cases",
    }


def evaluate_classifier(
    *,
    flow_name: str,
    target: EvaluationTarget,
    prediction_times: List[Tuple[int, int]],
    collector: ScalarsCollector,
    output_root: Path,
    classifier_input: Optional[ClassifierInput] = None,
) -> None:
    """Evaluate one classifier target and write outputs.

    Parameters
    ----------
    flow_name
        Classifier target name.
    target
        Evaluation target metadata.
    prediction_times
        Prediction times requested for evaluation.
    collector
        Scalar collector updated in place.
    output_root
        Root directory for plot outputs.
    classifier_input
        Optional classifier payload with trained models and evaluation dataframe.
        If absent, rows are recorded as non-evaluated.
    """
    class_dir = output_root / "classifiers" / flow_name.replace("/", "_")
    class_dir.mkdir(parents=True, exist_ok=True)

    if classifier_input is None:
        for prediction_time in prediction_times:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No classifier input provided",
                service_name=None,
            )
        return

    models = _extract_models(classifier_input.trained_models)
    if not models:
        for prediction_time in prediction_times:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="Classifier input must include trained_models",
                service_name=None,
            )
        return

    if classifier_input.visits_df is not None:
        fig = plot_madcap(
            trained_models=models,
            test_visits=classifier_input.visits_df,
            media_file_path=class_dir,
            file_name="madcap.png",
            return_figure=True,
            label_col=classifier_input.label_col,
        )
        _close_fig(fig)
        fig = plot_estimated_probabilities(
            trained_models=models,
            test_visits=classifier_input.visits_df,
            media_file_path=class_dir,
            file_name="discrimination.png",
            return_figure=True,
            label_col=classifier_input.label_col,
        )
        _close_fig(fig)
        fig = plot_calibration(
            trained_models=models,
            test_visits=classifier_input.visits_df,
            media_file_path=class_dir,
            file_name="calibration.png",
            return_figure=True,
            label_col=classifier_input.label_col,
        )
        _close_fig(fig)

    models_for_times = _models_by_time(models)
    for prediction_time in prediction_times:
        prediction_time_key = _format_prediction_time(prediction_time)
        model = models_for_times.get(prediction_time_key)
        if model is None:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No trained model available for this prediction time",
                service_name=None,
            )
            continue

        metrics = _metrics_from_training_artifacts(model)
        if metrics is None:
            _record_not_evaluated(
                collector,
                flow_name=flow_name,
                target=target,
                prediction_time=prediction_time,
                reason="No compatible stored classifier metrics found in model artifacts",
                service_name=None,
            )
            continue

        collector.record(
            flow=flow_name,
            service=None,
            component=target.component,
            prediction_time=prediction_time_key,
            flow_type=target.flow_type,
            aspirational=target.aspirational,
            evaluated=True,
            metrics=metrics,
        )


MODE_HANDLERS = {
    "distribution": evaluate_distribution,
    "arrival_deltas": evaluate_arrival_deltas,
    "survival_curve": evaluate_survival_curve,
    "aspirational_skip": evaluate_aspirational_skip,
    "classifier": evaluate_classifier,
}
