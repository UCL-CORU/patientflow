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

from patientflow.evaluate import calc_mae_mpe
from patientflow.evaluation.adapters import to_legacy_prob_dist_dict_all
from patientflow.evaluation.constants import RELIABILITY_THRESHOLDS
from patientflow.evaluation.scalars import ScalarsCollector
from patientflow.evaluation.types import (
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
    """
    service_dir = output_root / "services" / service_name.replace("/", "_")
    service_dir.mkdir(parents=True, exist_ok=True)

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
    """Temporary compatibility path for classifier metrics extraction."""
    from patientflow.evaluate import _metrics_from_training_artifacts as _legacy_metrics

    metrics = _legacy_metrics(trained_model)
    if metrics is None:
        return None
    return _flatten_reliability(metrics)


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
