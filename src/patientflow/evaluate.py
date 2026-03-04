"""Evaluation orchestration and scalar utilities for patient flow models.

This module provides:

1. Backwards-compatible scalar helpers used by existing notebooks:
   ``calculate_results`` and ``calc_mae_mpe``.
2. A target-driven evaluation runner (``run_evaluation``) that writes
   plots and scalar summaries to a structured output directory.
"""

from typing import Dict, List, Any, Union, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import re
import matplotlib.pyplot as plt

from patientflow.load import get_model_key
from patientflow.viz.madcap import plot_madcap
from patientflow.viz.estimated_probabilities import plot_estimated_probabilities
from patientflow.viz.calibration import plot_calibration
from patientflow.viz.epudd import plot_epudd
from patientflow.viz.observed_against_expected import plot_deltas, plot_arrival_deltas
from patientflow.viz.survival_curve import plot_admission_time_survival_curve
from patientflow.predict.types import FlowSelection


def calculate_results(
    expected_values: List[Union[int, float]], observed_values: List[float]
) -> Dict[str, Union[List[Union[int, float]], float]]:
    """Calculate evaluation metrics based on expected and observed values.

    Parameters
    ----------
    expected_values : List[Union[int, float]]
        List of expected values.
    observed_values : List[float]
        List of observed values.

    Returns
    -------
    Dict[str, Union[List[Union[int, float]], float]]
        Dictionary containing:
        - expected : List[Union[int, float]]
            Original expected values
        - observed : List[float]
            Original observed values
        - mae : float
            Mean Absolute Error
        - mpe : float
            Mean Percentage Error
    """
    expected_array: np.ndarray = np.array(expected_values)
    observed_array: np.ndarray = np.array(observed_values)

    if len(expected_array) == 0 or len(observed_array) == 0:
        return {
            "expected": expected_values,
            "observed": observed_values,
            "mae": 0.0,
            "mpe": 0.0,
        }

    absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
    mae: float = float(np.mean(absolute_errors)) if len(absolute_errors) > 0 else 0.0

    non_zero_mask: np.ndarray = observed_array != 0
    filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
    filtered_observed_array: np.ndarray = observed_array[non_zero_mask]

    mpe: float = 0.0
    if len(filtered_absolute_errors) > 0 and len(filtered_observed_array) > 0:
        percentage_errors: np.ndarray = (
            filtered_absolute_errors / filtered_observed_array * 100
        )
        mpe = float(np.mean(percentage_errors))

    return {
        "expected": expected_values,
        "observed": observed_values,
        "mae": mae,
        "mpe": mpe,
    }


def calc_mae_mpe(
    prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]],
    use_most_probable: bool = False,
) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """Calculate MAE and MPE for all prediction times in the given probability distribution dictionary.

    Parameters
    ----------
    prob_dist_dict_all : Dict[Any, Dict[Any, Dict[str, Any]]]
        Nested dictionary containing probability distributions.
    use_most_probable : bool, optional
        Whether to use the most probable value or mathematical expectation of the distribution.
        Default is False.

    Returns
    -------
    Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]
        Dictionary of results sorted by prediction time, containing:
        - expected : List[Union[int, float]]
            Expected values for each prediction
        - observed : List[float]
            Observed values for each prediction
        - mae : float
            Mean Absolute Error
        - mpe : float
            Mean Percentage Error
    """
    # Create temporary results dictionary
    unsorted_results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

    # Process results as before
    for _prediction_time in prob_dist_dict_all.keys():
        expected_values: List[Union[int, float]] = []
        observed_values: List[float] = []

        for dt in prob_dist_dict_all[_prediction_time].keys():
            preds: Dict[str, Any] = prob_dist_dict_all[_prediction_time][dt]

            expected_value: Union[int, float] = (
                int(preds["agg_predicted"].idxmax().values[0])
                if use_most_probable
                else float(
                    np.dot(
                        preds["agg_predicted"].index,
                        preds["agg_predicted"].values.flatten(),
                    )
                )
            )

            observed_value: float = float(preds["agg_observed"])

            expected_values.append(expected_value)
            observed_values.append(observed_value)

        unsorted_results[_prediction_time] = calculate_results(
            expected_values, observed_values
        )

    # Sort results by prediction time
    def get_time_value(key: str) -> int:
        # Extract trailing HHMM/HMM token if present (e.g. admissions_1530, flow_x_930).
        match = re.search(r"(\d{3,4})$", key)
        if match:
            return int(match.group(1))
        return 99999

    # Create sorted dictionary
    sorted_results = dict(
        sorted(unsorted_results.items(), key=lambda x: get_time_value(x[0]))
    )

    return sorted_results


@dataclass(frozen=True)
class EvaluationTarget:
    """Configuration for evaluating one named target.

    Parameters
    ----------
    name : str
        Stable target name used in output paths and scalar keys.
    flow_type : str
        Semantic type of the target (for example ``"classifier"``, ``"pmf"``,
        ``"poisson"``, or ``"special"``).
    aspirational : bool
        Whether the target uses aspirational assumptions and should therefore
        skip observed-vs-predicted diagnostics.
    components : list of str, optional
        Explicit components to evaluate from ``{"arrivals", "departures",
        "net_flow"}``. If ``None``, components are inferred from
        ``flow_selection``.
    flow_selection : FlowSelection, optional
        Flow-family selector used to infer components and document intent.
    evaluation_mode : str
        Evaluation strategy name. Supported values are currently
        ``"classifier"``, ``"distribution"``, ``"arrival_deltas"``,
        ``"survival_curve"``, and ``"aspirational_skip"``.
    """

    name: str
    flow_type: str
    aspirational: bool
    components: Optional[List[str]]
    flow_selection: Optional[FlowSelection]
    evaluation_mode: str


RELIABILITY_THRESHOLDS: Dict[str, int] = {
    "classifier_positive_cases": 50,
    "distribution_snapshots": 30,
    "transition_transfers": 10,
}


def get_default_evaluation_targets() -> Dict[str, EvaluationTarget]:
    """Return default evaluation target registry.

    Returns
    -------
    Dict[str, EvaluationTarget]
        Mapping of target name to evaluation target configuration.

    Notes
    -----
    The default targets encode the currently supported flows and diagnostic
    modes for the evaluation pipeline.
    """
    return {
        "ed_current_admission_classifier": EvaluationTarget(
            name="ed_current_admission_classifier",
            flow_type="classifier",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="classifier",
        ),
        "discharge_classifier": EvaluationTarget(
            name="discharge_classifier",
            flow_type="classifier",
            aspirational=False,
            components=["departures"],
            flow_selection=None,
            evaluation_mode="classifier",
        ),
        "ed_current_beds": EvaluationTarget(
            name="ed_current_beds",
            flow_type="pmf",
            aspirational=False,
            components=None,
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=False,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
            ),
            evaluation_mode="distribution",
        ),
        "ed_current_window_prob": EvaluationTarget(
            name="ed_current_window_prob",
            flow_type="special",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="survival_curve",
        ),
        "ed_current_window_prob_aspirational": EvaluationTarget(
            name="ed_current_window_prob_aspirational",
            flow_type="special",
            aspirational=True,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="aspirational_skip",
        ),
        "ed_current_window_beds": EvaluationTarget(
            name="ed_current_window_beds",
            flow_type="pmf",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="distribution",
        ),
        "ed_current_window_beds_aspirational": EvaluationTarget(
            name="ed_current_window_beds_aspirational",
            flow_type="pmf",
            aspirational=True,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="aspirational_skip",
        ),
        "ed_yta_arrival_rates": EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="arrival_deltas",
        ),
        "ed_yta_beds": EvaluationTarget(
            name="ed_yta_beds",
            flow_type="poisson",
            aspirational=False,
            components=None,
            flow_selection=FlowSelection.custom(
                include_ed_current=False,
                include_ed_yta=True,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
            ),
            evaluation_mode="distribution",
        ),
        "ed_yta_beds_aspirational": EvaluationTarget(
            name="ed_yta_beds_aspirational",
            flow_type="poisson",
            aspirational=True,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="aspirational_skip",
        ),
        "non_ed_yta_beds": EvaluationTarget(
            name="non_ed_yta_beds",
            flow_type="poisson",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="distribution",
        ),
        "elective_yta_beds": EvaluationTarget(
            name="elective_yta_beds",
            flow_type="poisson",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="distribution",
        ),
        "discharge_emergency": EvaluationTarget(
            name="discharge_emergency",
            flow_type="pmf",
            aspirational=False,
            components=["departures"],
            flow_selection=None,
            evaluation_mode="distribution",
        ),
        "discharge_elective": EvaluationTarget(
            name="discharge_elective",
            flow_type="pmf",
            aspirational=False,
            components=["departures"],
            flow_selection=None,
            evaluation_mode="distribution",
        ),
        "combined_emergency_arrivals": EvaluationTarget(
            name="combined_emergency_arrivals",
            flow_type="pmf",
            aspirational=True,
            components=["arrivals"],
            flow_selection=FlowSelection.emergency_only(),
            evaluation_mode="aspirational_skip",
        ),
        "combined_elective_arrivals": EvaluationTarget(
            name="combined_elective_arrivals",
            flow_type="pmf",
            aspirational=False,
            components=["arrivals"],
            flow_selection=FlowSelection.elective_only(),
            evaluation_mode="distribution",
        ),
        "combined_net_emergency": EvaluationTarget(
            name="combined_net_emergency",
            flow_type="pmf",
            aspirational=True,
            components=["net_flow"],
            flow_selection=FlowSelection.emergency_only(),
            evaluation_mode="aspirational_skip",
        ),
        "combined_net_elective": EvaluationTarget(
            name="combined_net_elective",
            flow_type="pmf",
            aspirational=False,
            components=["net_flow"],
            flow_selection=FlowSelection.elective_only(),
            evaluation_mode="distribution",
        ),
    }


def _format_prediction_time(prediction_time: Tuple[int, int]) -> str:
    hour, minute = prediction_time
    return f"{hour:02d}{minute:02d}"


def _extract_models(
    models: Union[List[Any], Dict[str, Any], None],
) -> List[Any]:
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


def _close_fig(fig: Any) -> None:
    if fig is not None:
        plt.close(fig)


def _safe_path_fragment(value: str) -> str:
    return value.replace("/", "_")


def _infer_components_from_flow_selection(
    flow_selection: FlowSelection,
) -> List[str]:
    components: List[str] = []
    has_inflow = any(
        [
            flow_selection.include_ed_current,
            flow_selection.include_ed_yta,
            flow_selection.include_non_ed_yta,
            flow_selection.include_elective_yta,
            flow_selection.include_transfers_in,
        ]
    )
    if has_inflow:
        components.append("arrivals")
    if flow_selection.include_departures:
        components.append("departures")
    if has_inflow and flow_selection.include_departures:
        components.append("net_flow")
    return components


def _resolve_target_components(evaluation_target: EvaluationTarget) -> List[str]:
    if evaluation_target.components:
        return evaluation_target.components
    if evaluation_target.flow_selection is not None:
        return _infer_components_from_flow_selection(evaluation_target.flow_selection)
    return []


def _upsert_scalar_metadata(
    scalars: Dict[str, Any],
    flow_name: str,
    evaluation_target: EvaluationTarget,
    prediction_time_key: str,
    evaluated: bool,
    reason: Optional[str] = None,
    service_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    component_name: Optional[str] = None,
) -> None:
    flow_node = scalars.setdefault(flow_name, {})
    flow_node["flow_type"] = evaluation_target.flow_type
    flow_node["aspirational"] = evaluation_target.aspirational
    flow_node["evaluation_mode"] = evaluation_target.evaluation_mode
    if service_name is None:
        component_node = flow_node.setdefault("components", {}).setdefault(
            component_name or "unspecified", {}
        )
        by_time = component_node.setdefault("prediction_times", {})
    else:
        service_node = flow_node.setdefault("services", {}).setdefault(service_name, {})
        component_node = service_node.setdefault("components", {}).setdefault(
            component_name or "unspecified", {}
        )
        by_time = component_node.setdefault("prediction_times", {})
    time_node = by_time.setdefault(prediction_time_key, {})
    time_node["evaluated"] = evaluated
    if reason:
        time_node["reason"] = reason
    if metrics:
        time_node.update(metrics)


def _metrics_from_training_artifacts(trained_model: Any) -> Optional[Dict[str, Any]]:
    training_results = getattr(trained_model, "training_results", None)
    if training_results is None:
        return None

    selected = getattr(trained_model, "selected_eval_metrics", {}) or {}
    # Backward compatibility for models created before top-level selected_eval_metrics.
    if not selected:
        selected = getattr(training_results, "selected_eval_metrics", {}) or {}
    logloss = selected.get("log_loss")
    auroc = selected.get("auroc")
    auprc = selected.get("auprc")
    n_samples = selected.get("n_samples")
    n_positive = selected.get("n_positive_cases")

    # Backward compatibility for models trained before selected_eval_metrics existed.
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
                key=lambda t: _trial_cv_results(t).get("valid_logloss", float("inf")),
            )
            cv = _trial_cv_results(best_trial)
            logloss = cv.get("valid_logloss", logloss)
            auroc = cv.get("valid_auc", auroc)
            auprc = cv.get("valid_auprc", auprc)

            dataset_info = training_info.get("dataset_info", {})
            split_sizes = dataset_info.get("train_valid_test_set_no", {})
            split_positives = dataset_info.get("train_valid_test_positive_cases", {})
            if n_samples is None:
                n_samples = split_sizes.get("valid_set_no")
            if n_positive is None:
                n_positive = split_positives.get("valid_positive_cases")
            if n_positive is None:
                split_balances = dataset_info.get("train_valid_test_class_balance", {})
                valid_balance = split_balances.get("y_valid_class_balance")
                if valid_balance is not None and n_samples is not None:
                    positive_rate = valid_balance.get(1, valid_balance.get("1"))
                    if positive_rate is not None:
                        n_positive = int(round(float(n_samples) * float(positive_rate)))

    if logloss is None or auroc is None or auprc is None:
        return None

    reliability = {
        "is_reliable": None,
        "threshold": RELIABILITY_THRESHOLDS["classifier_positive_cases"],
        "basis": "positive_cases",
    }
    if n_positive is not None:
        reliability["is_reliable"] = (
            int(n_positive) >= RELIABILITY_THRESHOLDS["classifier_positive_cases"]
        )

    return {
        "log_loss": float(logloss),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_samples": int(n_samples) if n_samples is not None else None,
        "n_positive_cases": int(n_positive) if n_positive is not None else None,
        "reliability": reliability,
    }


def evaluate_classifier(
    classifier_name: str,
    prediction_times: List[Tuple[int, int]],
    scalars: Dict[str, Any],
    evaluation_target: EvaluationTarget,
    output_root: Path,
    classifier_input: Optional[Dict[str, Any]] = None,
) -> None:
    """Evaluate one classifier target and write outputs.

    Parameters
    ----------
    classifier_name : str
        Classifier target name used in outputs and scalar keys.
    prediction_times : list
        Prediction times to evaluate.
    scalars : dict
        Mutable scalar output structure updated in-place.
    evaluation_target : EvaluationTarget
        Target metadata controlling evaluation behaviour.
    output_root : pathlib.Path
        Root output directory for this run.
    classifier_input : dict, optional
        Input payload containing ``trained_models`` and any plotting data
        required by the visual diagnostics.
    """
    class_dir = output_root / "classifiers" / _safe_path_fragment(classifier_name)
    class_dir.mkdir(parents=True, exist_ok=True)

    if not classifier_input:
        for prediction_time in prediction_times:
            prediction_time_key = _format_prediction_time(prediction_time)
            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=classifier_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason="No classifier input provided",
                component_name="classifier",
            )
        return

    models = _extract_models(classifier_input.get("trained_models"))
    visits_df = classifier_input.get("visits_df")
    label_col = classifier_input.get("label_col", "is_admitted")
    if not models:
        for prediction_time in prediction_times:
            prediction_time_key = _format_prediction_time(prediction_time)
            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=classifier_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason="Classifier input must include trained_models",
                component_name="classifier",
            )
        return

    if visits_df is not None:
        fig = plot_madcap(
            trained_models=models,
            test_visits=visits_df,
            media_file_path=class_dir,
            file_name="madcap.png",
            return_figure=True,
            label_col=label_col,
        )
        _close_fig(fig)
        fig = plot_estimated_probabilities(
            trained_models=models,
            test_visits=visits_df,
            media_file_path=class_dir,
            file_name="discrimination.png",
            return_figure=True,
            label_col=label_col,
        )
        _close_fig(fig)
        fig = plot_calibration(
            trained_models=models,
            test_visits=visits_df,
            media_file_path=class_dir,
            file_name="calibration.png",
            return_figure=True,
            label_col=label_col,
        )
        _close_fig(fig)

    models_by_time = _models_by_time(models)
    for prediction_time in prediction_times:
        prediction_time_key = _format_prediction_time(prediction_time)
        model = models_by_time.get(prediction_time_key)
        if model is None:
            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=classifier_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason="No trained model available for this prediction time",
                component_name="classifier",
            )
            continue

        metrics = _metrics_from_training_artifacts(model)
        if metrics is None:
            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=classifier_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason="No compatible stored classifier metrics found in model artifacts",
                component_name="classifier",
            )
            continue
        _upsert_scalar_metadata(
            scalars=scalars,
            flow_name=classifier_name,
            evaluation_target=evaluation_target,
            prediction_time_key=prediction_time_key,
            evaluated=True,
            metrics=metrics,
            component_name="classifier",
        )


def evaluate_flow(
    service_name: str,
    flow_name: str,
    prediction_times: List[Tuple[int, int]],
    scalars: Dict[str, Any],
    evaluation_target: EvaluationTarget,
    output_root: Path,
    flow_input: Optional[Dict[Tuple[int, int], Any]] = None,
) -> None:
    """Evaluate one non-classifier target for one service.

    Parameters
    ----------
    service_name : str
        Hospital service identifier (for example ``"medical"``).
    flow_name : str
        Flow target name.
    prediction_times : list
        Prediction times to evaluate.
    scalars : dict
        Mutable scalar output structure updated in-place.
    evaluation_target : EvaluationTarget
        Target metadata controlling components and diagnostics.
    output_root : pathlib.Path
        Root output directory for this run.
    flow_input : dict, optional
        Mapping from prediction time to flow payload for this service/target.
    """
    service_dir = output_root / "services" / _safe_path_fragment(service_name)
    service_dir.mkdir(parents=True, exist_ok=True)
    by_time_input = flow_input or {}
    target_components = _resolve_target_components(evaluation_target)

    for prediction_time in prediction_times:
        prediction_time_key = _format_prediction_time(prediction_time)
        if not target_components:
            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=flow_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason="No explicit or implied components for this target",
                service_name=service_name,
                component_name="unspecified",
            )
            continue
        if evaluation_target.aspirational:
            for component_name in target_components:
                _upsert_scalar_metadata(
                    scalars=scalars,
                    flow_name=flow_name,
                    evaluation_target=evaluation_target,
                    prediction_time_key=prediction_time_key,
                    evaluated=False,
                    reason="Aspirational flow: observed-vs-predicted diagnostics skipped",
                    service_name=service_name,
                    component_name=component_name,
                )

    if not target_components or evaluation_target.aspirational:
        return

    for component_name in target_components:
        if evaluation_target.evaluation_mode == "distribution":
            prob_dist_dict_all: Dict[str, Dict[Any, Dict[str, Any]]] = {}

            for prediction_time in prediction_times:
                prediction_time_key = _format_prediction_time(prediction_time)
                payload = by_time_input.get(prediction_time)
                if payload is None:
                    _upsert_scalar_metadata(
                        scalars=scalars,
                        flow_name=flow_name,
                        evaluation_target=evaluation_target,
                        prediction_time_key=prediction_time_key,
                        evaluated=False,
                        reason="No flow input provided for this service/time",
                        service_name=service_name,
                        component_name=component_name,
                    )
                    continue

                component_payload = payload
                if (
                    isinstance(payload, dict)
                    and component_name in payload
                    and isinstance(payload[component_name], dict)
                ):
                    component_payload = payload[component_name]

                model_key = get_model_key(flow_name, prediction_time)
                prob_dist_dict_all[model_key] = component_payload

            if not prob_dist_dict_all:
                continue

            available_prediction_times = [
                pt
                for pt in prediction_times
                if get_model_key(flow_name, pt) in prob_dist_dict_all
            ]

            fig = plot_epudd(
                prediction_times=available_prediction_times,
                prob_dist_dict_all=prob_dist_dict_all,
                model_name=flow_name,
                media_file_path=service_dir,
                file_name=f"{flow_name}_{component_name}_epudd.png",
                return_figure=True,
            )
            _close_fig(fig)

            results = calc_mae_mpe(prob_dist_dict_all)
            fig = plot_deltas(
                results1=results,
                media_file_path=service_dir,
                file_name=f"{flow_name}_{component_name}_obs_exp.png",
                return_figure=True,
            )
            _close_fig(fig)

            for prediction_time in prediction_times:
                prediction_time_key = _format_prediction_time(prediction_time)
                model_key = get_model_key(flow_name, prediction_time)
                if model_key not in results:
                    continue
                component_payload = prob_dist_dict_all[model_key]
                result_for_time = results[model_key]
                n_snapshots = int(len(component_payload))
                mae_val = result_for_time["mae"]
                mpe_val = result_for_time["mpe"]
                assert isinstance(mae_val, (int, float))
                assert isinstance(mpe_val, (int, float))
                metrics = {
                    "mae": float(mae_val),
                    "mpe": float(mpe_val),
                    "n_snapshots": n_snapshots,
                    "reliability": {
                        "is_reliable": n_snapshots
                        >= RELIABILITY_THRESHOLDS["distribution_snapshots"],
                        "threshold": RELIABILITY_THRESHOLDS["distribution_snapshots"],
                        "basis": "snapshots",
                    },
                }
                _upsert_scalar_metadata(
                    scalars=scalars,
                    flow_name=flow_name,
                    evaluation_target=evaluation_target,
                    prediction_time_key=prediction_time_key,
                    evaluated=True,
                    service_name=service_name,
                    metrics=metrics,
                    component_name=component_name,
                )
            continue

        for prediction_time in prediction_times:
            prediction_time_key = _format_prediction_time(prediction_time)
            payload = by_time_input.get(prediction_time)
            if payload is None:
                _upsert_scalar_metadata(
                    scalars=scalars,
                    flow_name=flow_name,
                    evaluation_target=evaluation_target,
                    prediction_time_key=prediction_time_key,
                    evaluated=False,
                    reason="No flow input provided for this service/time",
                    service_name=service_name,
                    component_name=component_name,
                )
                continue

            component_payload = payload
            if (
                isinstance(payload, dict)
                and component_name in payload
                and isinstance(payload[component_name], dict)
            ):
                component_payload = payload[component_name]

            if evaluation_target.evaluation_mode == "arrival_deltas":
                required_delta_keys = ("df", "snapshot_dates", "prediction_window")
                if not (
                    isinstance(component_payload, dict)
                    and all(key in component_payload for key in required_delta_keys)
                ):
                    _upsert_scalar_metadata(
                        scalars=scalars,
                        flow_name=flow_name,
                        evaluation_target=evaluation_target,
                        prediction_time_key=prediction_time_key,
                        evaluated=False,
                        reason=(
                            "Arrival delta flow input must include df, snapshot_dates, "
                            "and prediction_window"
                        ),
                        service_name=service_name,
                        component_name=component_name,
                    )
                    continue
                fig = plot_arrival_deltas(
                    df=component_payload["df"],
                    prediction_time=prediction_time,
                    snapshot_dates=component_payload["snapshot_dates"],
                    prediction_window=component_payload["prediction_window"],
                    yta_time_interval=component_payload.get(
                        "yta_time_interval", timedelta(minutes=15)
                    ),
                    suptitle=f"{flow_name} \u2014 {service_name}",
                    media_file_path=service_dir,
                    file_name=f"{flow_name}_{component_name}_{prediction_time_key}_deltas.png",
                    return_figure=True,
                )
                _close_fig(fig)
                _upsert_scalar_metadata(
                    scalars=scalars,
                    flow_name=flow_name,
                    evaluation_target=evaluation_target,
                    prediction_time_key=prediction_time_key,
                    evaluated=True,
                    service_name=service_name,
                    component_name=component_name,
                    metrics={
                        "n_snapshots": int(len(component_payload["snapshot_dates"])),
                        "reliability": {
                            "is_reliable": int(len(component_payload["snapshot_dates"]))
                            >= RELIABILITY_THRESHOLDS["distribution_snapshots"],
                            "threshold": RELIABILITY_THRESHOLDS[
                                "distribution_snapshots"
                            ],
                            "basis": "snapshots",
                        },
                    },
                )
                continue

            if evaluation_target.evaluation_mode == "survival_curve":
                required_survival_keys = ("train_df", "test_df")
                if not (
                    isinstance(component_payload, dict)
                    and all(key in component_payload for key in required_survival_keys)
                ):
                    _upsert_scalar_metadata(
                        scalars=scalars,
                        flow_name=flow_name,
                        evaluation_target=evaluation_target,
                        prediction_time_key=prediction_time_key,
                        evaluated=False,
                        reason=(
                            "Survival curve flow input must include train_df and test_df"
                        ),
                        service_name=service_name,
                        component_name=component_name,
                    )
                    continue
                fig = plot_admission_time_survival_curve(
                    df=[component_payload["train_df"], component_payload["test_df"]],
                    labels=["train", "test"],
                    start_time_col=component_payload.get(
                        "start_time_col", "arrival_datetime"
                    ),
                    end_time_col=component_payload.get(
                        "end_time_col", "departure_datetime"
                    ),
                    media_file_path=service_dir,
                    file_name=f"{flow_name}_{component_name}_survival.png",
                    return_figure=True,
                )
                _close_fig(fig)
                _upsert_scalar_metadata(
                    scalars=scalars,
                    flow_name=flow_name,
                    evaluation_target=evaluation_target,
                    prediction_time_key=prediction_time_key,
                    evaluated=True,
                    service_name=service_name,
                    component_name=component_name,
                )
                continue

            _upsert_scalar_metadata(
                scalars=scalars,
                flow_name=flow_name,
                evaluation_target=evaluation_target,
                prediction_time_key=prediction_time_key,
                evaluated=False,
                reason=(
                    f"Unsupported evaluation mode: {evaluation_target.evaluation_mode}"
                ),
                service_name=service_name,
                component_name=component_name,
            )


def run_evaluation(
    output_root: Union[str, Path],
    prediction_times: Optional[List[Tuple[int, int]]] = None,
    config_path: Optional[Union[str, Path]] = None,
    run_label: Optional[str] = None,
    evaluation_targets: Optional[Dict[str, EvaluationTarget]] = None,
    classifier_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    flow_inputs_by_service: Optional[
        Dict[str, Dict[str, Dict[Tuple[int, int], Any]]]
    ] = None,
    services: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run evaluation targets and write plots/scalars to disk.

    Parameters
    ----------
    output_root : str or pathlib.Path
        Root directory under which the run folder is created.
    prediction_times : list
        Prediction times to evaluate. Must be provided explicitly.
    config_path : str or pathlib.Path, optional
        Optional config file copied to ``config.yaml`` in run output.
    run_label : str, optional
        Run folder name. If omitted, a UTC timestamp label is generated.
    evaluation_targets : dict, optional
        Mapping of target name to ``EvaluationTarget``. Defaults to
        ``get_default_evaluation_targets()``.
    classifier_inputs : dict, optional
        Classifier input payloads keyed by classifier target name.
    flow_inputs_by_service : dict, optional
        Explicit non-classifier input payloads keyed by
        ``service -> flow -> prediction_time``.
    services : list of str, optional
        Services to evaluate. If omitted, inferred from provided flow inputs.

    Returns
    -------
    dict
        Run summary containing output paths, counts, and evaluated times.

    Notes
    -----
    - Classifier inputs are passed via ``classifier_inputs`` keyed by classifier name.
    - Service flow inputs are passed in ``flow_inputs_by_service`` keyed by
      ``service -> flow -> prediction_time``.
    """
    if prediction_times is None or len(prediction_times) == 0:
        raise ValueError("prediction_times must be provided explicitly.")

    registry = evaluation_targets or get_default_evaluation_targets()
    times = prediction_times

    root = Path(output_root)
    effective_run_label = run_label or datetime.now(timezone.utc).strftime(
        "run_%Y%m%d_%H%M%S"
    )
    root = root / effective_run_label
    root.mkdir(parents=True, exist_ok=True)
    (root / "classifiers").mkdir(exist_ok=True)
    (root / "services").mkdir(exist_ok=True)

    if config_path is not None:
        source = Path(config_path)
        destination = root / "config.yaml"
        if source.exists():
            shutil.copy2(source, destination)

    scalars: Dict[str, Any] = {
        "_meta": {
            "schema_version": "phase2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reliability_thresholds": RELIABILITY_THRESHOLDS,
            "phase": 2,
        }
    }

    classifier_inputs = classifier_inputs or {}
    flow_inputs_by_service = flow_inputs_by_service or {}

    services_to_process = services or sorted(flow_inputs_by_service.keys())

    for flow_name, evaluation_target in registry.items():
        if evaluation_target.flow_type == "classifier":
            evaluate_classifier(
                classifier_name=flow_name,
                prediction_times=times,
                scalars=scalars,
                evaluation_target=evaluation_target,
                output_root=root,
                classifier_input=classifier_inputs.get(flow_name),
            )
        else:
            for service_name in services_to_process:
                service_flow_input = flow_inputs_by_service.get(service_name, {}).get(
                    flow_name
                )
                evaluate_flow(
                    service_name=service_name,
                    flow_name=flow_name,
                    prediction_times=times,
                    scalars=scalars,
                    evaluation_target=evaluation_target,
                    output_root=root,
                    flow_input=service_flow_input,
                )

    scalars_path = root / "scalars.json"
    scalars_path.write_text(json.dumps(scalars, indent=2, sort_keys=True))

    return {
        "output_root": str(root),
        "run_label": effective_run_label,
        "scalars_path": str(scalars_path),
        "n_flows": len(registry),
        "n_services": len(services_to_process),
        "prediction_times": [_format_prediction_time(pt) for pt in times],
    }
