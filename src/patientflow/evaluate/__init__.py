"""Evaluate patient flow model predictions.

This package provides a structured interface for evaluation workflows:

- typed input payloads and target definitions
- builders for assembling evaluation inputs
- per-mode handlers and a top-level runner
- flat scalar output helpers
- adapters for plotting-ready distribution dictionaries

The symbols re-exported here are the preferred import surface for
notebooks and application code.
"""

from patientflow.evaluate.adapters import (
    from_legacy_prob_dist_dict,
    to_legacy_prob_dist_dict_all,
)
from patientflow.evaluate.builder import EvaluationInputsBuilder
from patientflow.evaluate.constants import RELIABILITY_THRESHOLDS
from patientflow.evaluate.handlers import MODE_HANDLERS, evaluate_distribution
from patientflow.evaluate.legacy_api import calc_mae_mpe, calculate_results
from patientflow.evaluate.observations import (
    count_observed,
    count_observed_at_some_point,
    count_observed_in_window,
)
from patientflow.evaluate.runner import run_evaluation
from patientflow.evaluate.scalars import ScalarsCollector, default_scalars_meta
from patientflow.evaluate.targets import (
    convert_legacy_target,
    convert_legacy_targets,
    get_default_evaluation_targets,
)
from patientflow.evaluate.types import (
    ArrivalDeltaPayload,
    ClassifierInput,
    EvaluationInputs,
    EvaluationTarget,
    FlowInputPayload,
    OBSERVATION_MODES,
    SnapshotResult,
    SurvivalCurvePayload,
)

__all__ = [
    "ArrivalDeltaPayload",
    "ClassifierInput",
    "EvaluationInputs",
    "EvaluationTarget",
    "FlowInputPayload",
    "OBSERVATION_MODES",
    "SnapshotResult",
    "SurvivalCurvePayload",
    "convert_legacy_target",
    "convert_legacy_targets",
    "calc_mae_mpe",
    "calculate_results",
    "count_observed",
    "count_observed_at_some_point",
    "count_observed_in_window",
    "default_scalars_meta",
    "EvaluationInputsBuilder",
    "get_default_evaluation_targets",
    "from_legacy_prob_dist_dict",
    "MODE_HANDLERS",
    "RELIABILITY_THRESHOLDS",
    "evaluate_distribution",
    "run_evaluation",
    "ScalarsCollector",
    "to_legacy_prob_dist_dict_all",
]
