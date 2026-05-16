"""Tests for the patientflow.evaluate package (inputs, runner, handlers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from patientflow.evaluate.handlers import (
    _arrival_delta_suptitle,
    _classifier_diagnostics_suptitle,
    _classifier_quality_suptitle,
    _distribution_epudd_suptitle,
)
from patientflow.evaluate.inputs import (
    EVAL_SPLITS,
    EvaluationInputs,
    EvaluationInputsBuilder,
    EvaluationTarget,
    eval_split_label,
)
from patientflow.evaluate.runner import write_evaluation_run_manifest
from patientflow.predict.demand import FlowSelection


# --- eval_split: inputs and manifest ---


def test_eval_split_label():
    assert eval_split_label("valid") == "validation set"
    assert eval_split_label("test") == "test set"
    assert eval_split_label(None) == "evaluation cohort"
    assert eval_split_label("other") == "evaluation cohort"


def test_builder_defaults_eval_split_valid():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_times=[(6, 0)],
    )
    inputs = builder.build()
    assert inputs.eval_split == "valid"


def test_builder_set_eval_split_test():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_times=[(6, 0)],
        eval_split="test",
    ).set_eval_split("test")
    inputs = builder.build()
    assert inputs.eval_split == "test"


def test_builder_rejects_unknown_eval_split():
    with pytest.raises(ValueError, match="Unknown eval_split"):
        EvaluationInputsBuilder(eval_split="holdout")  # type: ignore[arg-type]

    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_times=[(6, 0)],
    )
    with pytest.raises(ValueError, match="Unknown eval_split"):
        builder.set_eval_split("train")  # type: ignore[arg-type]


def test_eval_splits_constant():
    assert EVAL_SPLITS == ("valid", "test")


def test_manifest_records_eval_split(tmp_path: Path):
    flow = FlowSelection.emergency_only()
    inputs = EvaluationInputs(
        flow_selection=flow,
        prediction_times=[(6, 0)],
        evaluation_targets=[],
        eval_split="test",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = write_evaluation_run_manifest(
        run_dir,
        output_root=tmp_path,
        run_name="run",
        inputs=inputs,
        project_config_path=tmp_path / "missing_config.yaml",
    )
    text = manifest_path.read_text(encoding="utf-8")
    assert "eval_split: test" in text


# --- eval_split: handler suptitles ---


def _classifier_probability_quality_target() -> EvaluationTarget:
    return EvaluationTarget(
        flow_name="ed_admissions_cls",
        name="t",
        flow_type="admissions",
        evaluation_mode="classifier_probability_quality",
        component="c",
        observation_mode="admitted_at_some_point",
    )


def test_classifier_quality_suptitle_uses_eval_split():
    title = _classifier_quality_suptitle(
        _classifier_probability_quality_target(),
        "discrimination",
        eval_split="test",
    )
    assert "(test set)" in title
    assert "validation" not in title


def test_distribution_epudd_suptitle_uses_eval_split():
    dist_target = EvaluationTarget(
        flow_name="ed_current_beds",
        name="d",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="epudd",
        observation_mode="admitted_at_some_point",
    )
    title = _distribution_epudd_suptitle(
        dist_target, "medical", eval_split="valid"
    )
    assert "(validation set)" in title


def test_classifier_diagnostics_suptitle_includes_cohort_and_clock():
    target = _classifier_probability_quality_target()
    target = EvaluationTarget(
        flow_name=target.flow_name,
        name=target.name,
        flow_type=target.flow_type,
        evaluation_mode="classifier_model_diagnostics",
        component="model_diagnostics",
        observation_mode=target.observation_mode,
    )
    title = _classifier_diagnostics_suptitle(
        target,
        "feature importances",
        eval_split="valid",
    )
    assert "(validation set)" in title
    assert "feature importances" in title

    clock_title = _classifier_diagnostics_suptitle(
        target,
        "MADCAP by age group",
        eval_split="test",
        prediction_time=(6, 0),
    )
    assert "(test set)" in clock_title
    assert "at 06:00" in clock_title


def test_arrival_delta_suptitle_uses_eval_split():
    target = EvaluationTarget(
        flow_name="ed_yta_arrival_rates",
        name="arrivals",
        flow_type="admissions",
        evaluation_mode="arrival_deltas",
        component="cumulative_arrival_delta",
        observation_mode="arrived_in_window",
    )
    title = _arrival_delta_suptitle(
        target, "medical", (9, 30), eval_split="valid"
    )
    assert "(validation set)" in title
    assert "medical" in title
    assert "09:30" in title
