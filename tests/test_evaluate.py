"""Tests for the patientflow.evaluate package (inputs, runner, handlers)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patientflow.evaluate.handlers import (
    _arrival_delta_suptitle,
    _classifier_diagnostics_suptitle,
    _classifier_quality_suptitle,
    _distribution_comparison_suptitle,
    evaluate_distribution,
)
from patientflow.evaluate.inputs import (
    EVAL_SPLITS,
    EvaluationInputs,
    EvaluationInputsBuilder,
    EvaluationTarget,
    eval_split_label,
)
from patientflow.evaluate.runner import write_evaluation_run_manifest
from patientflow.evaluate.scalars import ScalarsCollector
from patientflow.load import get_model_key
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


def test_distribution_comparison_suptitle_uses_eval_split():
    dist_target = EvaluationTarget(
        flow_name="ed_current_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_current",
        observation_mode="admitted_at_some_point",
    )
    title = _distribution_comparison_suptitle(
        dist_target, "medical", eval_split="valid"
    )
    assert "(validation set)" in title


def test_classifier_diagnostics_suptitle_includes_cohort_and_clock():
    target = _classifier_probability_quality_target()
    target = EvaluationTarget(
        flow_name=target.flow_name,
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
        flow_type="admissions",
        evaluation_mode="arrival_deltas",
        component="cumulative_arrival_delta",
        observation_mode="arrived_in_window",
    )
    title = _arrival_delta_suptitle(target, "medical", (9, 30), eval_split="valid")
    assert "(validation set)" in title
    assert "medical" in title
    assert "09:30" in title


# --- distribution: observation contexts and recompute ---


def _distribution_leaf(agg_observed: int | None = None) -> dict:
    leaf: dict = {"agg_predicted": {"agg_proba": np.array([0.7, 0.2, 0.1])}}
    if agg_observed is not None:
        leaf["agg_observed"] = agg_observed
    return leaf


def _model_key_distribution(
    model_name: str,
    prediction_time: tuple[int, int],
    snapshot_date: date,
    leaf: dict,
) -> dict:
    mk = get_model_key(model_name, prediction_time)
    return {mk: {snapshot_date: leaf}}


def test_add_distribution_observations_requires_a_frame_mapping():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_times=[(10, 0)],
    )
    with pytest.raises(ValueError, match="at least one of"):
        builder.add_distribution_observations(
            "flow",
            prediction_window=timedelta(hours=8),
        )


def test_add_distribution_observations_stores_distinct_frame_keys():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_times=[(10, 0)],
    )
    ed = pd.DataFrame({"snapshot_date": [date(2024, 1, 1)]})
    arrivals = pd.DataFrame({"arrival_datetime": [datetime(2024, 1, 1, 11, 0)]})
    builder.add_distribution_observations(
        "ed_flow",
        prediction_window=timedelta(hours=8),
        ed_visits_by_service={"medical": ed},
    )
    builder.add_distribution_observations(
        "yta_flow",
        prediction_window=timedelta(hours=8),
        inpatient_arrivals_by_service={"medical": arrivals},
    )
    inputs = builder.build()
    assert "ed_visits" in inputs.observation_contexts["ed_flow"]["medical"]
    assert "inpatient_arrivals" not in inputs.observation_contexts["ed_flow"]["medical"]
    assert "inpatient_arrivals" in inputs.observation_contexts["yta_flow"]["medical"]
    assert "ed_visits" not in inputs.observation_contexts["yta_flow"]["medical"]


def test_evaluate_distribution_recomputes_yta_from_inpatient_arrivals(
    tmp_path: Path,
):
    moment = datetime(2024, 1, 1, 10, 0, 0)
    snapshot = date(2024, 1, 1)
    prediction_time = (10, 0)
    model_name = "beds"
    leaf = _distribution_leaf()

    ed_visits = pd.DataFrame(
        {
            "arrival_datetime": [moment + timedelta(hours=1)],
            "snapshot_date": [snapshot],
            "prediction_time": [prediction_time],
        }
    )
    inpatient_arrivals = pd.DataFrame(
        {"arrival_datetime": [moment + timedelta(hours=1)]}
    )

    target = EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[prediction_time],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    model_name, prediction_time, snapshot, leaf
                )
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_yta_beds",
            prediction_window=timedelta(hours=8),
            ed_visits_by_service={"medical": ed_visits},
            inpatient_arrivals_by_service={"medical": inpatient_arrivals},
        )
        .build()
    )

    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=ScalarsCollector(),
    )
    assert leaf["agg_observed"] == 1


def test_evaluate_distribution_yta_pre_filtered_cohort_not_specialty_column(
    tmp_path: Path,
):
    """Per-service YTA frames define the cohort (e.g. is_child), not specialty=."""
    snapshot = date(2031, 9, 4)
    prediction_time = (6, 0)
    model_name = "beds"
    moment = datetime(2031, 9, 4, 6, 0, 0)
    window = timedelta(hours=8)

    inpatient_arrivals = pd.DataFrame(
        {
            "arrival_datetime": [
                moment + timedelta(hours=1),
                moment + timedelta(hours=2),
            ],
            "specialty": ["paediatric", "medical"],
            "is_child": [True, True],
        }
    )
    paediatric_cohort = inpatient_arrivals[inpatient_arrivals["is_child"]]
    leaf = _distribution_leaf(agg_observed=2)

    target = EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[prediction_time],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "paediatric": _model_key_distribution(
                    model_name, prediction_time, snapshot, leaf
                )
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_yta_beds",
            prediction_window=window,
            inpatient_arrivals_by_service={"paediatric": paediatric_cohort},
        )
        .build()
    )

    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=ScalarsCollector(),
    )
    assert leaf["agg_observed"] == 2


def test_evaluate_distribution_ed_current_applies_specialty_on_shared_frame(
    tmp_path: Path,
):
    snapshot = date(2024, 1, 1)
    prediction_time = (10, 0)
    model_name = "beds"
    ed_visits = pd.DataFrame(
        [
            {
                "snapshot_date": snapshot,
                "prediction_time": prediction_time,
                "is_admitted": 1,
                "specialty": "medical",
            },
            {
                "snapshot_date": snapshot,
                "prediction_time": prediction_time,
                "is_admitted": 1,
                "specialty": "surgical",
            },
        ]
    )
    medical_leaf = _distribution_leaf()
    surgical_leaf = _distribution_leaf()
    target = EvaluationTarget(
        flow_name="ed_current_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_current",
        observation_mode="admitted_at_some_point",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[prediction_time],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    model_name, prediction_time, snapshot, medical_leaf
                ),
                "surgical": _model_key_distribution(
                    model_name, prediction_time, snapshot, surgical_leaf
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            prediction_window=timedelta(hours=8),
            ed_visits_by_service={
                "medical": ed_visits,
                "surgical": ed_visits,
            },
        )
        .build()
    )

    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=ScalarsCollector(),
    )
    assert medical_leaf["agg_observed"] == 1
    assert surgical_leaf["agg_observed"] == 1


def test_evaluate_distribution_asserts_on_leaf_mismatch(tmp_path: Path):
    snapshot = date(2024, 1, 1)
    prediction_time = (10, 0)
    model_name = "beds"
    leaf = _distribution_leaf(agg_observed=99)
    ed_visits = pd.DataFrame(
        [
            {
                "snapshot_date": snapshot,
                "prediction_time": prediction_time,
                "is_admitted": 1,
                "specialty": "medical",
            }
        ]
    )
    target = EvaluationTarget(
        flow_name="ed_current_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_current",
        observation_mode="admitted_at_some_point",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[prediction_time],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    model_name, prediction_time, snapshot, leaf
                )
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            prediction_window=timedelta(hours=8),
            ed_visits_by_service={"medical": ed_visits},
        )
        .build()
    )
    with pytest.raises(AssertionError, match="does not match recomputed"):
        evaluate_distribution(
            inputs,
            target,
            distributions_dir=tmp_path / "distributions",
            collector=ScalarsCollector(),
        )


def test_evaluate_distribution_departures_admission_type_filter(tmp_path: Path):
    snapshot = date(2024, 1, 1)
    prediction_time = (10, 0)
    model_name = "dep"
    leaf = _distribution_leaf()
    inpatient_visits = pd.DataFrame(
        [
            {
                "snapshot_date": snapshot,
                "prediction_time": prediction_time,
                "left_subspecialty_in_window": True,
                "admission_type": "elective",
                "specialty": "medical",
            },
            {
                "snapshot_date": snapshot,
                "prediction_time": prediction_time,
                "left_subspecialty_in_window": True,
                "admission_type": "emergency",
                "specialty": "medical",
            },
        ]
    )
    target = EvaluationTarget(
        flow_name="departures_elective",
        flow_type="departures",
        evaluation_mode="distribution",
        component="departures_elective",
        observation_mode="departed_in_window",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[prediction_time],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "departures_elective",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    model_name, prediction_time, snapshot, leaf
                )
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "departures_elective",
            prediction_window=timedelta(hours=8),
            inpatient_visits_by_service={"medical": inpatient_visits},
        )
        .build()
    )
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=ScalarsCollector(),
    )
    assert leaf["agg_observed"] == 1


def test_evaluate_distribution_raises_when_observation_frame_missing():
    target = EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_times=[(10, 0)],
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    "beds",
                    (10, 0),
                    date(2024, 1, 1),
                    _distribution_leaf(),
                )
            },
            model_name="beds",
        )
        .add_distribution_observations(
            "ed_yta_beds",
            prediction_window=timedelta(hours=8),
            ed_visits_by_service={"medical": pd.DataFrame()},
        )
        .build()
    )
    with pytest.raises(ValueError, match="inpatient_arrivals"):
        evaluate_distribution(
            inputs,
            target,
            distributions_dir=Path("/tmp/unused"),
            collector=ScalarsCollector(),
        )
