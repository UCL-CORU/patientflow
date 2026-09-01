"""Tests for the patientflow.evaluate package (inputs, runner, handlers)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from patientflow.evaluate.handlers import (
    _arrival_delta_suptitle,
    _classifier_diagnostics_suptitle,
    _classifier_quality_suptitle,
    _distribution_comparison_suptitle,
    evaluate_classifier_model_diagnostics,
    evaluate_classifier_probability_quality,
    evaluate_distribution,
)
from patientflow.evaluate.inputs import (
    EVAL_SPLITS,
    DEFAULT_MADCAP_GROUPINGS,
    EvaluationInputs,
    EvaluationInputsBuilder,
    EvaluationTarget,
    MadcapGrouping,
    eval_split_label,
    normalize_prediction_dict,
    resolve_madcap_groupings,
    standard_ed_targets,
)
from patientflow.evaluate.runner import (
    evaluation_targets_for_manifest,
    prediction_dict_for_manifest,
    write_evaluation_run_manifest,
)
from patientflow.evaluate.scalars import (
    RELIABILITY_MIN_POSITIVE_CASES,
    ScalarsCollector,
    classifier_reliable,
    scalar_target_fields,
)
from patientflow.load import get_model_key
from patientflow.model_artifacts import TrainedClassifier, TrainingResults
from patientflow.predict.demand import FlowSelection


def _uniform_prediction_dict(
    times: list[tuple[int, int]], *, hours: float = 8.0
) -> dict[tuple[int, int], timedelta]:
    window = timedelta(hours=hours)
    return {pt: window for pt in times}


# --- eval_split: inputs and manifest ---


def test_eval_split_label():
    assert eval_split_label("valid") == "validation set"
    assert eval_split_label("test") == "test set"
    assert eval_split_label(None) == "evaluation cohort"
    assert eval_split_label("other") == "evaluation cohort"


def test_builder_defaults_eval_split_valid():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
    )
    inputs = builder.build()
    assert inputs.eval_split == "valid"


def test_builder_set_eval_split_test():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
        eval_split="test",
    ).set_eval_split("test")
    inputs = builder.build()
    assert inputs.eval_split == "test"


def test_builder_rejects_unknown_eval_split():
    with pytest.raises(ValueError, match="Unknown eval_split"):
        EvaluationInputsBuilder(eval_split="holdout")  # type: ignore[arg-type]

    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
    )
    with pytest.raises(ValueError, match="Unknown eval_split"):
        builder.set_eval_split("train")  # type: ignore[arg-type]


def test_eval_splits_constant():
    assert EVAL_SPLITS == ("valid", "test")


def test_builder_requires_non_empty_prediction_dict():
    with pytest.raises(ValueError, match="prediction_dict must not be empty"):
        normalize_prediction_dict({})


def test_prediction_times_derived_sorted():
    inputs = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict={(22, 0): timedelta(hours=8), (6, 0): timedelta(hours=4)},
    ).build()
    assert inputs.prediction_times == [(6, 0), (22, 0)]


def test_manifest_records_eval_split_and_prediction_dict(tmp_path: Path):
    flow = FlowSelection.emergency_only()
    prediction_dict = _uniform_prediction_dict([(6, 0), (12, 0)])
    inputs = EvaluationInputs(
        flow_selection=flow,
        prediction_dict=prediction_dict,
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
    )
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert "training" not in loaded
    assert loaded["evaluation"]["eval_split"] == "test"
    assert loaded["evaluation"]["prediction_dict"] == prediction_dict_for_manifest(
        prediction_dict
    )
    assert "patientflow_version" in loaded["evaluation"]


def test_manifest_records_evaluation_targets_with_observation_mode(tmp_path: Path):
    target = _classifier_probability_quality_target()
    inputs = EvaluationInputs(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
        evaluation_targets=[target],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_evaluation_run_manifest(
        run_dir,
        output_root=tmp_path,
        run_name="run",
        inputs=inputs,
    )
    loaded = yaml.safe_load((run_dir / "evaluation_run.yaml").read_text())
    assert loaded["evaluation"][
        "evaluation_targets"
    ] == evaluation_targets_for_manifest(inputs)
    assert (
        loaded["evaluation"]["evaluation_targets"][0]["observation_mode"]
        == "admitted_at_some_point"
    )


def test_scalar_target_fields_includes_observation_mode():
    target = _classifier_probability_quality_target()
    fields = scalar_target_fields(target)
    assert fields["observation_mode"] == target.observation_mode
    assert fields["flow"] == target.flow_name


def test_classifier_reliable_by_split():
    pos = {
        "train_positive_cases": 100,
        "valid_positive_cases": RELIABILITY_MIN_POSITIVE_CASES,
        "test_positive_cases": RELIABILITY_MIN_POSITIVE_CASES - 1,
    }
    assert classifier_reliable({"split": "valid"}, pos)
    assert not classifier_reliable({"split": "test"}, pos)
    assert classifier_reliable(
        {"split": "cv_train", "n_positive_cases": RELIABILITY_MIN_POSITIVE_CASES},
        pos,
    )
    assert not classifier_reliable(
        {"split": "cv_train", "n_positive_cases": RELIABILITY_MIN_POSITIVE_CASES - 1},
        pos,
    )
    assert not classifier_reliable({"split": "unknown"}, pos)
    assert not classifier_reliable({}, pos)


def test_manifest_optional_training_metadata(tmp_path: Path):
    inputs = EvaluationInputs(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
        evaluation_targets=[],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_evaluation_run_manifest(
        run_dir,
        output_root=tmp_path,
        run_name="run",
        inputs=inputs,
        training_metadata={"start_validation_set": "2025-10-01"},
    )
    loaded = yaml.safe_load((run_dir / "evaluation_run.yaml").read_text())
    assert loaded["training_metadata"]["start_validation_set"] == "2025-10-01"


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
        component="arrival_delta",
        observation_mode="arrived_in_window",
    )
    title = _arrival_delta_suptitle(target, "medical", eval_split="valid")
    assert title == "Arrival delta plots for medical service (validation set)"
    assert "09:30" not in title


def test_evaluate_classifier_model_diagnostics_per_clock_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Multi-clock diagnostics write features.png plus features_HHMM.png."""
    feature_calls: list[dict] = []

    def _capture_features(models, *args, **kwargs):
        n = len(models) if not isinstance(models, dict) else len(models)
        feature_calls.append(
            {
                "n_models": n,
                "file_name": kwargs.get("file_name"),
                "suptitle": kwargs.get("suptitle"),
            }
        )

    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_features", _capture_features
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.SHAP_AVAILABLE", False)
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_shap", None)

    visits = pd.DataFrame({"is_admitted": [0, 1]})
    target = EvaluationTarget(
        flow_name="ed_admissions_cls",
        flow_type="admissions",
        evaluation_mode="classifier_model_diagnostics",
        component="classifier_model_diagnostics",
        observation_mode="admitted_at_some_point",
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0), (15, 30)]),
            eval_split="valid",
        )
        .add_classifier(
            target.flow_name,
            [
                _minimal_trained_classifier((6, 0)),
                _minimal_trained_classifier((15, 30)),
            ],
            visits,
            "is_admitted",
        )
        .with_evaluation_targets([target])
        .build()
    )
    collector = ScalarsCollector()
    evaluate_classifier_model_diagnostics(
        inputs,
        target,
        classifiers_dir=tmp_path / "classifiers",
        collector=collector,
    )

    assert [c["file_name"] for c in feature_calls] == [
        "features.png",
        "features_0600.png",
        "features_1530.png",
    ]
    assert feature_calls[0]["n_models"] == 2
    assert feature_calls[1]["n_models"] == 1
    assert feature_calls[2]["n_models"] == 1
    assert "at 06:00" in feature_calls[1]["suptitle"]
    assert "at 15:30" in feature_calls[2]["suptitle"]
    assert len(collector.as_list()) == 2


# --- classifier MADCAP groupings ---


def _minimal_trained_classifier(
    prediction_time: tuple[int, int] = (6, 0),
) -> TrainedClassifier:
    return TrainedClassifier(
        training_results=TrainingResults(
            prediction_time=prediction_time,
            training_info={
                "dataset_info": {
                    "train_valid_test_positive_cases": {
                        "train": 10,
                        "valid": 5,
                        "test": 5,
                    }
                }
            },
        ),
        selected_eval_metrics={
            "auroc": 0.8,
            "auprc": 0.7,
            "log_loss": 0.4,
            "split": "valid",
            "n_samples": 20,
            "n_positive_cases": 5,
        },
    )


def test_resolve_madcap_groupings_default_and_explicit():
    assert resolve_madcap_groupings(None) == list(DEFAULT_MADCAP_GROUPINGS)
    custom = [MadcapGrouping("sex", "Sex", "madcap_by_sex")]
    assert resolve_madcap_groupings(custom) == custom
    assert resolve_madcap_groupings([]) == []


def test_add_classifier_defaults_madcap_groupings_to_age():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
    )
    visits = pd.DataFrame({"is_admitted": [0, 1], "age_group": ["18-24", "65-74"]})
    builder.add_classifier(
        "ed_admissions_cls",
        [_minimal_trained_classifier()],
        visits,
        "is_admitted",
    )
    block = builder.build().classifier_by_flow["ed_admissions_cls"]
    assert block["madcap_groupings"] == [
        MadcapGrouping("age_group", "Age group", "madcap_by_age")
    ]


def test_add_classifier_stores_explicit_madcap_groupings():
    groupings = [
        MadcapGrouping("age_group", "Age group", "madcap_by_age"),
        MadcapGrouping("ethnicity", "Ethnicity", "madcap_by_ethnicity"),
    ]
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
    )
    visits = pd.DataFrame(
        {
            "is_admitted": [0, 1],
            "age_group": ["18-24", "65-74"],
            "ethnicity": ["A", "B"],
        }
    )
    builder.add_classifier(
        "ed_admissions_cls",
        [_minimal_trained_classifier()],
        visits,
        "is_admitted",
        madcap_groupings=groupings,
    )
    block = builder.build().classifier_by_flow["ed_admissions_cls"]
    assert block["madcap_groupings"] == groupings


def test_evaluate_classifier_probability_quality_madcap_groupings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Emit one MADCAP-by-group call per present grouping; skip missing columns."""
    by_group_calls: list[dict] = []

    def _noop_plot(*_args, **_kwargs):
        return None

    def _capture_by_group(*args, **kwargs):
        by_group_calls.append(
            {
                "grouping_var": kwargs.get("grouping_var"),
                "grouping_var_name": kwargs.get("grouping_var_name"),
                "file_name": kwargs.get("file_name"),
                "suptitle": kwargs.get("suptitle"),
            }
        )

    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_estimated_probabilities", _noop_plot
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_madcap", _noop_plot)
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_madcap_by_group", _capture_by_group
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_calibration", _noop_plot)

    visits = pd.DataFrame(
        {
            "is_admitted": [0, 1, 0, 1],
            "age_group": ["18-24", "65-74", "18-24", "65-74"],
            "ethnicity": ["GroupA", "GroupB", "GroupA", "GroupB"],
        }
    )
    groupings = [
        MadcapGrouping("age_group", "Age group", "madcap_by_age"),
        MadcapGrouping("ethnicity", "Ethnicity", "madcap_by_ethnicity"),
        MadcapGrouping("missing_col", "Missing", "madcap_by_missing"),
    ]
    target = _classifier_probability_quality_target()
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0)]),
            eval_split="test",
        )
        .add_classifier(
            target.flow_name,
            [_minimal_trained_classifier((6, 0))],
            visits,
            "is_admitted",
            madcap_groupings=groupings,
        )
        .with_evaluation_targets([target])
        .build()
    )
    collector = ScalarsCollector()
    out_dir = tmp_path / "classifiers"
    evaluate_classifier_probability_quality(
        inputs, target, classifiers_dir=out_dir, collector=collector
    )

    assert [(c["grouping_var"], c["file_name"]) for c in by_group_calls] == [
        ("age_group", "madcap_by_age.png"),
        ("ethnicity", "madcap_by_ethnicity.png"),
    ]
    assert by_group_calls[0]["grouping_var_name"] == "Age group"
    assert "MADCAP by age group" in by_group_calls[0]["suptitle"]
    assert "MADCAP by ethnicity" in by_group_calls[1]["suptitle"]
    assert collector.as_list()[0]["charts_generated"] is True


def test_evaluate_classifier_probability_quality_default_age_madcap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    by_group_calls: list[dict] = []

    def _noop_plot(*_args, **_kwargs):
        return None

    def _capture_by_group(*_args, **kwargs):
        by_group_calls.append({"file_name": kwargs.get("file_name")})

    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_estimated_probabilities", _noop_plot
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_madcap", _noop_plot)
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_madcap_by_group", _capture_by_group
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_calibration", _noop_plot)

    visits = pd.DataFrame({"is_admitted": [0, 1], "age_group": ["18-24", "65-74"]})
    target = _classifier_probability_quality_target()
    # Omit madcap_groupings key (legacy block shape) — handler should still default.
    inputs = EvaluationInputs(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(6, 0)]),
        evaluation_targets=[target],
        eval_split="valid",
        classifier_by_flow={
            target.flow_name: {
                "trained_models": [_minimal_trained_classifier()],
                "visits_df": visits,
                "label_col": "is_admitted",
                "model_name": "admissions",
            }
        },
    )
    evaluate_classifier_probability_quality(
        inputs,
        target,
        classifiers_dir=tmp_path / "classifiers",
        collector=ScalarsCollector(),
    )
    assert by_group_calls == [{"file_name": "madcap_by_age.png"}]


def test_evaluate_classifier_probability_quality_multi_clock_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    by_group_calls: list[dict] = []

    def _noop_plot(*_args, **_kwargs):
        return None

    def _capture_by_group(*_args, **kwargs):
        by_group_calls.append({"file_name": kwargs.get("file_name")})

    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_estimated_probabilities", _noop_plot
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_madcap", _noop_plot)
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_madcap_by_group", _capture_by_group
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_calibration", _noop_plot)

    visits = pd.DataFrame(
        {
            "is_admitted": [0, 1],
            "age_group": ["18-24", "65-74"],
            "ethnicity": ["A", "B"],
        }
    )
    target = _classifier_probability_quality_target()
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0), (15, 30)]),
        )
        .add_classifier(
            target.flow_name,
            [
                _minimal_trained_classifier((6, 0)),
                _minimal_trained_classifier((15, 30)),
            ],
            visits,
            "is_admitted",
            madcap_groupings=[
                MadcapGrouping("ethnicity", "Ethnicity", "madcap_by_ethnicity"),
            ],
        )
        .with_evaluation_targets([target])
        .build()
    )
    evaluate_classifier_probability_quality(
        inputs,
        target,
        classifiers_dir=tmp_path / "classifiers",
        collector=ScalarsCollector(),
    )
    assert [c["file_name"] for c in by_group_calls] == [
        "madcap_by_ethnicity_0600.png",
        "madcap_by_ethnicity_1530.png",
    ]


def test_evaluate_classifier_probability_quality_passes_madcap_figsize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    by_group_calls: list[dict] = []

    def _noop_plot(*_args, **_kwargs):
        return None

    def _capture_by_group(*_args, **kwargs):
        by_group_calls.append({"figsize": kwargs.get("figsize")})

    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_estimated_probabilities", _noop_plot
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_madcap", _noop_plot)
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.plot_madcap_by_group", _capture_by_group
    )
    monkeypatch.setattr("patientflow.evaluate.handlers.plot_calibration", _noop_plot)

    visits = pd.DataFrame({"is_admitted": [0, 1], "age_group": ["18-24", "65-74"]})
    target = _classifier_probability_quality_target()
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0)]),
            eval_split="test",
        )
        .add_classifier(
            target.flow_name,
            [_minimal_trained_classifier((6, 0))],
            visits,
            "is_admitted",
        )
        .with_evaluation_targets([target])
        .build()
    )
    evaluate_classifier_probability_quality(
        inputs,
        target,
        classifiers_dir=tmp_path / "classifiers",
        collector=ScalarsCollector(),
        madcap_figsize=(12, 2),
    )
    assert by_group_calls == [{"figsize": (12, 2)}]


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
        prediction_dict=_uniform_prediction_dict([(10, 0)]),
    )
    with pytest.raises(ValueError, match="at least one of"):
        builder.add_distribution_observations("flow")


def test_add_distribution_observations_stores_distinct_frame_keys():
    builder = EvaluationInputsBuilder(
        flow_selection=FlowSelection.emergency_only(),
        prediction_dict=_uniform_prediction_dict([(10, 0)]),
    )
    ed = pd.DataFrame({"snapshot_date": [date(2024, 1, 1)]})
    arrivals = pd.DataFrame({"arrival_datetime": [datetime(2024, 1, 1, 11, 0)]})
    builder.add_distribution_observations(
        "ed_flow",
        ed_visits_by_service={"medical": ed},
    )
    builder.add_distribution_observations(
        "yta_flow",
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
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


def test_evaluate_distribution_uses_per_time_prediction_window(tmp_path: Path):
    """Each clock time uses its own horizon from prediction_dict."""
    snapshot = date(2024, 1, 1)
    pt_short = (10, 0)
    pt_long = (14, 0)
    model_name = "beds"
    moment_short = datetime(2024, 1, 1, 10, 0, 0)
    moment_long = datetime(2024, 1, 1, 14, 0, 0)
    # 5 h after 10:00 is outside a 4 h window; 5 h after 14:00 is inside 8 h.
    arrivals_short = pd.DataFrame(
        {"arrival_datetime": [moment_short + timedelta(hours=5)]}
    )
    arrivals_long = pd.DataFrame(
        {"arrival_datetime": [moment_long + timedelta(hours=5)]}
    )
    leaf_short = _distribution_leaf()
    leaf_long = _distribution_leaf()

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
            prediction_dict={
                pt_short: timedelta(hours=4),
                pt_long: timedelta(hours=8),
            },
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "svc_a": _model_key_distribution(
                    model_name, pt_short, snapshot, leaf_short
                ),
                "svc_b": _model_key_distribution(
                    model_name, pt_long, snapshot, leaf_long
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_yta_beds",
            inpatient_arrivals_by_service={
                "svc_a": arrivals_short,
                "svc_b": arrivals_long,
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
    assert leaf_short["agg_observed"] == 0
    assert leaf_long["agg_observed"] == 1


def test_evaluate_distribution_yta_pre_filtered_cohort_not_specialty_column(
    tmp_path: Path,
):
    """Per-service YTA frames define the cohort (e.g. is_child), not specialty=."""
    snapshot = date(2031, 9, 4)
    prediction_time = (6, 0)
    model_name = "beds"
    moment = datetime(2031, 9, 4, 6, 0, 0)

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
            prediction_dict={prediction_time: timedelta(hours=8)},
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
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
            prediction_dict=_uniform_prediction_dict([(10, 0)]),
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


def _two_snapshot_model_key_dist(
    model_name: str,
    prediction_time: tuple[int, int],
    leaves: tuple[dict, dict],
) -> dict:
    snap1, snap2 = date(2024, 1, 1), date(2024, 1, 2)
    mk = get_model_key(model_name, prediction_time)
    return {mk: {snap1: leaves[0], snap2: leaves[1]}}


def _n_snapshot_model_key_dist(
    model_name: str,
    prediction_time: tuple[int, int],
    n: int,
) -> dict:
    """Build ``n`` snapshot leaves under one model key (observed filled at eval)."""
    mk = get_model_key(model_name, prediction_time)
    series = {
        date(2024, 1, 1) + timedelta(days=i): _distribution_leaf() for i in range(n)
    }
    return {mk: series}


def _ed_visits_for_snapshots(
    prediction_time: tuple[int, int],
    n: int,
    *,
    specialty: str = "medical",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": date(2024, 1, 1) + timedelta(days=i),
                "prediction_time": prediction_time,
                "is_admitted": 1 if i % 2 == 0 else 0,
                "specialty": specialty,
            }
            for i in range(n)
        ]
    )


def test_evaluate_distribution_insufficient_observations_skips_epudd(
    tmp_path: Path,
):
    prediction_time = (10, 0)
    model_name = "beds"
    snapshot = date(2024, 1, 1)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _model_key_distribution(
                    model_name, prediction_time, snapshot, _distribution_leaf(1)
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={
                "medical": pd.DataFrame(
                    {
                        "snapshot_date": [snapshot],
                        "prediction_time": [prediction_time],
                        "is_admitted": [1],
                        "specialty": ["medical"],
                    }
                ),
            },
        )
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
    )
    row = collector.as_list()[0]
    assert row["skip_reason"] == "insufficient_observations"
    assert row["charts_generated"] is False
    assert "rpit_cvm_mean_w2" not in row
    assert not (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_evaluate_distribution_custom_admissions_label_col(tmp_path: Path):
    prediction_time = (10, 0)
    model_name = "beds"
    leaf_a = _distribution_leaf()
    leaf_b = _distribution_leaf()
    ed_visits = pd.DataFrame(
        [
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": prediction_time,
                "was_admitted": True,
                "specialty": "medical",
            },
            {
                "snapshot_date": date(2024, 1, 2),
                "prediction_time": prediction_time,
                "was_admitted": False,
                "specialty": "medical",
            },
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _two_snapshot_model_key_dist(
                    model_name, prediction_time, (leaf_a, leaf_b)
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .add_distribution_benchmark_cohort(
            admissions_ed_visits=ed_visits,
            admissions_label_col="was_admitted",
        )
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
    )
    assert leaf_a["agg_observed"] == 1
    assert leaf_b["agg_observed"] == 0


def test_evaluate_distribution_rpit_cvm_and_benchmark_scalars(tmp_path: Path):
    prediction_time = (10, 0)
    model_name = "beds"
    n_snap = 30
    ed_visits = _ed_visits_for_snapshots(prediction_time, n_snap)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _n_snapshot_model_key_dist(
                    model_name, prediction_time, n_snap
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .add_distribution_benchmark_cohort(admissions_ed_visits=ed_visits)
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
        charts="all",
    )
    row = collector.as_list()[0]
    assert row["observation_mode"] == "admitted_at_some_point"
    assert row["charts_generated"] is True
    assert row["reliable"] is True
    assert "rpit_cvm_mean_w2" in row
    assert "rpit_cvm_w2" not in row
    assert "rpit_cvm_benchmark_mean_w2" in row
    assert "rpit_cvm_benchmark_w2" not in row
    assert "rpit_cvm_w2_reduction" in row
    assert (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_evaluate_distribution_gate_a_keeps_rpit_without_chart(tmp_path: Path):
    """n_snap in [MIN_DISTRIBUTION_SNAPSHOTS, Gate A) still scores rPIT; no PNG."""
    prediction_time = (10, 0)
    model_name = "beds"
    leaf_a = _distribution_leaf(1)
    leaf_b = _distribution_leaf(0)
    ed_visits = _ed_visits_for_snapshots(prediction_time, 2)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _two_snapshot_model_key_dist(
                    model_name, prediction_time, (leaf_a, leaf_b)
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .add_distribution_benchmark_cohort(admissions_ed_visits=ed_visits)
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
        charts="all",
    )
    row = collector.as_list()[0]
    assert row["n_snapshots"] == 2
    assert "rpit_cvm_mean_w2" in row
    assert row["reliable"] is False
    assert row["charts_generated"] is False
    assert row["skip_reason"] == "insufficient_observations"
    assert not (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_evaluate_distribution_flagged_skips_unflagged_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prediction_time = (10, 0)
    model_name = "beds"
    n_snap = 30
    ed_visits = _ed_visits_for_snapshots(prediction_time, n_snap)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _n_snapshot_model_key_dist(
                    model_name, prediction_time, n_snap
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .build()
    )
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.distribution_chart_flagged",
        lambda row: False,
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
        charts="flagged",
    )
    row = collector.as_list()[0]
    assert row["chart_flagged"] is False
    assert row["charts_generated"] is False
    assert row["skip_reason"] == "not_flagged"
    assert not (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_evaluate_distribution_charts_none(tmp_path: Path):
    prediction_time = (10, 0)
    model_name = "beds"
    n_snap = 30
    ed_visits = _ed_visits_for_snapshots(prediction_time, n_snap)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _n_snapshot_model_key_dist(
                    model_name, prediction_time, n_snap
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
        charts="none",
    )
    row = collector.as_list()[0]
    assert "rpit_cvm_mean_w2" in row
    assert row["charts_generated"] is False
    assert row["skip_reason"] == "charts_disabled"
    assert not (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_evaluate_distribution_flagged_writes_when_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prediction_time = (10, 0)
    model_name = "beds"
    n_snap = 30
    ed_visits = _ed_visits_for_snapshots(prediction_time, n_snap)
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _n_snapshot_model_key_dist(
                    model_name, prediction_time, n_snap
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .build()
    )
    monkeypatch.setattr(
        "patientflow.evaluate.handlers.distribution_chart_flagged",
        lambda row: True,
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
        charts="flagged",
    )
    row = collector.as_list()[0]
    assert row["chart_flagged"] is True
    assert row["charts_generated"] is True
    assert (
        tmp_path
        / "distributions"
        / "ed_current_beds"
        / "medical"
        / "bed_demand_ed_current.png"
    ).exists()


def test_manifest_records_charts_mode(tmp_path: Path):
    from patientflow.evaluate.runner import run_evaluation

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
            prediction_dict=_uniform_prediction_dict([(10, 0)]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _n_snapshot_model_key_dist("beds", (10, 0), 2),
            },
            model_name="beds",
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={
                "medical": _ed_visits_for_snapshots((10, 0), 2),
            },
        )
        .build()
    )
    result = run_evaluation(tmp_path, inputs, run_name="charts_mode", charts="none")
    loaded = yaml.safe_load(result["manifest_path"].read_text())
    assert loaded["evaluation"]["charts"] == "none"


def test_evaluate_distribution_yta_has_no_benchmark_fields(tmp_path: Path):
    prediction_time = (10, 0)
    model_name = "beds"
    leaf_a, leaf_b = _distribution_leaf(), _distribution_leaf()
    moment = datetime(2024, 1, 1, 10, 0, 0)
    arrivals = pd.DataFrame(
        {
            "arrival_datetime": [
                moment + timedelta(hours=1),
                moment + timedelta(hours=2),
            ],
        }
    )
    target = EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    )
    ed_visits = pd.DataFrame(
        {
            "snapshot_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "prediction_time": [prediction_time, prediction_time],
            "is_admitted": [0, 1],
        }
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "medical": _two_snapshot_model_key_dist(
                    model_name, prediction_time, (leaf_a, leaf_b)
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_yta_beds",
            inpatient_arrivals_by_service={"medical": arrivals},
        )
        .add_distribution_benchmark_cohort(admissions_ed_visits=ed_visits)
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
    )
    row = collector.as_list()[0]
    assert row["observation_mode"] == "arrived_in_window"
    assert "rpit_cvm_mean_w2" in row
    assert "rpit_cvm_benchmark_mean_w2" not in row


def test_standard_ed_targets_default_four_targets():
    targets = standard_ed_targets()
    assert len(targets) == 4
    assert {t.flow_name for t in targets} == {
        "ed_admissions_cls",
        "ed_current_beds",
        "ed_yta_arrival_rates",
    }


def test_evaluate_distribution_specialty_proportions_benchmark(tmp_path: Path):
    prediction_time = (10, 0)
    model_name = "beds"
    leaf_a = _distribution_leaf(1)
    leaf_b = _distribution_leaf(0)
    alt_a = _distribution_leaf(0)
    alt_b = _distribution_leaf(1)
    ed_visits = pd.DataFrame(
        [
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": prediction_time,
                "is_admitted": 1,
                "specialty": "medical",
            },
            {
                "snapshot_date": date(2024, 1, 2),
                "prediction_time": prediction_time,
                "is_admitted": 0,
                "specialty": "medical",
            },
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
            prediction_dict=_uniform_prediction_dict([prediction_time]),
        )
        .with_evaluation_targets([target])
        .add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _two_snapshot_model_key_dist(
                    model_name, prediction_time, (leaf_a, leaf_b)
                ),
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_current_beds",
            ed_visits_by_service={"medical": ed_visits},
        )
        .add_distribution_benchmark_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service={
                "medical": _two_snapshot_model_key_dist(
                    model_name, prediction_time, (alt_a, alt_b)
                ),
            },
            benchmark_kind="specialty_proportions",
        )
        .build()
    )
    collector = ScalarsCollector()
    evaluate_distribution(
        inputs,
        target,
        distributions_dir=tmp_path / "distributions",
        collector=collector,
    )
    row = collector.as_list()[0]
    assert "rpit_cvm_specialty_proportions_mean_w2" in row
    assert "rpit_cvm_specialty_proportions_w2_reduction" in row
