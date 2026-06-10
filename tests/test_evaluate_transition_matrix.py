"""Tests for transition-matrix evaluation mode (handler and runner integration).

Integration and handler tests for evaluate_transition_matrix: scalar row shape,
skip semantics, builder validation, and coexistence with other evaluation modes.
Routing-vector unit tests live in test_transfer_arrivals; GoF unit tests in
test_goodness_of_fit.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from patientflow.evaluate.goodness_of_fit import multinomial_gof_montecarlo
from patientflow.evaluate.handlers import evaluate_transition_matrix
from patientflow.evaluate.inputs import (
    EvaluationInputsBuilder,
    EvaluationTarget,
)
from patientflow.evaluate.runner import run_evaluation
from patientflow.evaluate.scalars import ScalarsCollector, scalar_merge_key
from patientflow.predict.demand import FlowSelection
from patientflow.predict.transfers import build_per_patient_probabilities
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator


def _uniform_prediction_dict(
    times: list[tuple[int, int]], *, hours: float = 8.0
) -> dict[tuple[int, int], timedelta]:
    return {pt: timedelta(hours=hours) for pt in times}


def _build_sex_split_model(services: list[str]) -> TransferProbabilityEstimator:
    """Estimator where cardiology females route to gynae and males to surgery."""
    females = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 20,
            "next_subspecialty": ["gynae"] * 20,
            "admission_type": ["emergency"] * 20,
            "age_on_arrival": [30] * 20,
            "sex": ["F"] * 20,
        }
    )
    males = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 20,
            "next_subspecialty": ["surgery"] * 20,
            "admission_type": ["emergency"] * 20,
            "age_on_arrival": [30] * 20,
            "sex": ["M"] * 20,
        }
    )
    discharges = pd.DataFrame(
        {
            "current_subspecialty": ["medicine"] * 10,
            "next_subspecialty": [None] * 10,
            "admission_type": ["emergency"] * 10,
            "age_on_arrival": [50] * 10,
            "sex": ["M"] * 10,
        }
    )
    X = pd.concat([females, males, discharges], ignore_index=True)
    model = TransferProbabilityEstimator(cohort_col="admission_type")
    model.fit(X, set(services))
    return model


def _transition_target(flow_name: str = "inpatient_transfers") -> EvaluationTarget:
    return EvaluationTarget(
        flow_name=flow_name,
        flow_type="departures",
        evaluation_mode="transition_matrix",
        component="transition_matrix_row",
    )


def test_multinomial_gof_miscalibrated_source_low_p_value():
    """Deliberately wrong destinations at high N yield a low p-value end-to-end."""
    services = ["cardiology", "surgery", "gynae"]
    model = _build_sex_split_model(services)
    destinations = list(model.get_transition_matrix("emergency").columns)

    # Model expects females from cardiology to go to gynae; observe surgery only.
    events = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 100,
            "next_subspecialty": ["surgery"] * 100,
            "admission_type": ["emergency"] * 100,
            "age_on_arrival": [30] * 100,
            "sex": ["F"] * 100,
        }
    )
    pp = build_per_patient_probabilities(
        events, "cardiology", "emergency", destinations, model
    )
    n_obs = np.zeros(len(destinations), dtype=int)
    n_obs[destinations.index("surgery")] = 100

    result = multinomial_gof_montecarlo(
        pp.P, n_obs, destinations, n_simulations=2000, seed=7
    )
    assert result.p_value < 0.05


def test_handler_skipped_rows(tmp_path: Path):
    """Handler emits skip_reason rows without pearson_x2/p_value when appropriate."""
    services = ["cardiology", "surgery", "gynae", "medicine"]
    model = _build_sex_split_model(services)
    target = _transition_target()

    # One unmatched adult at cardiology (all-discharge expected) and no medicine traffic.
    events = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"],
            "next_subspecialty": [None],
            "admission_type": ["emergency"],
            "age_on_arrival": [40],
            "sex": [None],
        }
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0)]),
        )
        .with_evaluation_targets([target])
        .add_transition_matrix(
            "inpatient_transfers",
            model,
            events,
            cohort="emergency",
            n_simulations=100,
            seed=1,
        )
        .build()
    )
    collector = ScalarsCollector()
    evaluate_transition_matrix(
        inputs,
        target,
        transitions_dir=tmp_path / "transitions",
        collector=collector,
    )
    rows = {row["service"]: row for row in collector.as_list()}

    assert rows["medicine"]["skip_reason"] == "no_observed_departures"
    assert rows["medicine"]["n_departures"] == 0
    assert "pearson_x2" not in rows["medicine"]

    assert rows["cardiology"]["skip_reason"] == "all_discharge_expected"
    assert rows["cardiology"]["n_departures"] == 1
    assert "p_value" not in rows["cardiology"]


def test_builder_requires_transition_matrix_block_for_target():
    """build() fails when a transition_matrix target has no add_transition_matrix block."""
    target = _transition_target()
    with pytest.raises(ValueError, match="add_transition_matrix"):
        (
            EvaluationInputsBuilder(
                flow_selection=FlowSelection.emergency_only(),
                prediction_dict=_uniform_prediction_dict([(6, 0)]),
            )
            .with_evaluation_targets([target])
            .build()
        )


def test_integration_run_evaluation_emits_rows_and_service_summary(tmp_path: Path):
    """run_evaluation writes one row per estimator source and a by_slice summary."""
    services = ["cardiology", "surgery", "gynae", "medicine"]
    model = _build_sex_split_model(services)
    target = _transition_target()

    events = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 4 + ["surgery"] * 2,
            "next_subspecialty": ["gynae", "gynae", "surgery", "surgery", None, None],
            "admission_type": ["emergency"] * 6,
            "age_on_arrival": [30, 30, 30, 30, 40, 40],
            "sex": ["F", "F", "M", "M", "M", "M"],
        }
    )
    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict=_uniform_prediction_dict([(6, 0)]),
        )
        .with_evaluation_targets([target])
        .add_transition_matrix(
            "inpatient_transfers",
            model,
            events,
            cohort="emergency",
            n_simulations=200,
            seed=99,
        )
        .build()
    )
    result = run_evaluation(tmp_path, inputs, run_name="tm_run")
    payload = json.loads((result["scalars_path"]).read_text(encoding="utf-8"))
    rows = payload["evaluation_rows"]
    tm_rows = [r for r in rows if r["evaluation_mode"] == "transition_matrix"]

    # Complete coverage (D19): every estimator source gets a row, active or skipped.
    assert len(tm_rows) == len(services)
    active = [r for r in tm_rows if "pearson_x2" in r]
    skipped = [r for r in tm_rows if "skip_reason" in r]
    assert len(active) + len(skipped) == len(services)

    cardiology = next(r for r in tm_rows if r["service"] == "cardiology")
    assert cardiology["n_departures"] == 4
    assert cardiology["model_name"] == "transfers_emergency"
    assert cardiology["prediction_time"] is None
    assert len(cardiology["destinations"]) == len(cardiology["expected_counts"])

    summary = payload["_service_summary"]["by_slice"][
        "transition_matrix/inpatient_transfers/transition_matrix_row"
    ]
    assert summary["n_active_sources"] == len(active)
    assert summary["n_skipped_sources"] == len(skipped)
    assert "skipped_by_reason" in summary


def _distribution_leaf(agg_observed: int | None = None) -> dict:
    leaf: dict = {"agg_predicted": {"agg_proba": np.array([0.7, 0.2, 0.1])}}
    if agg_observed is not None:
        leaf["agg_observed"] = agg_observed
    return leaf


def test_integration_alongside_distribution_without_clash(tmp_path: Path):
    """Transition-matrix rows coexist in one run with distribution rows; keys stay unique."""
    from patientflow.load import get_model_key

    services = ["cardiology", "surgery", "gynae", "medicine"]
    model = _build_sex_split_model(services)
    snapshot = date(2024, 1, 1)
    prediction_time = (6, 0)
    model_name = "beds"
    leaf = _distribution_leaf(agg_observed=1)

    dist_target = EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    )
    tm_target = _transition_target()

    events = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology", "cardiology"],
            "next_subspecialty": ["gynae", "surgery"],
            "admission_type": ["emergency"] * 2,
            "age_on_arrival": [30, 30],
            "sex": ["F", "M"],
        }
    )
    arrivals = pd.DataFrame(
        {
            "arrival_datetime": [datetime(2024, 1, 1, 7, 0)],
            "specialty": ["medical"],
            "is_child": [False],
        }
    )

    inputs = (
        EvaluationInputsBuilder(
            flow_selection=FlowSelection.emergency_only(),
            prediction_dict={prediction_time: timedelta(hours=8)},
        )
        .with_evaluation_targets([dist_target, tm_target])
        .add_distributions_from_service_dict(
            "ed_yta_beds",
            prob_dist_by_service={
                "medical": {
                    get_model_key(model_name, prediction_time): {
                        snapshot: leaf,
                    }
                }
            },
            model_name=model_name,
        )
        .add_distribution_observations(
            "ed_yta_beds",
            inpatient_arrivals_by_service={"medical": arrivals},
        )
        .add_transition_matrix(
            "inpatient_transfers",
            model,
            events,
            cohort="emergency",
            n_simulations=100,
            seed=5,
        )
        .build()
    )

    run_result = run_evaluation(tmp_path, inputs, run_name="mixed_run")
    payload = json.loads(run_result["scalars_path"].read_text(encoding="utf-8"))
    modes = {row["evaluation_mode"] for row in payload["evaluation_rows"]}
    assert "distribution" in modes
    assert "transition_matrix" in modes

    keys = {scalar_merge_key(row) for row in payload["evaluation_rows"]}
    assert len(keys) == len(payload["evaluation_rows"])
