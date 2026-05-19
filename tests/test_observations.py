"""Tests for patientflow.evaluate.observations."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from patientflow.evaluate.observations import (
    OBSERVATION_MODES,
    count_observed,
    count_observed_admitted_at_some_point,
    count_observed_admitted_in_window,
    count_observed_applies_specialty_filter,
    count_observed_arrived_and_admitted_in_window,
    count_observed_arrived_in_window,
    count_observed_departed_in_window,
    validate_observation_mode_for_component,
)


def test_count_observed_applies_specialty_filter():
    assert count_observed_applies_specialty_filter("admitted_at_some_point")
    assert count_observed_applies_specialty_filter("admitted_in_window")
    assert not count_observed_applies_specialty_filter("arrived_in_window")
    assert not count_observed_applies_specialty_filter("departed_in_window")


def _ed_row(**kwargs):
    base = dict(
        snapshot_date=date(2024, 1, 1),
        prediction_time=(10, 0),
        is_admitted=1,
        specialty="medical",
    )
    base.update(kwargs)
    return base


def test_admitted_at_some_point():
    df = pd.DataFrame(
        [
            _ed_row(),
            _ed_row(is_admitted=0),
            _ed_row(specialty="surgical"),
        ]
    )
    n = count_observed_admitted_at_some_point(
        df,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
        specialty="medical",
    )
    assert n == 1


def test_admitted_in_window_requires_column():
    df = pd.DataFrame([_ed_row()])
    with pytest.raises(ValueError, match="departure_datetime"):
        count_observed_admitted_in_window(
            df,
            date(2024, 1, 1),
            (10, 0),
            timedelta(hours=8),
        )


def test_admitted_in_window_counts_departure_on_snapshot_cohort():
    """Counts in-ED snapshot rows with is_admitted and leave-ED in window."""
    moment = datetime(2024, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                **_ed_row(),
                "departure_datetime": moment + timedelta(hours=2),
            },
            {
                **_ed_row(),
                "departure_datetime": moment + timedelta(hours=12),
            },
            {
                **_ed_row(is_admitted=0),
                "departure_datetime": moment + timedelta(hours=1),
            },
        ]
    )
    n = count_observed_admitted_in_window(
        df,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
    )
    assert n == 1


def test_admitted_in_window_excludes_admission_datetime_semantics():
    """admission_datetime in window does not count without departure_datetime."""
    moment = datetime(2024, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                **_ed_row(),
                "admission_datetime": moment + timedelta(hours=2),
            },
        ]
    )
    with pytest.raises(ValueError, match="departure_datetime"):
        count_observed_admitted_in_window(
            df,
            date(2024, 1, 1),
            (10, 0),
            timedelta(hours=8),
        )


def test_arrived_in_window_on_arrivals_frame():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    arrivals = pd.DataFrame(
        {
            "arrival_datetime": [
                moment + timedelta(hours=1),
                moment + timedelta(hours=9),
            ],
        }
    )
    n = count_observed_arrived_in_window(
        arrivals,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
    )
    assert n == 1


def test_arrived_in_window_does_not_use_ed_visits_fallback():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    ed_visits = pd.DataFrame(
        {
            "arrival_datetime": [moment + timedelta(hours=1)],
            "snapshot_date": [date(2024, 1, 1)],
            "prediction_time": [(10, 0)],
        }
    )
    n = count_observed(
        "arrived_in_window",
        snapshot_date=date(2024, 1, 1),
        prediction_time=(10, 0),
        prediction_window=timedelta(hours=8),
        ed_visits=ed_visits,
        inpatient_arrivals=None,
    )
    assert n == 0


def test_departed_in_window_requires_outcome_column():
    df = pd.DataFrame([{"snapshot_date": date(2024, 1, 1), "prediction_time": (10, 0)}])
    with pytest.raises(ValueError, match="left_subspecialty_in_window"):
        count_observed_departed_in_window(
            df,
            date(2024, 1, 1),
            (10, 0),
            timedelta(hours=8),
        )


def test_departed_in_window_counts_label_not_datetime():
    df = pd.DataFrame(
        [
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": (10, 0),
                "left_subspecialty_in_window": True,
                "specialty": "medical",
            },
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": (10, 0),
                "left_subspecialty_in_window": False,
                "specialty": "medical",
            },
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": (10, 0),
                "left_subspecialty_in_window": True,
                "specialty": "surgical",
            },
        ]
    )
    n = count_observed_departed_in_window(
        df,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
        specialty="medical",
    )
    assert n == 1


def test_arrived_and_admitted_in_window_on_arrivals():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    arrivals = pd.DataFrame(
        [
            {
                "arrival_datetime": moment - timedelta(hours=1),
                "departure_datetime": moment + timedelta(hours=2),
            },
            {
                "arrival_datetime": moment + timedelta(hours=1),
                "departure_datetime": moment + timedelta(hours=2),
            },
            {
                "arrival_datetime": moment + timedelta(hours=1),
                "departure_datetime": moment + timedelta(hours=12),
            },
        ]
    )
    n = count_observed_arrived_and_admitted_in_window(
        arrivals,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
    )
    assert n == 1


@pytest.mark.parametrize("mode", OBSERVATION_MODES)
def test_count_observed_dispatcher_accepts_each_mode(mode):
    df = pd.DataFrame([_ed_row()])
    if mode == "admitted_in_window":
        df["departure_datetime"] = datetime(2024, 1, 1, 12, 0, 0)
    kwargs = dict(
        snapshot_date=date(2024, 1, 1),
        prediction_time=(10, 0),
        prediction_window=timedelta(hours=8),
        ed_visits=None,
        inpatient_arrivals=None,
        inpatient_visits=None,
    )
    if mode in ("admitted_at_some_point", "admitted_in_window"):
        kwargs["ed_visits"] = df
    if mode == "departed_in_window":
        kwargs["inpatient_visits"] = pd.DataFrame(
            [
                {
                    "snapshot_date": date(2024, 1, 1),
                    "prediction_time": (10, 0),
                    "left_subspecialty_in_window": True,
                }
            ]
        )
    if mode == "arrived_in_window":
        kwargs["inpatient_arrivals"] = pd.DataFrame(
            {"arrival_datetime": [datetime(2024, 1, 1, 11, 0, 0)]}
        )
    if mode == "arrived_and_admitted_in_window":
        kwargs["inpatient_arrivals"] = pd.DataFrame(
            {
                "arrival_datetime": [datetime(2024, 1, 1, 11, 0, 0)],
                "departure_datetime": [datetime(2024, 1, 1, 12, 0, 0)],
            }
        )
    count_observed(mode, **kwargs)


def test_count_observed_unknown_mode():
    with pytest.raises(ValueError, match="Unknown observation_mode"):
        count_observed(
            "not_a_mode",
            snapshot_date=date(2024, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=1),
        )


def test_departed_in_window_filters_admission_type():
    df = pd.DataFrame(
        [
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": (10, 0),
                "left_subspecialty_in_window": True,
                "admission_type": "elective",
                "specialty": "medical",
            },
            {
                "snapshot_date": date(2024, 1, 1),
                "prediction_time": (10, 0),
                "left_subspecialty_in_window": True,
                "admission_type": "emergency",
                "specialty": "medical",
            },
        ]
    )
    n = count_observed_departed_in_window(
        df,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
        specialty="medical",
        admission_type="elective",
    )
    assert n == 1


def test_validate_observation_mode_for_component():
    validate_observation_mode_for_component("arrivals", "admitted_in_window")
    validate_observation_mode_for_component("departures", "departed_in_window")
    with pytest.raises(ValueError, match="arrivals"):
        validate_observation_mode_for_component("arrivals", "departed_in_window")
    with pytest.raises(ValueError, match="net_flow"):
        validate_observation_mode_for_component("net_flow", "admitted_at_some_point")


def test_evaluate_package_import_paths():
    from patientflow.evaluate.inputs import EvaluationInputsBuilder
    from patientflow.evaluate.legacy_api import calculate_results, calc_mae_mpe
    from patientflow.evaluate.runner import run_evaluation
    from patientflow.evaluate.scalars import ScalarsCollector

    assert callable(calculate_results)
    assert callable(calc_mae_mpe)
    assert callable(run_evaluation)
    assert callable(ScalarsCollector)
    assert EvaluationInputsBuilder is not None


def test_calculate_results_legacy_api():
    from patientflow.evaluate.legacy_api import calculate_results

    r = calculate_results([1, 2], [1.0, 4.0])
    assert r["mae"] == pytest.approx(1.0)
    assert r["mpe"] == pytest.approx(25.0)
