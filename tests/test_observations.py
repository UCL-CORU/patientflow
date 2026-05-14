"""Tests for patientflow.evaluate.observations."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from patientflow.evaluate.observations import (
    OBSERVATION_MODES,
    count_observed,
    count_observed_admitted_at_some_point,
    count_observed_admitted_in_window,
    count_observed_arrived_and_admitted_in_window,
    count_observed_arrived_in_window,
    count_observed_departed_in_window,
)


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
    with pytest.raises(ValueError, match="admission_datetime"):
        count_observed_admitted_in_window(
            df,
            date(2024, 1, 1),
            (10, 0),
            timedelta(hours=8),
        )


def test_admitted_in_window_counts():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                **_ed_row(),
                "admission_datetime": moment + timedelta(hours=2),
            },
            {
                **_ed_row(),
                "admission_datetime": moment + timedelta(hours=12),
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


def test_arrived_in_window_on_arrivals_frame():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    visits = pd.DataFrame(
        {
            "arrival_datetime": [
                moment + timedelta(hours=1),
                moment + timedelta(hours=9),
            ],
        }
    )
    n = count_observed_arrived_in_window(
        visits,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
    )
    assert n == 1


def test_departed_in_window_requires_column():
    df = pd.DataFrame([{"snapshot_date": date(2024, 1, 1), "prediction_time": (10, 0)}])
    with pytest.raises(ValueError, match="departure_datetime"):
        count_observed_departed_in_window(
            df,
            date(2024, 1, 1),
            (10, 0),
            timedelta(hours=8),
        )


def test_arrived_and_admitted_in_window():
    moment = datetime(2024, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                **_ed_row(),
                "arrival_datetime": moment - timedelta(hours=1),
                "admission_datetime": moment + timedelta(hours=2),
            },
            {
                **_ed_row(),
                "arrival_datetime": moment + timedelta(hours=1),
                "admission_datetime": moment + timedelta(hours=2),
            },
        ]
    )
    n = count_observed_arrived_and_admitted_in_window(
        df,
        date(2024, 1, 1),
        (10, 0),
        timedelta(hours=8),
    )
    assert n == 1


@pytest.mark.parametrize("mode", OBSERVATION_MODES)
def test_count_observed_dispatcher_accepts_each_mode(mode):
    df = pd.DataFrame([_ed_row()])
    if mode == "admitted_in_window":
        df["admission_datetime"] = datetime(2024, 1, 1, 12, 0, 0)
    if mode == "arrived_and_admitted_in_window":
        df["arrival_datetime"] = datetime(2024, 1, 1, 11, 0, 0)
        df["admission_datetime"] = datetime(2024, 1, 1, 12, 0, 0)
    kwargs = dict(
        snapshot_date=date(2024, 1, 1),
        prediction_time=(10, 0),
        prediction_window=timedelta(hours=8),
        ed_visits=df,
    )
    if mode == "departed_in_window":
        kwargs["inpatient_visits"] = pd.DataFrame(
            [
                {
                    "snapshot_date": date(2024, 1, 1),
                    "prediction_time": (10, 0),
                    "departure_datetime": datetime(2024, 1, 1, 12, 0, 0),
                }
            ]
        )
        del kwargs["ed_visits"]
    if mode == "arrived_in_window":
        kwargs["visits"] = pd.DataFrame(
            {"arrival_datetime": [datetime(2024, 1, 1, 11, 0, 0)]}
        )
        del kwargs["ed_visits"]
    count_observed(mode, **kwargs)


def test_count_observed_unknown_mode():
    with pytest.raises(ValueError, match="Unknown observation_mode"):
        count_observed(
            "not_a_mode",
            snapshot_date=date(2024, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=1),
        )


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
