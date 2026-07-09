"""Tests for patientflow.generate."""

from __future__ import annotations

import pandas as pd

from patientflow.generate import synthesise_departure_times


def test_synthesise_departure_times_ed_visits():
    df = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2024-01-01").date()],
            "prediction_time": [(10, 0)],
            "is_admitted": [True],
        }
    )
    out = synthesise_departure_times(df, kind="ed_visits", seed=0)
    assert "departure_datetime" in out.columns
    assert out["departure_datetime"].notna().all()


def test_synthesise_departure_times_inpatient_arrivals():
    df = pd.DataFrame(
        {
            "arrival_datetime": pd.to_datetime(["2024-01-01 08:00:00"], utc=True),
        }
    )
    out = synthesise_departure_times(df, kind="inpatient_arrivals", seed=0)
    assert "departure_datetime" in out.columns
    assert out["departure_datetime"].notna().all()
