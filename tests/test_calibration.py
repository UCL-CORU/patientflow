"""Tests for patientflow.evaluate.calibration."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from patientflow.evaluate.calibration import (
    MIN_DISTRIBUTION_SNAPSHOTS,
    benchmark_cohort_key_for_observation_mode,
    binomial_pmf_from_n_p,
    build_benchmark_prob_dist_dict,
    global_p_bar_by_prediction_time,
    rpit_cvm_calibration_score,
)


def _leaf(observed: int, proba: list[float]) -> dict:
    return {
        "agg_observed": observed,
        "agg_predicted": {"agg_proba": np.array(proba, dtype=float)},
    }


def test_benchmark_cohort_key_for_observation_mode():
    assert (
        benchmark_cohort_key_for_observation_mode("admitted_at_some_point")
        == "admissions"
    )
    assert (
        benchmark_cohort_key_for_observation_mode("departed_in_window") == "departures"
    )
    assert benchmark_cohort_key_for_observation_mode("arrived_in_window") is None


def test_global_p_bar_by_prediction_time():
    import pandas as pd

    df = pd.DataFrame(
        {
            "prediction_time": [(9, 0), (9, 0), (12, 0)],
            "is_admitted": [1, 0, 1],
        }
    )
    out = global_p_bar_by_prediction_time(
        df, [(9, 0), (12, 0)], label_col="is_admitted"
    )
    assert out[(9, 0)] == pytest.approx(0.5)
    assert out[(12, 0)] == pytest.approx(1.0)


def test_rpit_cvm_returns_none_when_n_too_small():
    snap = date(2024, 1, 1)
    dist = {snap: _leaf(0, [1.0, 0.0])}
    assert rpit_cvm_calibration_score(dist) is None
    assert MIN_DISTRIBUTION_SNAPSHOTS == 2


def test_rpit_cvm_reproducible_with_seed():
    snap1, snap2 = date(2024, 1, 1), date(2024, 1, 2)
    dist = {
        snap1: _leaf(1, [0.25, 0.5, 0.25]),
        snap2: _leaf(0, [0.7, 0.2, 0.1]),
    }
    a = rpit_cvm_calibration_score(dist, n_repetitions=20, seed=42)
    b = rpit_cvm_calibration_score(dist, n_repetitions=20, seed=42)
    assert a is not None and b is not None
    assert a.w2_values == b.w2_values


def test_well_calibrated_lower_w2_than_miscalibrated():
    snap1, snap2, snap3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
    n, p = 10, 0.3
    pmf = binomial_pmf_from_n_p(n, p, max_k=n)
    good = {
        snap1: _leaf(int(np.random.binomial(n, p)), pmf),
        snap2: _leaf(int(np.random.binomial(n, p)), pmf),
        snap3: _leaf(int(np.random.binomial(n, p)), pmf),
    }
    bad = {
        snap1: _leaf(0, pmf),
        snap2: _leaf(n, pmf),
        snap3: _leaf(0, pmf),
    }
    good_res = rpit_cvm_calibration_score(good, n_repetitions=100, seed=1)
    bad_res = rpit_cvm_calibration_score(bad, n_repetitions=100, seed=1)
    assert good_res is not None and bad_res is not None
    assert good_res.mean_w2 < bad_res.mean_w2


def test_benchmark_build_copies_observed():
    snap = date(2024, 1, 1)
    pf = {snap: _leaf(3, [0.1] * 11)}
    bench = build_benchmark_prob_dist_dict(pf, p_bar=0.3)
    assert bench[snap]["agg_observed"] == 3
    assert len(bench[snap]["agg_predicted"]["agg_proba"]) == 11


def test_patientflow_lower_w2_than_binomial_benchmark():
    """Heterogeneous observed counts vs uniform p: patientflow-shaped PMFs win."""
    snap1, snap2, snap3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
    n = 8
    p_bar = 0.25
    pf = {
        snap1: _leaf(2, binomial_pmf_from_n_p(n, 0.2, max_k=n)),
        snap2: _leaf(2, binomial_pmf_from_n_p(n, 0.35, max_k=n)),
        snap3: _leaf(2, binomial_pmf_from_n_p(n, 0.25, max_k=n)),
    }
    bench = build_benchmark_prob_dist_dict(pf, p_bar=p_bar)
    pf_res = rpit_cvm_calibration_score(pf, n_repetitions=80, seed=7)
    bm_res = rpit_cvm_calibration_score(bench, n_repetitions=80, seed=8)
    assert pf_res is not None and bm_res is not None
    assert pf_res.mean_w2 < bm_res.mean_w2
