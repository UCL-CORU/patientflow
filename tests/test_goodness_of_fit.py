"""Tests for patientflow.evaluate.goodness_of_fit.

Unit tests for the Pearson statistic and Monte Carlo p-value used by
transition-matrix evaluation. These use synthetic probability matrices unless
noted; they do not depend on TransferProbabilityEstimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from patientflow.evaluate.goodness_of_fit import (
    derive_seed_offset,
    multinomial_gof_montecarlo,
    pearson_x2,
)


def test_pearson_x2_hand_worked_and_zero_expected_cells():
    """Pearson sum uses only E_d > 0; all-zero expectation returns 0."""
    observed = np.array([3, 1, 2])
    expected = np.array([2.0, 0.0, 2.0])
    # Only cells with E > 0: (3-2)^2/2 + (2-2)^2/2 = 0.5
    assert pearson_x2(observed, expected) == pytest.approx(0.5)
    assert pearson_x2(np.array([0, 0]), np.array([0.0, 0.0])) == 0.0


def test_multinomial_gof_seed_reproducibility():
    """Same seed reproduces pearson_x2 and p_value across repeated runs."""
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    observed = np.array([1, 1])
    destinations = ["a", "b"]
    first = multinomial_gof_montecarlo(
        p, observed, destinations, n_simulations=200, seed=42
    )
    second = multinomial_gof_montecarlo(
        p, observed, destinations, n_simulations=200, seed=42
    )
    assert first.pearson_x2 == second.pearson_x2
    assert first.p_value == second.p_value


def test_derive_seed_offset_differs_by_source():
    """Per-source seed offsets differ so each subspecialty has its own RNG stream."""
    assert derive_seed_offset(10, "cardiology") != derive_seed_offset(10, "surgery")


def test_structural_violations_excluded_from_pearson_sum():
    """Observed traffic at E_d=0 counts as structural violation, not in Pearson sum."""
    p = np.array([[1.0, 0.0], [1.0, 0.0]])
    observed = np.array([1, 1])
    destinations = ["a", "b"]
    result = multinomial_gof_montecarlo(
        p, observed, destinations, n_simulations=50, seed=1
    )
    assert result.n_structural_violations == 1
    # Pearson uses only E_a = 2: (1 - 2)^2 / 2 = 0.5; the violation at b is excluded.
    assert result.pearson_x2 == pytest.approx(0.5)


def test_multinomial_gof_well_calibrated_p_values_approximately_uniform():
    """Under the null, p-values from independently simulated datasets are ~uniform."""
    destinations = ["a", "b", "c"]
    dest_indices = np.arange(len(destinations))
    p_values = []

    for rep in range(50):
        rng = np.random.default_rng(rep)
        n_events = 50
        # Heterogeneous per-event routing; observed counts drawn from the same p_i.
        p_matrix = rng.dirichlet(np.ones(len(destinations)), size=n_events)
        n_obs = np.zeros(len(destinations), dtype=int)
        for event_idx in range(n_events):
            draw = int(rng.choice(dest_indices, p=p_matrix[event_idx]))
            n_obs[draw] += 1

        result = multinomial_gof_montecarlo(
            p_matrix,
            n_obs,
            destinations,
            n_simulations=800,
            seed=10_000 + rep,
        )
        p_values.append(result.p_value)

    assert np.mean(p_values) == pytest.approx(0.5, abs=0.15)
