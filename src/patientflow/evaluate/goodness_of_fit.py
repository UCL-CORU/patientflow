"""Pearson and Monte Carlo goodness-of-fit helpers for evaluation.

Generic per-event Categorical simulation against fixed expected counts.
Transfer-specific routing vectors are built in
`patientflow.predict.transfers.build_per_patient_probabilities`.
Used by `patientflow.evaluate.handlers.evaluate_transition_matrix`.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class MultinomialGoFResult:
    """Pearson goodness-of-fit outcome for one source subspecialty.

    Attributes
    ----------
    pearson_x2 : float
        Observed Pearson statistic on the aligned count vectors.
    p_value : float
        Monte Carlo p-value: proportion of simulated statistics at least as
        extreme as `pearson_x2`.
    n_observed : int
        Total departures (`sum of observed_counts`).
    n_simulations : int
        Monte Carlo draws used for the p-value.
    n_structural_violations : int
        Destinations with `E_d == 0` but `n_obs_d > 0`; reported separately and
        excluded from the Pearson sum.
    destinations : list of str
        Labels aligned 1-1 with `expected_counts` and `observed_counts`.
    expected_counts : list of float
        Patient-level aggregate `E_d = sum_i p_i(d)`, not `N` times a pooled row.
    observed_counts : list of int
        Observed destination counts aligned with `destinations`.
    """

    pearson_x2: float
    p_value: float
    n_observed: int
    n_simulations: int
    n_structural_violations: int
    destinations: List[str]
    expected_counts: List[float]
    observed_counts: List[int]


def pearson_x2(observed: np.ndarray, expected: np.ndarray) -> float:
    """Return Pearson X² using only cells with strictly positive expectation.

    Parameters
    ----------
    observed, expected : numpy.ndarray
        Aligned count vectors over destinations.

    Returns
    -------
    float
        sum over destinations with E_d > 0 of (n_d - E_d)^2 / E_d. Returns 0.0
        when no cell has positive expectation.
    """
    mask = expected > 0
    if not np.any(mask):
        return 0.0
    diff = observed[mask].astype(float) - expected[mask]
    return float(np.sum(diff**2 / expected[mask]))


def count_structural_violations(
    observed: np.ndarray, expected: np.ndarray
) -> int:
    """Count destinations with zero expectation but positive observation."""
    return int(np.sum((expected == 0) & (observed > 0)))


def derive_seed_offset(base_seed: int | None, source: str) -> int | None:
    """Derive a deterministic per-source RNG seed from a slice-level seed."""
    if base_seed is None:
        return None
    offset = zlib.crc32(source.encode("utf-8"))
    return int((base_seed + offset) % (2**32 - 1))


def multinomial_gof_montecarlo(
    P: np.ndarray,
    observed_counts: np.ndarray,
    destinations: Sequence[str],
    *,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> MultinomialGoFResult:
    """Run a Monte Carlo Pearson test with per-event Categorical simulation.

    Compares observed destination counts to fixed patient-level expectations
    `E = sum(P, axis=0)`. Because each departure can have a different routing
    vector, the null is simulated by drawing each event's destination from its
    own row of `P`, not from a single shared `Multinomial(N, p_bar)`.

    Algorithm:

    1. Compute `T_obs = Pearson(observed, E)` using only destinations with
       `E_d > 0`.
    2. For each of `n_simulations` draws, sample one destination per event from
       `Categorical(p_i)`, aggregate simulated counts, and compute `T_sim`.
    3. Return `p_value = (1 + count(T_sim >= T_obs)) / (n_simulations + 1)`.

    Destinations with `E_d == 0` are omitted from the Pearson sum. Observed
    traffic at those destinations is counted in `n_structural_violations` only.

    Parameters
    ----------
    P : numpy.ndarray
        Per-event probability matrix with shape (n_events, n_destinations).
    observed_counts : numpy.ndarray
        Observed destination counts aligned with `destinations`.
    destinations : sequence of str
        Destination labels.
    n_simulations : int, optional
        Number of Monte Carlo draws (default 10_000).
    seed : int or None, optional
        RNG seed for reproducibility.

    Returns
    -------
    MultinomialGoFResult
        Observed statistic, Monte Carlo p-value, and aligned count vectors.
    """
    expected = P.sum(axis=0)
    observed = np.asarray(observed_counts, dtype=int)
    n_structural = count_structural_violations(observed, expected)
    t_obs = pearson_x2(observed, expected)

    rng = np.random.default_rng(seed)
    dest_indices = np.arange(len(destinations))
    n_events = P.shape[0]
    exceed = 0

    for _ in range(n_simulations):
        sim_counts = np.zeros(len(destinations), dtype=int)
        for event_idx in range(n_events):
            draw = int(rng.choice(dest_indices, p=P[event_idx]))
            sim_counts[draw] += 1
        t_sim = pearson_x2(sim_counts, expected)
        if t_sim >= t_obs:
            exceed += 1

    p_value = (1 + exceed) / (n_simulations + 1)
    return MultinomialGoFResult(
        pearson_x2=t_obs,
        p_value=float(p_value),
        n_observed=int(observed.sum()),
        n_simulations=n_simulations,
        n_structural_violations=n_structural,
        destinations=list(destinations),
        expected_counts=[float(x) for x in expected],
        observed_counts=[int(x) for x in observed],
    )
