"""Discrete distribution helpers shared by evaluate calibration and viz."""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd


def proba_array_from_agg_predicted(agg_predicted: Any) -> np.ndarray:
    """Return a 1-D probability mass array from an ``agg_predicted`` payload."""
    if isinstance(agg_predicted, dict) and "agg_proba" in agg_predicted:
        return np.asarray(agg_predicted["agg_proba"], dtype=float).flatten()
    if isinstance(agg_predicted, pd.DataFrame) and "agg_proba" in agg_predicted.columns:
        return np.asarray(agg_predicted["agg_proba"].to_numpy(), dtype=float).flatten()
    if isinstance(agg_predicted, pd.Series):
        return np.asarray(agg_predicted.values, dtype=float).flatten()
    raise TypeError(
        "agg_predicted must be a dict with 'agg_proba', a DataFrame with "
        "'agg_proba', or a Series"
    )


def cdf_from_proba_array(proba: np.ndarray) -> Callable[[float], float]:
    """Build a discrete CDF callable ``F(k) = P(X <= k)`` from a PMF array."""
    p = np.asarray(proba, dtype=float).flatten()
    if p.size == 0:
        return lambda x: 0.0
    cum = np.cumsum(p)
    values = np.arange(len(p))

    def cdf(x: float) -> float:
        if x < values[0]:
            return 0.0
        idx = int(np.floor(x))
        if idx >= len(cum):
            return float(cum[-1])
        return float(cum[idx])

    return cdf


def cdf_from_agg_predicted(agg_predicted: Any) -> Callable[[float], float]:
    """Build a CDF callable from an ``agg_predicted`` leaf field."""
    return cdf_from_proba_array(proba_array_from_agg_predicted(agg_predicted))


def prob_to_cdf(
    prob_dist: Union[pd.Series, pd.DataFrame, Dict, np.ndarray],
) -> Callable[[float], float]:
    """Convert a probability distribution to a CDF function (viz-compatible API)."""
    import pandas as pd

    if isinstance(prob_dist, pd.DataFrame):
        if len(prob_dist) > 0:
            prob_series = prob_dist.iloc[0]
        else:
            raise ValueError("Empty DataFrame provided")
        values = list(prob_series.index)
        probs = list(prob_series.values)
    elif isinstance(prob_dist, pd.Series):
        values = list(prob_dist.index)
        probs = list(prob_dist.values)
    elif isinstance(prob_dist, dict):
        sorted_items = sorted(prob_dist.items())
        values = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
    else:
        values = list(range(len(prob_dist)))
        probs = list(prob_dist)

    sorted_pairs = sorted(zip(values, probs))
    vals = [pair[0] for pair in sorted_pairs]
    probs_sorted = [pair[1] for pair in sorted_pairs]
    cum_probs = np.cumsum(probs_sorted)

    def cdf_function(x: float) -> float:
        if x < vals[0]:
            return 0.0
        for i, val in enumerate(vals):
            if x <= val:
                return float(cum_probs[i])
        return 1.0

    return cdf_function
