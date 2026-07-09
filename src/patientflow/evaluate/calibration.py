"""Randomised PIT + Cramér–von Mises calibration for discrete predictive distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom, cramervonmises

from patientflow.evaluate.distributions import (
    cdf_from_agg_predicted,
    proba_array_from_agg_predicted,
)

MIN_DISTRIBUTION_SNAPSHOTS = 2
DEFAULT_RPIT_REPETITIONS = 500

BenchmarkCohortKey = str  # "admissions" | "departures"


@dataclass(frozen=True)
class BenchmarkCohortSpec:
    """Eval-split cohort used to compute global p̄ per prediction clock."""

    visits_df: pd.DataFrame
    label_col: str
    prediction_time_col: str = "prediction_time"


@dataclass(frozen=True)
class RpitCvmResult:
    """Summary of rPIT + CvM calibration for one model on one prob_dist_dict slice."""

    mean_w2: float
    w2_values: List[float]
    std_w2: float
    n_observations: int
    n_repetitions: int


def benchmark_cohort_key_for_observation_mode(
    observation_mode: str,
) -> Optional[BenchmarkCohortKey]:
    """Return benchmark cohort key for *observation_mode*, or None if no binomial benchmark."""
    if observation_mode in ("admitted_at_some_point", "admitted_in_window"):
        return "admissions"
    if observation_mode == "departed_in_window":
        return "departures"
    return None


def global_p_bar_by_prediction_time(
    visits_df: pd.DataFrame,
    prediction_times: Sequence[Tuple[int, int]],
    *,
    label_col: str,
    prediction_time_col: str = "prediction_time",
) -> Dict[Tuple[int, int], float]:
    """Global class balance per prediction clock on the eval-split cohort."""
    if label_col not in visits_df.columns:
        raise ValueError(
            f"benchmark cohort requires column {label_col!r} on visits dataframe"
        )
    if prediction_time_col not in visits_df.columns:
        raise ValueError(
            f"benchmark cohort requires column {prediction_time_col!r} on visits dataframe"
        )

    out: Dict[Tuple[int, int], float] = {}
    labels = visits_df[label_col].astype(bool)
    for pt in prediction_times:
        mask = visits_df[prediction_time_col] == pt
        sub = labels[mask]
        if len(sub) == 0:
            out[pt] = 0.0
        else:
            out[pt] = float(sub.mean())
    return out


def binomial_pmf_from_n_p(
    n: int,
    p_bar: float,
    *,
    max_k: Optional[int] = None,
) -> np.ndarray:
    """Probability mass on ``0..max_k`` for ``Binomial(n, p_bar)``."""
    n = max(0, int(n))
    p_bar = float(np.clip(p_bar, 0.0, 1.0))
    upper = int(max_k if max_k is not None else n)
    upper = max(upper, n)
    support = np.arange(0, upper + 1, dtype=int)
    return np.asarray(binom.pmf(support, n, p_bar), dtype=float)


def cohort_size_from_leaf(leaf: Mapping[str, Any]) -> int:
    """Infer trial count *n* from patientflow PMF support (length − 1)."""
    proba = proba_array_from_agg_predicted(leaf.get("agg_predicted"))
    return max(0, len(proba) - 1)


def build_benchmark_prob_dist_dict(
    patientflow_prob_dist_dict: Mapping[Any, Mapping[str, Any]],
    *,
    p_bar: float,
) -> Dict[Any, Dict[str, Any]]:
    """Mirror snapshot keys with ``Binomial(n, p_bar)`` PMFs; copy ``agg_observed``."""
    out: Dict[Any, Dict[str, Any]] = {}
    for snap, leaf in patientflow_prob_dist_dict.items():
        if not isinstance(leaf, Mapping):
            continue
        n = cohort_size_from_leaf(leaf)
        proba = binomial_pmf_from_n_p(n, p_bar, max_k=n)
        out[snap] = {
            "agg_predicted": {"agg_proba": proba},
            "agg_observed": leaf.get("agg_observed", 0),
        }
    return out


def _iter_valid_leaves(
    prob_dist_dict: Mapping[Any, Mapping[str, Any]],
) -> List[Tuple[int, Callable[[float], float]]]:
    pairs: List[Tuple[int, Callable[[float], float]]] = []
    for _snap, leaf in prob_dist_dict.items():
        if not isinstance(leaf, Mapping):
            continue
        if "agg_predicted" not in leaf:
            continue
        try:
            x = int(leaf.get("agg_observed", 0) or 0)
            cdf = cdf_from_agg_predicted(leaf["agg_predicted"])
        except (TypeError, ValueError):
            continue
        pairs.append((x, cdf))
    return pairs


def _rpit_draws(
    observations: Sequence[int],
    cdfs: Sequence[Callable[[float], float]],
    rng: np.random.Generator,
) -> np.ndarray:
    u = np.empty(len(observations), dtype=float)
    for i, (x, cdf) in enumerate(zip(observations, cdfs)):
        p_lo = cdf(float(x - 1)) if x > 0 else 0.0
        p_hi = cdf(float(x))
        if p_hi <= p_lo:
            u[i] = p_lo
        else:
            u[i] = rng.uniform(p_lo, p_hi)
    return u


def cramervonmises_w2_uniform(u: np.ndarray) -> float:
    """One-sample CvM W² against Uniform(0, 1)."""
    return float(cramervonmises(np.asarray(u, dtype=float), cdf="uniform").statistic)


def rpit_cvm_calibration_score(
    prob_dist_dict: Mapping[Any, Mapping[str, Any]],
    *,
    n_repetitions: int = DEFAULT_RPIT_REPETITIONS,
    seed: Optional[int] = None,
) -> Optional[RpitCvmResult]:
    """Run rPIT + CvM; return None if fewer than two valid snapshot leaves."""
    pairs = _iter_valid_leaves(prob_dist_dict)
    n_obs = len(pairs)
    if n_obs < MIN_DISTRIBUTION_SNAPSHOTS:
        return None

    observations = [p[0] for p in pairs]
    cdfs = [p[1] for p in pairs]
    rng = np.random.default_rng(seed)
    w2_values: List[float] = []
    for _ in range(n_repetitions):
        u = _rpit_draws(observations, cdfs, rng)
        w2_values.append(cramervonmises_w2_uniform(u))

    arr = np.asarray(w2_values, dtype=float)
    return RpitCvmResult(
        mean_w2=float(arr.mean()),
        w2_values=w2_values,
        std_w2=float(arr.std(ddof=0)),
        n_observations=n_obs,
        n_repetitions=n_repetitions,
    )


def rpit_cvm_result_to_scalar_fields(
    result: RpitCvmResult,
    *,
    prefix: str = "rpit_cvm",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Map ``RpitCvmResult`` to scalar row fields with a given key prefix."""
    return {
        f"{prefix}_mean_w2": result.mean_w2,
        f"{prefix}_std_w2": result.std_w2,
        f"{prefix}_n_observations": result.n_observations,
        f"{prefix}_n_repetitions": result.n_repetitions,
        f"{prefix}_seed": seed,
    }
