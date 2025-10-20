"""Composable distribution algebra used for demand prediction.

This module introduces a minimal, immutable Distribution abstraction that wraps
discrete probability mass functions (PMFs) with an optional integer offset for
their support. It provides algebraic operations used throughout demand
prediction while mirroring the numerical behavior of the existing
implementation (k-sigma clamping and Poisson truncation).

Scope: Phase 1 only – no changes to existing call sites. This module is
standalone and does not modify current behavior elsewhere.
"""

from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np
from scipy.stats import poisson as _poisson
from scipy.stats import binom as _binom


def _validate_pmf(p: np.ndarray) -> None:
    if p.ndim != 1:
        raise ValueError("PMF must be a 1D array")
    if p.size == 0:
        raise ValueError("PMF must not be empty")
    if np.any(p < 0):
        raise ValueError("PMF must be non-negative")
    if np.any(~np.isfinite(p)):
        raise ValueError("PMF must be finite (no NaN or inf)")


@dataclass(frozen=True)
class Distribution:
    """Immutable discrete distribution with optional integer offset.

    The PMF is defined over integer support {offset, offset+1, ..., offset+N}.
    """

    probabilities: np.ndarray
    offset: int = 0

    # ---- Constructors ----
    @classmethod
    def from_pmf(cls, pmf: np.ndarray, offset: int = 0) -> "Distribution":
        p = np.asarray(pmf, dtype=float).copy()
        _validate_pmf(p)
        return cls(probabilities=p, offset=int(offset))

    @classmethod
    def from_poisson_with_cap(
        cls, lambda_param: float, max_k: int
    ) -> "Distribution":
        if lambda_param < 0:
            raise ValueError("Poisson lambda must be non-negative")
        if lambda_param == 0.0:
            return cls.from_pmf(np.array([1.0]))
        k = np.arange(int(max_k) + 1)
        p = _poisson.pmf(k, float(lambda_param))
        return cls.from_pmf(p)

    @staticmethod
    def poisson_cap_from_k_sigma(lambda_param: float, k_sigma: float) -> int:
        if lambda_param <= 0:
            return 0
        return int(np.ceil(lambda_param + k_sigma * np.sqrt(lambda_param)))

    # ---- Basic properties ----
    def __len__(self) -> int:
        return self.probabilities.shape[0]

    # ---- Algebraic operations (pure) ----
    def convolve(self, other: "Distribution") -> "Distribution":
        """Convolve two distributions (sum of independent integer-valued RVs)."""
        p = np.convolve(self.probabilities, other.probabilities)
        new_offset = self.offset + other.offset
        return Distribution.from_pmf(p, new_offset)

    def clamp_nonnegative(self, cap_max: int) -> "Distribution":
        """Clamp to non-negative support [offset, offset+cap_max] if offset>=0.

        For the common case where offset==0, this mirrors the current
        implementation's deterministic cap to limit array growth.
        """
        if cap_max < 0:
            # retain only the first mass; ensures non-empty result
            return Distribution.from_pmf(self.probabilities[:1], self.offset)
        max_len = cap_max + 1
        if len(self) <= max_len:
            return self
        return Distribution.from_pmf(self.probabilities[:max_len], self.offset)

    def renorm(self) -> "Distribution":
        total = float(np.sum(self.probabilities))
        if total <= 0.0:
            return self
        return Distribution.from_pmf(self.probabilities / total, self.offset)

    def thin(self, prob: float) -> "Distribution":
        """Binomial thinning: each individual is kept with probability `prob`.

        For a count K ~ self, the thinned count J|K ~ Binomial(K, prob). This
        returns the unconditional PMF of J. Mirrors the logic in
        scale_pmf_by_probability for general PMFs.
        """
        if not (0.0 <= prob <= 1.0):
            raise ValueError("Thinning probability must be in [0, 1]")

        n_max = len(self) - 1
        # Result support is [0, n_max] regardless of offset (thinning is defined
        # on the non-negative count component). Offset stays the same.
        result = np.zeros(n_max + 1)

        # P(0 successes) accumulates contributions from all k with j=0
        # We'll compute generically using binomial PMF for clarity.
        for k in range(0, n_max + 1):
            pk = self.probabilities[k]
            if pk == 0.0:
                continue
            # Vector of size k+1: j=0..k
            j_vals = np.arange(k + 1)
            bj = _binom.pmf(j_vals, k, prob)
            # Accumulate into result[0..k]
            result[: k + 1] += pk * bj

        return Distribution.from_pmf(result, 0)

    def net(self, other: "Distribution", k_sigma: float) -> "Distribution":
        """Distribution of the difference (self - other) with asymmetric clamp.

        The result support is in [-(max other), +(max self)] translated by an
        integer offset. Uses k-sigma truncation intersected with physical bounds,
        mirroring the behavior in DemandPredictor._compute_net_flow_pmf.
        """
        # Renormalize defensively to avoid drift
        p_a = self.renorm().probabilities
        p_d = other.renorm().probabilities

        max_a = len(p_a) - 1
        max_d = len(p_d) - 1

        # Net ranges from -max_d .. +max_a; index shift by +max_d
        net_size = max_a + max_d + 1
        p_net = np.zeros(net_size)

        for a in range(len(p_a)):
            if p_a[a] == 0.0:
                continue
            # Vectorized add for all d
            for d in range(len(p_d)):
                if p_d[d] == 0.0:
                    continue
                idx = (a - d) + max_d
                p_net[idx] += p_a[a] * p_d[d]

        # Asymmetric k-sigma clamping with physical bounds
        initial_offset = -max_d

        mean_a = float(np.sum(np.arange(len(p_a)) * p_a))
        var_a = float(np.sum((np.arange(len(p_a)) ** 2) * p_a) - mean_a * mean_a)
        mean_d = float(np.sum(np.arange(len(p_d)) * p_d))
        var_d = float(np.sum((np.arange(len(p_d)) ** 2) * p_d) - mean_d * mean_d)

        mean_net = mean_a - mean_d
        std_net = float(np.sqrt(max(0.0, var_a + var_d)))

        physical_min = -max_d
        physical_max = max_a

        left_bound = int(np.floor(mean_net - k_sigma * std_net))
        right_bound = int(np.ceil(mean_net + k_sigma * std_net))

        left_value = max(physical_min, left_bound)
        right_value = min(physical_max, right_bound)

        start_idx = max(0, left_value - physical_min)
        end_idx = min(len(p_net), right_value - physical_min + 1)

        if start_idx >= end_idx:
            start_idx = max(0, min(len(p_net) - 1, start_idx))
            end_idx = start_idx + 1

        p_trunc = p_net[start_idx:end_idx]
        final_offset = initial_offset + start_idx

        return Distribution.from_pmf(p_trunc, final_offset)

    # ---- Statistics ----
    def expected(self) -> float:
        idx = np.arange(len(self)) + self.offset
        return float(np.sum(idx * self.probabilities))

    def variance(self) -> float:
        idx = np.arange(len(self)) + self.offset
        mean = float(np.sum(idx * self.probabilities))
        mean_sq = float(np.sum((idx ** 2) * self.probabilities))
        return max(0.0, mean_sq - mean * mean)

    def percentiles(self, percentile_list: Iterable[int]) -> Dict[int, int]:
        cumsum = np.cumsum(self.probabilities)
        result: Dict[int, int] = {}
        for pct in percentile_list:
            idx = int(np.searchsorted(cumsum, pct / 100.0))
            result[int(pct)] = int(idx + self.offset)
        return result

    # ---- Pretty helpers ----
    def head(self, max_display: int = 10, precision: int = 3) -> str:
        p = self.probabilities
        if len(p) <= max_display:
            values = ", ".join(f"{v:.{precision}g}" for v in p)
            return f"PMF[{self.offset}:{self.offset+len(p)}]: [{values}]"
        # Center on mode
        mode_idx = int(np.argmax(p))
        half = max_display // 2
        start = max(0, mode_idx - half)
        end = min(len(p), start + max_display)
        if end - start < max_display:
            start = max(0, end - max_display)
        values = ", ".join(f"{v:.{precision}g}" for v in p[start:end])
        return f"PMF[{start + self.offset}:{end + self.offset}]: [{values}]"


