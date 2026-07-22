"""Chart emission policy for evaluation runs.

Scalars always emit; PNG charts are optional and gated by sample size, inactive
mass, and (in ``flagged`` mode) scalar drill-down signals.

See Also
--------
patientflow.evaluate.runner.run_evaluation
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from patientflow.evaluate.scalars import RELIABILITY_MIN_OBSERVATIONS_DISTRIBUTION

ChartMode = Literal["none", "flagged", "all"]

DEFAULT_CHART_MODE: ChartMode = "flagged"

# Gate A: enough observations for a human to trust a panel visual.
CHART_GATE_A_MIN_OBSERVATIONS: int = RELIABILITY_MIN_OBSERVATIONS_DISTRIBUTION

# Absolute W² fallback when no ``*_w2_reduction`` fields are present on the row.
CHART_ABS_W2_FLAG_THRESHOLD: float = 1.0

SKIP_INACTIVE_SERVICE = "inactive_service"
SKIP_INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
SKIP_NOT_FLAGGED = "not_flagged"
SKIP_CHARTS_DISABLED = "charts_disabled"

_VALID_CHART_MODES: frozenset[str] = frozenset({"none", "flagged", "all"})


def normalize_chart_mode(charts: str | ChartMode) -> ChartMode:
    """Return a validated :data:`ChartMode`.

    Parameters
    ----------
    charts :
        One of ``"none"``, ``"flagged"``, or ``"all"``.

    Returns
    -------
    ChartMode

    Raises
    ------
    ValueError
        If *charts* is not a recognised mode.
    """
    if charts not in _VALID_CHART_MODES:
        raise ValueError(
            f"Unknown charts mode {charts!r}; expected one of "
            f"{sorted(_VALID_CHART_MODES)}"
        )
    return charts  # type: ignore[return-value]


def distribution_chart_flagged(row: Mapping[str, Any]) -> bool:
    """Return whether a distribution scalar row asks for chart drill-down.

    Prefer relative signals: any present ``*_w2_reduction`` field that is
    strictly negative (model worse than that benchmark). When no reduction
    fields exist, fall back to ``rpit_cvm_mean_w2 >= CHART_ABS_W2_FLAG_THRESHOLD``.
    """
    reductions = [
        float(v)
        for k, v in row.items()
        if k.endswith("_w2_reduction") and v is not None
    ]
    if reductions:
        return any(r < 0.0 for r in reductions)
    w2 = row.get("rpit_cvm_mean_w2")
    if w2 is None:
        return False
    return float(w2) >= CHART_ABS_W2_FLAG_THRESHOLD


def sample_ok_for_chart(n_observations: int) -> bool:
    """Gate A: enough snapshot leaves / histogram days to trust a panel."""
    return int(n_observations) >= CHART_GATE_A_MIN_OBSERVATIONS


def decide_panelled_figure(
    charts: ChartMode,
    *,
    panel_sample_ok: Sequence[bool],
    panel_flagged: Sequence[bool],
) -> bool:
    """Return whether to write a panelled (one-file-per-service) figure.

    Parameters
    ----------
    charts :
        Chart emission mode.
    panel_sample_ok :
        Per-clock Gate A pass flags (same length as *panel_flagged*).
    panel_flagged :
        Per-clock Gate C drill-down flags.

    Notes
    -----
    When the figure is written, **all** Gate-A clocks are drawn as panels
    (not only flagged ones). Arrivals in ``flagged`` mode never write: callers
    pass all-``False`` *panel_flagged*.
    """
    if charts == "none":
        return False
    if not any(panel_sample_ok):
        return False
    if charts == "all":
        return True
    # flagged
    return any(ok and flagged for ok, flagged in zip(panel_sample_ok, panel_flagged))


def panelled_clock_chart_fields(
    charts: ChartMode,
    *,
    sample_ok: bool,
    write_figure: bool,
) -> dict[str, Any]:
    """Return ``charts_generated`` / optional ``skip_reason`` for one clock row.

    Call only for active (non-inactive) services after Gate A / mode decisions.
    Rows with fewer than the rPIT minimum snapshots should be handled by the
    caller before this helper (``insufficient_observations``, no rPIT).
    """
    if not sample_ok:
        return {
            "charts_generated": False,
            "skip_reason": SKIP_INSUFFICIENT_OBSERVATIONS,
        }
    if charts == "none":
        return {
            "charts_generated": False,
            "skip_reason": SKIP_CHARTS_DISABLED,
        }
    if write_figure:
        return {"charts_generated": True}
    return {
        "charts_generated": False,
        "skip_reason": SKIP_NOT_FLAGGED,
    }
