"""Unit tests for evaluation chart policy helpers."""

from __future__ import annotations

import pytest

from patientflow.evaluate.chart_policy import (
    CHART_ABS_W2_FLAG_THRESHOLD,
    DEFAULT_CHART_MODE,
    decide_panelled_figure,
    distribution_chart_flagged,
    normalize_chart_mode,
    panelled_clock_chart_fields,
    sample_ok_for_chart,
)


def test_default_chart_mode_is_flagged():
    assert DEFAULT_CHART_MODE == "flagged"


def test_normalize_chart_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown charts mode"):
        normalize_chart_mode("sometimes")


def test_distribution_chart_flagged_negative_reduction_wins():
    assert distribution_chart_flagged(
        {
            "rpit_cvm_mean_w2": 0.05,
            "rpit_cvm_w2_reduction": -0.01,
        }
    )
    assert not distribution_chart_flagged(
        {
            "rpit_cvm_mean_w2": 5.0,
            "rpit_cvm_w2_reduction": 0.5,
        }
    )


def test_distribution_chart_flagged_absolute_fallback_when_no_reduction():
    assert not distribution_chart_flagged(
        {"rpit_cvm_mean_w2": CHART_ABS_W2_FLAG_THRESHOLD - 0.01}
    )
    assert distribution_chart_flagged({"rpit_cvm_mean_w2": CHART_ABS_W2_FLAG_THRESHOLD})


def test_sample_ok_for_chart_gate_a():
    assert not sample_ok_for_chart(29)
    assert sample_ok_for_chart(30)


def test_decide_panelled_figure_modes():
    ok = [True, False, True]
    flagged = [False, False, True]
    assert not decide_panelled_figure("none", panel_sample_ok=ok, panel_flagged=flagged)
    assert decide_panelled_figure("all", panel_sample_ok=ok, panel_flagged=flagged)
    assert decide_panelled_figure("flagged", panel_sample_ok=ok, panel_flagged=flagged)
    assert not decide_panelled_figure(
        "flagged",
        panel_sample_ok=ok,
        panel_flagged=[False, False, False],
    )
    # Arrivals in flagged: all False flags → no figure
    assert not decide_panelled_figure(
        "flagged",
        panel_sample_ok=[True, True],
        panel_flagged=[False, False],
    )


def test_panelled_clock_chart_fields():
    assert panelled_clock_chart_fields("all", sample_ok=False, write_figure=True) == {
        "charts_generated": False,
        "skip_reason": "insufficient_observations",
    }
    assert panelled_clock_chart_fields("none", sample_ok=True, write_figure=False) == {
        "charts_generated": False,
        "skip_reason": "charts_disabled",
    }
    assert panelled_clock_chart_fields(
        "flagged", sample_ok=True, write_figure=False
    ) == {
        "charts_generated": False,
        "skip_reason": "not_flagged",
    }
    assert panelled_clock_chart_fields(
        "flagged", sample_ok=True, write_figure=True
    ) == {"charts_generated": True}
