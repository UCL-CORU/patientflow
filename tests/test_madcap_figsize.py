"""figsize and wrap behaviour for stratified MADCAP charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from patientflow.viz.madcap import (
    DEFAULT_MADCAP_BY_GROUP_FIGSIZE,
    MADCAP_MAX_COLS,
    _plot_madcap_by_group_single,
)


def _synthetic_groups(n_groups: int, n_per_group: int = 20):
    rng = np.random.default_rng(0)
    n = n_groups * n_per_group
    proba = rng.random(n)
    label = rng.integers(0, 2, n)
    group = np.array([f"G{i % n_groups}" for i in range(n)])
    return proba, label, group


def test_default_figsize_one_row_for_three_groups():
    proba, label, group = _synthetic_groups(3)
    fig = _plot_madcap_by_group_single(
        proba,
        label,
        group,
        (6, 0),
        "Age group",
        plot_difference=False,
        return_figure=True,
    )
    assert fig is not None
    assert tuple(fig.get_size_inches()) == DEFAULT_MADCAP_BY_GROUP_FIGSIZE
    assert len(fig.axes) == 3
    plt.close(fig)


def test_explicit_figsize_is_used():
    proba, label, group = _synthetic_groups(3)
    fig = _plot_madcap_by_group_single(
        proba,
        label,
        group,
        (6, 0),
        "Age group",
        plot_difference=False,
        return_figure=True,
        figsize=(12, 2),
    )
    assert fig is not None
    assert tuple(fig.get_size_inches()) == (12.0, 2.0)
    plt.close(fig)


def test_many_groups_wrap_onto_second_row():
    n_groups = MADCAP_MAX_COLS + 3
    proba, label, group = _synthetic_groups(n_groups)
    fig = _plot_madcap_by_group_single(
        proba,
        label,
        group,
        (6, 0),
        "Ethnicity",
        plot_difference=False,
        return_figure=True,
    )
    assert fig is not None
    n_rows = 2
    expected_height = DEFAULT_MADCAP_BY_GROUP_FIGSIZE[1] * n_rows
    assert tuple(fig.get_size_inches()) == (
        DEFAULT_MADCAP_BY_GROUP_FIGSIZE[0],
        expected_height,
    )
    assert len(fig.axes) == n_rows * MADCAP_MAX_COLS
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(hidden) == n_rows * MADCAP_MAX_COLS - n_groups
    plt.close(fig)
