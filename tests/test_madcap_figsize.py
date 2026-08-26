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
    madcap_by_group_figsize,
)


def _synthetic_groups(n_groups: int, n_per_group: int = 20):
    rng = np.random.default_rng(0)
    n = n_groups * n_per_group
    proba = rng.random(n)
    label = rng.integers(0, 2, n)
    group = np.array([f"G{i % n_groups}" for i in range(n)])
    return proba, label, group


def _visible_box_aspects(fig) -> list[float]:
    aspects = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        aspect = ax.get_box_aspect()
        if aspect is not None:
            aspects.append(float(aspect))
    return aspects


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
    assert tuple(fig.get_size_inches()) == madcap_by_group_figsize(1, 3)
    assert tuple(fig.get_size_inches()) == DEFAULT_MADCAP_BY_GROUP_FIGSIZE
    assert len(fig.axes) == 3
    assert _visible_box_aspects(fig) == [1.0, 1.0, 1.0]
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


def test_six_groups_use_two_by_three_grid():
    proba, label, group = _synthetic_groups(6)
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
    assert tuple(fig.get_size_inches()) == madcap_by_group_figsize(2, 3)
    assert len(fig.axes) == 6
    assert all(ax.get_visible() for ax in fig.axes)
    assert _visible_box_aspects(fig) == [1.0] * 6
    plt.close(fig)


def test_many_groups_wrap_onto_second_row():
    n_groups = MADCAP_MAX_COLS + 1
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
    n_cols = MADCAP_MAX_COLS
    assert tuple(fig.get_size_inches()) == madcap_by_group_figsize(n_rows, n_cols)
    assert len(fig.axes) == n_rows * n_cols
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(hidden) == n_rows * n_cols - n_groups
    plt.close(fig)
