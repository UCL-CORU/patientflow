"""Visualization module for plotting aspirational curves in patient flow analysis.

This module provides functionality for creating and customizing plots of aspirational
curves, which represent the probability of admission over time. These curves are
useful for setting aspirational targets in healthcare settings.

Functions
---------
plot_curve : function
    Plot an aspirational curve with specified points and optional annotations

Examples
--------
>>> plot_curve(
...     title="Admission Probability Curve",
...     x1=4,
...     y1=0.2,
...     x2=24,
...     y2=0.8,
...     include_titles=True
... )
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from patientflow.calculate.admission_in_prediction_window import (
    create_curve,
)
from patientflow.viz.utils import clean_title_for_filename


def plot_curve(
    title,
    x1,
    y1,
    x2,
    y2,
    figsize=(10, 5),
    include_titles=False,
    text_size=14,
    media_file_path=None,
    file_name=None,
    return_figure=False,
    annotate_points=False,
):
    """Plot an aspirational curve with specified points and optional annotations.

    This function creates a plot of an aspirational curve between two points,
    with options for customization of the visualization including titles,
    annotations, and saving to a file.

    Parameters
    ----------
    title : str
        The title of the plot.
    x1 : float
        x-coordinate of the first point.
    y1 : float
        y-coordinate of the first point (probability value).
    x2 : float
        x-coordinate of the second point.
    y2 : float
        y-coordinate of the second point (probability value).
    figsize : tuple of int, optional
        Figure size in inches (width, height), by default (10, 5).
    include_titles : bool, optional
        Whether to include axis labels and title, by default False.
    text_size : int, optional
        Font size for text elements, by default 14.
    media_file_path : str or Path, optional
        Path to save the plot image, by default None.
    file_name : str, optional
        Custom filename for saving the plot. If not provided, uses a cleaned version of the title.
    return_figure : bool, optional
        Whether to return the figure object instead of displaying it, by default False.
    annotate_points : bool, optional
        Whether to add coordinate annotations to the points, by default False.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if return_figure is True, otherwise None.

    Notes
    -----
    The function creates a curve between two points using the create_curve function
    and adds various visualization elements including grid lines, annotations,
    and optional titles.
    """
    gamma, lamda, a, x_values, y_values = create_curve(
        x1, y1, x2, y2, generate_values=True
    )

    # Plot the curve
    fig = plt.figure(figsize=figsize)

    plt.plot(x_values, y_values)
    plt.scatter(x1, y1, color="red")  # Mark the point (x1, y1)
    plt.scatter(x2, y2, color="red")  # Mark the point (x2, y2)

    if annotate_points:
        plt.annotate(
            f"({x1}, {y1:.2f})",
            (x1, y1),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=text_size,
        )
        plt.annotate(
            f"({x2}, {y2:.2f})",
            (x2, y2),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=text_size,
        )

    if text_size:
        plt.tick_params(axis="both", which="major", labelsize=text_size)

    x_ticks = np.arange(min(x_values), max(x_values) + 1, 2)
    plt.xticks(x_ticks)

    if include_titles:
        plt.title(title, fontsize=text_size)
        plt.xlabel("Hours since admission", fontsize=text_size)
        plt.ylabel("Probability of admission by this point", fontsize=text_size)

    plt.axhline(y=y1, color="green", linestyle="--", label=f"y ={int(y1*100)}%")
    plt.axvline(x=x1, color="gray", linestyle="--", label="x = 4 hours")
    plt.legend(fontsize=text_size)

    plt.tight_layout()

    if media_file_path:
        os.makedirs(media_file_path, exist_ok=True)
        if file_name:
            filename = file_name
        else:
            filename = clean_title_for_filename(title)
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
