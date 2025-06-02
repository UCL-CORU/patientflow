"""Visualization utilities for evaluating patient flow predictions.

This module provides functions for creating visualizations to evaluate the accuracy
and performance of patient flow predictions, particularly focusing on comparing
observed versus expected values.

Functions
---------
plot_observed_against_expected : function
    Plot histograms of observed minus expected values
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from patientflow.viz.utils import format_prediction_time


def plot_observed_against_expected(
    results1,
    results2=None,
    title1=None,
    title2=None,
    main_title="Histograms of Observed - Expected Values",
    xlabel="Observed minus expected",
    media_file_path=None,
    return_figure=False,
):
    """Plot histograms of observed minus expected values.

    Creates a grid of histograms showing the distribution of differences between
    observed and expected values for different prediction times. Optionally compares
    two sets of results side by side.

    Parameters
    ----------
    results1 : dict
        First set of results containing observed and expected values for different
        prediction times. Keys are prediction times, values are dicts with 'observed'
        and 'expected' arrays.
    results2 : dict, optional
        Second set of results for comparison, following the same format as results1.
    title1 : str, optional
        Title for the first set of results.
    title2 : str, optional
        Title for the second set of results.
    main_title : str, default="Histograms of Observed - Expected Values"
        Main title for the entire plot.
    xlabel : str, default="Observed minus expected"
        Label for the x-axis of each histogram.
    media_file_path : Path, optional
        Path where the plot should be saved. If provided, saves the plot as a PNG file.
    return_figure : bool, default=False
        If True, returns the matplotlib figure object instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if return_figure is True, otherwise None.

    Notes
    -----
    The function creates a grid of histograms with a maximum of 5 columns.
    Each histogram shows the distribution of differences between observed and
    expected values for a specific prediction time. A red dashed line at x=0
    indicates where observed equals expected.
    """
    # Calculate the number of subplots needed
    num_plots = len(results1)

    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_plots)  # Maximum of 5 columns
    num_rows = math.ceil(num_plots / num_cols)

    if results2:
        num_rows *= 2  # Double the number of rows if we have two result sets

    # Set a minimum width for the figure
    min_width = 8  # minimum width in inches
    width = max(min_width, 4 * num_cols)
    height = 4 * num_rows

    # Create the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height), squeeze=False)
    fig.suptitle(main_title, fontsize=14)

    # Flatten the axes array
    axes = axes.flatten()

    def plot_results(
        results, start_index, result_title, global_min, global_max, max_freq
    ):
        # Convert prediction times to minutes for sorting
        prediction_times_sorted = sorted(
            results.items(),
            key=lambda x: int(x[0].split("_")[-1][:2]) * 60
            + int(x[0].split("_")[-1][2:]),
        )

        # Create symmetric bins around zero
        bins = np.arange(global_min, global_max + 2) - 0.5

        for i, (_prediction_time, values) in enumerate(prediction_times_sorted):
            observed = np.array(values["observed"])
            expected = np.array(values["expected"])
            difference = observed - expected

            ax = axes[start_index + i]

            ax.hist(difference, bins=bins, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="r", linestyle="--", linewidth=1)

            # Format the prediction time
            formatted_time = format_prediction_time(_prediction_time)

            # Combine the result_title and formatted_time
            if result_title:
                ax.set_title(f"{result_title} {formatted_time}")
            else:
                ax.set_title(formatted_time)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")
            ax.set_xlim(global_min - 0.5, global_max + 0.5)
            ax.set_ylim(0, max_freq)

    # Calculate global min and max differences for consistent x-axis across both result sets
    all_differences = []
    max_counts = []

    # Gather all differences and compute histogram data for both result sets
    for results in [results1] + ([results2] if results2 else []):
        for _, values in results.items():
            observed = np.array(values["observed"])
            expected = np.array(values["expected"])
            differences = observed - expected
            all_differences.extend(differences)
            # Compute histogram data to find maximum frequency
            counts, _ = np.histogram(differences)
            max_counts.append(max(counts))

    # Find the symmetric range around zero
    abs_max = max(abs(min(all_differences)), abs(max(all_differences)))
    global_min = -math.ceil(abs_max)
    global_max = math.ceil(abs_max)

    # Find the maximum frequency across all histograms
    max_freq = math.ceil(max(max_counts) * 1.1)  # Add 10% padding

    # Plot the first results set
    plot_results(results1, 0, title1, global_min, global_max, max_freq)

    # Plot the second results set if provided
    if results2:
        plot_results(results2, num_plots, title2, global_min, global_max, max_freq)

    # Hide any unused subplots
    for j in range(num_plots * (2 if results2 else 1), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if media_file_path:
        plt.savefig(media_file_path / "observed_vs_expected.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
