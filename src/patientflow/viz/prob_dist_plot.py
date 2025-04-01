"""
Module: probability_distribution_visualization
==============================================

This module provides functionality to visualize probability distributions using bar plots.
The main function, `prob_dist_plot`, can handle both custom probability data, predefined
distributions such as the Poisson distribution, and dictionary input.

Functions
---------
prob_dist_plot(prob_dist_data, title, directory_path=None, figsize=(6, 3),
               include_titles=False, truncate_at_beds=(0, 20), text_size=None,
               bar_colour="#5B9BD5", file_name=None, min_beds_lines=None,
               plot_min_beds_lines=True, plot_bed_base=None, xlabel="Number of beds")
    Plots a bar chart of a probability distribution with optional customization for
    titles, labels, and additional markers.

Dependencies
------------
- numpy
- pandas
- matplotlib.pyplot
- scipy.stats
- itertools
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def prob_dist_plot(
    prob_dist_data,
    title,
    directory_path=None,
    figsize=(6, 3),
    include_titles=False,
    truncate_at_beds=None,
    text_size=None,
    bar_colour="#5B9BD5",
    file_name=None,
    min_beds_lines=None,
    plot_min_beds_lines=True,
    plot_bed_base=None,
    xlabel="Number of beds",
    return_figure=False,
):
    """
    Plot a probability distribution as a bar chart with enhanced plotting options.

    This function generates a bar plot for a given probability distribution, either
    as a pandas DataFrame, a scipy.stats distribution object (e.g., Poisson), or a
    dictionary. The plot can be customized with titles, axis labels, markers, and
    additional visual properties.

    Parameters
    ----------
    prob_dist_data : pandas.DataFrame, dict, scipy.stats distribution, or array-like
        The probability distribution data to be plotted. Can be:
        - pandas DataFrame
        - dictionary (keys are indices, values are probabilities)
        - scipy.stats distribution (e.g., Poisson). If a `scipy.stats` distribution is provided,
        the function computes probabilities for integer values within the specified range.
        - array-like of probabilities (indices will be 0 to len(array)-1)

    title : str
        The title of the plot, used for display and optionally as the file name.

    directory_path : str or pathlib.Path, optional
        Directory where the plot image will be saved. If not provided, the plot is
        displayed without saving.

    figsize : tuple of float, optional, default=(6, 3)
        The size of the figure, specified as (width, height).

    include_titles : bool, optional, default=False
        Whether to include titles and axis labels in the plot.

    truncate_at_beds : int or tuple of (int, int), optional, default=None
        Either a single number specifying the upper bound, or a tuple of
        (lower_bound, upper_bound) for the x-axis range. If None, the full
        range of the data will be displayed.

    text_size : int, optional
        Font size for plot text, including titles and tick labels.

    bar_colour : str, optional, default="#5B9BD5"
        The color of the bars in the plot.

    file_name : str, optional
        Name of the file to save the plot. If not provided, the title is used to generate
        a file name.

    min_beds_lines : dict, optional
        A dictionary where keys are percentages (as decimals, e.g., 0.5 for 50%) and values are
        the indices into the distribution's index array where vertical lines should be drawn.
        For example, {0.5: 5} would draw a line at the 5th position in the prob_dist_data.index array.

    plot_min_beds_lines : bool, optional, default=True
        Whether to plot the minimum beds lines if min_beds_lines is provided.

    plot_bed_base : dict, optional
        Dictionary of bed balance lines to plot in red.
        Keys are labels and values are x-axis positions.

    xlabel : str, optional, default="Number of beds"
        A label for the x axis

    return_figure : bool, optional
        If True, returns the matplotlib figure instead of displaying it (default is False)

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot
    """
    # Convert input data to standardized pandas DataFrame

    # Handle array-like input
    if isinstance(prob_dist_data, (np.ndarray, list)):
        array_length = len(prob_dist_data)
        prob_dist_data = pd.DataFrame(
            {"agg_proba": prob_dist_data}, index=range(array_length)
        )

    # Handle scipy.stats distribution input
    elif hasattr(prob_dist_data, "pmf") and callable(prob_dist_data.pmf):
        # Determine range for the distribution
        if truncate_at_beds is None:
            # Default range for distributions if not specified
            lower_bound = 0
            upper_bound = 20  # Reasonable default for most discrete distributions
        elif isinstance(truncate_at_beds, (int, float)):
            lower_bound = 0
            upper_bound = truncate_at_beds
        else:
            lower_bound, upper_bound = truncate_at_beds

        # Generate x values and probabilities
        x = np.arange(lower_bound, upper_bound + 1)
        probs = prob_dist_data.pmf(x)
        prob_dist_data = pd.DataFrame({"agg_proba": probs}, index=x)

        # No need to filter later
        truncate_at_beds = None

    # Handle dictionary input
    elif isinstance(prob_dist_data, dict):
        prob_dist_data = pd.DataFrame(
            {"agg_proba": list(prob_dist_data.values())},
            index=list(prob_dist_data.keys()),
        )

    # Apply truncation if specified
    if truncate_at_beds is not None:
        # Determine bounds
        if isinstance(truncate_at_beds, (int, float)):
            lower_bound = 0
            upper_bound = truncate_at_beds
        else:
            lower_bound, upper_bound = truncate_at_beds

        # Apply filtering
        mask = (prob_dist_data.index >= lower_bound) & (
            prob_dist_data.index <= upper_bound
        )
        filtered_data = prob_dist_data[mask]
    else:
        # Use all available data
        filtered_data = prob_dist_data

    # Create the plot
    fig = plt.figure(figsize=figsize)

    if not file_name:
        file_name = (
            title.replace(" ", "_").replace("/n", "_").replace("%", "percent") + ".png"
        )

    # Plot bars
    plt.bar(
        filtered_data.index,
        filtered_data["agg_proba"].values,
        color=bar_colour,
    )

    # Generate appropriate ticks based on data range
    if len(filtered_data) > 0:
        data_min = min(filtered_data.index)
        data_max = max(filtered_data.index)
        data_range = data_max - data_min

        if data_range <= 10:
            tick_step = 1
        elif data_range <= 50:
            tick_step = 5
        else:
            tick_step = 10

        tick_start = (data_min // tick_step) * tick_step
        tick_end = data_max + 1
        plt.xticks(np.arange(tick_start, tick_end, tick_step))

    # Plot minimum beds lines
    if plot_min_beds_lines and min_beds_lines:
        colors = itertools.cycle(
            plt.cm.gray(np.linspace(0.3, 0.7, len(min_beds_lines)))
        )
        for point in min_beds_lines:
            plt.axvline(
                x=prob_dist_data.index[min_beds_lines[point]],
                linestyle="--",
                linewidth=2,
                color=next(colors),
                label=f"{point*100:.0f}% probability",
            )
        plt.legend(loc="upper right", fontsize=14)

    # Add bed balance lines
    if plot_bed_base:
        for point in plot_bed_base:
            plt.axvline(
                x=plot_bed_base[point],
                linewidth=2,
                color="red",
                label=f"bed balance: {point}",
            )
        plt.legend(loc="upper right", fontsize=14)

    # Add text and labels
    if text_size:
        plt.tick_params(axis="both", which="major", labelsize=text_size)
        plt.xlabel(xlabel, fontsize=text_size)
        if include_titles:
            plt.title(title, fontsize=text_size)
            plt.ylabel("Probability", fontsize=text_size)
    else:
        plt.xlabel(xlabel)
        if include_titles:
            plt.title(title)
            plt.ylabel("Probability")

    plt.tight_layout()

    # Save or display the figure
    if directory_path:
        plt.savefig(directory_path / file_name.replace(" ", "_"), dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
