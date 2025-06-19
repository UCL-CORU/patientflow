"""
Module for generating probability distribution visualizations.

Functions
---------
plot_prob_dist : Plot a probability distribution as a bar chart with enhanced plotting options.
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from patientflow.predict.emergency_demand import find_probability_threshold_index


def _calculate_probability_thresholds(
    probability_sequence, probability_levels=[0.7, 0.9]
):
    """Calculate resource thresholds for given probability levels based on the probability distribution.

    This function determines resource thresholds where there's a specified probability
    that at least this many resources will be needed.

    Parameters
    ----------
    probability_sequence : array-like
        The probability mass function (PMF) of resource needs
    probability_levels : list of float, optional
        The desired probability levels (e.g., [0.7, 0.9] for 70% and 90%)
        Default is [0.7, 0.9]

    Returns
    -------
    dict
        Dictionary mapping probability levels to resource thresholds.
        For example, {0.9: 15, 0.7: 12} means that:
        - There is a 90% probability of needing at least 15 resources
        - There is a 70% probability of needing at least 12 resources

    Examples
    --------
    >>> pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]  # Probability of needing 0,1,2,3,4,5,6 beds
    >>> _calculate_probability_thresholds(pmf, [0.8, 0.9])
    {0.8: 4, 0.9: 5}
    # This means:
    # - There is an 80% probability of needing at least 4 beds
    # - There is a 90% probability of needing at least 5 beds
    """
    thresholds = {}
    for probability in probability_levels:
        thresholds[probability] = find_probability_threshold_index(
            probability_sequence, probability
        )

    return thresholds


def plot_prob_dist(
    prob_dist_data,
    title,
    media_file_path=None,
    figsize=(6, 3),
    include_titles=False,
    truncate_at_beds=None,
    text_size=None,
    bar_colour="#5B9BD5",
    file_name=None,
    probability_thresholds=None,
    show_probability_thresholds=True,
    probability_levels=None,
    plot_bed_base=None,
    xlabel="Number of beds",
    return_figure=False,
):
    """Plot a probability distribution as a bar chart with enhanced plotting options.

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
    media_file_path : str or pathlib.Path, optional
        Directory where the plot image will be saved. If not provided, the plot is
        displayed without saving.
    figsize : tuple of float, optional
        The size of the figure, specified as (width, height).
        Default is (6, 3)
    include_titles : bool, optional
        Whether to include titles and axis labels in the plot.
        Default is False
    truncate_at_beds : int or tuple of (int, int), optional
        Either a single number specifying the upper bound, or a tuple of
        (lower_bound, upper_bound) for the x-axis range. If None, the full
        range of the data will be displayed.
    text_size : int, optional
        Font size for plot text, including titles and tick labels.
    bar_colour : str, optional
        The color of the bars in the plot.
        Default is "#5B9BD5"
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to a generated name based on the title.
    probability_thresholds : dict, optional
        A dictionary where keys are points on the cumulative distribution function (as decimals, e.g., 0.9 for 90%)
        and values are the corresponding resource thresholds (bed counts).
        For example, {0.9: 15} indicates there is a 90% probability that
        at least 15 beds will be needed (represents the lower tail of the distribution).
    show_probability_thresholds : bool, optional
        Whether to show vertical lines indicating the resource requirements
        at different points on the cumulative distribution function.
        Default is True
    probability_levels : list of float, optional
        List of probability levels for automatic threshold calculation.
    plot_bed_base : dict, optional
        Dictionary of bed balance lines to plot in red.
        Keys are labels and values are x-axis positions.
    xlabel : str, optional
        A label for the x axis.
        Default is "Number of beds"
    return_figure : bool, optional
        If True, returns the matplotlib figure instead of displaying it.
        Default is False

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot

    Examples
    --------
    Basic usage with an array of probabilities:

    >>> probabilities = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    >>> plot_prob_dist(probabilities, "Bed Demand Distribution")

    With thresholds:

    >>> thresholds = _calculate_probability_thresholds(probabilities, [0.8, 0.95])
    >>> plot_prob_dist(probabilities, "Bed Demand with Confidence Levels",
    ...                probability_thresholds=thresholds)

    Using with a scipy stats distribution:

    >>> from scipy import stats
    >>> poisson_dist = stats.poisson(mu=5)  # Poisson with mean of 5
    >>> plot_prob_dist(poisson_dist, "Poisson Distribution (μ=5)",
    ...                truncate_at_beds=(0, 15))
    """

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

    # Calculate probability thresholds if probability_levels is provided
    if probability_thresholds is None and probability_levels is not None:
        probability_thresholds = _calculate_probability_thresholds(
            filtered_data["agg_proba"].values, probability_levels
        )

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

    # Plot probability threshold lines
    if show_probability_thresholds and probability_thresholds:
        colors = itertools.cycle(
            plt.cm.gray(np.linspace(0.3, 0.7, len(probability_thresholds)))
        )
        for probability, bed_count in probability_thresholds.items():
            plt.axvline(
                x=bed_count,
                linestyle="--",
                linewidth=2,
                color=next(colors),
                label=f"{probability*100:.0f}% probability of needing ≥ {bed_count} beds",
            )
        plt.legend(loc="upper right")

    # Add bed balance lines
    if plot_bed_base:
        for point in plot_bed_base:
            plt.axvline(
                x=plot_bed_base[point],
                linewidth=2,
                color="red",
                label=f"bed balance: {point}",
            )
        plt.legend(loc="upper right")

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
    if media_file_path:
        plt.savefig(media_file_path / file_name.replace(" ", "_"), dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
