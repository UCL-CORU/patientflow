import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from patientflow.load import get_model_key


def _prob_to_cdf(prob_dist):
    """Convert probability distribution to CDF function"""
    import pandas as pd

    if isinstance(prob_dist, pd.DataFrame):
        # Handle DataFrame: assume columns are values, data contains probabilities
        # Take the first row if it's a DataFrame with multiple rows
        if len(prob_dist) > 0:
            prob_series = prob_dist.iloc[0]  # Take first row
        else:
            raise ValueError("Empty DataFrame provided")
        values = list(prob_series.index)
        probs = list(prob_series.values)
    elif isinstance(prob_dist, pd.Series):
        # Handle Series: index is values, values are probabilities
        values = list(prob_dist.index)
        probs = list(prob_dist.values)
    elif isinstance(prob_dist, dict):
        # Sort by keys (values) to ensure proper cumulative calculation
        sorted_items = sorted(prob_dist.items())
        values = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
    else:
        # Array format: index = value, array[index] = probability
        values = list(range(len(prob_dist)))
        probs = prob_dist

    # Ensure values are sorted
    sorted_pairs = sorted(zip(values, probs))
    values = [pair[0] for pair in sorted_pairs]
    probs = [pair[1] for pair in sorted_pairs]

    # Calculate cumulative probabilities
    cum_probs = np.cumsum(probs)

    def cdf_function(x):
        # Return P(X <= x)
        if x < values[0]:
            return 0.0
        for i, val in enumerate(values):
            if x <= val:
                return cum_probs[i]
        return 1.0  # x is larger than all values

    return cdf_function


def plot_randomised_pit(
    prediction_times: List[Tuple[int, int]],
    prob_dist_dict_all: Dict[str, Dict],
    model_name: str = "admissions",
    return_figure: bool = False,
    return_dataframe: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    media_file_path: Optional[Path] = None,
    file_name: Optional[str] = None,
    n_bins: int = 10,
    seed: Optional[int] = 42,
) -> Union[
    plt.Figure, Dict[str, List[float]], Tuple[plt.Figure, Dict[str, List[float]]], None
]:
    """
    Generate randomised PIT histograms for multiple prediction times side by side.

    Parameters
    ----------
    prediction_times : list of tuple
        List of (hour, minute) tuples representing times for which predictions were made.
    prob_dist_dict_all : dict
        Dictionary of probability distributions keyed by model_key. Each entry contains
        information about predicted distributions and observed values for different
        snapshot dates.
    model_name : str, optional
        Base name of the model to construct model keys, by default "admissions".
    return_figure : bool, optional
        If True, returns the figure object instead of displaying it, by default False.
    return_dataframe : bool, optional
        If True, returns a dictionary of PIT values by model_key, by default False.
    figsize : tuple of (float, float), optional
        Size of the figure in inches as (width, height). If None, calculated automatically
        based on number of plots, by default None.
    suptitle : str, optional
        Super title for the entire figure, displayed above all subplots, by default None.
    media_file_path : Path, optional
        Path to save the plot, by default None. If provided, saves the plot as a PNG file.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "plot_randomised_pit.png".
    n_bins : int, optional
        Number of histogram bins, by default 10.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots, if return_figure is True.
    dict
        Dictionary of PIT values by model_key, if return_dataframe is True.
    tuple
        Tuple of (figure, pit_values_dict) if both return_figure and return_dataframe are True.
    None
        If neither return_figure nor return_dataframe is True, displays the plots and returns None.
    """
    if seed is not None:
        np.random.seed(seed)

    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60 + x[1],
    )

    # Calculate figure parameters
    num_plots = len(prediction_times_sorted)
    figsize = figsize or (num_plots * 5, 4)

    # Create subplot layout
    fig, axs = plt.subplots(1, num_plots, figsize=figsize)
    axs = [axs] if num_plots == 1 else axs

    all_pit_values: Dict[str, List[float]] = {}
    max_density = 0.0  # Track maximum density across all histograms

    # Process each subplot
    for i, prediction_time in enumerate(prediction_times_sorted):
        model_key = get_model_key(model_name, prediction_time)
        prob_dist_dict = prob_dist_dict_all[model_key]

        if not prob_dist_dict:
            continue

        observations = []
        cdf_functions = []

        # Extract data for each date
        for dt in prob_dist_dict:
            try:
                observation = prob_dist_dict[dt]["agg_observed"]
                predicted_dist = prob_dist_dict[dt]["agg_predicted"]["agg_proba"]

                # Convert probability distribution to CDF function
                cdf_func = _prob_to_cdf(predicted_dist)

                observations.append(observation)
                cdf_functions.append(cdf_func)

            except Exception as e:
                print(f"Skipping date {dt} due to error: {e}")
                continue

        if len(observations) == 0:
            continue

        # Generate PIT values
        pit_values = []

        for obs, cdf_func in zip(observations, cdf_functions):
            try:
                # Calculate PIT range bounds
                lower = cdf_func(obs - 1) if obs > 0 else 0.0
                upper = cdf_func(obs)

                # Sample randomly within the range
                pit_value = np.random.uniform(lower, upper)
                pit_values.append(pit_value)

            except Exception as e:
                print(f"Error processing observation {obs}: {e}")
                continue

        all_pit_values[model_key] = pit_values

        # Calculate histogram to get density
        hist, _ = np.histogram(pit_values, bins=n_bins, density=True)
        max_density = max(max_density, np.max(hist))

    # Now plot with consistent y-axis scale
    for i, prediction_time in enumerate(prediction_times_sorted):
        model_key = get_model_key(model_name, prediction_time)
        pit_values = all_pit_values.get(model_key, [])

        if not pit_values:
            continue

        # Plot histogram
        ax = axs[i]
        ax.hist(
            pit_values,
            bins=n_bins,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Randomised PIT",
        )

        # Add uniform reference line
        ax.axhline(
            y=1.0, color="red", linestyle="--", linewidth=2, label="Perfect Uniform"
        )

        # Set labels and title
        hour, minutes = prediction_time
        ax.set_xlabel("PIT Value")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT Histogram for {hour}:{minutes:02}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max_density * 1.1)  # Add 10% padding
        ax.grid(True, alpha=0.3)

        if i == 0:  # Only show legend on first subplot
            ax.legend()

    # Final plot configuration
    plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)
    if media_file_path:
        plt.savefig(media_file_path / (file_name or "plot_randomised_pit.png"), dpi=300)

    # Return based on flags
    if return_figure and return_dataframe:
        return fig, all_pit_values
    elif return_figure:
        return fig
    elif return_dataframe:
        plt.show()
        plt.close()
        return all_pit_values
    else:
        plt.show()
        plt.close()
        return None
