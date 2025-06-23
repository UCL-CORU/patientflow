"""Visualization tools for patient flow analysis using survival curves.

This module provides functions to create and analyze survival curves for
time-to-event analysis.

Functions
---------
plot_admission_time_survival_curve : function
    Create single or multiple survival curves for ward admission times

Notes
-----
* The survival curves show the proportion of patients who have not yet
  experienced an event (e.g., admission to ward) over time
* Time is measured in hours from the initial event (e.g., arrival)
* A 4-hour target line is included by default to show performance
  against common healthcare targets
* The curves are created without external survival analysis packages
  for simplicity and transparency
* Multiple curves can be plotted on the same figure for comparison

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from patientflow.calculate.survival_curve import calculate_survival_curve


def plot_admission_time_survival_curve(
    df,
    start_time_col="arrival_datetime",
    end_time_col="departure_datetime",
    title="Time to Event Survival Curve",
    target_hours=[4],
    xlabel="Elapsed time from start",
    ylabel="Proportion not yet experienced event",
    annotation_string="{:.1%} experienced event\nwithin {:.0f} hours",
    labels=None,
    media_file_path=None,
    file_name=None,
    return_figure=False,
    return_df=False,
):
    """Create a survival curve for time-to-event analysis.

    This function creates a survival curve showing the proportion of patients
    who have not yet experienced an event over time. Can plot single or multiple
    survival curves on the same plot.

    Parameters
    ----------
    df : pandas.DataFrame or list of pandas.DataFrame
        DataFrame(s) containing patient visit data. If a list is provided,
        multiple survival curves will be plotted on the same figure.
    start_time_col : str, default="arrival_datetime"
        Name of the column containing the start time (e.g., arrival time)
    end_time_col : str, default="admitted_to_ward_datetime"
        Name of the column containing the end time (e.g., admission time)
    title : str, default="Time to Event Survival Curve"
        Title for the plot
    target_hours : list of float, default=[4]
        List of target times in hours to show on the plot
    xlabel : str, default="Elapsed time from start"
        Label for the x-axis
    ylabel : str, default="Proportion not yet experienced event"
        Label for the y-axis
    annotation_string : str, default="{:.1%} experienced event\nwithin {:.0f} hours"
        String template for the text annotation. Use {:.1%} for the proportion and {:.0f} for the hours.
        Annotations are only shown for the first curve when plotting multiple curves.
    labels : list of str, optional
        Labels for each survival curve when plotting multiple curves.
        If None and multiple dataframes are provided, default labels will be used.
        Ignored when plotting a single curve.
    media_file_path : pathlib.Path, optional
        Path to save the plot. If None, the plot is not saved.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "survival_curve.png".
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it
    return_df : bool, default=False
        If True, returns a DataFrame containing the survival curve data.
        For multiple curves, returns a list of DataFrames.

    Returns
    -------
    matplotlib.figure.Figure or pandas.DataFrame or list or tuple or None
        - If return_figure is True and return_df is False: returns the figure object
        - If return_figure is False and return_df is True: returns the DataFrame(s) with survival curve data
        - If both return_figure and return_df are True: returns a tuple of (figure, DataFrame(s))
        - If both are False: returns None

    Notes
    -----
    The survival curve shows the proportion of patients who have not yet experienced
    the event at each time point. Vertical lines are drawn at each target hour
    to indicate the target times, with the corresponding proportion of patients
    who experienced the event within these timeframes.

    When plotting multiple curves, different colors are automatically assigned
    and a legend is displayed. Target line annotations are only shown for the
    first curve to avoid visual clutter.
    """
    # Handle single dataframe vs list of dataframes
    if isinstance(df, pd.DataFrame):
        dataframes = [df]
        is_single_curve = True
    else:
        dataframes = df
        is_single_curve = False

    # Handle labels
    if labels is None:
        if is_single_curve:
            curve_labels = [None]
        else:
            curve_labels = [f"Curve {i+1}" for i in range(len(dataframes))]
    else:
        curve_labels = labels

    # Validate inputs
    if len(dataframes) != len(curve_labels):
        raise ValueError("Number of dataframes must match number of labels")

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    # Define colors for multiple curves
    colors = plt.cm.Set1(np.linspace(0, 1, len(dataframes)))

    survival_dfs = []

    # Process each dataframe
    for idx, (current_df, label) in enumerate(zip(dataframes, curve_labels)):
        # Calculate survival curve using the extracted function
        survival_df = calculate_survival_curve(current_df, start_time_col, end_time_col)

        # Extract arrays for plotting
        unique_times = survival_df["time_hours"].values
        survival_prob = survival_df["survival_probability"].values

        # Store DataFrame if requested
        if return_df:
            survival_dfs.append(survival_df)

        # Plot the survival curve
        color = colors[idx] if not is_single_curve else None
        plt.step(
            unique_times,
            survival_prob,
            where="post",
            color=color,
            label=label if not is_single_curve else None,
        )

        # Plot target lines and annotations only for the first curve (or single curve)
        if idx == 0:
            # Plot target lines for each target hour
            for target_hour in target_hours:
                # Find the survival probability at target hours
                closest_time_idx = np.abs(unique_times - target_hour).argmin()
                if closest_time_idx < len(survival_prob):
                    survival_at_target = survival_prob[closest_time_idx]
                    event_at_target = 1 - survival_at_target

                    # Add text annotation to the plot (only for single curve or first curve)
                    if is_single_curve or len(dataframes) == 1:
                        plt.text(
                            target_hour + 0.5,
                            survival_at_target,
                            annotation_string.format(event_at_target, target_hour),
                            bbox=dict(facecolor="white", alpha=0.8),
                        )

                        # Draw a vertical line from x-axis to the curve at target hours
                        plt.plot(
                            [target_hour, target_hour],
                            [0, survival_at_target],
                            color="grey",
                            linestyle="--",
                            linewidth=2,
                        )

                        # Draw a horizontal line from the curve to the y-axis at the survival probability level
                        plt.plot(
                            [0, target_hour],
                            [survival_at_target, survival_at_target],
                            color="grey",
                            linestyle="--",
                            linewidth=2,
                        )

    # Configure the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Make axes meet at the origin
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Move spines to the origin
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))

    # Hide the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend for multiple curves
    if not is_single_curve:
        plt.legend()

    plt.tight_layout()

    if media_file_path:
        if file_name:
            plt.savefig(media_file_path / file_name, dpi=300)
        else:
            plt.savefig(media_file_path / "survival_curve.png", dpi=300)

    # Handle return values
    return_data = (
        survival_dfs[0]
        if (return_df and is_single_curve)
        else survival_dfs
        if return_df
        else None
    )

    if return_figure and return_df:
        return fig, return_data
    elif return_figure:
        return fig
    elif return_df:
        return return_data
    else:
        plt.show()
        plt.close()
