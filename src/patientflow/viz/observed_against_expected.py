"""Visualisation utilities for evaluating patient flow predictions.

This module provides functions for creating visualizations to evaluate the accuracy
and performance of patient flow predictions, particularly focusing on comparing
observed versus expected values.

Functions
---------
plot_deltas : function
    Plot histograms of observed minus expected values
plot_arrival_delta_single_instance : function
    Plot comparison between observed arrivals and expected arrival rates
plot_arrival_deltas : function
    Plot delta charts for multiple snapshot dates on the same figure
"""

from datetime import timedelta, datetime, time
from patientflow.calculate.arrival_rates import time_varying_arrival_rates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from patientflow.viz.utils import format_prediction_time


def plot_deltas(
    results1,
    results2=None,
    title1=None,
    title2=None,
    suptitle="Histograms of Observed - Expected Values",
    xlabel="Observed minus expected",
    media_file_path=None,
    file_name=None,
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
    suptitle : str, default="Histograms of Observed - Expected Values"
        Super title for the entire plot.
    xlabel : str, default="Observed minus expected"
        Label for the x-axis of each histogram.
    media_file_path : Path, optional
        Path where the plot should be saved. If provided, saves the plot as a PNG file.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "observed_vs_expected.png".
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
    fig.suptitle(suptitle, fontsize=14)

    # Flatten the axes array
    axes = axes.flatten()

    def plot_results(results, start_index, result_title, global_min, global_max):
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

    # Calculate global min and max differences for consistent x-axis across both result sets
    all_differences = []

    # Gather all differences for consistent x-axis scaling
    for results in [results1] + ([results2] if results2 else []):
        for _, values in results.items():
            observed = np.array(values["observed"])
            expected = np.array(values["expected"])
            differences = observed - expected
            all_differences.extend(differences)

    # Find the symmetric range around zero
    abs_max = max(abs(min(all_differences)), abs(max(all_differences)))
    global_min = -math.ceil(abs_max)
    global_max = math.ceil(abs_max)

    # Plot the first results set
    plot_results(results1, 0, title1, global_min, global_max)

    # Plot the second results set if provided
    if results2:
        plot_results(results2, num_plots, title2, global_min, global_max)

    # Hide any unused subplots
    for j in range(num_plots * (2 if results2 else 1), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if media_file_path:
        if file_name:
            plt.savefig(media_file_path / file_name, dpi=300)
        else:
            plt.savefig(media_file_path / "observed_vs_expected.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()


def _prepare_arrival_data(
    df, prediction_time, snapshot_date, prediction_window, yta_time_interval
):
    """Helper function to prepare arrival data for plotting."""
    prediction_time_obj = time(hour=prediction_time[0], minute=prediction_time[1])
    snapshot_datetime = pd.Timestamp(
        datetime.combine(snapshot_date, prediction_time_obj), tz="UTC"
    )

    default_date = datetime(2024, 1, 1)
    default_datetime = pd.Timestamp(
        datetime.combine(default_date, prediction_time_obj), tz="UTC"
    )

    df_copy = df.copy()
    if "arrival_datetime" in df_copy.columns:
        df_copy.set_index("arrival_datetime", inplace=True)
        # Ensure the index is timezone-aware to match snapshot_datetime
        if df_copy.index.tz is None:
            df_copy.index = df_copy.index.tz_localize("UTC")

    return df_copy, snapshot_datetime, default_datetime, prediction_time_obj


def _calculate_arrival_rates(
    df_copy, prediction_time_obj, prediction_window, yta_time_interval
):
    """Helper function to calculate arrival rates and prepare time points."""
    arrival_rates = time_varying_arrival_rates(
        df_copy, yta_time_interval=yta_time_interval
    )
    end_time = (
        datetime.combine(datetime.min, prediction_time_obj) + prediction_window
    ).time()

    mean_arrival_rates = {
        k: v
        for k, v in arrival_rates.items()
        if (k >= prediction_time_obj and k < end_time)
        or (
            end_time < prediction_time_obj
            and (k >= prediction_time_obj or k < end_time)
        )
    }

    return mean_arrival_rates


def _prepare_arrival_times(mean_arrival_rates, prediction_time_obj, default_date):
    """Helper function to prepare arrival times for plotting."""
    arrival_times_piecewise = []
    for t in mean_arrival_rates.keys():
        if t < prediction_time_obj:
            dt = datetime.combine(default_date + timedelta(days=1), t)
        else:
            dt = datetime.combine(default_date, t)
        if dt.tzinfo is None:
            dt = pd.Timestamp(dt, tz="UTC")
        arrival_times_piecewise.append(dt)

    arrival_times_piecewise.sort()
    return arrival_times_piecewise


def _calculate_cumulative_rates(arrival_times_piecewise, mean_arrival_rates):
    """Helper function to calculate cumulative arrival rates."""
    cumulative_rates = []
    current_sum = 0
    for t in arrival_times_piecewise:
        rate = mean_arrival_rates[t.time()]
        current_sum += rate
        cumulative_rates.append(current_sum)
    return cumulative_rates


def _create_combined_timeline(
    default_datetime, arrival_times_plot, prediction_window, arrival_times_piecewise
):
    """Helper function to create combined timeline for plotting."""
    all_times = sorted(
        set(
            [default_datetime]
            + arrival_times_plot
            + [default_datetime + prediction_window]
            + arrival_times_piecewise
        )
    )
    if all_times[0] != default_datetime:
        all_times = [default_datetime] + all_times
    return all_times


def _plot_arrival_delta_chart(
    ax,
    all_times,
    delta,
    prediction_time,
    prediction_window,
    snapshot_date,
    show_only_delta=False,
):
    """Helper function to plot arrival delta chart."""
    ax.step(all_times, delta, where="post", label="Actual - Expected", color="red")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Difference (Actual - Expected)")
    ax.set_title(
        f"Difference Between Actual and Expected Arrivals in the "
        f"{int(prediction_window.total_seconds()/3600)} hours after "
        f"{format_prediction_time(prediction_time)} on {snapshot_date}"
    )
    ax.legend()


def _format_time_axis(ax, all_times):
    """Helper function to format time axis."""
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    min_time = min(all_times)
    max_time = max(all_times)
    hourly_ticks = pd.date_range(start=min_time, end=max_time, freq="h")
    ax.set_xticks(hourly_ticks)
    ax.set_xlim(left=min_time)


def plot_arrival_delta_single_instance(
    df,
    prediction_time,
    snapshot_date,
    prediction_window: timedelta,
    yta_time_interval: timedelta = timedelta(minutes=15),
    show_delta=True,
    show_only_delta=False,
    media_file_path=None,
    file_name=None,
    return_figure=False,
    fig_size=(10, 4),
):
    """Plot comparison between observed arrivals and expected arrival rates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing arrival data
    prediction_time : tuple
        (hour, minute) of prediction time
    snapshot_date : datetime.date
        Date to analyze
    prediction_window : int
        Prediction window in minutes
    show_delta : bool, default=True
        If True, plot the difference between actual and expected arrivals
    show_only_delta : bool, default=False
        If True, only plot the delta between actual and expected arrivals
    yta_time_interval : int, default=15
        Time interval in minutes for calculating arrival rates
    media_file_path : Path, optional
        Path to save the plot
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "arrival_comparison.png"
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it
    fig_size : tuple, default=(10, 4)
        Figure size as (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if return_figure is True, otherwise None
    """
    # Prepare data
    df_copy, snapshot_datetime, default_datetime, prediction_time_obj = (
        _prepare_arrival_data(
            df, prediction_time, snapshot_date, prediction_window, yta_time_interval
        )
    )

    # Get arrivals within the prediction window
    arrivals = df_copy[
        (df_copy.index > snapshot_datetime)
        & (df_copy.index <= snapshot_datetime + prediction_window)
    ]

    # Sort arrivals by time and create cumulative count
    arrivals = arrivals.sort_values("arrival_datetime")
    arrivals["cumulative_count"] = range(1, len(arrivals) + 1)

    # Calculate arrival rates and prepare time points
    mean_arrival_rates = _calculate_arrival_rates(
        df_copy, prediction_time_obj, prediction_window, yta_time_interval
    )

    # Prepare arrival times
    arrival_times_piecewise = _prepare_arrival_times(
        mean_arrival_rates, prediction_time_obj, default_date=datetime(2024, 1, 1)
    )

    # Calculate cumulative rates
    cumulative_rates = _calculate_cumulative_rates(
        arrival_times_piecewise, mean_arrival_rates
    )

    # Create figure with subplots if showing delta
    if show_delta and not show_only_delta:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(fig_size[0], fig_size[1] * 2), sharex=True
        )
        ax = ax1
    else:
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    # Ensure arrivals index is timezone-aware
    if arrivals.index.tz is None:
        arrivals.index = arrivals.index.tz_localize("UTC")

    # Convert arrival times to use default date for plotting
    arrival_times_plot = [
        default_datetime + (t - snapshot_datetime) for t in arrivals.index
    ]

    # Create combined timeline
    all_times = _create_combined_timeline(
        default_datetime, arrival_times_plot, prediction_window, arrival_times_piecewise
    )

    # Interpolate both actual and expected to the combined timeline
    actual_counts = np.interp(
        [t.timestamp() for t in all_times],
        [
            t.timestamp()
            for t in [default_datetime]
            + arrival_times_plot
            + [default_datetime + prediction_window]
        ],
        [0]
        + list(arrivals["cumulative_count"])
        + [arrivals["cumulative_count"].iloc[-1] if len(arrivals) > 0 else 0],
    )

    expected_counts = np.interp(
        [t.timestamp() for t in all_times],
        [t.timestamp() for t in arrival_times_piecewise],
        cumulative_rates,
    )

    # Calculate delta
    delta = actual_counts - expected_counts
    delta[0] = 0  # Ensure delta starts at 0

    if not show_only_delta:
        # Plot actual and expected arrivals
        ax.step(
            [default_datetime]
            + arrival_times_plot
            + [default_datetime + prediction_window],
            [0]
            + list(arrivals["cumulative_count"])
            + [arrivals["cumulative_count"].iloc[-1] if len(arrivals) > 0 else 0],
            where="post",
            label="Actual Arrivals",
        )
        ax.scatter(
            arrival_times_piecewise,
            cumulative_rates,
            label="Expected Arrivals",
            color="orange",
        )

        ax.set_xlabel("Time")
        ax.set_title(
            f"Cumulative Arrivals in the {int(prediction_window.total_seconds()/3600)} hours after {format_prediction_time(prediction_time)} on {snapshot_date}"
        )
        ax.legend()

    if show_delta or show_only_delta:
        if show_only_delta:
            _plot_arrival_delta_chart(
                ax, all_times, delta, prediction_time, prediction_window, snapshot_date
            )
        else:
            _plot_arrival_delta_chart(
                ax2, all_times, delta, prediction_time, prediction_window, snapshot_date
            )
        plt.tight_layout()

    # Format time axis for all subplots
    for ax in plt.gcf().get_axes():
        _format_time_axis(ax, all_times)

    if media_file_path:
        filename = file_name if file_name else "arrival_comparison.png"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()


def _prepare_common_values(prediction_time):
    """Helper function to prepare common values used across all dates."""
    prediction_time_obj = time(hour=prediction_time[0], minute=prediction_time[1])
    default_date = datetime(2024, 1, 1)
    default_datetime = pd.Timestamp(
        datetime.combine(default_date, prediction_time_obj), tz="UTC"
    )
    return prediction_time_obj, default_datetime


def plot_arrival_deltas(
    df,
    prediction_time,
    snapshot_dates,
    prediction_window: timedelta,
    yta_time_interval: timedelta = timedelta(minutes=15),
    media_file_path=None,
    file_name=None,
    return_figure=False,
    fig_size=(15, 6),
):
    """Plot delta charts for multiple snapshot dates on the same figure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing arrival data
    prediction_time : tuple
        (hour, minute) of prediction time
    snapshot_dates : list
        List of datetime.date objects to analyze
    prediction_window : timedelta
        Prediction window in minutes
    yta_time_interval : int, default=15
        Time interval in minutes for calculating arrival rates
    media_file_path : Path, optional
        Path to save the plot
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "multiple_deltas.png"
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it
    fig_size : tuple, default=(15, 6)
        Figure size as (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if return_figure is True, otherwise None
    """
    # Create figure with subplots
    fig = plt.figure(figsize=fig_size)
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Store all deltas for averaging
    all_deltas = []
    all_times_list = []
    final_deltas = []  # Store final delta values for histogram

    # Calculate common values once
    prediction_time_obj, default_datetime = _prepare_common_values(prediction_time)

    for snapshot_date in snapshot_dates:
        # Prepare data for this date
        df_copy, snapshot_datetime, _, _ = _prepare_arrival_data(
            df, prediction_time, snapshot_date, prediction_window, yta_time_interval
        )

        # Get arrivals within the prediction window
        arrivals = df_copy[
            (df_copy.index > snapshot_datetime)
            & (df_copy.index <= snapshot_datetime + pd.Timedelta(prediction_window))
        ]

        if len(arrivals) == 0:
            continue

        # Sort arrivals by time and create cumulative count
        arrivals = arrivals.sort_values("arrival_datetime")
        arrivals["cumulative_count"] = range(1, len(arrivals) + 1)

        # Calculate arrival rates and prepare time points
        mean_arrival_rates = _calculate_arrival_rates(
            df_copy, prediction_time_obj, prediction_window, yta_time_interval
        )

        # Prepare arrival times
        arrival_times_piecewise = _prepare_arrival_times(
            mean_arrival_rates, prediction_time_obj, default_date=datetime(2024, 1, 1)
        )

        # Calculate cumulative rates
        cumulative_rates = _calculate_cumulative_rates(
            arrival_times_piecewise, mean_arrival_rates
        )

        # Convert arrival times to use default date for plotting
        arrival_times_plot = [
            default_datetime + (t - snapshot_datetime) for t in arrivals.index
        ]

        # Create combined timeline
        all_times = _create_combined_timeline(
            default_datetime,
            arrival_times_plot,
            prediction_window,
            arrival_times_piecewise,
        )

        # Interpolate both actual and expected to the combined timeline
        actual_counts = np.interp(
            [t.timestamp() for t in all_times],
            [
                t.timestamp()
                for t in [default_datetime]
                + arrival_times_plot
                + [default_datetime + pd.Timedelta(prediction_window)]
            ],
            [0]
            + list(arrivals["cumulative_count"])
            + [arrivals["cumulative_count"].iloc[-1]],
        )

        expected_counts = np.interp(
            [t.timestamp() for t in all_times],
            [t.timestamp() for t in arrival_times_piecewise],
            cumulative_rates,
        )

        # Calculate delta
        delta = actual_counts - expected_counts
        delta[0] = 0  # Ensure delta starts at 0

        # Store for averaging
        all_deltas.append(delta)
        all_times_list.append(all_times)

        # Store final delta value for histogram
        final_deltas.append(delta[-1])

        # Plot delta for this snapshot date
        ax1.step(all_times, delta, where="post", color="grey", alpha=0.5)

    # Calculate and plot average delta
    if all_deltas:
        # Find the common time points across all dates
        common_times = sorted(set().union(*[set(times) for times in all_times_list]))

        # Interpolate all deltas to common time points
        interpolated_deltas = []
        for times, delta in zip(all_times_list, all_deltas):
            # Only interpolate within the actual time range for each date
            min_time = min(times)
            max_time = max(times)
            valid_times = [t for t in common_times if min_time <= t <= max_time]

            if valid_times:
                interpolated = np.interp(
                    [t.timestamp() for t in valid_times],
                    [t.timestamp() for t in times],
                    delta,
                )
                # Pad with NaN for times outside the valid range
                padded = np.full(len(common_times), np.nan)
                valid_indices = [
                    i for i, t in enumerate(common_times) if t in valid_times
                ]
                padded[valid_indices] = interpolated
                interpolated_deltas.append(padded)

        # Calculate average delta, ignoring NaN values
        avg_delta = np.nanmean(interpolated_deltas, axis=0)

        # Plot average delta as a solid line
        # Only plot where we have valid data (not NaN)
        valid_mask = ~np.isnan(avg_delta)
        if np.any(valid_mask):
            ax1.step(
                [t for t, m in zip(common_times, valid_mask) if m],
                avg_delta[valid_mask],
                where="post",
                color="red",
                linewidth=2,
            )

    # Add horizontal line at y=0
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Format the main plot
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Difference (Actual - Expected)")
    ax1.set_title(
        f"Difference Between Actual and Expected Arrivals in the {(int(prediction_window.total_seconds()/3600))} hours after {format_prediction_time(prediction_time)} on all dates"
    )

    # Format time axis
    _format_time_axis(ax1, common_times)

    # Create histogram of final delta values
    if final_deltas:
        # Round values to nearest integer for binning
        rounded_deltas = np.round(final_deltas)
        unique_values = np.unique(rounded_deltas)

        # Create bins centered on integer values
        bin_edges = np.arange(unique_values.min() - 0.5, unique_values.max() + 1.5, 1)

        ax2.hist(final_deltas, bins=bin_edges, color="grey", alpha=0.7)
        ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Final Difference (Actual - Expected)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Final Differences")

        # Set x-axis ticks to integer values with appropriate spacing
        value_range = unique_values.max() - unique_values.min()
        step_size = max(1, int(value_range / 10))  # Aim for about 10 ticks
        ax2.set_xticks(
            np.arange(unique_values.min(), unique_values.max() + 1, step_size)
        )

    plt.tight_layout()

    if media_file_path:
        filename = file_name if file_name else "multiple_deltas.png"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
