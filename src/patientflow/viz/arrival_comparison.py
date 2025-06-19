"""Visualization functions for comparing actual and predicted patient arrivals.

This module provides functions to visualize and analyze the difference between
actual patient arrivals and predicted arrival rates over time.

Functions
---------
plot_arrival_comparison : function
    Plot comparison between observed arrivals and expected arrival rates
plot_multiple_deltas : function
    Plot delta charts for multiple snapshot dates on the same figure
"""

from datetime import timedelta, datetime, time
from patientflow.calculate.arrival_rates import time_varying_arrival_rates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from patientflow.viz.utils import format_prediction_time


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


def _plot_delta(
    ax,
    all_times,
    delta,
    prediction_time,
    prediction_window,
    snapshot_date,
    show_only_delta=False,
):
    """Helper function to plot delta chart."""
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


def plot_arrival_comparison(
    df,
    prediction_time,
    snapshot_date,
    prediction_window: timedelta,
    yta_time_interval: timedelta = timedelta(minutes=15),
    show_delta=True,
    show_only_delta=False,
    media_file_path=None,
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
            _plot_delta(
                ax, all_times, delta, prediction_time, prediction_window, snapshot_date
            )
        else:
            _plot_delta(
                ax2, all_times, delta, prediction_time, prediction_window, snapshot_date
            )
        plt.tight_layout()

    # Format time axis for all subplots
    for ax in plt.gcf().get_axes():
        _format_time_axis(ax, all_times)

    if media_file_path:
        plt.savefig(media_file_path / "arrival_comparison.png", dpi=300)

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


def plot_multiple_deltas(
    df,
    prediction_time,
    snapshot_dates,
    prediction_window: timedelta,
    yta_time_interval: timedelta = timedelta(minutes=15),
    media_file_path=None,
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
        plt.savefig(media_file_path / "multiple_deltas.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
