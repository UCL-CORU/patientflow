"""
Visualization functions for inpatient arrival rates and cumulative statistics.

This module provides functions to visualize time-varying arrival rates and cumulative arrivals,
over the course of a day.

Functions
---------
annotate_hour_line : function
    Annotate hour lines on a matplotlib plot
plot_arrival_rates : function
    Plot arrival rates for one or two datasets
plot_cumulative_arrival_rates : function
    Plot cumulative arrival rates with statistical distributions

"""

import matplotlib.pyplot as plt
import numpy as np

from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates,
    time_varying_arrival_rates_lagged,
    process_arrival_rates,
    unfettered_demand_by_hour,
)

from patientflow.viz.utils import clean_title_for_filename
import scipy.stats as stats


def annotate_hour_line(
    hour_line,
    y_value,
    hour_values,
    start_plot_index,
    line_styles,
    x_margin,
    annotation_prefix,
    text_y_offset=1,
    text_x_position=None,
    slope=None,
    x1=None,
    y1=None,
):
    """Annotate hour lines on a matplotlib plot with consistent formatting.

    Parameters
    ----------
    hour_line : int
        The hour to annotate on the plot.
    y_value : float
        The y-coordinate for annotation positioning.
    hour_values : list of int
        Hour values corresponding to the x-axis positions.
    start_plot_index : int
        Starting index for the plot's data.
    line_styles : dict
        Line styles for annotations keyed by hour.
    x_margin : float
        Margin added to x-axis for annotation positioning.
    annotation_prefix : str
        Prefix for the annotation text (e.g., "On average").
    text_y_offset : float, optional
        Vertical offset for the annotation text from the line, by default 1.
    text_x_position : float, optional
        Horizontal position for annotation text, by default None.
    slope : float, optional
        Slope of a line for extended annotations, by default None.
    x1 : float, optional
        Reference x-coordinate for slope-based annotation, by default None.
    y1 : float, optional
        Reference y-coordinate for slope-based annotation, by default None.
    """
    a = hour_values[hour_line - start_plot_index]
    if slope is not None and x1 is not None:
        y_a = slope * (a - x1) + y1
        plt.plot([a, a], [0, y_a], color="grey", linestyle=line_styles[hour_line])
        plt.plot(
            [0 - x_margin, a],
            [y_a, y_a],
            color="grey",
            linestyle=line_styles[hour_line],
        )
        annotation_text = (
            f"{annotation_prefix}, {int(y_a)} beds needed by {hour_line}:00"
        )
        y_position = y_a + text_y_offset
    else:
        plt.annotate(
            "",
            xy=(a, y_value),
            xytext=(a, 0),
            arrowprops=dict(
                arrowstyle="-", linestyle=line_styles[hour_line], color="grey"
            ),
        )
        plt.annotate(
            "",
            xy=(a, y_value),
            xytext=(hour_values[0] - x_margin, y_value),
            arrowprops=dict(
                arrowstyle="-", linestyle=line_styles[hour_line], color="grey"
            ),
        )
        annotation_text = (
            f"{annotation_prefix}, {int(y_value)} beds needed by {hour_line}:00"
        ).strip()  # strip() removes leading comma if prefix is empty
        y_position = y_value + text_y_offset

    # Use custom text x position if provided, otherwise use default
    x_position = (
        text_x_position if text_x_position is not None else (hour_values[1] - x_margin)
    )

    plt.annotate(
        annotation_text,
        xy=(a / 2 if slope is not None else a, y_value),
        xytext=(x_position, y_position),
        va="bottom",
        ha="left",
        fontsize=10,
    )


def plot_arrival_rates(
    inpatient_arrivals,
    title,
    inpatient_arrivals_2=None,
    labels=None,
    lagged_by=None,
    curve_params=None,
    time_interval=60,
    start_plot_index=0,
    x_margin=0.5,
    file_prefix="",
    media_file_path=None,
    file_name=None,
    num_days=None,
    num_days_2=None,
    return_figure=False,
):
    """Plot arrival rates for one or two datasets with optional lagged and spread rates.

    Parameters
    ----------
    inpatient_arrivals : array-like
        Primary dataset of inpatient arrivals.
    title : str
        Title of the plot.
    inpatient_arrivals_2 : array-like, optional
        Optional second dataset for comparison, by default None.
    labels : tuple of str, optional
        Labels for the datasets when comparing two datasets, by default None.
    lagged_by : int, optional
        Time lag in hours to apply to the arrival rates, by default None.
    curve_params : tuple of float, optional
        Parameters for spread arrival rates as (x1, y1, x2, y2), by default None.
    time_interval : int, optional
        Time interval in minutes for arrival rate calculations, by default 60.
    start_plot_index : int, optional
        Starting hour index for plotting, by default 0.
    x_margin : float, optional
        Margin on the x-axis, by default 0.5.
    file_prefix : str, optional
        Prefix for the saved file name, by default "".
    media_file_path : str or Path, optional
        Directory path to save the plot, by default None.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, uses file_prefix + cleaned title.
    num_days : int, optional
        Number of days in the first dataset, by default None.
    num_days_2 : int, optional
        Number of days in the second dataset, by default None.
    return_figure : bool, optional
        If True, returns the matplotlib figure instead of displaying it, by default False.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot.
    """
    is_dual_plot = inpatient_arrivals_2 is not None
    if is_dual_plot and labels is None:
        labels = ("Dataset 1", "Dataset 2")

    datasets = [(inpatient_arrivals, "C0", "o", num_days)]
    if is_dual_plot:
        datasets.append((inpatient_arrivals_2, "C1", "s", num_days_2))

    # Calculate and process arrival rates for all datasets
    processed_data = []
    max_y_values = []

    for dataset, color, marker, num_days in datasets:
        # Calculate base arrival rates
        arrival_rates_dict = time_varying_arrival_rates(
            dataset, time_interval, num_days=num_days
        )
        arrival_rates, hour_labels, hour_values = process_arrival_rates(
            arrival_rates_dict
        )
        max_y_values.append(max(arrival_rates))

        # Calculate lagged rates if needed
        arrival_rates_lagged = None
        if lagged_by is not None:
            arrival_rates_lagged_dict = time_varying_arrival_rates_lagged(
                dataset, lagged_by, yta_time_interval=time_interval, num_days=num_days
            )
            arrival_rates_lagged, _, _ = process_arrival_rates(
                arrival_rates_lagged_dict
            )
            max_y_values.append(max(arrival_rates_lagged))

        # Calculate spread rates if needed
        arrival_rates_spread = None
        if curve_params is not None:
            x1, y1, x2, y2 = curve_params
            arrival_rates_spread_dict = unfettered_demand_by_hour(
                dataset, x1, y1, x2, y2, num_days=num_days
            )
            arrival_rates_spread, _, _ = process_arrival_rates(
                arrival_rates_spread_dict
            )
            max_y_values.append(max(arrival_rates_spread))

        processed_data.append(
            {
                "arrival_rates": arrival_rates,
                "arrival_rates_lagged": arrival_rates_lagged,
                "arrival_rates_spread": arrival_rates_spread,
                "color": color,
                "marker": marker,
                "dataset_label": labels[len(processed_data)] if is_dual_plot else None,
            }
        )

    # Helper function to create cyclic data
    def get_cyclic_data(data):
        return data[start_plot_index:] + data[0:start_plot_index]

    # Plot setup
    fig = plt.figure(figsize=(10, 6))
    x_values = get_cyclic_data(hour_labels)

    # Plot data for each dataset
    for data in processed_data:
        dataset_suffix = f" ({data['dataset_label']})" if data["dataset_label"] else ""

        # Base arrival rates
        base_label = f"Arrival rates of admitted patients{dataset_suffix}"
        plt.plot(
            x_values,
            get_cyclic_data(data["arrival_rates"]),
            marker="x",
            color=data["color"],
            markersize=4,
            linestyle=":" if (curve_params or lagged_by) else "-",
            linewidth=1 if (curve_params or lagged_by) else None,
            label=base_label,
        )

        if lagged_by is not None:
            # Lagged arrival rates
            lagged_label = f"Average number of beds needed assuming admission\nexactly {lagged_by} hours after arrival{dataset_suffix}"
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates_lagged"]),
                marker="o",
                markersize=4,
                color=data["color"],
                linestyle="--",
                linewidth=1,
                label=lagged_label,
            )

        if curve_params is not None and data["arrival_rates_spread"] is not None:
            # Spread arrival rates
            spread_label = f"Average number of beds applying ED targets of {int(y1*100)}% in {int(x1)} hours{dataset_suffix}"
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates_spread"]),
                marker=data["marker"],  # Keep original dataset marker
                color=data["color"],  # Keep original dataset color
                label=spread_label,
            )

    # Set plot limits and labels
    plt.ylim(0, max(max_y_values) + 0.25)
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)

    plt.xlabel("Hour of day")
    plt.ylabel("Arrival Rate (patients per hour)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Always show legend if there are multiple datasets or multiple rate types
    if is_dual_plot or lagged_by is not None or curve_params is not None:
        plt.legend()

    plt.tight_layout()

    # Save if path provided
    if media_file_path:
        if file_name:
            filename = file_name
        else:
            filename = f"{file_prefix}{clean_title_for_filename(title)}"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()


def get_window_parameters(data, start_window, end_window, hour_values):
    """Calculate window parameters for visualization.

    Parameters
    ----------
    data : array-like
        Reindexed cumulative data
    start_window : int
        Start position in reindexed space
    end_window : int
        End position in reindexed space
    hour_values : array-like
        Original hour values for display

    Returns
    -------
    tuple
        (slope, x1, y1, x2, y2) where:
        - slope: float, The calculated slope of the line
        - x1: float, Start hour value
        - y1: float, Start y-value
        - x2: float, End hour value
        - y2: float, End y-value
    """
    y1 = data[start_window]
    y2 = data[-1]
    x1 = hour_values[start_window]  # Get display hour
    x2 = hour_values[end_window]  # Get display hour
    slope = (y2 - y1) / (x2 - x1)

    return slope, x1, y1, x2, y2


def draw_window_visualization(
    ax, hour_values, window_params, annotation_prefix, start_window, end_window
):
    """Draw the window visualization with annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    hour_values : array-like
        Hour labels for x-axis
    window_params : tuple
        (slope, x1, y1, y2) from get_window_parameters
    annotation_prefix : str
        Prefix for annotations
    start_window : int
        Start hour for window
    end_window : int
        End hour for window
    """
    slope, x1, y1, x2, y2 = window_params

    # Draw horizontal line
    ax.hlines(y=y2, xmin=x2, xmax=hour_values[-1], color="blue", linestyle="--")

    # Draw diagonal line
    ax.plot([x1, x2], [y1, y2], color="blue", linestyle="--")

    # Add annotation
    ax.annotate(
        f"{annotation_prefix}, {slope:.0f} beds need to be vacated\n"
        f"each hour between {start_window}:00 and {end_window}:00\n"
        f"to create capacity for all overnight arrivals\n"
        f"by {end_window}:00",
        xy=(hour_values[-1], y2 * 0.25),
        xytext=(hour_values[-1], y2 * 0.25),
        va="top",
        ha="right",
    )


def plot_cumulative_arrival_rates(
    inpatient_arrivals,
    title,
    curve_params=None,
    lagged_by=None,
    time_interval=60,
    start_plot_index=0,
    draw_window=None,
    x_margin=0.5,
    file_prefix="",
    set_y_lim=None,
    hour_lines=[12, 17],
    line_styles={12: "--", 17: ":", 20: "--"},
    annotation_prefix="On average",
    line_colour="red",
    media_file_path=None,
    file_name=None,
    plot_centiles=False,
    highlight_centile=0.9,
    centiles=[0.3, 0.5, 0.7, 0.9, 0.99],
    markers=["D", "s", "^", "o", "v"],
    line_styles_centiles=["-.", "--", ":", "-", "-"],
    bed_type_spec="",
    text_y_offset=1,
    num_days=None,
    return_figure=False,
):
    """Plot cumulative arrival rates with optional statistical distributions.

    Parameters
    ----------
    inpatient_arrivals : array-like
        Dataset of inpatient arrivals.
    title : str
        Title of the plot.
    curve_params : tuple of float, optional
        Parameters for spread rates as (x1, y1, x2, y2), by default None.
    lagged_by : int, optional
        Time lag in hours for cumulative rates, by default None.
    time_interval : int, optional
        Time interval in minutes for rate calculations, by default 60.
    start_plot_index : int, optional
        Starting hour index for plotting, by default 0.
    draw_window : tuple of int, optional
        Time window for detailed annotation, by default None.
    x_margin : float, optional
        Margin on the x-axis, by default 0.5.
    file_prefix : str, optional
        Prefix for the saved file name, by default "".
    set_y_lim : float, optional
        Upper limit for the y-axis, by default None.
    hour_lines : list of int, optional
        Specific hours to annotate, by default [12, 17].
    line_styles : dict, optional
        Line styles for hour annotations keyed by hour, by default {12: "--", 17: ":", 20: "--"}.
    annotation_prefix : str, optional
        Prefix for annotations, by default "On average".
    line_colour : str, optional
        Color for the main line plot, by default "red".
    media_file_path : str or Path, optional
        Directory path to save the plot, by default None.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, uses file_prefix + cleaned title.
    plot_centiles : bool, optional
        Whether to include percentile visualization, by default False.
    highlight_centile : float, optional
        Percentile to emphasize, by default 0.9. If 1.0 is provided, will use 0.9999 instead.
    centiles : list of float, optional
        List of percentiles to calculate, by default [0.3, 0.5, 0.7, 0.9, 0.99].
    markers : list of str, optional
        Marker styles for percentile lines, by default ["D", "s", "^", "o", "v"].
    line_styles_centiles : list of str, optional
        Line styles for percentile visualization, by default ["-.", "--", ":", "-", "-"].
    bed_type_spec : str, optional
        Specification for bed type in annotations, by default "".
    text_y_offset : float, optional
        Vertical offset for text annotations, by default 1.
    num_days : int, optional
        Number of days in the dataset, by default None.
    return_figure : bool, optional
        If True, returns the matplotlib figure instead of displaying it, by default False.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot.
    """

    # Handle edge case for highlight_centile = 1.0
    original_highlight_centile = highlight_centile
    if highlight_centile >= 1.0:
        highlight_centile = 0.9999  # Use a very high but not exactly 1.0 value

    # Ensure centiles are all valid (no 1.0 values)
    processed_centiles = [min(c, 0.9999) if c >= 1.0 else c for c in centiles]

    # Data processing
    if curve_params is not None:
        x1, y1, x2, y2 = curve_params
        arrival_rates_dict = unfettered_demand_by_hour(
            inpatient_arrivals, x1, y1, x2, y2, num_days=num_days
        )
    elif lagged_by is not None:
        arrival_rates_dict = time_varying_arrival_rates_lagged(
            inpatient_arrivals, lagged_by, time_interval, num_days=num_days
        )
    else:
        arrival_rates_dict = time_varying_arrival_rates(
            inpatient_arrivals, time_interval, num_days=num_days
        )

    # Process arrival rates
    arrival_rates, hour_labels, hour_values = process_arrival_rates(arrival_rates_dict)

    # Reindex based on start_plot_index
    rates_reindexed = (
        list(arrival_rates)[start_plot_index:] + list(arrival_rates)[0:start_plot_index]
    )
    labels_reindexed = (
        list(hour_labels)[start_plot_index:] + list(hour_labels)[0:start_plot_index]
    )

    # Calculate percentiles
    percentiles = [[] for _ in range(len(processed_centiles))]
    cumulative_value_at_centile = np.zeros(len(processed_centiles))

    for hour in range(len(rates_reindexed)):
        for i, centile in enumerate(processed_centiles):
            value_at_centile = stats.poisson.ppf(centile, rates_reindexed[hour])
            cumulative_value_at_centile[i] += value_at_centile
            percentiles[i].append(value_at_centile)

    # Set up plot
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot mean line
    label_suffix = f" {bed_type_spec} beds needed" if bed_type_spec else " beds needed"
    cumsum_rates = np.cumsum(rates_reindexed)

    plt.plot(
        labels_reindexed,
        cumsum_rates,
        marker="o",
        markersize=3,
        color=line_colour,
        linewidth=2,
        alpha=0.7,
        label=f"Average number of{label_suffix}",
    )

    # set max y value assuming centiles not plotted
    max_y = cumsum_rates[-1]

    if plot_centiles:
        # Calculate and plot percentiles
        percentiles = [[] for _ in range(len(processed_centiles))]
        cumulative_value_at_centile = np.zeros(len(processed_centiles))
        highlight_percentile_data = None

        # Find the index of highlight_centile in processed_centiles
        highlight_index = -1
        for i, c in enumerate(processed_centiles):
            if (
                abs(c - highlight_centile) < 0.0001
            ):  # Use small epsilon for float comparison
                highlight_index = i
                break

        # If highlight_centile is not in processed_centiles, add it
        if highlight_index == -1:
            processed_centiles.append(highlight_centile)
            percentiles.append([])
            cumulative_value_at_centile = np.append(cumulative_value_at_centile, 0)

        for hour in range(len(rates_reindexed)):
            for i, centile in enumerate(processed_centiles):
                try:
                    # Add error handling for ppf calculation
                    value_at_centile = stats.poisson.ppf(centile, rates_reindexed[hour])

                    # Apply a reasonable upper limit if the value is extremely large
                    if (
                        np.isinf(value_at_centile)
                        or value_at_centile > 1000 * rates_reindexed[hour]
                    ):
                        value_at_centile = 10 * rates_reindexed[hour]

                    cumulative_value_at_centile[i] += value_at_centile
                    percentiles[i].append(value_at_centile)
                except (ValueError, OverflowError, RuntimeError):
                    # Fallback if calculation fails
                    fallback_value = 10 * rates_reindexed[hour]
                    cumulative_value_at_centile[i] += fallback_value
                    percentiles[i].append(fallback_value)

                # Match the highlight centile to the processed value
                if (
                    abs(centile - highlight_centile) < 0.0001
                ):  # Use a small epsilon for floating point comparison
                    highlight_percentile_data = np.cumsum(percentiles[i])

        # Plot percentile lines
        for i, centile in enumerate(processed_centiles):
            marker = markers[i % len(markers)]
            line_style = line_styles_centiles[i % len(line_styles_centiles)]
            linewidth = 2 if centile == highlight_centile else 1
            alpha = 1.0 if centile == highlight_centile else 0.7

            # If the user requested 1.0, display as 99.99% since a Poisson distribution
            # cannot provide exact 100% probability with any finite value
            display_centile = processed_centiles[i]
            if centile == highlight_centile and original_highlight_centile >= 1.0:
                display_centile = (
                    0.9999  # Use 99.99% as the highest displayable probability
                )

            # Format the label text with appropriate precision
            if display_centile >= 0.999:
                # For very high probabilities, show as 99.9% or 99.99% to avoid implying exact 100%
                label_text = f"{display_centile*100:.2f}% probability"
            else:
                label_text = f"{display_centile*100:.0f}% probability"

            cumsum_percentile = np.cumsum(percentiles[i])
            plt.plot(
                labels_reindexed,
                cumsum_percentile,
                marker=marker,
                markersize=3,
                linestyle=line_style,
                color="C0",
                linewidth=linewidth,
                alpha=alpha,
                label=label_text,
            )
        # update max y
        max_y = max(cumulative_value_at_centile)

        # Draw window visualization if requested
        if draw_window:
            start_window, end_window = draw_window
            reindexed_start = (start_window - start_plot_index) % len(
                highlight_percentile_data
            )
            reindexed_end = (end_window - start_plot_index) % len(
                highlight_percentile_data
            )
            window_params = get_window_parameters(
                highlight_percentile_data, reindexed_start, reindexed_end, hour_values
            )
            draw_window_visualization(
                ax,
                hour_values,
                window_params,
                annotation_prefix,
                start_window,
                end_window,
            )
            slope, x1, y1, x2, y2 = window_params
            for hour_line in hour_lines:
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=y1,
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                    slope=slope,
                    x1=x1,
                    y1=y1,
                )

        else:
            # Regular percentile annotations
            for hour_line in hour_lines:
                # Check if highlight_percentile_data is available
                if highlight_percentile_data is None:
                    # Fall back to mean line if no highlight data
                    cumsum_at_hour = cumsum_rates[hour_line - start_plot_index]
                else:
                    cumsum_at_hour = highlight_percentile_data[
                        hour_line - start_plot_index
                    ]
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=cumsum_at_hour,
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                    text_y_offset=text_y_offset,
                )

        # Reverse legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc="upper left")
    else:
        plt.legend(loc="upper left")

        if draw_window:
            start_window, end_window = draw_window
            reindexed_start = (start_window - start_plot_index) % len(cumsum_rates)
            reindexed_end = (end_window - start_plot_index) % len(cumsum_rates)
            window_params = get_window_parameters(
                cumsum_rates, reindexed_start, reindexed_end, hour_values
            )
            draw_window_visualization(
                ax,
                hour_values,
                window_params,
                annotation_prefix,
                start_window,
                end_window,
            )
            slope, x1, y1, x2, y2 = window_params
            for hour_line in hour_lines:
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=y1,
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                    slope=slope,
                    x1=x1,
                    y1=y1,
                )
        else:
            # Regular mean line annotations
            for hour_line in hour_lines:
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=cumsum_rates[hour_line - start_plot_index],
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                )

    plt.xlabel("Hour of day")
    plt.ylabel("Cumulative number of beds needed")
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)
    plt.ylim(0, set_y_lim if set_y_lim else max(max_y + 2, max_y * 1.2))
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(5))

    plt.title(title)
    plt.tight_layout()

    if media_file_path:
        if file_name:
            filename = file_name
        else:
            filename = f"{file_prefix}{clean_title_for_filename(title)}"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
