"""Visualisation module for plotting data distributions.

This module provides functions for creating distribution plots of data variables
grouped by categories.

Functions
---------
plot_data_distribution : function
    Plot distributions of data variables grouped by categories
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_data_distribution(
    df,
    col_name,
    grouping_var,
    grouping_var_name,
    plot_type="both",
    title=None,
    rotate_x_labels=False,
    is_discrete=False,
    ordinal_order=None,
    media_file_path=None,
    file_name=None,
    return_figure=False,
    truncate_outliers=True,
    outlier_method="zscore",
    outlier_threshold=2.0,
):
    """Plot distributions of data variables grouped by categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot
    col_name : str
        Name of the column to plot distributions for
    grouping_var : str
        Name of the column to group the data by
    grouping_var_name : str
        Display name for the grouping variable
    plot_type : {'both', 'hist', 'kde'}, default='both'
        Type of plot to create. 'both' shows histogram with KDE, 'hist' shows
        only histogram, 'kde' shows only KDE plot
    title : str, optional
        Title for the plot
    rotate_x_labels : bool, default=False
        Whether to rotate x-axis labels by 90 degrees
    is_discrete : bool, default=False
        Whether the data is discrete
    ordinal_order : list, optional
        Order of categories for ordinal data
    media_file_path : Path, optional
        Path where the plot should be saved
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "data_distributions.png".
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it
    truncate_outliers : bool, default=True
        Whether to truncate the x-axis to exclude extreme outliers
    outlier_method : {'iqr', 'zscore'}, default='zscore'
        Method to detect outliers. 'iqr' uses interquartile range, 'zscore' uses z-score
    outlier_threshold : float, default=1.5
        Threshold for outlier detection. For IQR method, this is the multiplier.
        For z-score method, this is the number of standard deviations.

    Returns
    -------
    seaborn.FacetGrid or None
        If return_figure is True, returns the FacetGrid object. Otherwise,
        displays the plot and returns None.

    Raises
    ------
    ValueError
        If plot_type is not one of 'both', 'hist', or 'kde'
        If outlier_method is not one of 'iqr' or 'zscore'
    """
    sns.set_theme(style="whitegrid")

    if ordinal_order is not None:
        df[col_name] = pd.Categorical(
            df[col_name], categories=ordinal_order, ordered=True
        )

    # Calculate outlier bounds if truncation is requested
    x_limits = None
    if truncate_outliers:
        values = df[col_name].dropna()
        if pd.api.types.is_numeric_dtype(values) and len(values) > 0:
            # Check if data is actually discrete (all values are integers)
            is_actually_discrete = np.allclose(values, values.round())

            # Apply outlier truncation to continuous data OR discrete data with outliers
            # For discrete data, we still want to truncate if there are extreme outliers
            if outlier_method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
            elif outlier_method == "zscore":
                mean_val = values.mean()
                std_val = values.std()
                lower_bound = mean_val - outlier_threshold * std_val
                upper_bound = mean_val + outlier_threshold * std_val
            else:
                raise ValueError(
                    "Invalid outlier_method. Choose from 'iqr' or 'zscore'."
                )

            # Only apply truncation if there are actual outliers
            # For discrete data, ensure lower bound is at least 0
            if values.min() < lower_bound or values.max() > upper_bound:
                if is_actually_discrete:
                    # For discrete data, ensure bounds are reasonable
                    lower_bound = max(0, lower_bound)
                x_limits = (lower_bound, upper_bound)

    g = sns.FacetGrid(df, col=grouping_var, height=3, aspect=1.5)

    if is_discrete:
        valid_values = sorted([x for x in df[col_name].unique() if pd.notna(x)])
        min_val = min(valid_values)
        max_val = max(valid_values)
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    else:
        # Handle numeric data
        values = df[col_name].dropna()
        if pd.api.types.is_numeric_dtype(values):
            if np.allclose(values, values.round()):
                bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
            else:
                n_bins = min(100, max(10, int(np.sqrt(len(values)))))
                bins = n_bins
        else:
            bins = "auto"

    if plot_type == "both":
        g.map(sns.histplot, col_name, kde=True, bins=bins)
    elif plot_type == "hist":
        g.map(sns.histplot, col_name, kde=False, bins=bins)
    elif plot_type == "kde":
        g.map(sns.kdeplot, col_name, fill=True)
    else:
        raise ValueError("Invalid plot_type. Choose from 'both', 'hist', or 'kde'.")

    g.set_axis_labels(
        col_name, "Frequency" if plot_type != "kde" else "Density", fontsize=10
    )

    # Set facet titles with smaller font
    g.set_titles(col_template=f"{grouping_var}: {{col_name}}", size=11)

    # Add thousands separators to y-axis
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )

    if rotate_x_labels:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)

    if is_discrete:
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            # Apply outlier truncation if available, otherwise use default discrete limits
            if x_limits is not None:
                # Ensure discrete limits are reasonable: min ≥ 0, max ≥ 1, and use integers
                lower_limit = max(0, int(x_limits[0]))
                upper_limit = max(
                    1, int(x_limits[1] + 0.5)
                )  # Round up to ensure we include the max value
                ax.set_xlim(lower_limit - 0.5, upper_limit + 0.5)
            else:
                # Ensure default discrete limits are reasonable: min ≥ 0, max ≥ 1
                # Use the actual min/max values to center the bars properly
                lower_limit = max(0, min_val)
                upper_limit = max(1, max_val)
                ax.set_xlim(lower_limit - 0.5, upper_limit + 0.5)
    elif x_limits is not None:
        # Apply outlier truncation to x-axis
        for ax in g.axes.flat:
            ax.set_xlim(x_limits)
            # Ensure integer tick marks for numeric data with outliers
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        # Let matplotlib auto-scale the x-axis
        pass

    plt.subplots_adjust(top=0.80)
    if title:
        g.figure.suptitle(title, fontsize=14)
    else:
        g.figure.suptitle(
            f"Distribution of {col_name} grouped by {grouping_var_name}", fontsize=14
        )

    if media_file_path:
        if file_name:
            filename = file_name
        else:
            filename = "data_distributions.png"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return g
    else:
        plt.show()
        plt.close()
