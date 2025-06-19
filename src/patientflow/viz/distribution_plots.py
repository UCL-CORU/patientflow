"""Visualization module for plotting prediction and data distributions.

This module provides functions for creating various distribution plots, including
prediction distributions for trained models and data distributions for different
variables.

Functions
---------
plot_prediction_distributions : function
    Plot prediction distributions for multiple models
plot_data_distributions : function
    Plot distributions of data variables grouped by categories
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import prepare_patient_snapshots
from patientflow.model_artifacts import TrainedClassifier
from typing import Optional
from pathlib import Path


# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"


def plot_prediction_distributions(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits,
    exclude_from_training_data,
    bins=30,
    media_file_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    return_figure=False,
    label_col: str = "is_admitted",
):
    """Plot prediction distributions for multiple models.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of TrainedClassifier objects or dict with TrainedClassifier values
    test_visits : pandas.DataFrame
        DataFrame containing test visit data
    exclude_from_training_data : list
        Columns to exclude from the test data
    bins : int, default=30
        Number of bins for the histograms
    media_file_path : Path, optional
        Path where the plot should be saved
    suptitle : str, optional
        Optional super title for the entire figure
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it
    label_col : str, default="is_admitted"
        Name of the column containing the target labels

    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figure is True, returns the figure object. Otherwise, displays
        the plot and returns None.
    """
    # Convert dict to list if needed
    if isinstance(trained_models, dict):
        trained_models = list(trained_models.values())

    # Sort trained_models by prediction time
    trained_models_sorted = sorted(
        trained_models,
        key=lambda x: x.training_results.prediction_time[0] * 60
        + x.training_results.prediction_time[1],
    )
    num_plots = len(trained_models_sorted)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    for i, trained_model in enumerate(trained_models_sorted):
        # Use calibrated pipeline if available, otherwise use regular pipeline
        if (
            hasattr(trained_model, "calibrated_pipeline")
            and trained_model.calibrated_pipeline is not None
        ):
            pipeline = trained_model.calibrated_pipeline
        else:
            pipeline = trained_model.pipeline

        prediction_time = trained_model.training_results.prediction_time

        # Get test data for this prediction time
        X_test, y_test = prepare_patient_snapshots(
            df=test_visits,
            prediction_time=prediction_time,
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
            label_col=label_col,
        )

        X_test = add_missing_columns(pipeline, X_test)

        # Get predictions
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Separate predictions for positive and negative cases
        pos_preds = y_pred_proba[y_test == 1]
        neg_preds = y_pred_proba[y_test == 0]

        ax = axs[i]
        hour, minutes = prediction_time

        # Plot distributions
        ax.hist(
            neg_preds,
            bins=bins,
            alpha=0.5,
            color=primary_color,
            density=True,
            label="Negative Cases",
            histtype="step",
            linewidth=2,
        )
        ax.hist(
            pos_preds,
            bins=bins,
            alpha=0.5,
            color=secondary_color,
            density=True,
            label="Positive Cases",
            histtype="step",
            linewidth=2,
        )

        # Optional: Fill with lower opacity
        ax.hist(neg_preds, bins=bins, alpha=0.2, color=primary_color, density=True)
        ax.hist(pos_preds, bins=bins, alpha=0.2, color=secondary_color, density=True)

        ax.set_title(f"Prediction Distribution at {hour}:{minutes:02}", fontsize=14)
        ax.set_xlabel("Estimated Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend()

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle is not None:
        plt.suptitle(suptitle, y=1.05, fontsize=16)

    if media_file_path:
        plt.savefig(media_file_path / "prediction_distributions.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()


def plot_data_distributions(
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
        plt.savefig(media_file_path / "data_distributions.png", dpi=300)

    if return_figure:
        return g
    else:
        plt.show()
        plt.close()
