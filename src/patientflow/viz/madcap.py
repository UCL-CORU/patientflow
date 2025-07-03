"""
Module for generating MADCAP (Model Accuracy and Discriminative Calibration Plots) visualizations.

MADCAP plots compare model-predicted probabilities to observed outcomes, helping to assess
model calibration and discrimination. The plots can be generated for individual
prediction times or for specific groups (e.g., age groups).

Functions
---------
classify_age : function
    Classifies age into categories based on numeric values or age group strings.

plot_madcap : function
    Generates MADCAP plots for a list of trained models, comparing estimated probabilities
    to observed values.

_plot_madcap_subplot : function
    Plots a single MADCAP subplot showing cumulative predicted and observed values.

_plot_madcap_by_group_single : function
    Generates MADCAP plots for specific groups at a given prediction time.

plot_madcap_by_group(prediction_times, model_file_path, media_file_path, visits_csv_path, grouping_var, grouping_var_name)
    Generates MADCAP plots for groups (e.g., age groups) across a series of prediction times.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import prepare_patient_snapshots
from patientflow.model_artifacts import TrainedClassifier

exclude_from_training_data = [
    "visit_number",
    "snapshot_date",
    "prediction_time",
    "specialty",
    "consultation_sequence",
    "final_sequence",
]

# Default age categories for classification
DEFAULT_AGE_CATEGORIES = {
    "Children": {"numeric": {"max": 17}, "groups": ["0-17"]},
    "Adults < 65": {
        "numeric": {"min": 18, "max": 64},
        "groups": ["18-24", "25-34", "35-44", "45-54", "55-64"],
    },
    "Adults 65 or over": {"numeric": {"min": 65}, "groups": ["65-74", "75-115"]},
}


def classify_age(age, age_categories=None):
    """Classify age into categories based on numeric values or age group strings.

    Parameters
    ----------
    age : int, float, or str
        Age value (e.g., 30) or age group string (e.g., '18-24').
    age_categories : dict, optional
        Dictionary defining age categories and their ranges. If not provided, uses DEFAULT_AGE_CATEGORIES.
        Expected format:
        {
            "category_name": {
                "numeric": {"min": min_age, "max": max_age},
                "groups": ["age_group1", "age_group2", ...]
            }
        }

    Returns
    -------
    str
        Category name based on the age or age group, or 'unknown' for unexpected or invalid values.

    Examples
    --------
    >>> classify_age(25)
    'adults'
    >>> classify_age('65-74')
    '65 or over'
    """
    if age_categories is None:
        age_categories = DEFAULT_AGE_CATEGORIES

    if isinstance(age, (int, float)):
        for category, rules in age_categories.items():
            numeric_rules = rules.get("numeric", {})
            min_age = numeric_rules.get("min", float("-inf"))
            max_age = numeric_rules.get("max", float("inf"))

            if min_age <= age <= max_age:
                return category
        return "unknown"
    elif isinstance(age, str):
        for category, rules in age_categories.items():
            if age in rules.get("groups", []):
                return category
        return "unknown"
    else:
        return "unknown"


def plot_madcap(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits: pd.DataFrame,
    exclude_from_training_data: List[str],
    media_file_path: Optional[Path] = None,
    file_name: Optional[str] = None,
    suptitle: Optional[str] = None,
    return_figure: bool = False,
    label_col: str = "is_admitted",
) -> Optional[plt.Figure]:
    """Generate MADCAP plots for a list of trained models.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of trained classifier objects or dictionary with TrainedClassifier values.
    test_visits : pd.DataFrame
        DataFrame containing test visit data.
    exclude_from_training_data : List[str]
        List of columns to exclude from training data.
    media_file_path : Path, optional
        Directory path where the generated plots will be saved.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "madcap_plot.png".
    suptitle : str, optional
        Suptitle for the plot.
    return_figure : bool, default=False
        If True, returns the figure object instead of displaying it.
    label_col : str, default="is_admitted"
        Name of the column containing the target labels.

    Returns
    -------
    Optional[plt.Figure]
        The figure if return_figure is True, None otherwise.
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

    # Calculate the number of rows and columns for the subplots
    num_cols = min(num_plots, 5)  # Maximum 5 columns
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_plots * 5, 4))

    # Handle the case of a single plot differently
    if num_plots == 1:
        # When there's only one plot, axes is a single Axes object, not an array
        trained_model = trained_models_sorted[0]

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
        predict_proba = pipeline.predict_proba(X_test)[:, 1]

        # Plot directly on the single axes
        _plot_madcap_subplot(predict_proba, y_test, prediction_time, axes)
    else:
        # For multiple plots, ensure axes is always a 2D array
        if num_rows == 1:
            axes = axes.reshape(1, -1)

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
            predict_proba = pipeline.predict_proba(X_test)[:, 1]

            row = i // num_cols
            col = i % num_cols
            _plot_madcap_subplot(predict_proba, y_test, prediction_time, axes[row, col])

        # Hide any unused subplots
        for j in range(i + 1, num_rows * num_cols):
            row = j // num_cols
            col = j % num_cols
            axes[row, col].axis("off")

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.05)
        # Adjust layout to accommodate suptitle
        plt.subplots_adjust(top=0.85)

    if media_file_path:
        plot_name = file_name if file_name else "madcap_plot.png"
        madcap_plot_path = Path(media_file_path) / plot_name
        plt.savefig(madcap_plot_path, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)
        return None


def _plot_madcap_subplot(predict_proba, label, _prediction_time, ax):
    """Plot a single MADCAP subplot showing cumulative estimated and observed values.

    Parameters
    ----------
    predict_proba : array-like
        Array of predicted probabilities.
    label : array-like
        Array of true labels.
    _prediction_time : tuple
        Prediction time as (hour, minute).
    ax : matplotlib.axes.Axes
        The axis on which the subplot will be drawn.

    Notes
    -----
    The plot shows:

    * X-axis: Cases ordered by estimated probability
    * Y-axis: Cumulative count of positive outcomes
    * Two lines: predicted (blue) and observed (orange) cumulative counts
    """
    hour, minutes = _prediction_time
    # Ensure inputs are numpy arrays
    predict_proba = np.array(predict_proba)
    label = np.array(label)

    # Sort by predict_proba
    sorted_indices = np.argsort(predict_proba)
    sorted_proba = predict_proba[sorted_indices]
    sorted_label = label[sorted_indices]

    # Compute unique probabilities and their mean labels
    unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
    mean_labels = np.zeros_like(unique_probs)

    np.add.at(mean_labels, inverse_indices, sorted_label)
    counts = np.bincount(inverse_indices)
    mean_labels = mean_labels / counts

    # Cumulative sums for model and observed
    model = np.cumsum(sorted_proba)
    observed = np.cumsum(mean_labels[inverse_indices])

    x = np.arange(len(sorted_proba))

    # Plot
    ax.plot(x, model, label="predicted")
    ax.plot(x, observed, label="observed")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_xlabel("Cases ordered by estimated probability", fontsize=12)
    ax.set_ylabel("Cumulative count of positive outcomes", fontsize=12)
    ax.set_title(f"MADCAP Plot for {hour}:{minutes:02}", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize="x-small")


def _plot_madcap_by_group_single(
    predict_proba,
    label,
    group,
    _prediction_time,
    group_name,
    media_path=None,
    file_name: Optional[str] = None,
    plot_difference=True,
    return_figure=False,
):
    """Generate MADCAP plots for specific groups at a given prediction time.

    Parameters
    ----------
    predict_proba : array-like
        Array of estimated probabilities.
    label : array-like
        Array of true labels.
    group : array-like
        Array of group labels for each case (e.g., age group).
    _prediction_time : tuple
        Prediction time as (hour, minute).
    group_name : str
        Name of the group variable being plotted (e.g., 'Age Group').
    media_path : str or Path, optional
        Path to save the generated plot.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to a generated name based on group and time.
    plot_difference : bool, default=True
        If True, includes an additional plot showing the difference between predicted
        and observed outcomes.
    return_figure : bool, default=False
        If True, returns the figure object instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot and returns None.

    Notes
    -----
    For each group, generates:

    * A MADCAP plot showing predicted vs observed cumulative counts
    * Optionally, a difference plot showing predicted minus observed counts
    """
    # Remove those with unknown age
    mask_known = group != "unknown"
    predict_proba = predict_proba[mask_known]
    label = label[mask_known]
    group = group[mask_known]

    hour, minutes = _prediction_time

    predict_proba, label, group = map(np.array, (predict_proba, label, group))
    unique_groups = [grp for grp in np.unique(group) if grp != "unknown"]

    fig_size = (10, 8) if plot_difference else (9, 3)
    fig, ax = plt.subplots(
        2 if plot_difference else 1, len(unique_groups), figsize=fig_size
    )
    ax = ax.reshape(-1, len(unique_groups)) if plot_difference else ax.reshape(1, -1)

    for i, grp in enumerate(unique_groups):
        mask = group == grp
        sorted_indices = np.argsort(predict_proba[mask])
        sorted_proba = predict_proba[mask][sorted_indices]
        sorted_label = label[mask][sorted_indices]

        unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
        mean_labels = np.bincount(inverse_indices, weights=sorted_label) / np.bincount(
            inverse_indices
        )

        model = np.cumsum(sorted_proba)
        observed = np.cumsum(mean_labels[inverse_indices])
        x = np.arange(len(sorted_proba))

        ax[0, i].plot(x, model, label="predicted")
        ax[0, i].plot(x, observed, label="observed")
        ax[0, i].legend(loc="upper left", fontsize=8)
        ax[0, i].set_xlabel("Cases ordered by estimated probability", fontsize=8)
        ax[0, i].set_ylabel("Cumulative count of positive outcomes", fontsize=8)
        ax[0, i].set_title(f"{group_name}: {grp!s}", fontsize=8)
        ax[0, i].tick_params(axis="both", which="major", labelsize=8)

        if plot_difference:
            ax[1, i].plot(x, model - observed)
            ax[1, i].set_xlabel("Cases ordered by estimated probability", fontsize=8)
            ax[1, i].set_ylabel("Predicted - observed count", fontsize=8)
            ax[1, i].set_title(f"{group_name}: {grp!s}", fontsize=8)
            ax[1, i].tick_params(axis="both", which="major", labelsize=8)

    # Adjust layout first
    fig.tight_layout(pad=1.08)

    # Then add super title
    fig.suptitle(
        f"MADCAP Plots by {group_name} for {hour}:{minutes:02}", fontsize=10, y=1.04
    )

    # Fine-tune the layout
    fig.subplots_adjust(top=0.90)

    # fig.tight_layout(pad=1.08, rect=[0, 0.03, 1, 0.95])

    if media_path:
        plot_name = (
            file_name
            if file_name
            else f"madcap_plot_by_{group_name.replace(' ', '_')}_{hour}{minutes:02}.png"
        )
        madcap_plot_path = Path(media_path) / plot_name
        plt.savefig(madcap_plot_path, dpi=300, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_madcap_by_group(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits: pd.DataFrame,
    exclude_from_training_data: List[str],
    grouping_var: str,
    grouping_var_name: str,
    media_file_path: Optional[Path] = None,
    file_name: Optional[str] = None,
    plot_difference: bool = False,
    return_figure: bool = False,
    label_col: str = "is_admitted",
) -> Optional[List[plt.Figure]]:
    """Generate MADCAP plots for different groups across multiple prediction times.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of trained classifier objects or dictionary with TrainedClassifier values.
    test_visits : pd.DataFrame
        DataFrame containing the test visit data.
    exclude_from_training_data : List[str]
        List of columns to exclude from training data.
    grouping_var : str
        The column name in the dataset that defines the grouping variable.
    grouping_var_name : str
        A descriptive name for the grouping variable, used in plot titles.
    media_file_path : Path, optional
        Directory path where the generated plots will be saved.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to a generated name based on group and time.
    plot_difference : bool, default=False
        If True, includes difference plot between predicted and observed outcomes.
    return_figure : bool, default=False
        If True, returns a list of figure objects instead of displaying them.
    label_col : str, default="is_admitted"
        Name of the column containing the target labels.

    Returns
    -------
    Optional[List[plt.Figure]]
        List of figures if return_figure is True, None otherwise.
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

    figures = []
    for trained_model in trained_models_sorted:
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

        # Check if the grouping variable exists in X_test columns
        if grouping_var not in X_test.columns:
            raise ValueError(f"'{grouping_var}' not found in the dataset columns.")

        X_test = add_missing_columns(pipeline, X_test)
        predict_proba = pipeline.predict_proba(X_test)[:, 1]

        # Apply classification based on the grouping variable
        if grouping_var == "age_group":
            group = X_test["age_group"].apply(classify_age)
        elif grouping_var == "age_on_arrival":
            group = X_test["age_on_arrival"].apply(classify_age)
        else:
            group = X_test[grouping_var]

        fig = _plot_madcap_by_group_single(
            predict_proba,
            y_test,
            group,
            prediction_time,
            grouping_var_name,
            media_file_path,
            file_name=file_name,
            plot_difference=plot_difference,
            return_figure=True,
        )
        if return_figure:
            figures.append(fig)

    if return_figure:
        return figures
    else:
        return None
