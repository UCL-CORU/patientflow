"""Visualization module for plotting estimated probabilities from trained models.

This module provides functions for creating distribution plots of estimated
probabilities from trained classification models.

Functions
---------
plot_estimated_probabilities : function
    Plot estimated probability distributions for multiple models
"""

import matplotlib.pyplot as plt
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import prepare_patient_snapshots
from patientflow.model_artifacts import TrainedClassifier
from typing import Optional
from pathlib import Path


# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"


def plot_estimated_probabilities(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits,
    exclude_from_training_data,
    bins=30,
    media_file_path: Optional[Path] = None,
    file_name=None,
    suptitle: Optional[str] = None,
    return_figure=False,
    label_col: str = "is_admitted",
):
    """Plot estimated probability distributions for multiple models.

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
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "estimated_probabilities.png".
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

        ax.set_title(
            f"Distribution of Estimated Probabilities at {hour}:{minutes:02}",
            fontsize=14,
        )
        ax.set_xlabel("Estimated Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend()

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle is not None:
        plt.suptitle(suptitle, y=1.05, fontsize=16)

    if media_file_path:
        if file_name:
            filename = file_name
        else:
            filename = "estimated_probabilities.png"
        plt.savefig(media_file_path / filename, dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
