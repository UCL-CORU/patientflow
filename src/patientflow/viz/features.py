"""Visualisation module for plotting feature importances from trained models.

This module provides functionality to visualize feature importances from trained
classifiers, allowing for comparison across different prediction time points.

Functions
---------
plot_features : function
    Plot feature importance for multiple models
"""

import numpy as np
import matplotlib.pyplot as plt
from patientflow.model_artifacts import TrainedClassifier
from sklearn.pipeline import Pipeline
from typing import Optional
from pathlib import Path


def plot_features(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    media_file_path: Optional[Path] = None,
    file_name=None,
    top_n: int = 20,
    suptitle: Optional[str] = None,
    return_figure: bool = False,
) -> Optional[plt.Figure]:
    """Plot feature importance for multiple models.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of TrainedClassifier objects or dictionary with TrainedClassifier values.
    media_file_path : Path, optional
        Path where the plot should be saved. If None, the plot is only displayed.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "feature_importance_plots.png".
    top_n : int, default=20
        Number of top features to display.
    suptitle : str, optional
        Super title for the entire figure.
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it.

    Returns
    -------
    plt.Figure or None
        The matplotlib figure if return_figure is True, otherwise None.

    Notes
    -----
    The function sorts models by prediction time and creates a horizontal bar plot
    for each model showing the top N most important features. Feature names are
    truncated to 25 characters for better display.
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
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 6, 12))

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    for i, trained_model in enumerate(trained_models_sorted):
        # Always use regular pipeline
        pipeline: Pipeline = trained_model.pipeline
        prediction_time = trained_model.training_results.prediction_time

        # Get feature names from the pipeline
        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:25] for col in transformed_cols]

        # Get feature importances
        feature_importances = pipeline.named_steps["classifier"].feature_importances_
        indices = np.argsort(feature_importances)[
            -top_n:
        ]  # Get indices of the top N features

        # Plot for this prediction time
        ax = axs[i]
        hour, minutes = prediction_time
        ax.barh(range(len(indices)), feature_importances[indices], align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(np.array(truncated_cols)[indices])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title(f"Feature Importances for {hour}:{minutes:02}")

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle is not None:
        plt.suptitle(suptitle, y=1.05, fontsize=16)

    if media_file_path:
        # Save and display plot
        if file_name:
            feature_plot_path = media_file_path / file_name
        else:
            feature_plot_path = media_file_path / "feature_importance_plots.png"
        plt.savefig(feature_plot_path, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
        return None
