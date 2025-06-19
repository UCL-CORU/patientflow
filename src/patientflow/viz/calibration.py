"""Calibration plot visualization module.

This module creates calibration plots for trained models,
showing how well the predicted probabilities align with actual outcomes.

Functions
---------
plot_calibration : function
    Plot calibration curves for multiple models
"""

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import prepare_patient_snapshots
from patientflow.model_artifacts import TrainedClassifier
from typing import Optional
from pathlib import Path

# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#aec7e8"


def plot_calibration(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits,
    exclude_from_training_data,
    strategy="uniform",
    media_file_path: Optional[Path] = None,
    file_name=None,
    suptitle=None,
    return_figure=False,
    label_col: str = "is_admitted",
):
    """Plot calibration curves for multiple models.

    A calibration plot shows how well the predicted probabilities from a model
    align with the actual outcomes. The plot compares the mean predicted probability
    with the fraction of positive outcomes for different probability bins.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of TrainedClassifier objects or dictionary with TrainedClassifier values.
    test_visits : pandas.DataFrame
        DataFrame containing test visit data.
    exclude_from_training_data : list
        Columns to exclude from the test data.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy for calibration curve binning.
        - 'uniform': Bins are of equal width
        - 'quantile': Bins have equal number of samples
    media_file_path : Path, optional
        Path where the plot should be saved.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "calibration_plot.png".
    suptitle : str, optional
        Optional super title for the entire figure.
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it.
    label_col : str, default='is_admitted'
        Name of the column containing the target labels.

    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figure is True, returns the figure object. Otherwise, displays
        the plot and returns None.

    Notes
    -----
    The function creates a subplot for each trained model, sorted by prediction time.
    Each subplot shows the calibration curve and a reference line for perfect calibration.
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

        prob_true, prob_pred = calibration_curve(
            y_test, pipeline.predict_proba(X_test)[:, 1], n_bins=10, strategy=strategy
        )

        ax = axs[i]
        hour, minutes = prediction_time

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1,
            label="Predictions",
            color=primary_color,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfectly calibrated",
            color=secondary_color,
        )
        ax.set_title(f"Calibration Plot for {hour}:{minutes:02}", fontsize=14)
        ax.set_xlabel("Mean Estimated Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.legend()

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)

    if media_file_path:
        if file_name:
            calib_plot_path = media_file_path / file_name
        else:
            calib_plot_path = media_file_path / "calibration_plot.png"
        plt.savefig(calib_plot_path)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
