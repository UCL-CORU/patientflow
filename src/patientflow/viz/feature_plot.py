import numpy as np
import matplotlib.pyplot as plt
from patientflow.model_artifacts import TrainedClassifier
from sklearn.pipeline import Pipeline
from typing import Optional
from pathlib import Path


def plot_features(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    media_file_path: Optional[Path] = None,
    top_n: int = 20,
    suptitle: Optional[str] = None,
    return_figure: bool = False,
):
    """
    Plot feature importance for multiple models.

    Args:
        trained_models: List of TrainedClassifier objects or dict with TrainedClassifier values
        media_file_path: Path where the plot should be saved
        top_n: Number of top features to display (default: 20)
        suptitle: Optional super title for the entire figure (default: None)
        return_figure: If True, returns the figure instead of displaying it
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
        feature_plot_path = media_file_path / "feature_importance_plots.png"
        plt.savefig(feature_plot_path, bbox_inches="tight")

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
