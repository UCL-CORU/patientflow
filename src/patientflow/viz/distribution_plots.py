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
):
    """
    Plot prediction distributions for multiple models.

    Args:
        trained_models: List of TrainedClassifier objects or dict with TrainedClassifier values
        test_visits: DataFrame containing test visit data
        exclude_from_training_data: Columns to exclude from the test data
        bins: Number of bins for the histogram (default: 30)
        media_file_path: Path to save the plot (default: None)
        suptitle: Optional super title for the plot
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
        ax.set_xlabel("Predicted Probability", fontsize=12)
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
):
    sns.set_theme(style="whitegrid")

    if ordinal_order is not None:
        df[col_name] = pd.Categorical(
            df[col_name], categories=ordinal_order, ordered=True
        )

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

    if rotate_x_labels:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)

    if is_discrete:
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlim(min_val - 0.5, max_val + 0.5)

    plt.subplots_adjust(top=0.85)
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
