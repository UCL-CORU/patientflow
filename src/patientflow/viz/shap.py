"""SHAP (SHapley Additive exPlanations) visualization module.

This module provides functionality for generating SHAP plots. These are useful for
visualizing feature importance and their impact on model decisions.

Functions
---------
plot_shap : function
    Generate SHAP plots for multiple trained models.
"""

from matplotlib import pyplot as plt
from patientflow.prepare import prepare_patient_snapshots
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.model_artifacts import TrainedClassifier
import scipy.sparse
import numpy as np
from sklearn.pipeline import Pipeline
from typing import List, Optional
from pathlib import Path

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when shap is absent
    shap = None  # type: ignore
    SHAP_AVAILABLE = False


def plot_shap(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits,
    media_file_path: Optional[Path] = None,
    file_name: Optional[str] = None,
    return_figure=False,
    label_col: str = "is_admitted",
    *,
    exclude_from_training_data: Optional[List[str]] = None,
    show: bool = False,
):
    """Generate SHAP plots for multiple trained models.

    This function creates SHAP (SHapley Additive exPlanations) summary plots for each
    trained model, showing the impact of features on model predictions. The plots can
    be saved to a specified media file path or displayed directly.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] or dict[str, TrainedClassifier]
        List of trained classifier objects or dictionary with TrainedClassifier values.
    test_visits : pandas.DataFrame
        DataFrame containing the test visit data.
    media_file_path : Path, optional
        Directory path where the generated plots will be saved. If None, plots are
        only displayed.
    file_name : str, optional
        Filename when saving a single model (e.g. ``"shap_summary.png"``). When
        several ``trained_models`` are passed, each plot is saved as
        ``{stem}_{HHMM}{suffix}`` so clocks do not overwrite the same path. If
        omitted, uses ``shap_plot_{HHMM}.png`` per model.
    return_figure : bool, default=False
        If True, returns the figure instead of displaying.
    label_col : str, default="is_admitted"
        Name of the column containing the target labels.
    exclude_from_training_data : List[str], optional, deprecated
        This parameter is deprecated and ignored. Column selection is now handled
        automatically by the pipeline's FeatureColumnTransformer.
    show : bool, default=False
        If True, call ``matplotlib.pyplot.show()`` when not using ``return_figure``.
        Default False avoids blocking or inline display in notebooks when saving only.

    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figure is True, returns the generated figure. Otherwise, returns None.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    # Convert dict to list if needed
    if isinstance(trained_models, dict):
        trained_models = list(trained_models.values())

    # Sort trained_models by prediction time
    trained_models_sorted = sorted(
        trained_models,
        key=lambda x: x.training_results.prediction_time[0] * 60
        + x.training_results.prediction_time[1],
    )

    for trained_model in trained_models_sorted:
        fig, ax = plt.subplots(figsize=(8, 12))

        # use non-calibrated pipeline
        pipeline: Pipeline = trained_model.pipeline
        prediction_time = trained_model.training_results.prediction_time

        # Get test data for this prediction time
        X_test, _ = prepare_patient_snapshots(
            df=test_visits,
            prediction_time=prediction_time,
            single_snapshot_per_visit=False,
            label_col=label_col,
        )

        # Pipeline handles column selection automatically via FeatureColumnTransformer
        # For legacy models, add_missing_columns is called internally if needed
        if "feature_columns" not in pipeline.named_steps:
            X_test = add_missing_columns(pipeline, X_test)
        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:45] for col in transformed_cols]

        # Transform features
        X_test = pipeline.named_steps["feature_transformer"].transform(X_test)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])

        # Convert sparse matrix to dense if necessary
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()

        shap_values = explainer.shap_values(X_test)

        # Print prediction distribution
        predictions = pipeline.named_steps["classifier"].predict(X_test)
        print(
            "Predicted classification (not admitted, admitted): ",
            np.bincount(predictions),
        )

        # Print mean SHAP values for each class
        if isinstance(shap_values, list):
            print("SHAP values shape:", [arr.shape for arr in shap_values])
            print("Mean SHAP values (class 0):", np.abs(shap_values[0]).mean(0))
            print("Mean SHAP values (class 1):", np.abs(shap_values[1]).mean(0))

        # Create SHAP summary plot
        rng = np.random.default_rng()
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=truncated_cols,
            show=False,
            rng=rng,
        )

        hour, minutes = prediction_time
        ax.set_title(f"SHAP Values for Time of Day: {hour}:{minutes:02}")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()

        if media_file_path:
            # Save plot — disambiguate file_name when looping over several models
            if file_name:
                fn = Path(file_name)
                if len(trained_models_sorted) > 1:
                    clock = f"{hour:02d}{minutes:02d}"
                    unique_name = f"{fn.stem}_{clock}{fn.suffix}"
                else:
                    unique_name = fn.name
                shap_plot_path = str(media_file_path / unique_name)
            else:
                shap_plot_path = str(
                    media_file_path / f"shap_plot_{hour:02}{minutes:02}.png"
                )
            plt.savefig(shap_plot_path)

        if return_figure:
            return fig
        else:
            if show:
                plt.show()
            plt.close(fig)
