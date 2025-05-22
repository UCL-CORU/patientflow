from matplotlib import pyplot as plt
from patientflow.prepare import prepare_patient_snapshots
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.model_artifacts import TrainedClassifier
import shap
import scipy.sparse
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Optional
from pathlib import Path


def plot_shap(
    trained_models: list[TrainedClassifier] | dict[str, TrainedClassifier],
    test_visits,
    exclude_from_training_data,
    media_file_path: Optional[Path] = None,
    return_figure=False,
):
    """
    Generate SHAP plots for multiple trained models.

    Parameters
    ----------
    trained_models : list[TrainedClassifier] | dict[str, TrainedClassifier]
        List of trained classifier objects or dictionary with TrainedClassifier values
    media_file_path : Path
        Directory path where the generated plots will be saved
    test_visits : pd.DataFrame
        DataFrame containing the test visit data
    exclude_from_training_data : list[str]
        List of columns to exclude from training data
    return_figure : bool
        If True, returns the figure instead of displaying it
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

    for trained_model in trained_models_sorted:
        fig, ax = plt.subplots(figsize=(8, 12))

        # use non-calibrated pipeline
        pipeline: Pipeline = trained_model.pipeline
        prediction_time = trained_model.training_results.prediction_time

        # Get test data for this prediction time
        X_test, _ = prepare_patient_snapshots(
            df=test_visits,
            prediction_time=prediction_time,
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
        )

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
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=truncated_cols,
            show=False,
        )

        hour, minutes = prediction_time
        ax.set_title(f"SHAP Values for Time of Day: {hour}:{minutes:02}")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()

        if media_file_path:
            # Save plot
            shap_plot_path = str(
                media_file_path / f"shap_plot_{hour:02}{minutes:02}.png"
            )
            plt.savefig(shap_plot_path)

        if return_figure:
            return fig
        else:
            plt.show()
            plt.close(fig)
