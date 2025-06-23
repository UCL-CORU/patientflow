"""
Charts for hyperparameter optimisation trials.

This module provides tools to visualise the performance metrics of multiple hyperparameter tuning trials,
highlighting the best trials for each metric.

Functions
---------
plot_trial_results : function
    Plot selected performance metrics for a list of hyperparameter trials.
"""

import matplotlib.pyplot as plt
from typing import List, Optional

from patientflow.model_artifacts import HyperParameterTrial


def plot_trial_results(
    trials_list: List[HyperParameterTrial],
    metrics: Optional[List[str]] = None,
    media_file_path=None,
    file_name=None,
    return_figure=False,
):
    """
    Plot selected performance metrics from hyperparameter trials as scatter plots.

    This function visualizes the performance metrics of a series of hyperparameter trials.
    It creates scatter plots for each selected metric, with the best-performing trial
    highlighted and annotated with its hyperparameters.

    Optionally, the plot can be saved to disk or returned as a figure object.

    Parameters
    ----------
    trials_list : List[HyperParameterTrial]
        A list of `HyperParameterTrial` instances containing validation set results
        (not cross-validation fold results) and hyperparameter settings. Each trial's
        `cv_results` dictionary contains metrics such as 'valid_auc' and 'valid_logloss',
        which are computed on a held-out validation set for each hyperparameter configuration.
    metrics : List[str], optional
        List of metric names to plot. If None, defaults to ["valid_auc", "valid_logloss"].
        Each metric should be a key in the trial's cv_results dictionary.
    media_file_path : pathlib.Path or None, optional
        Directory path where the generated plot image will be saved as "trial_results.png".
        If None, the plot is not saved.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "trial_results.png".
    return_figure : bool, optional
        If True, the matplotlib figure is returned instead of being displayed directly.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure or None
        The matplotlib figure object if `return_figure` is True; otherwise, None.

    Notes
    -----
    - Assumes that each `HyperParameterTrial` in `trials_list` has a `cv_results` dictionary
      containing the requested metrics, which are computed on the validation set.
    - Parameters from the best-performing trials are shown in the plots.
    """
    # Set default metrics if none provided
    if metrics is None:
        metrics = ["valid_auc", "valid_logloss"]

    # Extract metrics from trials
    metric_values = {
        metric: [trial.cv_results.get(metric, 0) for trial in trials_list]
        for metric in metrics
    }

    # Create trial indices
    trial_indices = list(range(len(trials_list)))

    # Create figure with subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, (metric, values) in enumerate(metric_values.items()):
        ax = axes[idx]

        # Plot metric as dots
        ax.scatter(trial_indices, values, s=50, alpha=0.7)
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(metric.replace("valid_", "").upper())
        ax.set_title(metric.replace("valid_", "").replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.7)

        # Set x-axis to display integers
        ax.set_xticks(trial_indices)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: str(int(x))))

        # Set y-axis limits
        if "loss" in metric.lower():
            best_idx = values.index(min(values))
            ax.set_ylim(bottom=0, top=max(values) * 1.1)
        else:
            best_idx = values.index(max(values))
            ax.set_ylim(bottom=0, top=max(values) * 1.1)

        # Highlight best value
        highlight_color = "green" if "loss" not in metric.lower() else "darkred"
        ax.scatter(
            [best_idx],
            [values[best_idx]],
            color=highlight_color,
            s=150,
            edgecolor="black",
            zorder=5,
        )

        # Add annotation with best parameters
        best_trial = trials_list[best_idx]
        param_text = "\n".join([f"{k}: {v}" for k, v in best_trial.parameters.items()])
        best_value = values[best_idx]
        ax.text(
            0.05,
            0.05,
            f"Best {metric.replace('valid_', '').upper()}: {best_value:.4f}\n\nParameters:\n{param_text}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=9,
        )

    # Add overall title
    fig.suptitle("Hyperparameter Trial Results", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    if media_file_path:
        if file_name:
            plt.savefig(media_file_path / file_name, dpi=300)
        else:
            plt.savefig(media_file_path / "trial_results.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
