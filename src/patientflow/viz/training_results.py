import matplotlib.pyplot as plt
from typing import List

from patientflow.model_artifacts import HyperParameterTrial


def plot_trial_results(
    trials_list: List[HyperParameterTrial],
    media_file_path=None,
    return_figure=False,
):
    """
    Function to plot AUROC and log loss for hyperparameter trials as dots.

    Parameters
    ----------
    trials_list : List[HyperParameterTrial]
        List of hyperparameter trials to visualize
    media_file_path : Path, optional
        Path to save the plot
    return_figure : bool, optional
        If True, returns the figure instead of displaying it
    """
    # Extract metrics from trials
    auroc_values = [trial.cv_results.get("valid_auc", 0) for trial in trials_list]
    logloss_values = [trial.cv_results.get("valid_logloss", 0) for trial in trials_list]

    # Create trial indices
    trial_indices = list(range(len(trials_list)))

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot AUROC as dots
    ax1.scatter(trial_indices, auroc_values, color="blue", s=50, alpha=0.7)
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("AUROC")
    ax1.set_title("Area Under ROC Curve")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_ylim(bottom=0, top=max(auroc_values) * 1.1)

    # Plot Log Loss as dots
    ax2.scatter(trial_indices, logloss_values, color="red", s=50, alpha=0.7)
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Log Loss")
    ax2.set_title("Log Loss")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_ylim(bottom=0, top=max(logloss_values) * 1.1)

    # Highlight best AUROC
    best_auroc_idx = auroc_values.index(max(auroc_values))
    ax1.scatter(
        [best_auroc_idx],
        [auroc_values[best_auroc_idx]],
        color="green",
        s=150,
        edgecolor="black",
        zorder=5,
    )

    # Highlight best Log Loss
    best_logloss_idx = logloss_values.index(min(logloss_values))
    ax2.scatter(
        [best_logloss_idx],
        [logloss_values[best_logloss_idx]],
        color="darkred",
        s=150,
        edgecolor="black",
        zorder=5,
    )

    # Add annotation with best parameters
    best_auroc_trial = trials_list[best_auroc_idx]
    best_logloss_trial = trials_list[best_logloss_idx]

    # Format parameter text for best AUROC
    param_text_auroc = "\n".join(
        [f"{k}: {v}" for k, v in best_auroc_trial.parameters.items()]
    )
    ax1.text(
        0.05,
        0.05,
        f"Best AUROC: {auroc_values[best_auroc_idx]:.4f}\n\nParameters:\n{param_text_auroc}",
        transform=ax1.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=9,
    )

    # Format parameter text for best Log Loss
    param_text_logloss = "\n".join(
        [f"{k}: {v}" for k, v in best_logloss_trial.parameters.items()]
    )
    ax2.text(
        0.05,
        0.05,
        f"Best Log Loss: {logloss_values[best_logloss_idx]:.4f}\n\nParameters:\n{param_text_logloss}",
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=9,
    )

    # Add overall title
    fig.suptitle("Hyperparameter Trial Results", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    if media_file_path:
        plt.savefig(media_file_path / "trial_results.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
