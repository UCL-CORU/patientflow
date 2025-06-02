"""Visualization tools for patient flow analysis using survival curves.

This module provides functions to create and analyze survival curves for
time-to-event analysis.

Functions
---------
plot_admission_time_survival_curve : function
    Create a survival curve for ward admission times

Notes
-----
* The survival curves show the proportion of patients who have not yet
  experienced an event (e.g., admission to ward) over time
* Time is measured in hours from the initial event (e.g., arrival)
* A 4-hour target line is included by default to show performance
  against common healthcare targets
* The curves are created without external survival analysis packages
  for simplicity and transparency

Examples
--------
>>> import pandas as pd
>>> from patientflow.viz.survival_curves import plot_admission_time_survival_curve
>>> df = pd.DataFrame({
...     'arrival_datetime': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00']),
...     'admitted_to_ward_datetime': pd.to_datetime(['2023-01-01 12:00', '2023-01-01 14:00'])
... })
>>> plot_admission_time_survival_curve(df, title='Admission Times')
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_admission_time_survival_curve(
    df,
    title,
    media_file_path=None,
    return_figure=False,
):
    """Create a survival curve for patient ward admission times.

    This function creates a survival curve showing the proportion of patients
    who have not yet been admitted to a ward over time, without using the
    lifelines package.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing patient visit data with columns:
        * arrival_datetime: when the patient arrived
        * admitted_to_ward_datetime: when the patient was admitted to a ward
    title : str
        Title for the plot
    media_file_path : pathlib.Path, optional
        Path to save the plot. If None, the plot is not saved.
    return_figure : bool, default=False
        If True, returns the figure instead of displaying it

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if return_figure is True, otherwise None

    Notes
    -----
    The survival curve shows the proportion of patients who have not yet been
    admitted to a ward at each time point. A vertical line is drawn at 4 hours
    to indicate the target admission time, with the corresponding proportion
    of patients admitted within this timeframe.
    """
    # Calculate the wait time in hours
    df["wait_time_hours"] = (
        df["admitted_to_ward_datetime"] - df["arrival_datetime"]
    ).dt.total_seconds() / 3600

    # Drop any rows with missing wait times
    df_clean = df.dropna(subset=["wait_time_hours"]).copy()

    # Sort the data by wait time
    df_clean = df_clean.sort_values("wait_time_hours")

    # Calculate the number of patients
    n_patients = len(df_clean)

    # Calculate the survival function manually
    # For each time point, calculate proportion of patients who are still waiting
    unique_times = np.sort(df_clean["wait_time_hours"].unique())
    survival_prob = []

    for t in unique_times:
        # Number of patients admitted after this time point
        n_admitted_after = sum(df_clean["wait_time_hours"] > t)
        # Proportion of patients still waiting
        survival_prob.append(n_admitted_after / n_patients)

    # Add zero hours wait time (everyone is waiting at time 0)
    unique_times = np.insert(unique_times, 0, 0)
    survival_prob = np.insert(survival_prob, 0, 1.0)

    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.step(
        unique_times, survival_prob, where="post", label="Admission Survival Curve"
    )

    # Configure the plot
    if title:
        plt.title(title)
    else:
        plt.title("Time to Ward Admission Survival Curve")
    plt.xlabel("Elapsed time from arrival")
    plt.ylabel("Proportion not yet admitted")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Make axes meet at the origin
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Move spines to the origin
    ax = plt.gca()
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))

    # Hide the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add lines at 4 hours target
    target_hours = 4

    # Find the survival probability at 4 hours
    closest_time_idx = np.abs(unique_times - target_hours).argmin()
    if closest_time_idx < len(survival_prob):
        survival_at_target = survival_prob[closest_time_idx]
        admitted_at_target = 1 - survival_at_target

        # Draw a vertical line from x-axis to the curve at 4 hours
        plt.plot(
            [target_hours, target_hours],
            [0, survival_at_target],
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Draw a horizontal line from the curve to the y-axis at the survival probability level
        plt.plot(
            [0, target_hours],
            [survival_at_target, survival_at_target],
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Add text annotation to the plot
        plt.text(
            target_hours + 0.5,
            survival_at_target,
            f"{admitted_at_target:.1%} admitted\nwithin 4 hours",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        print(
            f"Proportion of patients admitted within {target_hours} hours: {admitted_at_target:.2%}"
        )

    plt.tight_layout()

    if media_file_path:
        plt.savefig(media_file_path / "survival_curve.png", dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close()
