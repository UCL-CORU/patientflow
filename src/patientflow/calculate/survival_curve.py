import numpy as np
import pandas as pd


def calculate_survival_curve(df, start_time_col, end_time_col):
    """Calculate survival curve data from patient visit data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing patient visit data
    start_time_col : str
        Name of the column containing the start time (e.g., arrival_datetime)
    end_time_col : str
        Name of the column containing the end time (e.g., departure_datetime)

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - time_hours: Time points in hours
        - survival_probability: Survival probabilities at each time point
        - event_probability: Event probabilities (1 - survival_probability)
    """
    # Calculate the wait time in hours
    df = df.copy()
    df["wait_time_hours"] = (
        df[end_time_col] - df[start_time_col]
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
        # Number of patients who experienced the event after this time point
        n_event_after = sum(df_clean["wait_time_hours"] > t)
        # Proportion of patients still waiting
        survival_prob.append(n_event_after / n_patients)

    # Add zero hours wait time (everyone is waiting at time 0)
    unique_times = np.insert(unique_times, 0, 0)
    survival_prob = np.insert(survival_prob, 0, 1.0)

    # Return structured DataFrame
    return pd.DataFrame(
        {
            "time_hours": unique_times,
            "survival_probability": survival_prob,
            "event_probability": 1 - survival_prob,
        }
    )
