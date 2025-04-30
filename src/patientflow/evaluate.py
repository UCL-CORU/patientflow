"""
Patient Flow Evaluation Module

This module provides functions for evaluating and comparing different prediction models
for patient admissions in a healthcare setting. It includes utilities for calculating
metrics such as Mean Absolute Error (MAE) and Mean Percentage Error (MPE), as well as
functions for predicting admissions based on historical data and combining different
prediction models.

Key Features:
- Evaluation of probability distribution-based prediction models
- Calculation of observed admissions based on ED targets
- Prediction using historical data from previous weeks
- Combination and evaluation of multiple prediction models

Main Functions:
- calc_mae_mpe: Calculate MAE and MPE for probability distribution predictions
- calculate_weighted_observed: Calculate actual admissions assuming ED targets are met
- predict_using_previous_weeks: Predict admissions using average from previous weeks
- evaluate_six_week_average: Evaluate the six-week average prediction model
- evaluate_combined_model: Evaluate a combined prediction model

"""

from typing import Dict, List, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal
from patientflow.calculate.admission_in_prediction_window import (
    get_y_from_aspirational_curve,
)
from patientflow.load import get_model_key


def calculate_results(
    expected_values: List[Union[int, float]], observed_values: List[float]
) -> Dict[str, Union[List[Union[int, float]], float]]:
    """
    Calculate evaluation metrics based on expected and observed values.

    Args:
        expected_values (List[Union[int, float]]): List of expected values.
        observed_values (List[float]): List of observed values.

    Returns:
        Dict[str, Union[List[Union[int, float]], float]]: Dictionary containing expected values, observed values, MAE, and MPE.
    """
    expected_array: np.ndarray = np.array(expected_values)
    observed_array: np.ndarray = np.array(observed_values)

    if len(expected_array) == 0 or len(observed_array) == 0:
        return {
            "expected": expected_values,
            "observed": observed_values,
            "mae": 0.0,
            "mpe": 0.0,
        }

    absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
    mae: float = float(np.mean(absolute_errors)) if len(absolute_errors) > 0 else 0.0

    non_zero_mask: np.ndarray = observed_array != 0
    filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
    filtered_observed_array: np.ndarray = observed_array[non_zero_mask]

    mpe: float = 0.0
    if len(filtered_absolute_errors) > 0 and len(filtered_observed_array) > 0:
        percentage_errors: np.ndarray = (
            filtered_absolute_errors / filtered_observed_array * 100
        )
        mpe = float(np.mean(percentage_errors))

    return {
        "expected": expected_values,
        "observed": observed_values,
        "mae": mae,
        "mpe": mpe,
    }


def calc_mae_mpe(
    prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]],
    use_most_probable: bool = False,
) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Calculate MAE and MPE for all prediction times in the given probability distribution dictionary.
    Results are sorted by prediction time.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        use_most_probable (bool, optional): Whether to use the most probable value or mathematical expectation of the distribution. Defaults to False.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Dictionary of results sorted by prediction time.
    """
    # Create temporary results dictionary
    unsorted_results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

    # Process results as before
    for _prediction_time in prob_dist_dict_all.keys():
        expected_values: List[Union[int, float]] = []
        observed_values: List[float] = []

        for dt in prob_dist_dict_all[_prediction_time].keys():
            preds: Dict[str, Any] = prob_dist_dict_all[_prediction_time][dt]

            expected_value: Union[int, float] = (
                int(preds["agg_predicted"].idxmax().values[0])
                if use_most_probable
                else float(
                    np.dot(
                        preds["agg_predicted"].index,
                        preds["agg_predicted"].values.flatten(),
                    )
                )
            )

            observed_value: float = float(preds["agg_observed"])

            expected_values.append(expected_value)
            observed_values.append(observed_value)

        unsorted_results[_prediction_time] = calculate_results(
            expected_values, observed_values
        )

    # Sort results by prediction time
    def get_time_value(key: str) -> int:
        # Extract time from key (e.g., 'admissions_1530' -> 1530)
        time_str = key.split("_")[1]
        return int(time_str)

    # Create sorted dictionary
    sorted_results = dict(
        sorted(unsorted_results.items(), key=lambda x: get_time_value(x[0]))
    )

    return sorted_results


def calculate_admission_probs_relative_to_prediction(
    df, prediction_datetime, prediction_window, x1, y1, x2, y2, is_before=True
):
    """
    Calculate admission probabilities for arrivals relative to a prediction time window

    Parameters:
    df: DataFrame containing arrival_datetime column
    prediction_datetime: datetime for prediction window start
    prediction_window: window length in minutes
    x1, y1, x2, y2: parameters for aspirational curve
    is_before: boolean indicating if arrivals are before prediction time

    Returns:
    DataFrame with added probability columns
    """
    result = df.copy()

    if is_before:
        result["hours_before_pred_window"] = result["arrival_datetime"].apply(
            lambda x: (prediction_datetime - x).seconds / 3600
        )
        result["prob_admission_before_pred_window"] = result[
            "hours_before_pred_window"
        ].apply(lambda x: get_y_from_aspirational_curve(x, x1, y1, x2, y2))
        result["prob_admission_in_pred_window"] = result[
            "hours_before_pred_window"
        ].apply(
            lambda x: get_y_from_aspirational_curve(
                x + prediction_window / 60, x1, y1, x2, y2
            )
            - get_y_from_aspirational_curve(x, x1, y1, x2, y2)
        )
    else:
        result["hours_after_pred_window"] = result["arrival_datetime"].apply(
            lambda x: (x - prediction_datetime).seconds / 3600
        )
        result["prob_admission_in_pred_window"] = result[
            "hours_after_pred_window"
        ].apply(
            lambda x: get_y_from_aspirational_curve(
                (prediction_window / 60) - x, x1, y1, x2, y2
            )
        )

    return result


def get_arrivals_with_admission_probs(
    df,
    prediction_datetime,
    prediction_window,
    prediction_time,
    x1,
    y1,
    x2,
    y2,
    date_range=None,
    target_date=None,
    target_weekday=None,
):
    """
    Get arrivals before and after prediction time with their admission probabilities

    Parameters:
    df: DataFrame with arrival_datetime column
    prediction_datetime: datetime for prediction window start
    prediction_window: window length in minutes
    prediction_time: tuple of (hour, minute)
    x1, y1, x2, y2: parameters for aspirational curve
    date_range: optional tuple of (start_date, end_date)
    target_date: optional specific date to analyze
    target_weekday: optional specific weekday to filter for

    Returns:
    tuple of (arrived_before, arrived_after) DataFrames for specified time period
    """
    hour, minute = prediction_time

    # Create base time masks
    after_mask = create_time_mask(df, hour, minute)
    before_mask = ~after_mask

    # Add date and weekday conditions if specified
    if date_range:
        start_date, end_date = date_range
        date_mask = (df["arrival_datetime"].dt.date >= start_date) & (
            df["arrival_datetime"].dt.date < end_date
        )
        if target_weekday is not None:
            date_mask &= df["arrival_datetime"].dt.weekday == target_weekday

        after_mask &= date_mask
        before_mask &= date_mask

    if target_date:
        target_mask = df["arrival_datetime"].dt.date == target_date
        after_mask &= target_mask
        before_mask &= target_mask

    # Calculate probabilities for filtered groups
    arrived_before = calculate_admission_probs_relative_to_prediction(
        df[before_mask],
        prediction_datetime,
        prediction_window,
        x1,
        y1,
        x2,
        y2,
        is_before=True,
    )

    arrived_after = calculate_admission_probs_relative_to_prediction(
        df[after_mask],
        prediction_datetime,
        prediction_window,
        x1,
        y1,
        x2,
        y2,
        is_before=False,
    )

    return arrived_before, arrived_after


def calculate_weighted_observed(
    df, dt, prediction_window, x1, y1, x2, y2, prediction_time
):
    """
    Calculate weighted observed admissions for a specific date and prediction window

    Parameters:
    df: DataFrame with arrival_datetime column
    dt: target date
    prediction_window: window length in minutes
    x1, y1, x2, y2: parameters for aspirational curve
    prediction_time: tuple of (hour, minute)
    """
    # Create prediction datetime
    prediction_datetime = pd.to_datetime(dt).replace(
        hour=prediction_time[0], minute=prediction_time[1]
    )

    # Filter for target date and get arrivals with probabilities
    filtered_df = df[df["arrival_datetime"].dt.date == dt]
    arrived_before, arrived_after = get_arrivals_with_admission_probs(
        filtered_df,
        prediction_datetime,
        prediction_window,
        prediction_time,
        x1,
        y1,
        x2,
        y2,
        target_date=dt,
    )

    # Calculate weighted sum
    weighted_observed = (
        arrived_before["prob_admission_in_pred_window"].sum()
        + arrived_after["prob_admission_in_pred_window"].sum()
    )

    return weighted_observed


def create_time_mask(df, hour, minute):
    """Create a mask for times before/after a specific hour:minute"""
    return (df["arrival_datetime"].dt.hour > hour) | (
        (df["arrival_datetime"].dt.hour == hour)
        & (df["arrival_datetime"].dt.minute > minute)
    )


def predict_using_previous_weeks(
    df: pd.DataFrame,
    dt: datetime,
    prediction_window: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    prediction_time: Tuple[int, int],
    num_weeks: int,
    weighted=True,
) -> float:
    """
    Calculate predicted admissions remaining until midnight.
    Args:
        df (pd.DataFrame): DataFrame containing patient data.
        dt (datetime): Date for prediction.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.
        weighted(bool): Whether to weight the numbers according to aspirational ED targets
    Returns:
        float: Predicted number of admissions remaining until midnight.
    """
    prediction_datetime = pd.to_datetime(dt).replace(
        hour=prediction_time[0], minute=prediction_time[1]
    )
    target_day_of_week = dt.weekday()

    end_date = dt - timedelta(days=1)
    start_date = end_date - timedelta(weeks=num_weeks)

    if weighted:
        # Create mask for historical data
        historical_mask = (
            (df["arrival_datetime"].dt.date >= start_date)
            & (df["arrival_datetime"].dt.date <= end_date)
            & (df["arrival_datetime"].dt.weekday == target_day_of_week)
        )

        # Create explicit copy of filtered data
        historical_data = df[historical_mask].copy()

        # Calculate minutes until midnight
        midnight_times = (
            historical_data["arrival_datetime"].dt.normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=1)
        )
        historical_data.loc[:, "minutes_to_midnight"] = (
            midnight_times - historical_data["arrival_datetime"]
        ).dt.total_seconds() / 60

        # Calculate admission probabilities
        historical_data.loc[:, "admission_probability"] = historical_data[
            "minutes_to_midnight"
        ].apply(lambda x: get_y_from_aspirational_curve(x / 60, x1, y1, x2, y2))

        # Group by date and calculate average
        historical_daily_sums = historical_data.groupby(
            historical_data["arrival_datetime"].dt.date
        )["admission_probability"].sum()
        historical_average = historical_daily_sums.mean()

        # Create mask for today's data
        today_mask = (df["arrival_datetime"].dt.date == dt) & (
            df["arrival_datetime"] < prediction_datetime
        )

        # Create explicit copy of today's filtered data
        today_data = df[today_mask].copy()

        # Calculate minutes until midnight for today's data
        midnight_today = (
            pd.to_datetime(dt).normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=1)
        )
        today_data.loc[:, "minutes_to_midnight"] = (
            midnight_today - today_data["arrival_datetime"]
        ).dt.total_seconds() / 60

        # Calculate admission probabilities for today
        today_data.loc[:, "admission_probability"] = today_data[
            "minutes_to_midnight"
        ].apply(lambda x: get_y_from_aspirational_curve(x / 60, x1, y1, x2, y2))

        today_sum = today_data["admission_probability"].sum()

        still_to_admit = max(historical_average - today_sum, 0)

    else:
        # Original unweighted logic with explicit copies
        historical_mask = (
            (df["arrival_datetime"].dt.date >= start_date)
            & (df["arrival_datetime"].dt.date < end_date)
            & (df["arrival_datetime"].dt.weekday == target_day_of_week)
        )
        historical_df = df[historical_mask].copy()
        average_count = len(historical_df) / num_weeks

        target_mask = (df["arrival_datetime"].dt.date == dt) & (
            df["arrival_datetime"] < prediction_datetime
        )
        target_date_count = len(df[target_mask])

        still_to_admit = max(average_count - target_date_count, 0)

    return still_to_admit


def evaluate_six_week_average(
    prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]],
    df: pd.DataFrame,
    prediction_window: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    prediction_time: Tuple[int, int],
    num_weeks: int,
    model_name: str,
) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Evaluate the six-week average prediction model.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        df (pd.DataFrame): DataFrame containing patient data.
        prediction_window (int): Prediction window in minutes.
        x1 (float), y1 (float), x2 (float), y2 (float): Parameters for aspirational curve.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.
        model_name (str): Name of the model.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Evaluation results.
    """
    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_key(model_name, prediction_time)

    for dt in prob_dist_dict_all[model_name].keys():
        expected_value: float = float(
            predict_using_previous_weeks(
                df, dt, prediction_window, x1, y1, x2, y2, prediction_time, num_weeks
            )
        )
        observed_value: float = float(
            calculate_weighted_observed(
                df, dt, prediction_window, x1, y1, x2, y2, prediction_time
            )
        )

        expected_values.append(expected_value)
        observed_values.append(observed_value)

    results = {model_name: calculate_results(expected_values, observed_values)}
    return results


def combine_distributions(dist1: pd.DataFrame, dist2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two probability distributions using convolution.

    Args:
        dist1 (pd.DataFrame): First probability distribution.
        dist2 (pd.DataFrame): Second probability distribution.

    Returns:
        pd.DataFrame: Combined probability distribution.
    """
    arr1 = dist1.values
    arr2 = dist2.values

    combined = signal.convolve(arr1, arr2)
    new_index = range(len(combined))

    combined_df = pd.DataFrame(combined, index=new_index, columns=["agg_predicted"])
    combined_df["agg_predicted"] = (
        combined_df["agg_predicted"] / combined_df["agg_predicted"].sum()
    )

    return combined_df


def evaluate_combined_model(
    prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]],
    df: pd.DataFrame,
    yta_preds: pd.DataFrame,
    prediction_window: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    prediction_time: Tuple[int, int],
    num_weeks: int,
    model_name: str,
    use_most_probable: bool = True,
) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Evaluate the combined prediction model.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        df (pd.DataFrame): DataFrame containing patient data.
        yta_preds (pd.DataFrame): Yet-to-arrive predictions.
        prediction_window (int): Prediction window in minutes.
        x1 (float), y1 (float), x2 (float), y2 (float): Parameters for aspirational curve.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.
        model_name (str): Name of the model.
        use_most_probable (bool, optional): Whether to use the most probable value or expected value. Defaults to True.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Evaluation results.
    """
    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_key(model_name, prediction_time)

    for dt in prob_dist_dict_all[model_name].keys():
        in_ed_preds: Dict[str, Any] = prob_dist_dict_all[model_name][dt]
        combined = combine_distributions(yta_preds, in_ed_preds["agg_predicted"])

        expected_value: Union[int, float] = (
            int(combined["agg_predicted"].idxmax())
            if use_most_probable
            else float(
                np.dot(
                    combined["agg_predicted"].index,
                    combined["agg_predicted"].values.flatten(),
                )
            )
        )

        observed_value: float = float(
            calculate_weighted_observed(
                df, dt, prediction_window, x1, y1, x2, y2, prediction_time
            )
        )

        expected_values.append(expected_value)
        observed_values.append(observed_value)

    results = {model_name: calculate_results(expected_values, observed_values)}
    return results
