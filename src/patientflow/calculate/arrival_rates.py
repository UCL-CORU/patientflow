"""
Calculate and process time-varying arrival rates and admission probabilities.

This module provides functions for calculating arrival rates, admission probabilities,
and unfettered demand rates for inpatient arrivals using an aspirational approach.

Functions
---------
time_varying_arrival_rates : function
    Calculate arrival rates for each time interval across the dataset's date range.
time_varying_arrival_rates_lagged : function
    Create lagged arrival rates based on time intervals.
admission_probabilities : function
    Compute cumulative and hourly admission probabilities using aspirational curves.
weighted_arrival_rates : function
    Aggregate weighted arrival rates for specific time intervals.
unfettered_demand_by_hour : function
    Estimate inpatient demand by hour using historical data and aspirational curves.
count_yet_to_arrive : function
    Count patients who arrived after prediction times and were admitted within prediction windows.

Notes
-----
- All times are handled in local timezone
- Arrival rates are normalized by the number of unique days in the dataset
- Demand calculations consider both historical patterns and admission probabilities
- Time intervals must divide evenly into 24 hours
- Aspirational curves use (x1,y1) and (x2,y2) coordinates to model admission probabilities

Examples
--------
>>> # Generate random arrival times over a week
>>> np.random.seed(42)  # For reproducibility
>>> n_arrivals = 1000
>>> random_times = [
...     pd.Timestamp('2024-01-01') +
...     pd.Timedelta(days=np.random.randint(0, 7)) +
...     pd.Timedelta(hours=np.random.randint(0, 24)) +
...     pd.Timedelta(minutes=np.random.randint(0, 60))
...     for _ in range(n_arrivals)
... ]
>>> df = pd.DataFrame(index=sorted(random_times))
>>>
>>> # Calculate various rates and demand
>>> rates = time_varying_arrival_rates(df, yta_time_interval=60)
>>> lagged_rates = time_varying_arrival_rates_lagged(df, lagged_by=4)
>>> demand = unfettered_demand_by_hour(df, x1=4, y1=0.8, x2=8, y2=0.95)
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from pandas import DataFrame
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union
from patientflow.calculate.admission_in_prediction_window import (
    get_y_from_aspirational_curve,
)


def time_varying_arrival_rates(
    df: DataFrame,
    yta_time_interval: Union[int, timedelta],
    num_days: Optional[int] = None,
    verbose: bool = False,
) -> OrderedDict[time, float]:
    """Calculate the time-varying arrival rates for a dataset indexed by datetime.

    This function computes the arrival rates for each time interval specified, across
    the entire date range present in the dataframe. The arrival rate is calculated as
    the number of entries in the dataframe for each time interval, divided by the
    number of days in the dataset's timespan.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame indexed by datetime, representing the data for which arrival rates
        are to be calculated. The index of the DataFrame should be of datetime type.
    yta_time_interval : int or timedelta
        The time interval for which the arrival rates are to be calculated.
        If int, assumed to be in minutes. If timedelta, will be converted to minutes.
        For example, if `yta_time_interval=60`, the function will calculate hourly
        arrival rates.
    num_days : int, optional
        The number of days that the DataFrame spans. If not provided, the number of
        days is calculated from the date of the min and max arrival datetimes.
    verbose : bool, optional
        If True, enable info-level logging. Defaults to False.

    Returns
    -------
    OrderedDict[datetime.time, float]
        A dictionary mapping times to arrival rates, where times are datetime.time
        objects and rates are float values.

    Raises
    ------
    TypeError
        If 'df' is not a pandas DataFrame, 'yta_time_interval' is not an integer or timedelta,
        or the DataFrame index is not a DatetimeIndex.
    ValueError
        If 'yta_time_interval' is less than or equal to 0 or does not divide evenly
        into 24 hours.

    Notes
    -----
    The minimum and maximum dates in the dataset are used to determine the timespan
    if num_days is not provided.
    """
    import logging
    import sys

    if verbose:
        # Create logger with a unique name
        logger = logging.getLogger(f"{__name__}.time_varying_arrival_rates")

        # Only set up handlers if they don't exist
        if not logger.handlers:
            logger.setLevel(logging.INFO if verbose else logging.WARNING)

            # Create handler that writes to sys.stdout
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO if verbose else logging.WARNING)

            # Create a formatting configuration
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(handler)

            # Prevent propagation to root logger
            logger.propagate = False

    # Input validation
    if not isinstance(df, DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")
    if not isinstance(yta_time_interval, (int, timedelta)):
        raise TypeError(
            "The parameter 'yta_time_interval' must be an integer or timedelta."
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a pandas DatetimeIndex.")

    # Handle both timedelta and numeric inputs for yta_time_interval
    if isinstance(yta_time_interval, timedelta):
        yta_time_interval_minutes = int(yta_time_interval.total_seconds() / 60)
    elif isinstance(yta_time_interval, int):
        yta_time_interval_minutes = yta_time_interval
    else:
        raise TypeError("yta_time_interval must be a timedelta object or integer")

    # Validate time interval
    minutes_in_day = 24 * 60
    if yta_time_interval_minutes <= 0:
        raise ValueError("The parameter 'yta_time_interval' must be positive.")
    if minutes_in_day % yta_time_interval_minutes != 0:
        raise ValueError(
            f"Time interval ({yta_time_interval_minutes} minutes) must divide evenly into 24 hours."
        )

    if num_days is None:
        # Calculate total days between first and last date
        if verbose and logger:
            logger.info("Inferring number of days from dataset")
        start_date = df.index.date.min()
        end_date = df.index.date.max()
        num_days = (end_date - start_date).days + 1

    if num_days == 0:
        raise ValueError("DataFrame contains no data.")

    if verbose and logger:
        logger.info(
            f"Calculating time-varying arrival rates for data provided, which spans {num_days} unique dates"
        )

    arrival_rates_dict = OrderedDict()

    # Initialize a time object to iterate through one day in the specified intervals
    _start_datetime = datetime(1970, 1, 1, 0, 0, 0, 0)
    _stop_datetime = _start_datetime + timedelta(days=1)

    # Iterate over each interval in a single day to calculate the arrival rate
    while _start_datetime != _stop_datetime:
        _start_time = _start_datetime.time()
        _end_time = (
            _start_datetime + timedelta(minutes=yta_time_interval_minutes)
        ).time()

        # Filter the dataframe for entries within the current time interval
        _df = df.between_time(_start_time, _end_time, inclusive="left")

        # Calculate and store the arrival rate for the interval
        arrival_rates_dict[_start_time] = _df.shape[0] / num_days

        # Move to the next interval
        _start_datetime = _start_datetime + timedelta(minutes=yta_time_interval_minutes)

    return arrival_rates_dict


def time_varying_arrival_rates_lagged(
    df: DataFrame,
    lagged_by: int,
    num_days: Optional[int] = None,
    yta_time_interval: Union[int, timedelta] = 60,
) -> OrderedDict[time, float]:
    """Calculate lagged time-varying arrival rates for a dataset indexed by datetime.

    This function first calculates the basic arrival rates and then adjusts them by
    a specified lag time, returning the rates sorted by the lagged times.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame indexed by datetime, representing the data for which arrival rates
        are to be calculated. The index must be a DatetimeIndex.
    lagged_by : int
        Number of hours to lag the arrival times.
    num_days : int, optional
        The number of days that the DataFrame spans. If not provided, the number of
        days is calculated from the date of the min and max arrival datetimes.
    yta_time_interval : int or timedelta, optional
        The time interval for which the arrival rates are to be calculated.
        If int, assumed to be in minutes. If timedelta, will be converted to minutes.
        Defaults to 60.

    Returns
    -------
    OrderedDict[datetime.time, float]
        A dictionary mapping lagged times (datetime.time objects) to their
        corresponding arrival rates.

    Raises
    ------
    TypeError
        If df is not a DataFrame, lagged_by is not an integer, yta_time_interval is not an integer or timedelta,
        or DataFrame index is not DatetimeIndex.
    ValueError
        If lagged_by is negative or yta_time_interval is not positive.

    Notes
    -----
    The lagged times are calculated by adding the specified number of hours to each
    time in the original arrival rates dictionary.
    """
    # Input validation
    if not isinstance(df, DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if not isinstance(lagged_by, int):
        raise TypeError("The parameter 'lagged_by' must be an integer.")

    if not isinstance(yta_time_interval, (int, timedelta)):
        raise TypeError(
            "The parameter 'yta_time_interval' must be an integer or timedelta."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a pandas DatetimeIndex.")

    if lagged_by < 0:
        raise ValueError("The parameter 'lagged_by' must be non-negative.")

    # Handle both timedelta and numeric inputs for yta_time_interval
    if isinstance(yta_time_interval, timedelta):
        yta_time_interval_minutes = int(yta_time_interval.total_seconds() / 60)
    elif isinstance(yta_time_interval, int):
        yta_time_interval_minutes = yta_time_interval
    else:
        raise TypeError("yta_time_interval must be a timedelta object or integer")

    if yta_time_interval_minutes <= 0:
        raise ValueError("The parameter 'yta_time_interval' must be positive.")

    # Calculate base arrival rates
    arrival_rates_dict = time_varying_arrival_rates(
        df, yta_time_interval, num_days=num_days
    )

    # Apply lag to the times
    lagged_dict = OrderedDict()
    reference_date = datetime(2000, 1, 1)  # Use arbitrary reference date

    for base_time, rate in arrival_rates_dict.items():
        # Combine with reference date and apply lag
        lagged_datetime = datetime.combine(reference_date, base_time) + timedelta(
            hours=lagged_by
        )
        lagged_dict[lagged_datetime.time()] = rate

    # Sort by lagged times
    return OrderedDict(sorted(lagged_dict.items()))


def process_arrival_rates(
    arrival_rates_dict: Dict[time, float],
) -> Tuple[List[float], List[str], List[int]]:
    """Process arrival rates dictionary into formats needed for plotting.

    Parameters
    ----------
    arrival_rates_dict : Dict[datetime.time, float]
        Mapping of times to arrival rates.

    Returns
    -------
    Tuple[List[float], List[str], List[int]]
        A tuple containing:
        - List[float]: Arrival rate values
        - List[str]: Formatted hour range strings (e.g., "09-\n10")
        - List[int]: Integers for x-axis positioning

    Notes
    -----
    The hour labels are formatted with line breaks for better plot readability.
    """
    # Extract hours and rates
    hours = list(arrival_rates_dict.keys())
    arrival_rates = list(arrival_rates_dict.values())

    # Create formatted hour labels with line breaks for better plot readability
    hour_labels = [
        f'{hour.strftime("%H")}-\n{str((hour.hour + 1) % 24).zfill(2)}'
        for hour in hours
    ]

    # Generate numerical values for x-axis positioning
    hour_values = list(range(len(hour_labels)))

    return arrival_rates, hour_labels, hour_values


def admission_probabilities(
    hours_since_arrival: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate probability of admission for each hour since arrival.

    Parameters
    ----------
    hours_since_arrival : np.ndarray
        Array of hours since arrival.
    x1 : float
        First x-coordinate of the aspirational curve.
    y1 : float
        First y-coordinate of the aspirational curve.
    x2 : float
        Second x-coordinate of the aspirational curve.
    y2 : float
        Second y-coordinate of the aspirational curve.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - np.ndarray: Cumulative admission probabilities
        - np.ndarray: Hourly admission probabilities

    Notes
    -----
    The aspirational curve is defined by two points (x1,y1) and (x2,y2) and is used
    to model the probability of admission over time.
    """
    prob_admission_by_hour = np.array(
        [
            get_y_from_aspirational_curve(hour, x1, y1, x2, y2)
            for hour in hours_since_arrival
        ]
    )
    prob_admission_within_hour = np.diff(prob_admission_by_hour)

    return prob_admission_by_hour, prob_admission_within_hour


def weighted_arrival_rates(
    weighted_rates: np.ndarray, elapsed_hours: range, hour_idx: int, num_intervals: int
) -> float:
    """Calculate sum of weighted arrival rates for a specific time interval.

    Parameters
    ----------
    weighted_rates : np.ndarray
        Array of weighted arrival rates.
    elapsed_hours : range
        Range of elapsed hours to consider.
    hour_idx : int
        Current interval index.
    num_intervals : int
        Total number of intervals in a day.

    Returns
    -------
    float
        Sum of weighted arrival rates.

    Notes
    -----
    The function calculates the sum of weighted arrival rates by iterating through
    the elapsed hours and considering the appropriate interval index for each hour.
    """
    total = 0
    for elapsed_hour in elapsed_hours:
        interval_index = (hour_idx - elapsed_hour) % num_intervals
        total += weighted_rates[elapsed_hour][interval_index]
    return total


def unfettered_demand_by_hour(
    df: DataFrame,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    yta_time_interval: Union[int, timedelta] = 60,
    max_hours_since_arrival: int = 10,
    num_days: Optional[int] = None,
) -> OrderedDict[time, float]:
    """Calculate true inpatient demand by hour based on historical arrival data.

    This function estimates demand rates using historical arrival data and an aspirational
    curve for admission probabilities. It takes a DataFrame of historical arrivals and
    parameters defining an aspirational curve to calculate hourly demand rates.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame indexed by datetime, representing historical arrival data.
        The index must be a DatetimeIndex.
    x1 : float
        First x-coordinate of the aspirational curve.
    y1 : float
        First y-coordinate of the aspirational curve (0-1).
    x2 : float
        Second x-coordinate of the aspirational curve.
    y2 : float
        Second y-coordinate of the aspirational curve (0-1).
    yta_time_interval : int or timedelta, optional
        Time interval for which the arrival rates are to be calculated.
        If int, assumed to be in minutes. If timedelta, will be converted to minutes.
        Defaults to 60.
    max_hours_since_arrival : int, optional
        Maximum hours since arrival to consider. Defaults to 10.
    num_days : int, optional
        The number of days that the DataFrame spans. If not provided, the number of
        days is calculated from the date of the min and max arrival datetimes.

    Returns
    -------
    OrderedDict[datetime.time, float]
        A dictionary mapping times (datetime.time objects) to their corresponding
        demand rates.

    Raises
    ------
    TypeError
        If df is not a DataFrame, coordinates are not floats, or DataFrame index
        is not DatetimeIndex.
    ValueError
        If coordinates are outside valid ranges, yta_time_interval is not positive,
        or doesn't divide evenly into 24 hours.

    Notes
    -----
    The function combines historical arrival patterns with admission probabilities
    to estimate true inpatient demand. The aspirational curve is used to model
    how admission probabilities change over time.
    """
    # Input validation
    if not isinstance(df, DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a pandas DatetimeIndex.")

    if not all(isinstance(x, (int, float)) for x in [x1, y1, x2, y2]):
        raise TypeError("Curve coordinates must be numeric values.")

    if not isinstance(yta_time_interval, (int, timedelta)):
        raise TypeError(
            "The parameter 'yta_time_interval' must be an integer or timedelta."
        )

    if not isinstance(max_hours_since_arrival, int):
        raise TypeError("The parameter 'max_hours_since_arrival' must be an integer.")

    # Handle both timedelta and numeric inputs for yta_time_interval
    if isinstance(yta_time_interval, timedelta):
        yta_time_interval_minutes = int(yta_time_interval.total_seconds() / 60)
    elif isinstance(yta_time_interval, int):
        yta_time_interval_minutes = yta_time_interval
    else:
        raise TypeError("yta_time_interval must be a timedelta object or integer")

    # Validate time interval
    minutes_in_day = 24 * 60
    if yta_time_interval_minutes <= 0:
        raise ValueError("The parameter 'yta_time_interval' must be positive.")
    if minutes_in_day % yta_time_interval_minutes != 0:
        raise ValueError(
            f"Time interval ({yta_time_interval_minutes} minutes) must divide evenly into 24 hours."
        )

    if max_hours_since_arrival <= 0:
        raise ValueError("The parameter 'max_hours_since_arrival' must be positive.")

    if not (0 <= y1 <= 1 and 0 <= y2 <= 1):
        raise ValueError("Y-coordinates must be between 0 and 1.")

    if x1 >= x2:
        raise ValueError("x1 must be less than x2.")

    # Calculate number of intervals in a day
    num_intervals = minutes_in_day // yta_time_interval_minutes

    # Calculate admission probabilities
    hours_since_arrival = np.arange(max_hours_since_arrival + 1)
    _, prob_admission_within_hour = admission_probabilities(
        hours_since_arrival, x1, y1, x2, y2
    )

    # Calculate base arrival rates from historical data
    arrival_rates_dict = time_varying_arrival_rates(
        df, yta_time_interval_minutes, num_days=num_days
    )

    # Convert dict to arrays while preserving order
    hour_keys = list(arrival_rates_dict.keys())
    arrival_rates = np.array([arrival_rates_dict[hour] for hour in hour_keys])

    # Initialize array for weighted arrival rates
    weighted_rates = np.zeros((max_hours_since_arrival, len(arrival_rates)))

    # Calculate weighted arrival rates for each hour and elapsed time
    for hour_idx, _ in enumerate(hour_keys):
        arrival_rate = arrival_rates[hour_idx]
        weighted_rates[:, hour_idx] = (
            arrival_rate * prob_admission_within_hour[:max_hours_since_arrival]
        )

    # Calculate summed demand rates for each hour
    demand_by_hour = OrderedDict()
    elapsed_hours = range(max_hours_since_arrival)

    for hour_idx, hour_key in enumerate(hour_keys):
        demand_by_hour[hour_key] = weighted_arrival_rates(
            weighted_rates, elapsed_hours, hour_idx, num_intervals
        )

    return demand_by_hour


def count_yet_to_arrive(
    df: DataFrame,
    snapshot_dates: List,
    prediction_times: List,
    prediction_window_hours: float,
) -> DataFrame:
    """Count patients who arrived after prediction times and were admitted within prediction windows.

    This function counts patients who arrived after specified prediction times and were
    admitted to a ward within the specified prediction window for each combination of
    snapshot date and prediction time.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing patient data with 'arrival_datetime',
        'admitted_to_ward_datetime', and 'patient_id' columns.
    snapshot_dates : list
        List of dates (datetime.date objects) to analyze.
    prediction_times : list
        List of (hour, minute) tuples representing prediction times.
    prediction_window_hours : float
        Length of prediction window in hours after the prediction time.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'snapshot_date': The date of the snapshot
        - 'prediction_time': Tuple of (hour, minute) for the prediction time
        - 'count': Number of unique patients who arrived after prediction time
                  and were admitted within the prediction window

    Raises
    ------
    TypeError
        If df is not a DataFrame or if required columns are missing.
    ValueError
        If prediction_window_hours is not positive.

    Notes
    -----
    This function is useful for analyzing historical patterns of patient arrivals
    and admissions to inform predictive models for emergency department demand.
    Only patients with non-null admitted_to_ward_datetime are counted.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import date, time
    >>> prediction_times = [(12, 0), (15, 30)]
    >>> snapshot_dates = [date(2023, 1, 1), date(2023, 1, 2)]
    >>> results = count_yet_to_arrive(df, snapshot_dates, prediction_times, 8.0)
    """
    # Input validation
    if not isinstance(df, DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    required_columns = ["arrival_datetime", "admitted_to_ward_datetime", "patient_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise TypeError(f"DataFrame missing required columns: {missing_columns}")

    if (
        not isinstance(prediction_window_hours, (int, float))
        or prediction_window_hours <= 0
    ):
        raise ValueError("prediction_window_hours must be a positive number.")

    # Create an empty list to store results
    results = []

    # For each combination of date and time
    for date_val in snapshot_dates:
        for hour, minute in prediction_times:
            # Create the prediction datetime
            prediction_datetime = pd.Timestamp(
                datetime.combine(date_val, time(hour=hour, minute=minute))
            )

            # Calculate the end of the prediction window
            prediction_window_end = prediction_datetime + pd.Timedelta(
                hours=prediction_window_hours
            )

            # Count patients who arrived after prediction time and were admitted within the window
            admitted_within_window = len(
                df[
                    (df["arrival_datetime"] > prediction_datetime)
                    & (df["admitted_to_ward_datetime"] <= prediction_window_end)
                ]
            )

            # Store the result
            results.append(
                {
                    "snapshot_date": date_val,
                    "prediction_time": (hour, minute),
                    "count": admitted_within_window,
                }
            )

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
