"""
Weighted Poisson Predictor for Hospital Admissions Forecasting.

This module implements a custom predictor to estimate the number of hospital admissions
within a specified prediction window using historical admission data. It applies Poisson
and binomial distributions to forecast future admissions, excluding already arrived patients.
The predictor accommodates different data filters for tailored predictions across various
hospital settings.

Classes
-------
WeightedPoissonPredictor : BaseEstimator, TransformerMixin
    Predicts the number of admissions within a given prediction window based on historical
    data and Poisson-binomial distribution.

Notes
-----
The predictor uses a combination of Poisson and binomial distributions to model the
probability of admissions within a prediction window. It takes into account historical
data patterns and can be filtered for specific hospital settings or specialties.

Examples
--------
>>> predictor = WeightedPoissonPredictor(filters={
...     'medical': {'specialty': 'medical'},
...     'surgical': {'specialty': 'surgical'},
...     'haem_onc': {'specialty': 'haem/onc'},
...     'paediatric': {'specialty': 'paediatric'}
... })
>>> predictor.fit(train_data, prediction_window=120, yta_time_interval=30, prediction_times=[8, 12, 16])
>>> predictions = predictor.predict(prediction_context, x1=2, y1=0.5, x2=4, y2=0.9)
"""

import warnings
from datetime import datetime, timedelta

import numpy as np

import pandas as pd
from typing import Dict, List, Optional

# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from patientflow.calculate.admission_in_prediction_window import (
    get_y_from_aspirational_curve,
)

# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates,
)


# Import utility functions for time adjustment
# from edmodel.utils.time_utils import adjust_for_model_specific_times
# Import sklearn base classes for custom transformer creation
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import binom, poisson


def weighted_poisson_binomial(i, lam, theta):
    """Calculate weighted probabilities using Poisson and Binomial distributions.

    Parameters
    ----------
    i : int
        The upper bound of the range for the binomial distribution.
    lam : float
        The lambda parameter for the Poisson distribution.
    theta : float
        The probability of success for the binomial distribution.

    Returns
    -------
    numpy.ndarray
        An array of weighted probabilities.

    Raises
    ------
    ValueError
        If i < 0, lam < 0, or theta is not between 0 and 1.
    """
    if i < 0 or lam < 0 or not 0 <= theta <= 1:
        raise ValueError("Ensure i >= 0, lam >= 0, and 0 <= theta <= 1.")

    arr_seq = np.arange(i + 1)
    probabilities = binom.pmf(arr_seq, i, theta)
    return poisson.pmf(i, lam) * probabilities


def aggregate_probabilities(lam, kmax, theta, time_index):
    """Aggregate probabilities for a range of values using the weighted Poisson-Binomial distribution.

    Parameters
    ----------
    lam : numpy.ndarray
        An array of lambda values for each time interval.
    kmax : int
        The maximum number of events to consider.
    theta : numpy.ndarray
        An array of theta values for each time interval.
    time_index : int
        The current time index for which to calculate probabilities.

    Returns
    -------
    numpy.ndarray
        Aggregated probabilities for the given time index.

    Raises
    ------
    ValueError
        If kmax < 0, time_index < 0, or array lengths are invalid.
    """
    if kmax < 0 or time_index < 0 or len(lam) <= time_index or len(theta) <= time_index:
        raise ValueError("Invalid kmax, time_index, or array lengths.")

    probabilities_matrix = np.zeros((kmax + 1, kmax + 1))
    for i in range(kmax + 1):
        probabilities_matrix[: i + 1, i] = weighted_poisson_binomial(
            i, lam[time_index], theta[time_index]
        )
    return probabilities_matrix.sum(axis=1)


def convolute_distributions(dist_a, dist_b):
    """Convolutes two probability distributions represented as dataframes.

    Parameters
    ----------
    dist_a : pd.DataFrame
        The first distribution with columns ['sum', 'prob'].
    dist_b : pd.DataFrame
        The second distribution with columns ['sum', 'prob'].

    Returns
    -------
    pd.DataFrame
        The convoluted distribution.

    Raises
    ------
    ValueError
        If DataFrames do not contain required 'sum' and 'prob' columns.
    """
    if not {"sum", "prob"}.issubset(dist_a.columns) or not {
        "sum",
        "prob",
    }.issubset(dist_b.columns):
        raise ValueError("DataFrames must contain 'sum' and 'prob' columns.")

    sums = [x + y for x in dist_a["sum"] for y in dist_b["sum"]]
    probs = [x * y for x in dist_a["prob"] for y in dist_b["prob"]]
    result = pd.DataFrame(zip(sums, probs), columns=["sum", "prob"])
    return result.groupby("sum")["prob"].sum().reset_index()


def poisson_binom_generating_function(NTimes, arrival_rates, theta, epsilon):
    """Generate a distribution based on the aggregate of Poisson and Binomial distributions.

    Parameters
    ----------
    NTimes : int
        The number of time intervals.
    arrival_rates : numpy.ndarray
        An array of lambda values for each time interval.
    theta : numpy.ndarray
        An array of theta values for each time interval.
    epsilon : float
        The desired error threshold.

    Returns
    -------
    pd.DataFrame
        The generated distribution.

    Raises
    ------
    ValueError
        If NTimes <= 0 or epsilon is not between 0 and 1.
    """

    if NTimes <= 0 or epsilon <= 0 or epsilon >= 1:
        raise ValueError("Ensure NTimes > 0 and 0 < epsilon < 1.")

    maxlam = max(arrival_rates)
    kmax = int(poisson.ppf(1 - epsilon, maxlam))
    distribution = np.zeros((kmax + 1, NTimes))

    for j in range(NTimes):
        distribution[:, j] = aggregate_probabilities(arrival_rates, kmax, theta, j)

    df_list = [
        pd.DataFrame({"sum": range(kmax + 1), "prob": distribution[:, j]})
        for j in range(NTimes)
    ]
    total_distribution = df_list[0]

    for df in df_list[1:]:
        total_distribution = convolute_distributions(total_distribution, df)

    total_distribution = total_distribution.rename(
        columns={"prob": "agg_proba"}
    ).set_index("sum")

    return total_distribution


def find_nearest_previous_prediction_time(requested_time, prediction_times):
    """Find the nearest previous time of day in prediction_times relative to requested time.

    Parameters
    ----------
    requested_time : tuple
        The requested time as (hour, minute).
    prediction_times : list
        List of available prediction times.

    Returns
    -------
    tuple
        The closest previous time of day from prediction_times.

    Notes
    -----
    If the requested time is earlier than all times in prediction_times,
    returns the latest time in prediction_times.
    """
    if requested_time in prediction_times:
        return requested_time

    original_prediction_time = requested_time
    requested_datetime = datetime.strptime(
        f"{requested_time[0]:02d}:{requested_time[1]:02d}", "%H:%M"
    )
    closest_prediction_time = max(
        prediction_times,
        key=lambda prediction_time_time: datetime.strptime(
            f"{prediction_time_time[0]:02d}:{prediction_time_time[1]:02d}",
            "%H:%M",
        ),
    )
    min_diff = float("inf")

    for prediction_time_time in prediction_times:
        prediction_time_datetime = datetime.strptime(
            f"{prediction_time_time[0]:02d}:{prediction_time_time[1]:02d}",
            "%H:%M",
        )
        diff = (requested_datetime - prediction_time_datetime).total_seconds()

        # If the difference is negative, it means the prediction_time_time is ahead of the requested_time,
        # hence we calculate the difference by considering a day's wrap around.
        if diff < 0:
            diff += 24 * 60 * 60  # Add 24 hours in seconds

        if 0 <= diff < min_diff:
            closest_prediction_time = prediction_time_time
            min_diff = diff

    warnings.warn(
        f"Time of day requested of {original_prediction_time} was not in model training. "
        f"Reverting to predictions for {closest_prediction_time}."
    )

    return closest_prediction_time


class WeightedPoissonPredictor(BaseEstimator, TransformerMixin):
    """A predictor for estimating hospital admissions within a prediction window.

    This predictor uses a combination of Poisson and binomial distributions to forecast
    future admissions, excluding patients who have already arrived. The prediction is
    based on historical data and can be filtered for specific hospital settings.

    Parameters
    ----------
    filters : dict, optional
        Optional filters for data categorization. If None, no filtering is applied.
    verbose : bool, default=False
        Whether to enable verbose logging.

    Attributes
    ----------
    filters : dict
        Filters for data categorization.
    verbose : bool
        Verbose logging flag.
    metrics : dict
        Stores metadata about the model and training data.
    weights : dict
        Model parameters computed during fitting.

    Notes
    -----
    The predictor implements scikit-learn's BaseEstimator and TransformerMixin
    interfaces for compatibility with scikit-learn pipelines.
    """

    def __init__(self, filters=None, verbose=False):
        """
        Initialize the WeightedPoissonPredictor with optional filters.

        Args:
            filters (dict, optional): A dictionary defining filters for different categories or specialties.
                                    If None or empty, no filtering will be applied.
            verbose (bool, optional): If True, enable info-level logging. Defaults to False.
        """
        self.filters = filters if filters else {}
        self.verbose = verbose
        self.metrics = {}  # Add metrics dictionary to store metadata

        if verbose:
            # Configure logging for Jupyter notebook compatibility
            import logging
            import sys

            # Create logger
            self.logger = logging.getLogger(f"{__name__}.WeightedPoissonPredictor")

            # Only set up handlers if they don't exist
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

                # Create handler that writes to sys.stdout
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.INFO if verbose else logging.WARNING)

                # Create a formatting configuration
                formatter = logging.Formatter("%(message)s")
                handler.setFormatter(formatter)

                # Add the handler to the logger
                self.logger.addHandler(handler)

                # Prevent propagation to root logger
                self.logger.propagate = False

        # Apply filters
        self.filters = filters if filters else {}

    def filter_dataframe(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply a set of filters to a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to filter.
        filters : dict
            A dictionary where keys are column names and values are the criteria
            or function to filter by.

        Returns
        -------
        pandas.DataFrame
            A filtered DataFrame.
        """
        filtered_df = df
        for column, criteria in filters.items():
            if callable(criteria):  # If the criteria is a function, apply it directly
                filtered_df = filtered_df[filtered_df[column].apply(criteria)]
            else:  # Otherwise, assume the criteria is a value or list of values for equality check
                filtered_df = filtered_df[filtered_df[column] == criteria]
        return filtered_df

    def _calculate_parameters(
        self, df, prediction_window, yta_time_interval, prediction_times, num_days
    ):
        """Calculate parameters required for the model.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to process.
        prediction_window : int, float, or timedelta
            The total prediction window for prediction. If timedelta, operations will convert to minutes as needed.
        yta_time_interval : int, float, or timedelta
            The interval for splitting the prediction window. If timedelta, operations will convert to minutes as needed.
        prediction_times : list
            Times of day at which predictions are made.
        num_days : int
            Number of days over which to calculate time-varying arrival rates.

        Returns
        -------
        dict
            Calculated arrival_rates parameters organized by time of day.
        """
        
        # Calculate Ntimes - Python handles the division naturally
        if isinstance(prediction_window, timedelta) and isinstance(yta_time_interval, timedelta):
            Ntimes = int(prediction_window / yta_time_interval)
        elif isinstance(prediction_window, timedelta):
            Ntimes = int(prediction_window.total_seconds() / 60 / yta_time_interval)
        elif isinstance(yta_time_interval, timedelta):
            Ntimes = int(prediction_window / (yta_time_interval.total_seconds() / 60))
        else:
            Ntimes = int(prediction_window / yta_time_interval)
        
        # Pass original type to time_varying_arrival_rates (it handles both int and timedelta)
        arrival_rates_dict = time_varying_arrival_rates(
            df, yta_time_interval, num_days, verbose=self.verbose
        )
        prediction_time_dict = {}

        for prediction_time_ in prediction_times:
            prediction_time_hr, prediction_time_min = (
                (prediction_time_, 0)
                if isinstance(prediction_time_, int)
                else prediction_time_
            )
            arrival_rates = [
                arrival_rates_dict[
                    (
                        datetime(1970, 1, 1, prediction_time_hr, prediction_time_min)
                        + i * (yta_time_interval if isinstance(yta_time_interval, timedelta) 
                               else timedelta(minutes=yta_time_interval))
                    ).time()
                ]
                for i in range(Ntimes)
            ]
            prediction_time_dict[(prediction_time_hr, prediction_time_min)] = {
                "arrival_rates": arrival_rates
            }

        return prediction_time_dict

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window,
        yta_time_interval,
        prediction_times: List[float],
        num_days: int,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
    ) -> "WeightedPoissonPredictor":
        """Fit the model to the training data.

        Parameters
        ----------
        train_df : pandas.DataFrame
            The training dataset with historical admission data.
        prediction_window : int or timedelta
            The prediction window in minutes. If timedelta, will be converted to minutes.
            If int, assumed to be in minutes.
        yta_time_interval : int or timedelta
            The interval in minutes for splitting the prediction window. If timedelta, will be converted to minutes.
            If int, assumed to be in minutes.
        prediction_times : list
            Times of day at which predictions are made, in hours.
        num_days : int
            The number of days that the train_df spans.
        epsilon : float, default=1e-7
            A small value representing acceptable error rate to enable calculation
            of the maximum value of the random variable representing number of beds.
        y : None, optional
            Ignored, present for compatibility with scikit-learn's fit method.

        Returns
        -------
        WeightedPoissonPredictor
            The instance itself, fitted with the training data.

        Raises
        ------
        ValueError
            If prediction_window/yta_time_interval is not greater than 1.
        """

        # Validate inputs - no conversions needed
        if isinstance(prediction_window, timedelta):
            if prediction_window.total_seconds() <= 0:
                raise ValueError("prediction_window must be positive")
        elif isinstance(prediction_window, (int, float)):
            if prediction_window <= 0:
                raise ValueError("prediction_window must be positive")
            if not np.isfinite(prediction_window):
                raise ValueError("prediction_window must be finite")
        else:
            raise TypeError("prediction_window must be a timedelta object or numeric value")
        
        if isinstance(yta_time_interval, timedelta):
            if yta_time_interval.total_seconds() <= 0:
                raise ValueError("yta_time_interval must be positive")
            if yta_time_interval.total_seconds() > 4 * 3600:  # 4 hours in seconds
                warnings.warn("yta_time_interval appears to be longer than 4 hours")
        elif isinstance(yta_time_interval, (int, float)):
            if yta_time_interval <= 0:
                raise ValueError("yta_time_interval must be positive")
            if yta_time_interval > 4 * 60:  # 4 hours in minutes
                warnings.warn("yta_time_interval appears to be longer than 4 hours")
            if not np.isfinite(yta_time_interval):
                raise ValueError("yta_time_interval must be finite")
        else:
            raise TypeError("yta_time_interval must be a timedelta object or numeric value")

        # Validate the ratio makes sense (convert only for this check)
        if isinstance(prediction_window, timedelta) and isinstance(yta_time_interval, timedelta):
            ratio = prediction_window / yta_time_interval
        elif isinstance(prediction_window, timedelta):
            ratio = prediction_window.total_seconds() / 60 / yta_time_interval
        elif isinstance(yta_time_interval, timedelta):
            ratio = prediction_window / (yta_time_interval.total_seconds() / 60)
        else:
            ratio = prediction_window / yta_time_interval
            
        if int(ratio) == 0:
            raise ValueError("prediction_window must be significantly larger than yta_time_interval")

        # Store original types - no conversion!
        self.prediction_window = prediction_window
        self.yta_time_interval = yta_time_interval
        self.epsilon = epsilon
        self.prediction_times = [
            tuple(x)
            if isinstance(x, (list, np.ndarray))
            else (x, 0)
            if isinstance(x, (int, float))
            else x
            for x in prediction_times
        ]

        # Initialize yet_to_arrive_dict
        self.weights = {}

        # If there are filters specified, calculate and store the parameters directly with the respective spec keys
        if self.filters:
            for spec, filters in self.filters.items():
                self.weights[spec] = self._calculate_parameters(
                    self.filter_dataframe(train_df, filters),
                    prediction_window,
                    yta_time_interval,
                    prediction_times,
                    num_days,
                )
        else:
            # If there are no filters, store the parameters with a generic key of 'unfiltered'
            self.weights["unfiltered"] = self._calculate_parameters(
                train_df,
                prediction_window,
                yta_time_interval,
                prediction_times,
                num_days,
            )

        if self.verbose:
            self.logger.info(
                f"Weighted Poisson Predictor trained for these times: {prediction_times}"
            )
            self.logger.info(
                f"using prediction window of {prediction_window} after the time of prediction"
            )
            self.logger.info(
                f"and time interval of {yta_time_interval} within the prediction window."
            )
            self.logger.info(f"The error value for prediction will be {epsilon}")
            self.logger.info(
                "To see the weights saved by this model, used the get_weights() method"
            )

        # Store metrics about the training data
        self.metrics["train_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.metrics["train_set_no"] = len(train_df)
        self.metrics["start_date"] = train_df.index.min().date()
        self.metrics["end_date"] = train_df.index.max().date()
        self.metrics["num_days"] = num_days

        return self

    def get_weights(self):
        """Get the weights computed by the fit method.

        Returns
        -------
        dict
            The weights computed during model fitting.
        """
        return self.weights

    def predict(
        self, prediction_context: Dict, x1: float, y1: float, x2: float, y2: float
    ) -> Dict:
        """Predict the number of admissions for the given context.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        x1 : float
            The x-coordinate of the first transition point on the aspirational curve,
            where the growth phase ends and the decay phase begins.
        y1 : float
            The y-coordinate of the first transition point (x1), representing the target
            proportion of patients admitted by time x1.
        x2 : float
            The x-coordinate of the second transition point on the curve, beyond which
            all but a few patients are expected to be admitted.
        y2 : float
            The y-coordinate of the second transition point (x2), representing the target
            proportion of patients admitted by time x2.

        Returns
        -------
        dict
            A dictionary with predictions for each specified context.

        Raises
        ------
        ValueError
            If filter key is not recognized or prediction_time is not provided.
        KeyError
            If required keys are missing from the prediction context.
        """
        predictions = {}

        # Calculate Ntimes 
        if isinstance(self.prediction_window, timedelta) and isinstance(self.yta_time_interval, timedelta):
            NTimes = int(self.prediction_window / self.yta_time_interval)
        elif isinstance(self.prediction_window, timedelta):
            NTimes = int(self.prediction_window.total_seconds() / 60 / self.yta_time_interval)
        elif isinstance(self.yta_time_interval, timedelta):
            NTimes = int(self.prediction_window / (self.yta_time_interval.total_seconds() / 60))
        else:
            NTimes = int(self.prediction_window / self.yta_time_interval)

        # Convert to hours only for numpy operations (which require numeric types)
        prediction_window_hours = (
            self.prediction_window.total_seconds() / 3600
            if isinstance(self.prediction_window, timedelta)
            else self.prediction_window / 60
        )
        yta_time_interval_hours = (
            self.yta_time_interval.total_seconds() / 3600
            if isinstance(self.yta_time_interval, timedelta)
            else self.yta_time_interval / 60
        )

        # Calculate theta, probability of admission in prediction window
        # for each time interval, calculate time remaining before end of window
        time_remaining_before_end_of_window = prediction_window_hours - np.arange(
            0, prediction_window_hours, yta_time_interval_hours
        )

        theta = get_y_from_aspirational_curve(
            time_remaining_before_end_of_window, x1, y1, x2, y2
        )

        for filter_key, filter_values in prediction_context.items():
            try:
                if filter_key not in self.weights:
                    raise ValueError(
                        f"Filter key '{filter_key}' is not recognized in the model weights."
                    )

                prediction_time = filter_values.get("prediction_time")
                if prediction_time is None:
                    raise ValueError(
                        f"No 'prediction_time' provided for filter '{filter_key}'."
                    )

                if prediction_time not in self.prediction_times:
                    prediction_time = find_nearest_previous_prediction_time(
                        prediction_time, self.prediction_times
                    )

                arrival_rates = self.weights[filter_key][prediction_time].get(
                    "arrival_rates"
                )
                if arrival_rates is None:
                    raise ValueError(
                        f"No arrival_rates found for the time of day '{prediction_time}' under filter '{filter_key}'."
                    )

                predictions[filter_key] = poisson_binom_generating_function(
                    NTimes, arrival_rates, theta, self.epsilon
                )

            except KeyError as e:
                raise KeyError(f"Key error occurred: {e!s}")

        return predictions
