"""
Hospital Admissions Forecasting Predictors.

This module implements custom predictors to estimate the number of hospital admissions
within a specified prediction window using historical admission data. It provides three
approaches: direct admission prediction, parametric curves with Poisson-binomial distributions,
and empirical survival curves with convolution of Poisson distributions. All predictors
accommodate different data filters for tailored predictions across various hospital settings.

Classes
-------
AdmissionGeneratingFunction
    Unified generating function class for computation of admission prediction
    distributions using probability generating functions.

IncomingAdmissionPredictor : BaseEstimator, TransformerMixin
    Base class for admission predictors that handles filtering and arrival rate calculation.
    Now supports both generating function and traditional approaches via use_generating_functions flag.

DirectAdmissionPredictor : IncomingAdmissionPredictor
    Simplest predictor that assumes every arrival is admitted immediately. Uses direct
    Poisson distribution based on total arrival rates without any admission probability
    adjustments. Now supports both generating function and traditional approaches.

ParametricIncomingAdmissionPredictor : IncomingAdmissionPredictor
    Predicts the number of admissions within a given prediction window based on historical
    data and Poisson-binomial distribution using parametric aspirational curves.
    Now uses efficient generating functions for better performance.

EmpiricalIncomingAdmissionPredictor : IncomingAdmissionPredictor
    Predicts the number of admissions using empirical survival curves and convolution
    of Poisson distributions instead of parametric curves. Now supports both generating
    function and traditional approaches.

Notes
-----
The DirectAdmissionPredictor is the simplest approach, summing all arrival rates across time
intervals and creating a single Poisson distribution. It assumes 100% admission rate,
making it useful for scenarios where immediate admission is expected or as a baseline
for comparison with more complex models.

The ParametricIncomingAdmissionPredictor uses a combination of Poisson and binomial distributions to
model the probability of admissions within a prediction window using parametric curves
defined by transition points (x1, y1, x2, y2).

The EmpiricalIncomingAdmissionPredictor inherits the arrival rate calculation and filtering logic
but replaces the parametric approach with empirical survival probabilities and convolution
of individual Poisson distributions for each time interval.

All predictors take into account historical data patterns and can be filtered for
specific hospital settings or specialties.

"""

import warnings
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

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

from patientflow.calculate.survival_curve import (
    calculate_survival_curve,
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


class AdmissionGeneratingFunction:
    """Unified generating function for admission prediction distributions.
    
    This class provides efficient computation of probability distributions for
    different admission prediction approaches using simple convolution methods.
    
    For typical use cases (32 intervals, max ~50), simple convolution is more
    appropriate than FFT-based methods which add unnecessary complexity.
    
    Parameters
    ----------
    arrival_rates : numpy.ndarray
        Array of arrival rates for each time interval
    admission_probs : numpy.ndarray
        Array of admission probabilities for each time interval
    method : str, default='empirical'
        Method to use: 'direct', 'parametric', or 'empirical'
    
    Attributes
    ----------
    arrival_rates : numpy.ndarray
        Arrival rates for each time interval
    admission_probs : numpy.ndarray
        Admission probabilities for each time interval
    method : str
        The prediction method being used
    """
    
    def __init__(self, arrival_rates, admission_probs, method='empirical'):
        self.arrival_rates = np.array(arrival_rates)
        self.admission_probs = np.array(admission_probs)
        self.method = method
        
        if len(self.arrival_rates) != len(self.admission_probs):
            raise ValueError("arrival_rates and admission_probs must have the same length")
    
    def get_distribution(self, max_value=50):
        """Get the probability distribution for the specified method.
        
        Parameters
        ----------
        max_value : int, default=50
            Maximum value for the discrete distribution support
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with 'sum' and 'agg_proba' columns representing the distribution
        """
        if self.method == 'direct':
            return self._direct_distribution(max_value)
        elif self.method == 'empirical':
            return self._empirical_distribution(max_value)
        elif self.method == 'parametric':
            return self._parametric_distribution(max_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _direct_distribution(self, max_value=50):
        """Simple Poisson distribution for direct method."""
        total_rate = np.sum(self.arrival_rates)
        x_values = np.arange(max_value)
        probabilities = poisson.pmf(x_values, total_rate)
        
        result_df = pd.DataFrame({
            'sum': range(len(probabilities)), 
            'agg_proba': probabilities
        })
        
        # Filter and normalize
        result_df = result_df[result_df['agg_proba'] > 1e-10]
        if len(result_df) > 0:
            result_df['agg_proba'] = result_df['agg_proba'] / result_df['agg_proba'].sum()
        
        return result_df.set_index('sum')
    
    def _empirical_distribution(self, max_value=50):
        """Weighted Poisson convolution for empirical method."""
        # Create weighted Poisson distributions for each time interval
        weighted_rates = self.arrival_rates * self.admission_probs
        x_values = np.arange(max_value)
        
        # Start with first distribution
        probabilities = poisson.pmf(x_values, weighted_rates[0])
        
        # Convolve with remaining distributions
        for rate in weighted_rates[1:]:
            pmf_i = poisson.pmf(x_values, rate)
            probabilities = np.convolve(probabilities, pmf_i)[:max_value]
        
        result_df = pd.DataFrame({
            'sum': range(len(probabilities)), 
            'agg_proba': probabilities
        })
        
        # Filter and normalize
        result_df = result_df[result_df['agg_proba'] > 1e-10]
        if len(result_df) > 0:
            result_df['agg_proba'] = result_df['agg_proba'] / result_df['agg_proba'].sum()
        
        return result_df.set_index('sum')
    
    def _parametric_distribution(self, max_value=50):
        """Poisson-binomial compound distribution for parametric method."""
        # For parametric case, use the existing proven method
        # This is complex enough that the existing implementation is appropriate
        NTimes = len(self.arrival_rates)
        epsilon = 1e-7
        
        # Use the existing poisson_binom_generating_function
        return poisson_binom_generating_function(
            NTimes, self.arrival_rates, self.admission_probs, epsilon
        )


class IncomingAdmissionPredictor(BaseEstimator, TransformerMixin, ABC):
    """Base class for admission predictors that handles filtering and arrival rate calculation.

    This abstract base class provides the common functionality for predicting hospital
    admissions, including data filtering, arrival rate calculation, and basic prediction
    infrastructure. Subclasses implement specific prediction strategies.

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

    def __init__(self, filters=None, verbose=False, use_generating_functions=True):
        """
        Initialize the IncomingAdmissionPredictor with optional filters.

        Args:
            filters (dict, optional): A dictionary defining filters for different categories or specialties.
                                    If None or empty, no filtering will be applied.
            verbose (bool, optional): If True, enable info-level logging. Defaults to False.
            use_generating_functions (bool, optional): If True, use the new generating function approach
                                                      for better performance. Defaults to True.
        """
        self.filters = filters if filters else {}
        self.verbose = verbose
        self.use_generating_functions = use_generating_functions
        self.metrics = {}  # Add metrics dictionary to store metadata

        if verbose:
            # Configure logging for Jupyter notebook compatibility
            import logging
            import sys

            # Create logger
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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

    # -------------------------
    # Shared helper utilities
    # -------------------------
    def _validate_only_kwargs(self, kwargs: Dict, allowed: set, context: str) -> None:
        unexpected = set(kwargs.keys()) - allowed
        if unexpected:
            raise ValueError(
                f"{context} only accepts these parameters: {allowed}. Remove these unexpected parameters: {unexpected}"
            )

    def _ensure_required_kwargs(self, kwargs: Dict, required: set, context: str) -> Dict:
        missing = [k for k in required if kwargs.get(k) is None]
        if missing:
            raise ValueError(
                f"{context} requires these parameters: {missing}."
            )
        return {k: kwargs[k] for k in required}

    def _iter_prediction_inputs(self, prediction_context: Dict):
        """Yield (filter_key, resolved_prediction_time, arrival_rates_np).

        Centralizes validation, time resolution, and array conversion.
        """
        for filter_key, filter_values in prediction_context.items():
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

            yield filter_key, prediction_time, np.array(arrival_rates)

    def _get_window_and_interval_hours(self):
        """Return (prediction_window_hours, interval_hours, NTimes)."""
        return (
            self.prediction_window_hours,
            self.yta_time_interval_hours,
            self.NTimes,
        )

    def _default_max_value(self, arrival_rates, floor: int = 20, cap: int = 200) -> int:
        """Unified default max_value based on epsilon and max arrival rate.

        Applies a floor and cap to ensure stable PMF support.
        """
        try:
            mv = int(poisson.ppf(1 - self.epsilon, float(np.max(arrival_rates))))
        except Exception:
            mv = floor
        mv = max(floor, mv)
        mv = min(cap, mv)
        return mv

    def _pmf_to_dataframe(self, probabilities: np.ndarray) -> pd.DataFrame:
        """Convert PMF array to standardized DataFrame indexed by 'sum' with 'agg_proba'."""
        result_df = pd.DataFrame({"sum": range(len(probabilities)), "agg_proba": probabilities})
        result_df = result_df[result_df["agg_proba"] > 1e-10]
        if len(result_df) > 0:
            result_df["agg_proba"] = result_df["agg_proba"] / result_df["agg_proba"].sum()
        return result_df.set_index("sum")

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
        self,
        df,
        prediction_window: timedelta,
        yta_time_interval: timedelta,
        prediction_times,
        num_days,
    ):
        """Calculate parameters required for the model.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to process.
        prediction_window : timedelta
            The total prediction window for prediction.
        yta_time_interval : timedelta
            The interval for splitting the prediction window.
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
        Ntimes = int(prediction_window / yta_time_interval)

        # Pass original type to time_varying_arrival_rates
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
                        + i * yta_time_interval
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
        prediction_window: timedelta,
        yta_time_interval: timedelta,
        prediction_times: List[float],
        num_days: int,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
    ) -> "IncomingAdmissionPredictor":
        """Fit the model to the training data.

        Parameters
        ----------
        train_df : pandas.DataFrame
            The training dataset with historical admission data.
        prediction_window : timedelta
            The prediction window as a timedelta object.
        yta_time_interval : timedelta
            The interval for splitting the prediction window as a timedelta object.
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
        IncomingAdmissionPredictor
            The instance itself, fitted with the training data.

        Raises
        ------
        TypeError
            If prediction_window or yta_time_interval are not timedelta objects.
        ValueError
            If prediction_window/yta_time_interval is not greater than 1.
        """

        # Validate inputs
        if not isinstance(prediction_window, timedelta):
            raise TypeError("prediction_window must be a timedelta object")
        if not isinstance(yta_time_interval, timedelta):
            raise TypeError("yta_time_interval must be a timedelta object")

        if prediction_window.total_seconds() <= 0:
            raise ValueError("prediction_window must be positive")
        if yta_time_interval.total_seconds() <= 0:
            raise ValueError("yta_time_interval must be positive")
        if yta_time_interval.total_seconds() > 4 * 3600:  # 4 hours in seconds
            warnings.warn("yta_time_interval appears to be longer than 4 hours")

        # Validate the ratio makes sense
        ratio = prediction_window / yta_time_interval
        if int(ratio) == 0:
            raise ValueError(
                "prediction_window must be significantly larger than yta_time_interval"
            )

        # Store original types
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
                f"{self.__class__.__name__} trained for these times: {prediction_times}"
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

        # Cache commonly used interval metadata
        self.NTimes = int(self.prediction_window / self.yta_time_interval)
        self.prediction_window_hours = self.prediction_window.total_seconds() / 3600
        self.yta_time_interval_hours = self.yta_time_interval.total_seconds() / 3600

        return self

    def get_weights(self):
        """Get the weights computed by the fit method.

        Returns
        -------
        dict
            The weights computed during model fitting.
        """
        return self.weights

    @abstractmethod
    def predict(self, prediction_context: Dict, **kwargs) -> Dict:
        """Predict the number of admissions for the given context.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        **kwargs
            Additional keyword arguments specific to the prediction method.

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
        pass


class DirectAdmissionPredictor(IncomingAdmissionPredictor):
    """A predictor that assumes every arrival is admitted immediately.

    This predictor uses only the arrival rates calculated from historical data
    and assumes 100% admission probability for all arrivals. No survival curves
    or parametric models are used - it's a direct Poisson distribution based
    on the arrival rates.

    Parameters
    ----------
    filters : dict, optional
        Optional filters for data categorization. If None, no filtering is applied.
    verbose : bool, default=False
        Whether to enable verbose logging.

    Notes
    -----
    This is the simplest predictor that directly uses arrival rates without
    any admission probability adjustments. It sums all arrival rates across
    time intervals and creates a single Poisson distribution, making it useful
    for scenarios where immediate admission is expected or as a baseline
    for comparison with more complex models.
    """

    def predict(self, prediction_context: Dict, **kwargs) -> Dict:
        """Predict the number of admissions assuming 100% admission rate.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        **kwargs
            Additional keyword arguments for prediction configuration:

            max_value : int, default=50
                Maximum value for the discrete distribution support.

        Returns
        -------
        dict
            A dictionary with predictions for each specified context.

        Raises
        ------
        ValueError
            If filter key is not recognized or prediction_time is not provided.
            If parametric parameters (x1, y1, x2, y2) are provided (not used by direct predictor).
        KeyError
            If required keys are missing from the prediction context.
        """
        # Be lenient: ignore unrelated kwargs (e.g., parametric args passed by a higher-level API)

        from scipy import stats

        # Extract parameters from kwargs with defaults (unified default if not provided)
        max_value = kwargs.get("max_value")

        predictions = {}

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(prediction_context):

                if self.use_generating_functions:
                    # Use the new generating function approach for consistency
                    # For direct case, admission probabilities are all 1.0 (100% admission rate)
                    admission_probs = np.ones(len(arrival_rates))
                    
                    gf = AdmissionGeneratingFunction(
                        arrival_rates, admission_probs, method='direct'
                    )
                    mv = max_value if max_value is not None else self._default_max_value(arrival_rates)
                    predictions[filter_key] = gf.get_distribution(max_value=mv)
                else:
                    # Use the original approach for backward compatibility
                    total_arrival_rate = arrival_rates.sum()
                    poisson_dist = stats.poisson(total_arrival_rate)
                    mv = max_value if max_value is not None else self._default_max_value(arrival_rates)
                    x_values = np.arange(mv)
                    probabilities = poisson_dist.pmf(x_values)
                    predictions[filter_key] = self._pmf_to_dataframe(probabilities)

                if self.verbose:
                    total_arrival_rate = arrival_rates.sum()
                    self.logger.info(
                        f"Direct prediction for {filter_key} at {prediction_time}: "
                        f"Expected admissions = {total_arrival_rate:.2f}"
                    )

        return predictions


class ParametricIncomingAdmissionPredictor(IncomingAdmissionPredictor):
    """A predictor for estimating hospital admissions using parametric curves.

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

    def predict(self, prediction_context: Dict, **kwargs) -> Dict:
        """Predict the number of admissions for the given context using parametric curves.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        **kwargs
            Additional keyword arguments for parametric curve configuration:

            x1 : float, required
                The x-coordinate of the first transition point on the aspirational curve,
                where the growth phase ends and the decay phase begins.
            y1 : float, required
                The y-coordinate of the first transition point (x1), representing the target
                proportion of patients admitted by time x1.
            x2 : float, required
                The x-coordinate of the second transition point on the curve, beyond which
                all but a few patients are expected to be admitted.
            y2 : float, required
                The y-coordinate of the second transition point (x2), representing the target
                proportion of patients admitted by time x2.
            max_value : int, optional, default=50
                Maximum value for the discrete distribution support.

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
        # Validate/collect kwargs
        required = {'x1', 'y1', 'x2', 'y2'}
        params = self._ensure_required_kwargs(kwargs, required, self.__class__.__name__)
        self._validate_only_kwargs(kwargs, required | {'max_value'}, context=self.__class__.__name__)

        predictions = {}

        # Use cached metadata
        prediction_window_hours, yta_time_interval_hours, NTimes = self._get_window_and_interval_hours()

        # Calculate theta, probability of admission in prediction window
        # for each time interval, calculate time remaining before end of window
        time_remaining_before_end_of_window = prediction_window_hours - np.arange(
            0, prediction_window_hours, yta_time_interval_hours
        )

        theta = get_y_from_aspirational_curve(
            time_remaining_before_end_of_window, params['x1'], params['y1'], params['x2'], params['y2']
        )

        max_value = kwargs.get('max_value')

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(prediction_context):
            if self.use_generating_functions:
                # Use the new generating function approach for better performance
                gf = AdmissionGeneratingFunction(
                    arrival_rates, theta, method='parametric'
                )
                mv = max_value if max_value is not None else self._default_max_value(arrival_rates)
                predictions[filter_key] = gf.get_distribution(max_value=mv)
            else:
                # Use the original approach for backward compatibility
                predictions[filter_key] = poisson_binom_generating_function(
                    NTimes, arrival_rates, theta, self.epsilon
                )

        return predictions


class EmpiricalIncomingAdmissionPredictor(IncomingAdmissionPredictor):
    """A predictor that uses empirical survival curves instead of parameterised curves.

    This predictor inherits all the arrival rate calculation and filtering logic from
    IncomingAdmissionPredictor but uses empirical survival probabilities and convolution
    of Poisson distributions for prediction instead of the Poisson-binomial approach.

    The survival curve is automatically calculated from the training data during the
    fit process by analysing time-to-admission patterns.

    Parameters
    ----------
    filters : dict, optional
        Optional filters for data categorization. If None, no filtering is applied.
    verbose : bool, default=False
        Whether to enable verbose logging.

    Attributes
    ----------
    survival_df : pandas.DataFrame
        The survival data calculated from training data, containing time-to-event
        information for empirical probability calculations.
    """

    def __init__(self, filters=None, verbose=False, use_generating_functions=True):
        """Initialize the EmpiricalIncomingAdmissionPredictor."""
        super().__init__(filters, verbose, use_generating_functions)
        self.survival_df = None

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window,
        yta_time_interval,
        prediction_times: List[float],
        num_days: int,
        epsilon=10**-7,
        y=None,
        start_time_col="arrival_datetime",
        end_time_col="departure_datetime",
    ) -> "EmpiricalIncomingAdmissionPredictor":
        """Fit the model to the training data and calculate empirical survival curve.

        Parameters
        ----------
        train_df : pandas.DataFrame
            The training dataset with historical admission data.
            Expected to have start_time_col as the index and end_time_col as a column.
            Alternatively, both can be regular columns.
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
        start_time_col : str, default='arrival_datetime'
            Name of the column containing the start time (e.g., arrival time).
            Expected to be the DataFrame index, but can also be a regular column.
        end_time_col : str, default='departure_datetime'
            Name of the column containing the end time (e.g., departure time).

        Returns
        -------
        EmpiricalIncomingAdmissionPredictor
            The instance itself, fitted with the training data.
        """
        # Calculate survival curve from training data using existing function
        # Handle case where start_time_col is in the index
        if start_time_col in train_df.columns:
            # start_time_col is a regular column
            df_for_survival = train_df
        else:
            # start_time_col is likely the index, reset it to make it a column
            df_for_survival = train_df.reset_index()
            # Verify that start_time_col is now available
            if start_time_col not in df_for_survival.columns:
                raise ValueError(
                    f"Column '{start_time_col}' not found in DataFrame columns or index"
                )

        self.survival_df = calculate_survival_curve(
            df_for_survival, start_time_col=start_time_col, end_time_col=end_time_col
        )

        # Verify survival curve was calculated and saved successfully
        if self.survival_df is None or len(self.survival_df) == 0:
            raise RuntimeError("Failed to calculate survival curve from training data")

        # Ensure train_df has start_time_col as index for parent fit method
        if start_time_col in train_df.columns:
            train_df = train_df.set_index(start_time_col)

        # Call parent fit method to handle arrival rate calculation and validation
        super().fit(
            train_df,
            prediction_window,
            yta_time_interval,
            prediction_times,
            num_days,
            epsilon=epsilon,
            y=y,
        )

        if self.verbose:
            self.logger.info(
                f"EmpiricalIncomingAdmissionPredictor has been fitted with survival curve containing {len(self.survival_df)} time points"
            )

        return self

    def get_survival_curve(self):
        """Get the survival curve calculated during fitting.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the survival curve with columns:
            - time_hours: Time points in hours
            - survival_probability: Survival probabilities at each time point
            - event_probability: Event probabilities (1 - survival_probability)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.survival_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.survival_df.copy()

    def _calculate_survival_probabilities(self, prediction_window, yta_time_interval):
        """Calculate survival probabilities for each time interval.

        Parameters
        ----------
        prediction_window : int or timedelta
            The prediction window.
        yta_time_interval : int or timedelta
            The time interval for splitting the prediction window.

        Returns
        -------
        numpy.ndarray
            Array of admission probabilities for each time interval.
        """
        # Use cached metadata for consistency
        prediction_window_hours, yta_time_interval_hours, NTimes = self._get_window_and_interval_hours()

        # Calculate admission probabilities for each time interval
        probabilities = []
        for i in range(NTimes):
            # Time remaining until end of prediction window
            time_remaining = prediction_window_hours - (i * yta_time_interval_hours)

            # Interpolate survival probability from survival curve
            if time_remaining <= 0:
                prob_admission = (
                    1.0  # If time remaining is 0 or negative, probability is 1
                )
            else:
                # Find the survival probability at this time point
                # Linear interpolation between points in survival curve
                survival_curve = self.survival_df
                if time_remaining >= survival_curve["time_hours"].max():
                    # If time is beyond our data, use the last survival probability
                    survival_prob = survival_curve["survival_probability"].iloc[-1]
                elif time_remaining <= survival_curve["time_hours"].min():
                    # If time is before our data, use the first survival probability
                    survival_prob = survival_curve["survival_probability"].iloc[0]
                else:
                    # Interpolate between points
                    survival_prob = np.interp(
                        time_remaining,
                        survival_curve["time_hours"],
                        survival_curve["survival_probability"],
                    )

                # Probability of admission = 1 - survival probability
                prob_admission = 1 - survival_prob

            probabilities.append(prob_admission)

        return np.array(probabilities)

    def _convolve_poisson_distributions(
        self, arrival_rates, probabilities, max_value=20
    ):
        """Convolve Poisson distributions for each time interval.

        This method can use either the new generating function approach or the original
        manual convolution method based on the use_generating_functions flag.

        Parameters
        ----------
        arrival_rates : numpy.ndarray
            Array of arrival rates for each time interval.
        probabilities : numpy.ndarray
            Array of admission probabilities for each time interval.
        max_value : int, default=20
            Maximum value for the discrete distribution support.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'sum' and 'agg_proba' columns representing the final distribution.
        """
        if self.use_generating_functions:
            # Use the new generating function approach for efficiency
            gf = AdmissionGeneratingFunction(
                arrival_rates, probabilities, method='empirical'
            )
            return gf.get_distribution(max_value=max_value)
        else:
            # Use the original approach for backward compatibility
            from scipy import stats

            # Create weighted Poisson distributions for each time interval
            weighted_rates = arrival_rates * probabilities
            poisson_dists = [stats.poisson(rate) for rate in weighted_rates]

            # Get PMF for each distribution
            x = np.arange(max_value)
            pmfs = [dist.pmf(x) for dist in poisson_dists]

            # Convolve all distributions together
            if len(pmfs) == 0:
                # Handle edge case of no distributions
                combined_pmf = np.zeros(max_value)
                combined_pmf[0] = 1.0  # All probability at 0
            else:
                combined_pmf = pmfs[0]
                for pmf in pmfs[1:]:
                    combined_pmf = np.convolve(combined_pmf, pmf)

            # Create result DataFrame
            result_df = pd.DataFrame(
                {"sum": range(len(combined_pmf)), "agg_proba": combined_pmf}
            )

            # Filter out near-zero probabilities and normalize
            result_df = result_df[result_df["agg_proba"] > 1e-10]
            result_df["agg_proba"] = result_df["agg_proba"] / result_df["agg_proba"].sum()

            return result_df.set_index("sum")

    def predict(self, prediction_context: Dict, **kwargs) -> Dict:
        """Predict the number of admissions using empirical survival curves.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        **kwargs
            Additional keyword arguments for prediction configuration:

            max_value : int, default=20
                Maximum value for the discrete distribution support.

        Returns
        -------
        dict
            A dictionary with predictions for each specified context.

        Raises
        ------
        ValueError
            If filter key is not recognized or prediction_time is not provided.
            If parametric parameters (x1, y1, x2, y2) are provided (not used by empirical predictor).
        KeyError
            If required keys are missing from the prediction context.
        RuntimeError
            If survival_df was not provided during fitting.
        """
        # Be lenient: ignore unrelated kwargs (e.g., parametric args passed by a higher-level API)
        
        if self.survival_df is None:
            raise RuntimeError(
                "No survival data available. Please call fit() method first to calculate survival curve from training data."
            )

        # Extract parameters from kwargs with unified default if not provided
        max_value = kwargs.get("max_value")

        predictions = {}

        # Calculate survival probabilities once (they're the same for all contexts)
        survival_probabilities = self._calculate_survival_probabilities(
            self.prediction_window, self.yta_time_interval
        )

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(prediction_context):

                # Generate prediction using convolution approach
                mv = max_value if max_value is not None else self._default_max_value(arrival_rates)
                predictions[filter_key] = self._convolve_poisson_distributions(
                    arrival_rates, survival_probabilities, max_value=mv
                )

        return predictions
