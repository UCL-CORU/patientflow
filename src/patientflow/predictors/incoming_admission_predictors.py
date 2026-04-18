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
    Uses generating functions for all predictors; the earlier flag is deprecated and ignored.

DirectAdmissionPredictor : IncomingAdmissionPredictor
    Simplest predictor that assumes every arrival is admitted immediately. Uses direct
    Poisson distribution based on total arrival rates without any admission probability
    adjustments. Implemented via generating functions.

ParametricIncomingAdmissionPredictor : IncomingAdmissionPredictor
    Predicts the number of admissions within a given prediction window based on historical
    data and Poisson-binomial distribution using parametric aspirational curves.
    Now uses efficient generating functions for better performance.

EmpiricalIncomingAdmissionPredictor : IncomingAdmissionPredictor
    Predicts the number of admissions using empirical survival curves and convolution
    of Poisson distributions; implemented via generating functions.

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

Assumptions
-----------
Parametric incoming-admissions use a simple per-slice Poisson arrivals model with
independent filtering. The following assumptions are required for the simplified
parametric route to be exact:

- Symbols and units:
    - λ_t: expected arrivals within time-slice t (the value produced by
      `time_varying_arrival_rates` for that slice; units match the slice length).
    - θ_t: probability an arrival in slice t is admitted within the prediction
      window (computed via `get_y_from_aspirational_curve`).

- Arrivals within each time-slice t follow a Poisson process with rate λ_t.

- Each arrival is independently admitted within the prediction window with
  probability θ_t, which is constant across that slice (given the model inputs).
  This independent filtering is sometimes called "Poisson thinning".

- Slices are independent for this filtering.

Under these assumptions, admitted arrivals in slice t are Poisson(λ_t θ_t) and
the total admitted count is Poisson(Σ_t λ_t θ_t). Intuition: filtering arrivals
with probability θ_t reduces the effective rate from λ_t to λ_t θ_t.

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

from scipy.stats import poisson


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

    def __init__(self, arrival_rates, admission_probs, method="empirical"):
        self.arrival_rates = np.array(arrival_rates)
        self.admission_probs = np.array(admission_probs)
        self.method = method

        if len(self.arrival_rates) != len(self.admission_probs):
            raise ValueError(
                "arrival_rates and admission_probs must have the same length"
            )

    def get_distribution(self, max_value):
        """Get the probability distribution for the specified method.

        Parameters
        ----------
        max_value : int
            Maximum value for the discrete distribution support.
            Typically determined using epsilon via _default_max_value(),
            but can be provided explicitly for custom truncation.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'sum' and 'agg_proba' columns representing the distribution
        """
        if self.method == "direct":
            return self._direct_distribution(max_value)
        elif self.method == "empirical":
            return self._empirical_distribution(max_value)
        elif self.method == "parametric":
            return self._parametric_distribution(max_value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _direct_distribution(self, max_value):
        """Simple Poisson distribution for direct method.

        Parameters
        ----------
        max_value : int
            Maximum value for the discrete distribution support.
            Typically determined using epsilon, but can be provided explicitly.
        """
        total_rate = np.sum(self.arrival_rates)
        x_values = np.arange(max_value)
        probabilities = poisson.pmf(x_values, total_rate)

        result_df = pd.DataFrame(
            {"sum": range(len(probabilities)), "agg_proba": probabilities}
        )

        return result_df.set_index("sum")

    def _empirical_distribution(self, max_value):
        """Weighted Poisson convolution for empirical method.

        Parameters
        ----------
        max_value : int
            Maximum value for the discrete distribution support.
            Typically determined using epsilon, but can be provided explicitly.
        """
        # Create weighted Poisson distributions for each time interval
        weighted_rates = self.arrival_rates * self.admission_probs
        x_values = np.arange(max_value)

        # Start with first distribution
        probabilities = poisson.pmf(x_values, weighted_rates[0])

        # Convolve with remaining distributions
        for rate in weighted_rates[1:]:
            pmf_i = poisson.pmf(x_values, rate)
            probabilities = np.convolve(probabilities, pmf_i)[:max_value]

        result_df = pd.DataFrame(
            {"sum": range(len(probabilities)), "agg_proba": probabilities}
        )

        return result_df.set_index("sum")

    def _parametric_distribution(self, max_value):
        """Deprecated: parametric GF route no longer used."""
        raise NotImplementedError(
            "Parametric GF route removed; use simplified Poisson route in predictor."
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
        self.filters = filters if filters else {}
        self.verbose = verbose
        # Always use generating-function path; keep flag only for backward compatibility
        # If user explicitly sets False, warn and proceed with GF implementation
        self.use_generating_functions = True
        if use_generating_functions is False:
            warnings.warn(
                "use_generating_functions=False is deprecated and ignored; generating-function implementation is always used.",
                DeprecationWarning,
                stacklevel=2,
            )
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

    def _ensure_required_kwargs(
        self, kwargs: Dict, required: set, context: str
    ) -> Dict:
        missing = [k for k in required if kwargs.get(k) is None]
        if missing:
            raise ValueError(f"{context} requires these parameters: {missing}.")
        return {k: kwargs[k] for k in required}

    def _resolve_prediction_window(self, prediction_window):
        """Return the effective prediction window.

        Resolution order:
        1. Explicit ``prediction_window`` argument if provided.
        2. ``prediction_window`` stored on the model at ``fit()`` time
           (deprecated; emits `DeprecationWarning`).
        3. Raise `ValueError`.
        """
        if prediction_window is not None:
            return prediction_window
        fit_window = getattr(self, "_deprecated_fit_prediction_window", None)
        if fit_window is not None:
            warnings.warn(
                "Relying on prediction_window stored at fit() time is deprecated. "
                "Pass prediction_window explicitly to predict() / predict_mean().",
                DeprecationWarning,
                stacklevel=3,
            )
            return fit_window
        raise ValueError(
            "prediction_window is required. Pass it as a keyword argument to "
            "predict() / predict_mean()."
        )

    def _snap_to_interval_boundary(self, prediction_time):
        """Snap a ``(hour, minute)`` tuple to the nearest ``yta_time_interval`` boundary within the 24-hour cycle."""
        interval_minutes = self.yta_time_interval.total_seconds() / 60
        if interval_minutes <= 0:
            return prediction_time
        total_minutes = prediction_time[0] * 60 + prediction_time[1]
        snapped = round(total_minutes / interval_minutes) * interval_minutes
        snapped_int = int(snapped) % (24 * 60)
        return (snapped_int // 60, snapped_int % 60)

    def _normalize_prediction_time(self, prediction_time):
        """Coerce a prediction_time into a ``(hour, minute)`` tuple."""
        if isinstance(prediction_time, (list, np.ndarray)):
            return tuple(prediction_time)
        if isinstance(prediction_time, (int, float)):
            return (int(prediction_time), 0)
        return prediction_time

    def _iter_prediction_inputs(self, prediction_context: Dict, prediction_window):
        """Yield ``(filter_key, resolved_prediction_time, arrival_rates_np)``.

        Slices ``Ntimes`` intervals from the stored full 24-hour arrival-rate
        dictionary, starting at the requested prediction time (snapped to the
        nearest ``yta_time_interval`` boundary).
        """
        Ntimes = int(prediction_window / self.yta_time_interval)
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

            prediction_time = self._normalize_prediction_time(prediction_time)
            snapped_prediction_time = self._snap_to_interval_boundary(prediction_time)
            if snapped_prediction_time != prediction_time:
                warnings.warn(
                    f"Requested prediction_time {prediction_time} does not fall on a "
                    f"yta_time_interval boundary; snapping to {snapped_prediction_time}.",
                    UserWarning,
                    stacklevel=3,
                )

            arrival_rates_dict = self.weights[filter_key].get("arrival_rates_dict")
            if arrival_rates_dict is None:
                raise ValueError(
                    f"No arrival_rates_dict found under filter '{filter_key}'. "
                    "Has the model been fit?"
                )

            hr, mn = snapped_prediction_time
            try:
                arrival_rates = [
                    arrival_rates_dict[
                        (
                            datetime(1970, 1, 1, hr, mn) + i * self.yta_time_interval
                        ).time()
                    ]
                    for i in range(Ntimes)
                ]
            except KeyError as e:
                raise ValueError(
                    f"No arrival_rates found for filter '{filter_key}' at "
                    f"prediction_time {snapped_prediction_time}: missing key {e}"
                )

            yield filter_key, snapped_prediction_time, np.array(arrival_rates)

    def _get_window_and_interval_hours(self, prediction_window):
        """Return ``(prediction_window_hours, interval_hours, NTimes)`` for the given window."""
        prediction_window_hours = prediction_window.total_seconds() / 3600
        NTimes = int(prediction_window / self.yta_time_interval)
        return (
            prediction_window_hours,
            self.yta_time_interval_hours,
            NTimes,
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
        result_df = pd.DataFrame(
            {"sum": range(len(probabilities)), "agg_proba": probabilities}
        )
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
        yta_time_interval: timedelta,
        num_days,
    ):
        """Calculate the full 24-hour arrival-rate dictionary for the given data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to process.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets.
        num_days : int
            Number of days over which to calculate time-varying arrival rates.

        Returns
        -------
        dict
            A dictionary with a single key ``"arrival_rates_dict"`` whose value
            is a dictionary mapping ``datetime.time`` to arrival rate,
            covering the full 24-hour cycle at ``yta_time_interval`` granularity.
        """
        arrival_rates_dict = time_varying_arrival_rates(
            df, yta_time_interval, num_days, verbose=self.verbose
        )
        return {"arrival_rates_dict": arrival_rates_dict}

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window: Optional[timedelta] = None,
        yta_time_interval: Optional[timedelta] = None,
        prediction_times: Optional[List[float]] = None,
        num_days: Optional[int] = None,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
    ) -> "IncomingAdmissionPredictor":
        """Fit the model to the training data.

        The underlying arrival-rate calculation is independent of the prediction
        window and the set of prediction times; these are now supplied at
        ``predict()`` time. ``prediction_window`` and ``prediction_times`` are
        therefore deprecated as ``fit()`` parameters: if provided, a
        `DeprecationWarning` is emitted and the values are stored as
        fall-back defaults for ``predict()``.

        Parameters
        ----------
        train_df : pandas.DataFrame
            The training dataset with historical admission data.
        prediction_window : timedelta, optional
            Deprecated. Prefer passing ``prediction_window`` to
            `predict()` / `predict_mean()` instead. If provided, will
            be stored as a fall-back default for predict-time use.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets. Required.
        prediction_times : list, optional
            Deprecated. Prediction times are no longer needed at fit time; any
            prediction time can be served at predict time (snapped to the
            nearest ``yta_time_interval`` boundary). Retained for backward
            compatibility only.
        num_days : int
            The number of days that the train_df spans. Required.
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
            If ``yta_time_interval`` is missing or not a timedelta, or if
            ``prediction_window`` (when supplied) is not a timedelta, or if
            ``num_days`` is missing.
        ValueError
            If ``yta_time_interval`` is not positive, or if
            ``prediction_window`` (when supplied) is not positive or is not
            significantly larger than ``yta_time_interval``.
        """

        # Validate required inputs
        if yta_time_interval is None:
            raise TypeError("yta_time_interval is required")
        if not isinstance(yta_time_interval, timedelta):
            raise TypeError("yta_time_interval must be a timedelta object")
        if yta_time_interval.total_seconds() <= 0:
            raise ValueError("yta_time_interval must be positive")
        if yta_time_interval.total_seconds() > 4 * 3600:  # 4 hours in seconds
            warnings.warn("yta_time_interval appears to be longer than 4 hours")
        if num_days is None:
            raise TypeError("num_days is required")

        # Deprecation handling for prediction_window / prediction_times
        if prediction_window is not None:
            warnings.warn(
                "Passing prediction_window to fit() is deprecated; pass it to "
                "predict() / predict_mean() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not isinstance(prediction_window, timedelta):
                raise TypeError("prediction_window must be a timedelta object")
            if prediction_window.total_seconds() <= 0:
                raise ValueError("prediction_window must be positive")
            ratio = prediction_window / yta_time_interval
            if int(ratio) == 0:
                raise ValueError(
                    "prediction_window must be significantly larger than yta_time_interval"
                )
        if prediction_times is not None:
            warnings.warn(
                "Passing prediction_times to fit() is deprecated; any prediction "
                "time can be served at predict time. This argument is retained "
                "for backward compatibility only and will be removed in a future "
                "release.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store required metadata
        self.yta_time_interval = yta_time_interval
        self.yta_time_interval_hours = yta_time_interval.total_seconds() / 3600
        self.epsilon = epsilon

        # Store deprecated fit-time values as predict-time fall-backs
        self._deprecated_fit_prediction_window = prediction_window
        if prediction_times is not None:
            normalized_times = [
                tuple(x)
                if isinstance(x, (list, np.ndarray))
                else (x, 0)
                if isinstance(x, (int, float))
                else x
                for x in prediction_times
            ]
        else:
            normalized_times = None
        self._deprecated_fit_prediction_times = normalized_times

        # Maintain legacy attributes for backward compatibility with downstream
        # "has been fit" checks and tests that read these directly. They are not
        # used by the prediction code path.
        self.prediction_window = prediction_window
        self.prediction_times = normalized_times
        self.prediction_window_hours: Optional[float] = (
            prediction_window.total_seconds() / 3600
            if prediction_window is not None
            else None
        )
        self.NTimes: Optional[int] = (
            int(prediction_window / yta_time_interval)
            if prediction_window is not None
            else None
        )

        # Initialise weights with the full 24-hour arrival-rate dictionary
        self.weights = {}
        if self.filters:
            for spec, filters in self.filters.items():
                self.weights[spec] = self._calculate_parameters(
                    self.filter_dataframe(train_df, filters),
                    yta_time_interval,
                    num_days,
                )
        else:
            self.weights["unfiltered"] = self._calculate_parameters(
                train_df,
                yta_time_interval,
                num_days,
            )

        if self.verbose:
            if normalized_times is not None:
                self.logger.info(
                    f"{self.__class__.__name__} fit-time prediction_times "
                    f"(deprecated): {normalized_times}"
                )
            if prediction_window is not None:
                self.logger.info(
                    f"fit-time prediction window (deprecated) of "
                    f"{prediction_window} after the time of prediction"
                )
            self.logger.info(
                f"Time interval of {yta_time_interval} used to bucket arrival rates."
            )
            self.logger.info(f"The error value for prediction will be {epsilon}")
            self.logger.info(
                "To see the weights saved by this model, use the get_weights() method"
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

    @abstractmethod
    def _get_admission_probabilities(self, **kwargs) -> np.ndarray:
        """Get admission probabilities for each time interval.

        This is an abstract method that must be implemented by subclasses.
        Each subclass implements its own logic for calculating admission probabilities
        based on their specific approach (direct, parametric, or empirical).

        Parameters
        ----------
        **kwargs
            Additional keyword arguments specific to the prediction method.

        Returns
        -------
        numpy.ndarray
            Array of admission probabilities for each time interval.
        """
        pass

    def predict_mean(
        self,
        prediction_context: Dict,
        prediction_window: Optional[timedelta] = None,
        **kwargs,
    ) -> float:
        """Return just the Poisson mean (expected value) for each context.

        This method extracts the underlying Poisson mean without computing
        the full probability distribution, making it suitable for later generation
        or when only the expected value is needed.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        prediction_window : timedelta, optional
            The prediction window. Required. If omitted, the value (if any)
            supplied to ``fit()`` is used and a `DeprecationWarning`
            is emitted.
        **kwargs
            Additional keyword arguments specific to the prediction method.
            These are passed to the subclass-specific ``_get_admission_probabilities`` method.

        Returns
        -------
        float
            The Poisson mean (expected value) for the single context specified in prediction_context.

        Raises
        ------
        ValueError
            If filter key is not recognized, ``prediction_time`` is not provided,
            or ``prediction_window`` cannot be resolved.
        KeyError
            If required keys are missing from the prediction context.
        """
        prediction_window = self._resolve_prediction_window(prediction_window)

        admission_probs = self._get_admission_probabilities(
            prediction_window=prediction_window, **kwargs
        )

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            prediction_context, prediction_window
        ):
            poisson_mean = float(
                np.sum(np.array(arrival_rates) * np.array(admission_probs))
            )
            return poisson_mean

        raise ValueError("No valid prediction context provided")


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

    def predict(
        self,
        prediction_context: Dict,
        prediction_window: Optional[timedelta] = None,
        **kwargs,
    ) -> Dict:
        """Predict the number of admissions assuming 100% admission rate.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        prediction_window : timedelta, optional
            The prediction window over which admissions are accumulated.
            Required. If omitted, the value (if any) supplied to ``fit()`` is
            used and a `DeprecationWarning` is emitted.
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
            If filter key is not recognized, ``prediction_time`` is not provided,
            or ``prediction_window`` cannot be resolved.
        KeyError
            If required keys are missing from the prediction context.
        """
        # Be lenient: ignore unrelated kwargs (e.g., parametric args passed by a higher-level API)
        prediction_window = self._resolve_prediction_window(prediction_window)

        max_value = kwargs.get("max_value")

        predictions = {}

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            prediction_context, prediction_window
        ):
            # For direct case, admission probabilities are all 1.0 (100% admission rate)
            admission_probs = np.ones(len(arrival_rates))

            gf = AdmissionGeneratingFunction(
                arrival_rates, admission_probs, method="direct"
            )
            mv = (
                max_value
                if max_value is not None
                else self._default_max_value(arrival_rates)
            )
            predictions[filter_key] = gf.get_distribution(max_value=mv)

            if self.verbose:
                total_arrival_rate = arrival_rates.sum()
                self.logger.info(
                    f"Direct prediction for {filter_key} at {prediction_time}: "
                    f"Expected admissions = {total_arrival_rate:.2f}"
                )

        return predictions

    def _get_admission_probabilities(
        self, prediction_window: Optional[timedelta] = None, **kwargs
    ) -> np.ndarray:
        """Get admission probabilities for direct method (always 1.0).

        For the direct predictor, all arrivals are assumed to be admitted immediately,
        so admission probabilities are always 1.0 for all time intervals.

        Parameters
        ----------
        prediction_window : timedelta, optional
            The prediction window. Used to determine the number of time
            intervals. Resolved via `_resolve_prediction_window()` if
            omitted.
        **kwargs
            Additional keyword arguments (ignored for direct method).

        Returns
        -------
        numpy.ndarray
            Array of ones with length equal to the number of time intervals.
        """
        prediction_window = self._resolve_prediction_window(prediction_window)
        _, _, NTimes = self._get_window_and_interval_hours(prediction_window)
        return np.ones(NTimes)


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

    Assumptions for simplified parametric route
    -------------------------------------------
    - Symbols: λ_t = expected arrivals in slice t; θ_t = probability an arrival in slice t is
      admitted within the prediction window.
    - Arrivals per time-slice t follow Poisson(λ_t).
    - Within a slice, each arrival is admitted independently with probability θ_t (constant per slice).
      This independent filtering is often called “Poisson thinning”.
    - Slices are independent.
    Result: admitted in slice t ~ Poisson(λ_t θ_t); total admitted ~ Poisson(Σ_t λ_t θ_t).
    """

    def predict(
        self,
        prediction_context: Dict,
        prediction_window: Optional[timedelta] = None,
        **kwargs,
    ) -> Dict:
        """Predict the number of admissions for the given context using parametric curves.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        prediction_window : timedelta, optional
            The prediction window over which admissions are accumulated.
            Required. If omitted, the value (if any) supplied to ``fit()`` is
            used and a `DeprecationWarning` is emitted.
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
            If filter key is not recognized, ``prediction_time`` is not provided,
            or ``prediction_window`` cannot be resolved.
        KeyError
            If required keys are missing from the prediction context.

        Assumptions
        -----------
        Uses the Poisson arrivals with independent filtering (often called "Poisson thinning"):
        - λ_t: expected arrivals in slice t; θ_t: probability of admission within window for arrivals in slice t
        - Arrivals per slice t follow Poisson(λ_t), slices are independent
        - Each arrival is admitted independently with probability θ_t (constant per slice)
        Result: admitted in slice t ~ Poisson(λ_t θ_t); total admitted ~ Poisson(Σ_t λ_t θ_t)
        """
        # Validate/collect kwargs
        required = {"x1", "y1", "x2", "y2"}
        params = self._ensure_required_kwargs(kwargs, required, self.__class__.__name__)
        self._validate_only_kwargs(
            kwargs, required | {"max_value"}, context=self.__class__.__name__
        )

        prediction_window = self._resolve_prediction_window(prediction_window)

        predictions = {}

        prediction_window_hours, yta_time_interval_hours, NTimes = (
            self._get_window_and_interval_hours(prediction_window)
        )

        # Calculate theta, probability of admission in prediction window,
        # for each time interval based on time remaining before end of window
        time_remaining_before_end_of_window = prediction_window_hours - np.arange(
            0, prediction_window_hours, yta_time_interval_hours
        )

        theta = get_y_from_aspirational_curve(
            time_remaining_before_end_of_window,
            params["x1"],
            params["y1"],
            params["x2"],
            params["y2"],
        )

        max_value = kwargs.get("max_value")

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            prediction_context, prediction_window
        ):
            # Simplified exact route under Poisson thinning: sum_t Poisson(lambda_t * theta_t) ~ Poisson(sum_t lambda_t * theta_t)
            mu = float(np.sum(np.array(arrival_rates) * np.array(theta)))
            mv = (
                max_value
                if max_value is not None
                else self._default_max_value(arrival_rates)
            )
            x_values = np.arange(mv)
            probabilities = poisson.pmf(x_values, mu)
            predictions[filter_key] = self._pmf_to_dataframe(probabilities)

        return predictions

    def _get_admission_probabilities(
        self, prediction_window: Optional[timedelta] = None, **kwargs
    ) -> np.ndarray:
        """Get admission probabilities for parametric method using aspirational curves.

        Uses the parametric aspirational curve to calculate admission probabilities
        for each time interval based on the remaining time in the prediction window.

        Parameters
        ----------
        prediction_window : timedelta, optional
            The prediction window. Resolved via
            `_resolve_prediction_window()` if omitted.
        **kwargs
            Additional keyword arguments. Must include:
            - x1, y1, x2, y2: Parameters for the aspirational curve

        Returns
        -------
        numpy.ndarray
            Array of admission probabilities for each time interval.

        Raises
        ------
        ValueError
            If required parameters (x1, y1, x2, y2) are missing.
        """
        required = {"x1", "y1", "x2", "y2"}
        params = self._ensure_required_kwargs(kwargs, required, self.__class__.__name__)

        prediction_window = self._resolve_prediction_window(prediction_window)

        prediction_window_hours, yta_time_interval_hours, NTimes = (
            self._get_window_and_interval_hours(prediction_window)
        )

        time_remaining_before_end_of_window = prediction_window_hours - np.arange(
            0, prediction_window_hours, yta_time_interval_hours
        )

        theta = get_y_from_aspirational_curve(
            time_remaining_before_end_of_window,
            params["x1"],
            params["y1"],
            params["x2"],
            params["y2"],
        )

        return theta


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
        super().__init__(filters, verbose, use_generating_functions)
        self.survival_df = None

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window: Optional[timedelta] = None,
        yta_time_interval: Optional[timedelta] = None,
        prediction_times: Optional[List[float]] = None,
        num_days: Optional[int] = None,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
        start_time_col: str = "arrival_datetime",
        end_time_col: str = "departure_datetime",
    ) -> "EmpiricalIncomingAdmissionPredictor":
        """Fit the model to the training data and calculate empirical survival curve.

        The survival curve is independent of the prediction window and the set
        of prediction times; these are now supplied at ``predict()`` time.
        ``prediction_window`` and ``prediction_times`` are therefore deprecated
        as ``fit()`` parameters; see `IncomingAdmissionPredictor.fit()`.

        Parameters
        ----------
        train_df : pandas.DataFrame
            The training dataset with historical admission data.
            Expected to have start_time_col as the index and end_time_col as a column.
            Alternatively, both can be regular columns.
        prediction_window : timedelta, optional
            Deprecated. Prefer passing ``prediction_window`` to
            `predict()` / `predict_mean()` instead.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets. Required.
        prediction_times : list, optional
            Deprecated. Retained for backward compatibility only.
        num_days : int
            The number of days that the train_df spans. Required.
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
        if start_time_col in train_df.columns:
            df_for_survival = train_df
        else:
            df_for_survival = train_df.reset_index()
            if start_time_col not in df_for_survival.columns:
                raise ValueError(
                    f"Column '{start_time_col}' not found in DataFrame columns or index"
                )

        self.survival_df = calculate_survival_curve(
            df_for_survival, start_time_col=start_time_col, end_time_col=end_time_col
        )

        if self.survival_df is None or len(self.survival_df) == 0:
            raise RuntimeError("Failed to calculate survival curve from training data")

        # Ensure train_df has start_time_col as index for parent fit method
        if start_time_col in train_df.columns:
            train_df = train_df.set_index(start_time_col)

        super().fit(
            train_df,
            prediction_window=prediction_window,
            yta_time_interval=yta_time_interval,
            prediction_times=prediction_times,
            num_days=num_days,
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
        prediction_window : timedelta
            The prediction window.
        yta_time_interval : timedelta
            The time interval for splitting the prediction window. Retained
            for API stability; derived from ``self.yta_time_interval`` if not
            otherwise needed.

        Returns
        -------
        numpy.ndarray
            Array of admission probabilities for each time interval.
        """
        prediction_window_hours, yta_time_interval_hours, NTimes = (
            self._get_window_and_interval_hours(prediction_window)
        )

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

    def _convolve_poisson_distributions(self, arrival_rates, probabilities, max_value):
        """Convolve Poisson distributions for each time interval using generating functions.

        Parameters
        ----------
        arrival_rates : numpy.ndarray
            Array of arrival rates for each time interval.
        probabilities : numpy.ndarray
            Array of admission probabilities for each time interval.
        max_value : int
            Maximum value for the discrete distribution support.
            Typically determined using epsilon via _default_max_value(),
            but can be provided explicitly for custom truncation.

        Notes
        -----
        Legacy manual convolution has been removed in favor of the unified
        AdmissionGeneratingFunction implementation.
        """
        gf = AdmissionGeneratingFunction(
            arrival_rates, probabilities, method="empirical"
        )
        return gf.get_distribution(max_value=max_value)

    def predict(
        self,
        prediction_context: Dict,
        prediction_window: Optional[timedelta] = None,
        **kwargs,
    ) -> Dict:
        """Predict the number of admissions using empirical survival curves.

        Parameters
        ----------
        prediction_context : dict
            A dictionary defining the context for which predictions are to be made.
            It should specify either a general context or one based on the applied filters.
        prediction_window : timedelta, optional
            The prediction window over which admissions are accumulated.
            Required. If omitted, the value (if any) supplied to ``fit()`` is
            used and a `DeprecationWarning` is emitted.
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
            If filter key is not recognized, ``prediction_time`` is not provided,
            or ``prediction_window`` cannot be resolved.
        KeyError
            If required keys are missing from the prediction context.
        RuntimeError
            If survival_df was not provided during fitting.
        """
        if self.survival_df is None:
            raise RuntimeError(
                "No survival data available. Please call fit() method first to calculate survival curve from training data."
            )

        prediction_window = self._resolve_prediction_window(prediction_window)

        max_value = kwargs.get("max_value")

        predictions = {}

        survival_probabilities = self._calculate_survival_probabilities(
            prediction_window, self.yta_time_interval
        )

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            prediction_context, prediction_window
        ):
            mv = (
                max_value
                if max_value is not None
                else self._default_max_value(arrival_rates)
            )
            predictions[filter_key] = self._convolve_poisson_distributions(
                arrival_rates, survival_probabilities, max_value=mv
            )

        return predictions

    def _get_admission_probabilities(
        self, prediction_window: Optional[timedelta] = None, **kwargs
    ) -> np.ndarray:
        """Get admission probabilities for empirical method using survival curves.

        Uses the empirical survival curve calculated during fitting to determine
        admission probabilities for each time interval based on the remaining time
        in the prediction window.

        Parameters
        ----------
        prediction_window : timedelta, optional
            The prediction window. Resolved via
            `_resolve_prediction_window()` if omitted.
        **kwargs
            Additional keyword arguments (ignored for empirical method).

        Returns
        -------
        numpy.ndarray
            Array of admission probabilities for each time interval.

        Raises
        ------
        RuntimeError
            If survival_df was not provided during fitting.
        """
        if self.survival_df is None:
            raise RuntimeError(
                "No survival data available. Please call fit() method first to calculate survival curve from training data."
            )

        prediction_window = self._resolve_prediction_window(prediction_window)

        survival_probabilities = self._calculate_survival_probabilities(
            prediction_window, self.yta_time_interval
        )

        return survival_probabilities
