"""
Hospital Admissions Forecasting Predictors.

This module implements custom predictors to estimate the number of hospital admissions
within a specified prediction window using historical admission data. It provides three
approaches: direct admission prediction (full admission probability), parametric
aspirational curves with a Poisson thinning approximation, and empirical survival curves
with convolution of Poisson distributions. All predictors accommodate different data
filters for tailored predictions across various hospital settings.

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
    Predicts admissions using parametric aspirational curves and a Poisson thinning
    (independent filtering) approximation; see Notes for assumptions.

EmpiricalIncomingAdmissionPredictor : IncomingAdmissionPredictor
    Predicts the number of admissions using empirical survival curves and convolution
    of Poisson distributions; implemented via generating functions.

Notes
-----
The DirectAdmissionPredictor is the simplest approach, summing all arrival rates across time
intervals and creating a single Poisson distribution. It assumes 100% admission rate,
making it useful for scenarios where immediate admission is expected or as a baseline
for comparison with more complex models.

The ParametricIncomingAdmissionPredictor uses Poisson arrivals per slice and
slice-wise admission probabilities from aspirational curves (parameters x1, y1, x2, y2),
under a Poisson thinning approximation; see *Assumptions*.

The EmpiricalIncomingAdmissionPredictor inherits the arrival rate calculation and filtering logic
but replaces the parametric approach with empirical survival probabilities and convolution
of individual Poisson distributions for each time interval.

All predictors take into account historical data patterns and can be filtered for
specific hospital settings or specialties.

Prediction API (``predict`` / ``predict_mean``)
---------------------------------------------
After ``fit()``, pass ``prediction_time`` and ``prediction_window`` at predict
time. Use ``filter_keys`` with ``predict()`` or ``filter_key`` with
``predict_mean()`` to name which hospital **service** (or other stratum) to use —
i.e. which key(s) appear in ``weights``, matching the names given in ``filters``
when the model was fitted with filters. If ``weights`` has only one key,
``filter_keys`` / ``filter_key`` may be omitted.

Optional ``prediction_date`` (a :class:`~datetime.date`) anchors the calendar day
when the model was fit with ``stratify_by_weekday=True``. Each prediction slice
then uses the arrival rate for that slice's **actual** weekday and time-of-day
(``datetime.weekday()``: Monday = 0, …, Sunday = 6). If ``prediction_date`` is
omitted, behaviour matches the legacy pooled 24-hour profile
(``arrival_rates_dict``) only.

The deprecated nested ``prediction_context`` dict (keyword or first positional
argument) is still accepted and emits ``DeprecationWarning``. It may include
``prediction_date`` per filter key (same date required for all keys).

Assumptions
-----------
Parametric incoming-admissions use a simple per-slice Poisson arrivals model with
independent filtering. The following assumptions are required for the simplified
parametric route to be exact:

- Symbols and units:
    - λ_t: expected arrivals within time-slice t (from the pooled profile or, when
      ``prediction_date`` is set and the model was fit with weekday stratification,
      from the slice's weekday-specific profile; units match the slice length).
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
from datetime import date, datetime, timedelta, time as dt_time
from abc import ABC, abstractmethod

import numpy as np

import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from patientflow.calculate.admission_in_prediction_window import (
    get_y_from_aspirational_curve,
)

# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates,
    time_varying_arrival_rates_by_weekday,
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
        Parameters computed during fitting, keyed like ``filters`` (often a hospital
        **service** name) or ``unfiltered`` when no filters were used.

    Notes
    -----
    The predictor implements scikit-learn's BaseEstimator and TransformerMixin
    interfaces for compatibility with scikit-learn pipelines.

    For ``predict`` / ``predict_mean`` arguments (``prediction_time``,
    ``filter_keys`` / ``filter_key``, legacy ``prediction_context``), see the
    module docstring section *Prediction API*.
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
        self.empty_filter_count = 0

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

    def _resolve_prediction_window(
        self, prediction_window: Optional[timedelta]
    ) -> timedelta:
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

    def _parse_legacy_prediction_context(
        self, prediction_context: Dict, *, for_predict_mean: bool
    ) -> Tuple[Tuple[int, int], List[str], Optional[date]]:
        """Parse deprecated ``prediction_context`` into one snapped time and filter keys.

        Raises ``ValueError`` if times differ after snapping, if keys are unknown,
        or if ``for_predict_mean`` and more than one filter key is present.
        Optional ``prediction_date`` must match across keys when provided.
        """
        if not isinstance(prediction_context, dict) or not prediction_context:
            raise ValueError("prediction_context must be a non-empty dict")

        if for_predict_mean and len(prediction_context) > 1:
            raise ValueError(
                "predict_mean legacy prediction_context must contain exactly one filter key; "
                "use prediction_time= and filter_key= instead."
            )

        snapped_times: List[Tuple[int, int]] = []
        filter_keys_order: List[str] = []
        prediction_dates_raw: List[Optional[date]] = []
        for filter_key, filter_values in prediction_context.items():
            if filter_key not in self.weights:
                raise ValueError(
                    f"Filter key '{filter_key}' is not recognized in the model weights."
                )
            if not isinstance(filter_values, dict):
                raise ValueError(
                    f"Values in prediction_context must be dicts; got {type(filter_values)!r} "
                    f"for filter '{filter_key}'."
                )
            prediction_time = filter_values.get("prediction_time")
            if prediction_time is None:
                raise ValueError(
                    f"No 'prediction_time' provided for filter '{filter_key}'."
                )
            normalized = self._normalize_prediction_time(prediction_time)
            snapped = self._snap_to_interval_boundary(normalized)
            if snapped != normalized:
                warnings.warn(
                    f"Requested prediction_time {normalized} does not fall on a "
                    f"yta_time_interval boundary; snapping to {snapped}.",
                    UserWarning,
                    stacklevel=4,
                )
            snapped_times.append(snapped)
            filter_keys_order.append(filter_key)

            pd_raw = filter_values.get("prediction_date")
            if pd_raw is not None:
                if isinstance(pd_raw, datetime):
                    pd_raw = pd_raw.date()
                if not isinstance(pd_raw, date):
                    raise TypeError(
                        f"'prediction_date' for filter '{filter_key}' must be datetime.date "
                        f"(or datetime.datetime), got {type(pd_raw)!r}."
                    )
            prediction_dates_raw.append(pd_raw)

        if len(set(snapped_times)) > 1:
            raise ValueError(
                "prediction_context must use the same prediction_time (after snapping to "
                "yta_time_interval) for all filter keys."
            )

        dates_defined = [d for d in prediction_dates_raw if d is not None]
        if not dates_defined:
            resolved_prediction_date = None
        elif len(dates_defined) != len(prediction_dates_raw):
            raise ValueError(
                "prediction_context must provide 'prediction_date' for every filter key "
                "when it is provided for any key."
            )
        elif len(set(dates_defined)) > 1:
            raise ValueError(
                "prediction_context must use the same prediction_date for all filter keys."
            )
        else:
            resolved_prediction_date = dates_defined[0]

        return snapped_times[0], filter_keys_order, resolved_prediction_date

    def _resolve_filter_keys(
        self, filter_keys: Optional[Union[str, Sequence[str]]]
    ) -> List[str]:
        """Resolve ``filter_keys`` for ``predict()``.

        If ``filter_keys`` is ``None`` and ``weights`` has a single key, that key
        is used. If ``weights`` has more than one key, ``filter_keys`` must be
        supplied explicitly.
        """
        if filter_keys is None:
            if len(self.weights) == 1:
                return [next(iter(self.weights.keys()))]
            raise ValueError(
                "filter_keys is required when weights has more than one key "
                "(e.g. multiple fitted services)."
            )
        if isinstance(filter_keys, str):
            keys = [filter_keys]
        else:
            keys = list(filter_keys)
        for fk in keys:
            if fk not in self.weights:
                raise ValueError(
                    f"Filter key '{fk}' is not recognized in the model weights."
                )
        return keys

    def _resolve_filter_key_for_mean(self, filter_key: Optional[str]) -> str:
        """Return the single weight key used by ``predict_mean()``."""
        if filter_key is None:
            if len(self.weights) == 1:
                return next(iter(self.weights.keys()))
            raise ValueError(
                "filter_key is required for predict_mean() when weights has more than "
                "one key (e.g. multiple fitted services)."
            )
        if filter_key not in self.weights:
            raise ValueError(
                f"Filter key '{filter_key}' is not recognized in the model weights."
            )
        return filter_key

    def _prepare_prediction_targets_for_predict(
        self,
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        filter_keys: Optional[Union[str, Sequence[str]]] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
    ) -> Tuple[Tuple[int, int], List[str], Optional[date]]:
        """Resolve snapped prediction time, filter keys, and optional calendar anchor."""
        if prediction_context is not None:
            if (
                prediction_time is not None
                or filter_keys is not None
                or prediction_date is not None
            ):
                raise ValueError(
                    "Pass either prediction_context (deprecated) or prediction_time with "
                    "filter_keys / prediction_date, not both."
                )
            if args:
                raise ValueError(
                    "When using prediction_context=, do not pass a dict as a positional argument."
                )
            warnings.warn(
                "prediction_context dict is deprecated; use prediction_time= and filter_keys= "
                "(omit filter_keys when weights has only one key).",
                DeprecationWarning,
                stacklevel=3,
            )
            snapped, keys, ctx_date = self._parse_legacy_prediction_context(
                prediction_context, for_predict_mean=False
            )
            return snapped, keys, ctx_date

        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise TypeError(
                    "predict() positional arguments are only accepted for the deprecated "
                    "prediction_context dict; pass prediction_time as a keyword argument."
                )
            if (
                prediction_time is not None
                or filter_keys is not None
                or prediction_date is not None
            ):
                raise ValueError(
                    "Do not combine a positional prediction_context dict with "
                    "prediction_time=, filter_keys=, or prediction_date=."
                )
            warnings.warn(
                "Passing prediction_context as the first positional argument is deprecated; "
                "use prediction_time= and filter_keys=.",
                DeprecationWarning,
                stacklevel=3,
            )
            snapped, keys, ctx_date = self._parse_legacy_prediction_context(
                cast(Dict, args[0]), for_predict_mean=False
            )
            return snapped, keys, ctx_date

        if prediction_time is None:
            raise TypeError(
                "predict() requires prediction_time= when not using prediction_context."
            )

        if prediction_date is not None and not isinstance(prediction_date, date):
            raise TypeError(
                "prediction_date must be a datetime.date (values inside prediction_context "
                "may use datetime.datetime, which is coerced to date)."
            )

        normalized = self._normalize_prediction_time(prediction_time)
        snapped = self._snap_to_interval_boundary(normalized)
        if snapped != normalized:
            warnings.warn(
                f"Requested prediction_time {normalized} does not fall on a "
                f"yta_time_interval boundary; snapping to {snapped}.",
                UserWarning,
                stacklevel=3,
            )
        resolved_keys = self._resolve_filter_keys(filter_keys)
        return snapped, resolved_keys, prediction_date

    def _prepare_prediction_targets_for_predict_mean(
        self,
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        filter_key: Optional[str] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
    ) -> Tuple[Tuple[int, int], List[str], Optional[date]]:
        """Resolve snapped prediction time and a single-element key list for ``predict_mean()``."""
        if prediction_context is not None:
            if (
                prediction_time is not None
                or filter_key is not None
                or prediction_date is not None
            ):
                raise ValueError(
                    "Pass either prediction_context (deprecated) or prediction_time with "
                    "filter_key / prediction_date, not both."
                )
            if args:
                raise ValueError(
                    "When using prediction_context=, do not pass a dict as a positional argument."
                )
            warnings.warn(
                "prediction_context dict is deprecated; use prediction_time= and filter_key=.",
                DeprecationWarning,
                stacklevel=3,
            )
            snapped, fk_list, ctx_date = self._parse_legacy_prediction_context(
                prediction_context, for_predict_mean=True
            )
            return snapped, fk_list, ctx_date

        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise TypeError(
                    "predict_mean() positional arguments are only accepted for the deprecated "
                    "prediction_context dict; pass prediction_time as a keyword argument."
                )
            if (
                prediction_time is not None
                or filter_key is not None
                or prediction_date is not None
            ):
                raise ValueError(
                    "Do not combine a positional prediction_context dict with "
                    "prediction_time=, filter_key=, or prediction_date=."
                )
            warnings.warn(
                "Passing prediction_context as the first positional argument is deprecated; "
                "use prediction_time= and filter_key=.",
                DeprecationWarning,
                stacklevel=3,
            )
            snapped, fk_list, ctx_date = self._parse_legacy_prediction_context(
                cast(Dict, args[0]), for_predict_mean=True
            )
            return snapped, fk_list, ctx_date

        if prediction_time is None:
            raise TypeError(
                "predict_mean() requires prediction_time= when not using prediction_context."
            )

        if prediction_date is not None and not isinstance(prediction_date, date):
            raise TypeError(
                "prediction_date must be a datetime.date (values inside prediction_context "
                "may use datetime.datetime, which is coerced to date)."
            )

        normalized = self._normalize_prediction_time(prediction_time)
        snapped = self._snap_to_interval_boundary(normalized)
        if snapped != normalized:
            warnings.warn(
                f"Requested prediction_time {normalized} does not fall on a "
                f"yta_time_interval boundary; snapping to {snapped}.",
                UserWarning,
                stacklevel=3,
            )
        fk = self._resolve_filter_key_for_mean(filter_key)
        return snapped, [fk], prediction_date

    def _iter_prediction_inputs(
        self,
        snapped_prediction_time: Tuple[int, int],
        prediction_window: timedelta,
        filter_keys: List[str],
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
    ):
        """Yield ``(filter_key, resolved_prediction_time, arrival_rates_np)``.

        Slices ``Ntimes`` intervals starting at ``snapped_prediction_time`` (on an
        interval boundary). When ``prediction_date`` is set and weights contain
        ``arrival_rates_by_weekday`` (from ``fit(..., stratify_by_weekday=True)``),
        each slice uses the rate for that slice's calendar weekday and time-of-day.
        If ``prediction_date`` is set but weekday profiles are missing, behaviour
        depends on ``strict_prediction_date``:
        - ``False`` (default): fall back to pooled ``arrival_rates_dict`` and warn.
        - ``True``: raise ``ValueError``.
        """
        Ntimes = int(prediction_window / self.yta_time_interval)
        hr, mn = snapped_prediction_time
        for filter_key in filter_keys:
            w = self.weights[filter_key]
            arrival_rates_dict = w.get("arrival_rates_dict")
            if arrival_rates_dict is None:
                raise ValueError(
                    f"No arrival_rates_dict found under filter '{filter_key}'. "
                    "Has the model been fit?"
                )
            by_weekday = w.get("arrival_rates_by_weekday")
            use_weekday = prediction_date is not None and by_weekday is not None
            if prediction_date is not None and by_weekday is None:
                message = (
                    "prediction_date was provided but no arrival_rates_by_weekday "
                    f"were found for filter '{filter_key}'. Fit with "
                    "stratify_by_weekday=True to enable weekday-stratified "
                    "arrival rates."
                )
                if strict_prediction_date:
                    raise ValueError(message)
                warnings.warn(
                    message + " Falling back to pooled arrival_rates_dict.",
                    UserWarning,
                    stacklevel=4,
                )

            try:
                if use_weekday:
                    assert prediction_date is not None
                    assert by_weekday is not None
                    anchor = datetime.combine(
                        prediction_date, dt_time(hour=hr, minute=mn)
                    )
                    arrival_rates = []
                    for i in range(Ntimes):
                        dt_i = anchor + i * self.yta_time_interval
                        d_i = dt_i.weekday()
                        t_i = dt_i.time()
                        arrival_rates.append(by_weekday[d_i][t_i])
                else:
                    arrival_rates = [
                        arrival_rates_dict[
                            (
                                datetime(1970, 1, 1, hr, mn)
                                + i * self.yta_time_interval
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

    def _resolve_num_days(self, df: pd.DataFrame, num_days: Optional[int]) -> int:
        """Return ``num_days`` when given, otherwise infer from the dataframe index."""
        if num_days is not None:
            if not isinstance(num_days, int):
                raise TypeError("num_days must be an integer or None")
            if num_days <= 0:
                raise ValueError("num_days must be positive")
            return num_days
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "When num_days is omitted, the DataFrame index must be a DatetimeIndex "
                "so the training span can be inferred."
            )
        if len(df.index) == 0:
            raise ValueError(
                "Cannot infer num_days from an empty DataFrame when num_days is omitted."
            )
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        inferred = (end_date - start_date).days + 1
        if inferred <= 0:
            raise ValueError("Could not infer a positive num_days from the index.")
        return inferred

    def _calculate_parameters(
        self,
        df,
        yta_time_interval: timedelta,
        num_days: Optional[int],
        stratify_by_weekday: bool = True,
    ):
        """Calculate the full 24-hour arrival-rate dictionary for the given data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to process.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets.
        num_days : int or None
            Divisor for pooled rates; if ``None``, inferred from ``df``'s index span.
        stratify_by_weekday : bool, default=True
            If True, also compute ``arrival_rates_by_weekday`` (keys ``0..6``,
            Monday=0) for use when ``prediction_date`` is passed at predict time.

        Returns
        -------
        dict
            Always contains ``arrival_rates_dict``. When ``stratify_by_weekday``,
            also ``arrival_rates_by_weekday``: ``dict[int, OrderedDict[time, float]]``.
        """
        if len(df.index) == 0:
            interval_minutes = int(yta_time_interval.total_seconds() / 60)
            full_day_slots = int((24 * 60) / interval_minutes)
            base_dt = datetime(1970, 1, 1, 0, 0)
            zero_rates = {
                (base_dt + timedelta(minutes=i * interval_minutes)).time(): 0.0
                for i in range(full_day_slots)
            }
            out: Dict = {"arrival_rates_dict": zero_rates}
            if stratify_by_weekday:
                out["arrival_rates_by_weekday"] = {
                    weekday: zero_rates.copy() for weekday in range(7)
                }
            return out

        effective_days = self._resolve_num_days(df, num_days)
        arrival_rates_dict = time_varying_arrival_rates(
            df, yta_time_interval, effective_days, verbose=self.verbose
        )
        out: Dict = {"arrival_rates_dict": arrival_rates_dict}
        if stratify_by_weekday:
            out["arrival_rates_by_weekday"] = time_varying_arrival_rates_by_weekday(
                df, yta_time_interval, num_days, verbose=self.verbose
            )
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window: Optional[timedelta] = None,
        yta_time_interval: Optional[timedelta] = None,
        prediction_times: Optional[List[float]] = None,
        num_days: Optional[int] = None,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
        stratify_by_weekday: bool = True,
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
            ``predict()`` / ``predict_mean()`` instead. If provided, will
            be stored as a fall-back default for predict-time use.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets. Required.
        prediction_times : list, optional
            Deprecated. Prediction times are no longer needed at fit time; any
            prediction time can be served at predict time (snapped to the
            nearest ``yta_time_interval`` boundary). Retained for backward
            compatibility only.
        num_days : int, optional
            Divisor for **pooled** arrival rates (``arrival_rates_dict``). If omitted,
            inferred from ``train_df``'s index as
            ``(last_date - first_date).days + 1``. Use an explicit value when
            the divisor should differ from that calendar span.
            Weekday-stratified profiles always normalise using the index date
            span for occurrence counts when fit uses
            ``time_varying_arrival_rates_by_weekday``.
        epsilon : float, default=1e-7
            A small value representing acceptable error rate to enable calculation
            of the maximum value of the random variable representing number of beds.
        y : None, optional
            Ignored, present for compatibility with scikit-learn's fit method.
        stratify_by_weekday : bool, default=True
            If True, fit an additional per-weekday arrival profile (``Monday=0`` …
            ``Sunday=6``). Pass ``prediction_date`` at predict time to slice the
            window using that profile; omit ``prediction_date`` to keep using the
            pooled 24-hour profile only.

        Returns
        -------
        IncomingAdmissionPredictor
            The instance itself, fitted with the training data.

        Raises
        ------
        TypeError
            If ``yta_time_interval`` is missing or not a timedelta, or if
            ``prediction_window`` (when supplied) is not a timedelta, or if
            ``num_days`` is omitted but cannot be inferred (e.g. index is not a
            ``DatetimeIndex``).
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
        self.stratify_by_weekday = stratify_by_weekday

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
        self.empty_filter_count = 0
        if self.filters:
            for spec, filters in self.filters.items():
                filtered_df = self.filter_dataframe(train_df, filters)
                if len(filtered_df.index) == 0:
                    self.empty_filter_count += 1
                self.weights[spec] = self._calculate_parameters(
                    filtered_df,
                    yta_time_interval,
                    num_days,
                    stratify_by_weekday=stratify_by_weekday,
                )
        else:
            self.weights["unfiltered"] = self._calculate_parameters(
                train_df,
                yta_time_interval,
                num_days,
                stratify_by_weekday=stratify_by_weekday,
            )

        effective_metadata_days = self._resolve_num_days(train_df, num_days)

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
        self.metrics["num_days"] = effective_metadata_days

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
    def predict(self, *args, **kwargs) -> Dict:
        """Predict the bed-demand count distribution per ``weights`` key; subclasses implement.

        Each key is typically a hospital **service** name (or ``unfiltered``). Callers
        should use ``prediction_time``, ``prediction_window``, and ``filter_keys``
        (see module *Prediction API*). Legacy ``prediction_context`` (keyword or dict
        as first positional) remains supported with ``DeprecationWarning``.

        Weekday contract
        ----------------
        When the model was fit with ``stratify_by_weekday=True``, callers must pass
        ``prediction_date`` for each call so the model can select the matching
        weekday-specific arrival profile. Calling without ``prediction_date`` (or
        with one while the model lacks weekday weights) silently falls back to the
        pooled 24-hour profile and emits a ``UserWarning``; set
        ``strict_prediction_date=True`` to raise instead.

        For per-snapshot evaluation, prefer
        :func:`patientflow.aggregate.get_prob_dist_by_service`, which threads
        ``prediction_date`` into ``build_service_data`` for every snapshot date and
        therefore honours weekday stratification without requiring callers to
        manage the contract themselves.
        """
        ...

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
        ...

    def predict_mean(
        self,
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        prediction_window: Optional[timedelta] = None,
        filter_key: Optional[str] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        **kwargs,
    ) -> float:
        """Return the Poisson mean (expected value) for a single ``weights`` key.

        Parameters
        ----------
        prediction_time : tuple or list or int, optional
            Time of day ``(hour, minute)`` for slicing arrival rates. Required
            unless using the deprecated ``prediction_context`` dict.
        prediction_window : timedelta, optional
            The prediction window. Required unless the value supplied at ``fit()``
            is used (deprecated; emits ``DeprecationWarning``).
        filter_key : str, optional
            Which hospital **service** (or other stratum) to use: a key present in
            ``weights``, usually the same name as in ``filters``. Required when
            ``weights`` has more than one key (unless using ``prediction_context``).
        prediction_context : dict, optional
            Deprecated. Former nested dict mapping filter key to
            ``{"prediction_time": ...}``. Must contain exactly one filter key.
        prediction_date : datetime.date, optional
            Calendar date at the snapped ``prediction_time``. When the model was fit
            with ``stratify_by_weekday=True``, selects per-slice weekday arrival rates;
            otherwise ignored for λ (pooled profile is used).
        strict_prediction_date : bool, default=False
            If ``True``, raise an error when ``prediction_date`` is provided but
            weekday-stratified arrival rates are unavailable for the selected key.
            If ``False``, warn and fall back to pooled arrival rates.
        **kwargs
            Passed to ``_get_admission_probabilities`` (e.g. ``x1``, ``y1``, …).

        Returns
        -------
        float
            Poisson mean for the requested service (``weights`` key).

        Raises
        ------
        TypeError
            If ``prediction_time`` is missing when not using ``prediction_context``.
        ValueError
            If ``filter_key`` is required but omitted, keys are unknown, or legacy
            ``prediction_context`` has multiple keys or inconsistent times.
        """
        prediction_window = self._resolve_prediction_window(prediction_window)

        snapped_time, filter_keys_list, resolved_date = (
            self._prepare_prediction_targets_for_predict_mean(
                *args,
                prediction_time=prediction_time,
                filter_key=filter_key,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
            )
        )

        admission_probs = self._get_admission_probabilities(
            prediction_window=prediction_window, **kwargs
        )

        for filter_key_i, _pt, arrival_rates in self._iter_prediction_inputs(
            snapped_time,
            prediction_window,
            filter_keys_list,
            prediction_date=resolved_date,
            strict_prediction_date=strict_prediction_date,
        ):
            return float(np.sum(np.array(arrival_rates) * np.array(admission_probs)))

        raise ValueError("No valid prediction targets provided")


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

    Day-of-week stratification is supported through the inherited ``fit()``
    interface:

    - Use ``fit(..., stratify_by_weekday=True)`` to store weekday-specific
      arrival profiles.
    - Use ``predict(..., prediction_date=...)`` (or ``predict_mean``) to
      activate weekday-aware slicing at predict time.
    - If ``prediction_date`` is omitted, pooled 24-hour arrival rates are used.
    """

    def predict(
        self,
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        prediction_window: Optional[timedelta] = None,
        filter_keys: Optional[Union[str, Sequence[str]]] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        **kwargs,
    ) -> Dict:
        """Predict the number of admissions assuming 100% admission rate.

        Parameters
        ----------
        prediction_time : tuple or list or int, optional
            Time of day for slicing arrival rates. Required unless using the
            deprecated ``prediction_context`` dict API.
        prediction_window : timedelta, optional
            The prediction window over which admissions are accumulated.
            Required unless the value supplied at ``fit()`` is used (deprecated).
        filter_keys : str or sequence of str, optional
            Service name(s) or other ``weights`` key(s). Required when ``weights``
            has more than one key (unless using ``prediction_context``).
        prediction_context : dict, optional
            Deprecated nested dict API.
        prediction_date : datetime.date, optional
            Calendar anchor for weekday-stratified arrival rates when fitted with
            ``stratify_by_weekday=True``. Otherwise λ uses the pooled profile only.
        strict_prediction_date : bool, default=False
            If ``True``, raise an error when ``prediction_date`` is provided but
            weekday-stratified arrival rates are unavailable for a selected key.
            If ``False``, warn and fall back to pooled arrival rates.
        **kwargs
            ``max_value`` : int, optional — maximum PMF support.

        Returns
        -------
        dict
            Keys match ``weights`` (e.g. hospital service). Values are DataFrames
            indexed by outcome count (``sum``) with column ``agg_proba`` (PMF).

        Raises
        ------
        TypeError
            If ``prediction_time`` is missing when not using legacy input.
        ValueError
            If ``filter_keys`` is required but omitted, keys are invalid, or legacy
            ``prediction_context`` has inconsistent times across keys.
        """
        # Be lenient: ignore unrelated kwargs (e.g., parametric args passed by a higher-level API)
        prediction_window = self._resolve_prediction_window(prediction_window)

        snapped_time, resolved_filter_keys, resolved_date = (
            self._prepare_prediction_targets_for_predict(
                *args,
                prediction_time=prediction_time,
                filter_keys=filter_keys,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
            )
        )

        max_value = kwargs.get("max_value")

        predictions = {}

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            snapped_time,
            prediction_window,
            resolved_filter_keys,
            prediction_date=resolved_date,
            strict_prediction_date=strict_prediction_date,
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

    This predictor uses Poisson-distributed arrivals per time slice and slice-wise
    admission probabilities from aspirational curves (Poisson thinning approximation).
    The prediction is based on historical data and can be filtered for specific
    hospital settings.

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
        Parameters computed during fitting, keyed like ``filters`` (often a hospital
        **service** name) or ``unfiltered`` when no filters were used.

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
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        prediction_window: Optional[timedelta] = None,
        filter_keys: Optional[Union[str, Sequence[str]]] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        **kwargs,
    ) -> Dict:
        """Predict admissions using parametric curves (Poisson thinning route).

        Parameters
        ----------
        prediction_time : tuple or list or int, optional
            Required unless using deprecated ``prediction_context``.
        prediction_window : timedelta, optional
            Required unless using deprecated fit-time default.
        filter_keys : str or sequence of str, optional
            Required when ``weights`` has more than one key unless using
            ``prediction_context``.
        prediction_context : dict, optional
            Deprecated nested dict API.
        prediction_date : datetime.date, optional
            Calendar anchor for weekday-stratified λ when fitted with
            ``stratify_by_weekday=True``.
        strict_prediction_date : bool, default=False
            If ``True``, raise an error when ``prediction_date`` is provided but
            weekday-stratified arrival rates are unavailable for a selected key.
            If ``False``, warn and fall back to pooled arrival rates.
        **kwargs
            ``x1``, ``y1``, ``x2``, ``y2`` (required); optional ``max_value``.

        Returns
        -------
        dict
            Service (``weights`` key) → PMF DataFrame (index ``sum``, column ``agg_proba``).

        Raises
        ------
        TypeError, ValueError
            Same rules as :meth:`DirectAdmissionPredictor.predict`.

        Notes
        -----
        Poisson thinning: admitted in slice t ~ Poisson(λ_t θ_t);
        total ~ Poisson(Σ_t λ_t θ_t).
        """
        # Validate/collect kwargs
        required = {"x1", "y1", "x2", "y2"}
        params = self._ensure_required_kwargs(kwargs, required, self.__class__.__name__)
        self._validate_only_kwargs(
            kwargs, required | {"max_value"}, context=self.__class__.__name__
        )

        prediction_window = self._resolve_prediction_window(prediction_window)

        snapped_time, resolved_filter_keys, resolved_date = (
            self._prepare_prediction_targets_for_predict(
                *args,
                prediction_time=prediction_time,
                filter_keys=filter_keys,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
            )
        )

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
            snapped_time,
            prediction_window,
            resolved_filter_keys,
            prediction_date=resolved_date,
            strict_prediction_date=strict_prediction_date,
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
    of Poisson distributions for prediction instead of the parametric curve route.

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
        stratify_by_weekday: bool = True,
        *,
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
            ``predict()`` / ``predict_mean()`` instead.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets. Required.
        prediction_times : list, optional
            Deprecated. Retained for backward compatibility only.
        num_days : int, optional
            Same as :meth:`IncomingAdmissionPredictor.fit`.
        epsilon : float, default=1e-7
            A small value representing acceptable error rate to enable calculation
            of the maximum value of the random variable representing number of beds.
        y : None, optional
            Ignored, present for compatibility with scikit-learn's fit method.
        stratify_by_weekday : bool, default=True
            Same as ``IncomingAdmissionPredictor.fit``.
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
            stratify_by_weekday=stratify_by_weekday,
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
        *args,
        prediction_time: Optional[Union[Tuple[int, int], List[int], int]] = None,
        prediction_window: Optional[timedelta] = None,
        filter_keys: Optional[Union[str, Sequence[str]]] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        **kwargs,
    ) -> Dict:
        """Predict admissions using empirical survival curves.

        Parameters
        ----------
        prediction_time : tuple or list or int, optional
            Required unless using deprecated ``prediction_context``.
        prediction_window : timedelta, optional
            Required unless using deprecated fit-time default.
        filter_keys : str or sequence of str, optional
            Required when ``weights`` has more than one key unless using
            ``prediction_context``.
        prediction_context : dict, optional
            Deprecated nested dict API.
        prediction_date : datetime.date, optional
            Calendar anchor for weekday-stratified λ when fitted with
            ``stratify_by_weekday=True``.
        strict_prediction_date : bool, default=False
            If ``True``, raise an error when ``prediction_date`` is provided but
            weekday-stratified arrival rates are unavailable for a selected key.
            If ``False``, warn and fall back to pooled arrival rates.
        **kwargs
            Optional ``max_value``.

        Returns
        -------
        dict
            Service (``weights`` key) → PMF DataFrame (index ``sum``, column ``agg_proba``).

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called (no survival curve).
        TypeError, ValueError
            Same rules as :meth:`DirectAdmissionPredictor.predict`.
        """
        if self.survival_df is None:
            raise RuntimeError(
                "No survival data available. Please call fit() method first to calculate survival curve from training data."
            )

        prediction_window = self._resolve_prediction_window(prediction_window)

        snapped_time, resolved_filter_keys, resolved_date = (
            self._prepare_prediction_targets_for_predict(
                *args,
                prediction_time=prediction_time,
                filter_keys=filter_keys,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
            )
        )

        max_value = kwargs.get("max_value")

        predictions = {}

        survival_probabilities = self._calculate_survival_probabilities(
            prediction_window, self.yta_time_interval
        )

        for filter_key, prediction_time, arrival_rates in self._iter_prediction_inputs(
            snapped_time,
            prediction_window,
            resolved_filter_keys,
            prediction_date=resolved_date,
            strict_prediction_date=strict_prediction_date,
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
