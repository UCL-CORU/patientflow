"""
Aggregate Prediction From Patient-Level Probabilities

This submodule provides functions to aggregate patient-level predicted probabilities into a probability distribution
using a generating function approach.

The module uses dynamic programming to compute the exact probability distribution of sums of
independent Bernoulli random variables, which is mathematically equivalent to multiplying their generating
functions but computationally much more efficient.

Computation Method Selection
----------------------------
For computational efficiency, the module automatically switches between two methods based on sample size:

- Exact computation (default for n ≤ 30): Uses dynamic programming to compute the exact probability
  distribution. This is the preferred method for smaller groups as it provides exact results.
- Normal approximation (default for n > 30): Uses a normal approximation with continuity correction
  for larger groups. The approximation is based on the central limit theorem, where the sum of independent
  Bernoulli random variables approaches a normal distribution for large n. The mean is the sum of probabilities
  and the variance is the sum of `p_i * (1 - p_i)`.

The threshold (default: 30) can be adjusted via the `normal_approx_threshold` parameter in functions that
accept it. Setting this parameter to `None` or a very large value forces exact computation for all sample sizes.

Functions
---------
BernoulliGeneratingFunction : class
    Generating function for sums of Bernoulli random variables.
    Automatically selects between exact computation and normal approximation based on sample size.

model_input_to_pred_proba : function
    Use a predictive model to convert model input data into predicted probabilities.

pred_proba_to_agg_predicted : function
    Convert individual probability predictions into aggregate predicted probability distribution using optional weights.

get_prob_dist_for_prediction_moment : function
    Calculate both predicted distributions and observed values for a given date using test data.

get_prob_dist : function
    Calculate probability distributions for each snapshot date based on given model predictions.

get_prob_dist_using_survival_curve : function
    Calculate probability distributions for each snapshot date based on given model predictions, using a survival curve to predict the probability of each patient being admitted within a given prediction window.

"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
from patientflow.predictors.incoming_admission_predictors import (
    EmpiricalIncomingAdmissionPredictor,
)


class BernoulliGeneratingFunction:
    """
    Generating function implementation for sums of independent Bernoulli random variables.

    This class is based on the mathematical principle that the generating function of
    a sum of independent random variables is the product of their individual generating functions.

    For Bernoulli random variables with success probabilities `p_i`, the generating function is:
    `G_i(z) = (1 - p_i) + p_i * z`

    The product `G_1(z) * G_2(z) * ... * G_n(z)` gives the generating function of the sum,
    and the coefficient of `z^k` gives `P(sum = k)`.

    The `get_distribution()` method automatically selects between exact computation and normal
    approximation based on sample size (see module-level documentation for details).

    Parameters
    ----------
    probabilities : List[float]
        List of success probabilities for each Bernoulli random variable
    weights : Optional[List[float]]
        Optional weights to apply to each probability (for weighted predictions)

    Attributes
    ----------
    probs : np.ndarray
        Array of (possibly weighted) success probabilities
    n : int
        Number of random variables
    """

    def __init__(
        self, probabilities: List[float], weights: Optional[List[float]] = None
    ):
        self.probs = np.array(probabilities)
        if weights is not None:
            self.probs = self.probs * np.array(weights)
        self.n = len(self.probs)

    def exact_distribution(self) -> Dict[int, float]:
        """
        Calculate exact probability distribution using dynamic programming.

        This method computes `P(sum = k)` for `k = 0, 1, ..., n` using the recurrence relation:
        `dp[i][k] = dp[i-1][k] * (1 - p_i) + dp[i-1][k-1] * p_i`

        Returns
        -------
        Dict[int, float]
            Dictionary mapping `{k: P(sum = k)}` for `k = 0, 1, ..., n`
        """
        if self.n == 0:
            return {0: 1.0}

        # Dynamic programming approach
        # `dp[i][k]` = probability that first i variables sum to k
        dp = np.zeros((self.n + 1, self.n + 1))
        dp[0][0] = 1.0  # Base case: 0 variables sum to 0 with probability 1

        for i in range(1, self.n + 1):
            p_i = self.probs[i - 1]
            for k in range(i + 1):  # Can't sum to more than i with i variables
                if k == 0:
                    # All previous variables are 0, current variable is 0
                    dp[i][k] = dp[i - 1][k] * (1 - p_i)
                else:
                    # Either: previous sum to k and current is 0, OR previous sum to k-1 and current is 1
                    dp[i][k] = dp[i - 1][k] * (1 - p_i) + dp[i - 1][k - 1] * p_i

        return {k: dp[self.n][k] for k in range(self.n + 1)}

    def normal_approximation(self) -> Dict[int, float]:
        """
        Use normal approximation with continuity correction for large datasets.

        For sums of independent Bernoulli random variables:
        - Mean = sum of probabilities
        - Variance = `sum of p_i * (1 - p_i)`

        Uses continuity correction: `P(X = k) ≈ P(k - 0.5 < Y < k + 0.5)`
        where `Y ~ Normal(mean, variance)`

        Returns
        -------
        Dict[int, float]
            Dictionary mapping `{k: P(sum = k)}` for `k = 0, 1, ..., n`
        """
        mean = self.probs.sum()
        variance = (self.probs * (1 - self.probs)).sum()

        if variance == 0:
            # Deterministic case: all probabilities are 0 or 1
            return {int(round(mean)): 1.0}

        std = np.sqrt(variance)
        result = {}

        for k in range(self.n + 1):
            if k == 0:
                p = norm.cdf(0.5, loc=mean, scale=std)
            elif k == self.n:
                p = 1 - norm.cdf(self.n - 0.5, loc=mean, scale=std)
            else:
                p = norm.cdf(k + 0.5, loc=mean, scale=std) - norm.cdf(
                    k - 0.5, loc=mean, scale=std
                )
            result[k] = max(0, p)  # Ensure non-negative probabilities

        # Normalize to ensure probabilities sum to 1
        total = sum(result.values())
        if total > 0:
            for k in result:
                result[k] /= total
        else:
            # Fallback to uniform distribution if something went wrong
            uniform_prob = 1.0 / (self.n + 1)
            result = {k: uniform_prob for k in range(self.n + 1)}

        return result

    def get_distribution(self, normal_approx_threshold: int = 30) -> Dict[int, float]:
        """
        Get probability distribution using exact or approximate method based on problem size.

        Parameters
        ----------
        normal_approx_threshold : int, optional (default=30)
            Use normal approximation if n > threshold for better performance.
            Set to None or a very large number to always use exact computation.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping {k: P(sum = k)} for all possible values k
        """
        if normal_approx_threshold is not None and self.n > normal_approx_threshold:
            return self.normal_approximation()
        else:
            return self.exact_distribution()


def model_input_to_pred_proba(model_input: Any, model: Any) -> pd.DataFrame:
    """
    Use a predictive model to convert model input data into predicted probabilities.

    Parameters
    ----------
    model_input : array-like
        The input data to the model, typically as features used for predictions.
    model : object
        A model object with a `predict_proba` method that computes probability estimates.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the predicted probabilities for the positive class,
        with one column labeled 'pred_proba'.
    """
    if len(model_input) == 0:
        return pd.DataFrame(columns=["pred_proba"])
    else:
        predictions = model.predict_proba(model_input)[:, 1]
        return pd.DataFrame(
            predictions, index=model_input.index, columns=["pred_proba"]
        )


def pred_proba_to_agg_predicted(
    predictions_proba: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    normal_approx_threshold: int = 30,
) -> pd.DataFrame:
    """
    Convert individual probability predictions into aggregate predicted probability distribution.

    This function uses a generating function approach based on dynamic programming
    to compute the exact probability distribution of the sum of independent Bernoulli random
    variables. For large datasets, it automatically switches to a normal approximation for
    better performance.

    Mathematical Background
    ----------------------
    Each patient has a probability `p_i` of needing a bed (Bernoulli random variable).
    The total number of beds needed is the sum of these Bernoulli variables.

    The generating function approach computes:
    `P(Total = k) = coefficient of z^k in ∏(1 - p_i + p_i * z)`

    This is computed efficiently using dynamic programming without symbolic expansion.

    Parameters
    ----------
    predictions_proba : pd.DataFrame
        A DataFrame containing the probability predictions; must have a single column named 'pred_proba'.
    weights : np.ndarray, optional
        An array of weights, of the same length as the DataFrame rows, to apply to each prediction.
        Useful for incorporating patient-specific factors or sampling weights.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in predictions_proba exceeds this threshold, use a Normal distribution
        approximation for better performance. Set to None or a very large number to always use
        exact computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a single column 'agg_proba' showing the aggregated probability distribution,
        indexed from `0` to `n`, where `n` is the number of predictions. Each row gives `P(total = k)`
        where `k` is the row index.

    """
    n = len(predictions_proba)

    if n == 0:
        return pd.DataFrame({"agg_proba": [1]}, index=[0])

    # Extract probabilities
    probabilities = predictions_proba["pred_proba"].values.tolist()
    weights_list = weights.tolist() if weights is not None else None

    # Use generating function approach
    gf = BernoulliGeneratingFunction(probabilities, weights_list)
    agg_predicted_dict = gf.get_distribution(normal_approx_threshold)

    agg_predicted = pd.DataFrame.from_dict(
        agg_predicted_dict, orient="index", columns=["agg_proba"]
    )
    return agg_predicted


def get_prob_dist_for_prediction_moment(
    X_test: Any,
    model: Any,
    weights: Optional[np.ndarray] = None,
    inference_time: bool = False,
    y_test: Optional[pd.Series] = None,
    category_filter: Optional[pd.Series] = None,
    normal_approx_threshold: int = 30,
) -> Dict[str, Any]:
    """
    Calculate both predicted distributions and observed values for a given snapshot date.

    Parameters
    ----------
    X_test : array-like
        Test features for a specific snapshot date.
    model : object or TrainedClassifier
        Either a predictive model which provides a `predict_proba` method,
        or a TrainedClassifier object containing a pipeline.
    weights : np.ndarray, optional
        Weights to apply to the predictions for aggregate calculation.
    inference_time : bool, optional (default=False)
        If True, do not calculate or return actual aggregate.
    y_test : array-like, optional
        Actual outcomes corresponding to the test features. Required if inference_time is False.
    category_filter : array-like, optional
        Boolean mask indicating which samples belong to the specific outcome category being analyzed.
        Should be the same length as y_test.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in X_test exceeds this threshold, use a Normal distribution approximation
        for better performance. Set to None or a very large number to always use exact computation.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys 'agg_predicted' and, if inference_time is False, 'agg_observed'.
        - 'agg_predicted': DataFrame with probability distribution
        - 'agg_observed': int with actual observed count

    Raises
    ------
    ValueError
        If y_test is not provided when inference_time is False.
        If model has no predict_proba method and is not a TrainedClassifier.
    """
    if not inference_time and y_test is None:
        raise ValueError("y_test must be provided if inference_time is False.")

    # Extract pipeline if model is a TrainedClassifier
    if hasattr(model, "calibrated_pipeline") and model.calibrated_pipeline is not None:
        model = model.calibrated_pipeline
    elif hasattr(model, "pipeline"):
        model = model.pipeline
    # Validate that model has predict_proba method
    elif not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model must either be a TrainedClassifier or have a predict_proba method"
        )

    prediction_moment_dict = {}

    if len(X_test) > 0:
        pred_proba = model_input_to_pred_proba(X_test, model)
        agg_predicted = pred_proba_to_agg_predicted(
            pred_proba, weights, normal_approx_threshold
        )
        prediction_moment_dict["agg_predicted"] = agg_predicted

        if not inference_time:
            # Calculate observed sum with optional category filter
            if y_test is None:
                raise ValueError("y_test must be provided if inference_time is False.")

            observed_bool = y_test.astype(bool)
            if category_filter is not None:
                observed_bool = observed_bool & category_filter.astype(bool)
            prediction_moment_dict["agg_observed"] = int(observed_bool.sum())
    else:
        prediction_moment_dict["agg_predicted"] = pd.DataFrame(
            {"agg_proba": [1]}, index=[0]
        )
        if not inference_time:
            prediction_moment_dict["agg_observed"] = 0

    return prediction_moment_dict


def get_prob_dist(
    snapshots_dict: Dict[date, List[int]],
    X_test: Any,
    y_test: Any,
    model: Any,
    weights: Optional[pd.Series] = None,
    verbose: bool = False,
    category_filter: Optional[Any] = None,
    normal_approx_threshold: int = 30,
) -> Dict[date, Dict[str, Any]]:
    """
    Calculate probability distributions for each snapshot date based on given model predictions.
    Parameters
    ----------
    snapshots_dict : Dict[date, List[int]]
        A dictionary mapping snapshot dates to indices in `X_test` and `y_test`.
        Must have datetime.date objects as keys and lists of indices as values.
    X_test : DataFrame or array-like
        Input test data to be passed to the model.
    y_test : array-like
        Observed target values.
    model : object or TrainedClassifier
        Either a predictive model which provides a `predict_proba` method,
        or a TrainedClassifier object containing a pipeline.
    weights : pd.Series, optional
        A Series containing weights for the test data points, which may influence the prediction,
        by default None. If provided, the weights should be indexed similarly to `X_test` and `y_test`.
    verbose : bool, optional (default=False)
        If True, print progress information.
    category_filter : array-like, optional
        Boolean mask indicating which samples belong to the specific outcome category being analyzed.
        Should be the same length as y_test.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in a snapshot exceeds this threshold, use a Normal distribution approximation
        for better performance. Set to None or a very large number to always use exact computation.

    Returns
    -------
    Dict[date, Dict[str, Any]]
        A dictionary mapping snapshot dates to probability distributions.
        Each value contains 'agg_predicted' (DataFrame) and 'agg_observed' (int).

    Raises
    ------
    ValueError
        If snapshots_dict is not properly formatted or empty.
        If model has no predict_proba method and is not a TrainedClassifier.

    """
    # Validate snapshots_dict format
    if not snapshots_dict:
        raise ValueError("snapshots_dict cannot be empty")

    for dt, indices in snapshots_dict.items():
        if not isinstance(dt, date):
            raise ValueError(
                f"snapshots_dict keys must be datetime.date objects, got {type(dt)}"
            )
        if not isinstance(indices, list):
            raise ValueError(
                f"snapshots_dict values must be lists, got {type(indices)}"
            )
        if indices and not all(isinstance(idx, int) for idx in indices):
            raise ValueError("All indices in snapshots_dict must be integers")

    # Extract pipeline if model is a TrainedClassifier
    if hasattr(model, "calibrated_pipeline") and model.calibrated_pipeline is not None:
        model = model.calibrated_pipeline
    elif hasattr(model, "pipeline"):
        model = model.pipeline
    # Validate that model has predict_proba method
    elif not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model must either be a TrainedClassifier or have a predict_proba method"
        )

    prob_dist_dict = {}
    if verbose:
        print(
            f"Calculating probability distributions for {len(snapshots_dict)} snapshot dates"
        )

        if len(snapshots_dict) > 10:
            print(
                "Using efficient generating function approach - much faster than before!"
            )

    # Initialize a counter for notifying the user every 10 snapshot dates processed
    count = 0

    for dt, snapshots_to_include in snapshots_dict.items():
        if len(snapshots_to_include) == 0:
            # Create an empty dictionary for the current snapshot date
            prob_dist_dict[dt] = {
                "agg_predicted": pd.DataFrame({"agg_proba": [1]}, index=[0]),
                "agg_observed": 0,
            }
        else:
            # Ensure the lengths of test features and outcomes are equal
            assert len(X_test.loc[snapshots_to_include]) == len(
                y_test.loc[snapshots_to_include]
            ), "Mismatch in lengths of X_test and y_test snapshots."

            if weights is None:
                prediction_moment_weights = None
            else:
                prediction_moment_weights = weights.loc[snapshots_to_include].values

            # Apply category filter
            if category_filter is None:
                prediction_moment_category_filter = None
            else:
                prediction_moment_category_filter = category_filter.loc[
                    snapshots_to_include
                ]

            # Use the refactored generating function approach
            prob_dist_dict[dt] = get_prob_dist_for_prediction_moment(
                X_test=X_test.loc[snapshots_to_include],
                y_test=y_test.loc[snapshots_to_include],
                model=model,
                weights=prediction_moment_weights,
                category_filter=prediction_moment_category_filter,
                normal_approx_threshold=normal_approx_threshold,
            )

        # Increment the counter and notify the user every 10 snapshot dates processed
        count += 1
        if verbose and count % 10 == 0 and count != len(snapshots_dict):
            print(f"Processed {count} snapshot dates")

    if verbose:
        print(f"Processed {len(snapshots_dict)} snapshot dates")

    return prob_dist_dict


def get_prob_dist_using_survival_curve(
    snapshot_dates: List[date],
    test_visits: pd.DataFrame,
    category: str,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    start_time_col: str,
    end_time_col: str,
    model: EmpiricalIncomingAdmissionPredictor,
    verbose: bool = False,
) -> Dict[date, Dict[str, Any]]:
    """
    Calculate probability distributions for each snapshot date using an EmpiricalIncomingAdmissionPredictor.

    Parameters
    ----------
    snapshot_dates : List[date]
        Array of dates for which to calculate probability distributions.
    test_visits : pd.DataFrame
        DataFrame containing test visit data. Must have either:
        - start_time_col as a column and end_time_col as a column, or
        - start_time_col as the index and end_time_col as a column
    category : str
        Category to use for predictions (e.g., 'medical', 'surgical')
    prediction_time : Tuple[int, int]
        Tuple of (hour, minute) representing the time of day for predictions
    prediction_window : timedelta
        The prediction window duration
    start_time_col : str
        Name of the column containing start times (or index name if using index)
    end_time_col : str
        Name of the column containing end times
    model : EmpiricalIncomingAdmissionPredictor
        A fitted instance of EmpiricalIncomingAdmissionPredictor
    verbose : bool, optional (default=False)
        If True, print progress information

    Returns
    -------
    Dict[date, Dict[str, Any]]
        A dictionary mapping snapshot dates to probability distributions.
        Each value contains 'agg_predicted' (DataFrame) and 'agg_observed' (int).

    Raises
    ------
    ValueError
        If test_visits does not have the required columns or if model is not fitted.
    """

    # Validate test_visits has required columns
    if start_time_col in test_visits.columns:
        # start_time_col is a regular column
        if end_time_col not in test_visits.columns:
            raise ValueError(f"Column '{end_time_col}' not found in DataFrame")
    else:
        # Check if start_time_col is the index
        if test_visits.index.name != start_time_col:
            raise ValueError(
                f"'{start_time_col}' not found in DataFrame columns or index (index.name is '{test_visits.index.name}')"
            )
        if end_time_col not in test_visits.columns:
            raise ValueError(f"Column '{end_time_col}' not found in DataFrame")

    # Validate model is fitted
    if not hasattr(model, "survival_df") or model.survival_df is None:
        raise ValueError(
            "Model must be fitted before calling get_prob_dist_using_survival_curve"
        )

    prob_dist_dict = {}
    if verbose:
        print(
            f"Calculating probability distributions for {len(snapshot_dates)} snapshot dates"
        )

    # Create prediction context that will be the same for all dates
    prediction_context = {category: {"prediction_time": prediction_time}}

    for dt in snapshot_dates:
        # Create prediction moment by combining snapshot date and prediction time
        prediction_moment = datetime.combine(
            dt, time(prediction_time[0], prediction_time[1])
        )
        # Convert to UTC if the test_visits timestamps are timezone-aware
        if start_time_col in test_visits.columns:
            if test_visits[start_time_col].dt.tz is not None:
                prediction_moment = prediction_moment.replace(tzinfo=timezone.utc)
        else:
            if test_visits.index.tz is not None:
                prediction_moment = prediction_moment.replace(tzinfo=timezone.utc)

        # Get predictions from model
        predictions = model.predict(prediction_context)
        prob_dist_dict[dt] = {"agg_predicted": predictions[category]}

        # Calculate observed values
        if start_time_col in test_visits.columns:
            # start_time_col is a regular column
            mask = (test_visits[start_time_col] > prediction_moment) & (
                test_visits[end_time_col] <= prediction_moment + prediction_window
            )
        else:
            # start_time_col is the index
            mask = (test_visits.index > prediction_moment) & (
                test_visits[end_time_col] <= prediction_moment + prediction_window
            )
        nrow = mask.sum()
        prob_dist_dict[dt]["agg_observed"] = int(nrow) if nrow > 0 else 0

    if verbose:
        print(f"Processed {len(snapshot_dates)} snapshot dates")

    return prob_dist_dict
