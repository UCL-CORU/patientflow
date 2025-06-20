"""
Aggregate Prediction From Patient-Level Probabilities

This submodule provides functions to aggregate patient-level predicted probabilities into a probability distribution.
The module uses symbolic mathematics to generate and manipulate expressions, enabling the computation of aggregate probabilities based on individual patient-level predictions.

Functions
---------
create_symbols : function
    Generate a sequence of symbolic objects intended for use in mathematical expressions.

compute_core_expression : function
    Compute a symbolic expression involving a basic mathematical operation with a symbol and a constant.

build_expression : function
    Construct a cumulative product expression by combining individual symbolic expressions.

expression_subs : function
    Substitute values into a symbolic expression based on a mapping from symbols to predictions.

return_coeff : function
    Extract the coefficient of a specified power from an expanded symbolic expression.

model_input_to_pred_proba : function
    Use a predictive model to convert model input data into predicted probabilities.

pred_proba_to_agg_predicted : function
    Convert individual probability predictions into aggregate predicted probability distribution using optional weights.

get_prob_dist_for_prediction_moment : function
    Calculate both predicted distributions and observed values for a given date using test data.

get_prob_dist : function
    Calculate probability distributions for each snapshot date based on given model predictions.

get_prob_dist_without_patient_snapshots : function
    Calculate probability distributions for each snapshot date using an EmpiricalSurvivalPredictor.

"""

import pandas as pd
import sympy as sym
from sympy import expand, symbols
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Tuple
from patientflow.predictors.incoming_admission_predictors import (
    EmpiricalIncomingAdmissionPredictor,
)


def create_symbols(n):
    """
    Generate a sequence of symbolic objects intended for use in mathematical expressions.

    Parameters
    ----------
    n : int
        Number of symbols to create.

    Returns
    -------
    tuple
        A tuple containing the generated symbolic objects.

    """
    return symbols(f"r0:{n}")


def compute_core_expression(ri, s):
    """
    Compute a symbolic expression involving a basic mathematical operation with a symbol and a constant.

    Parameters
    ----------
    ri : float
        The constant value to substitute into the expression.
    s : Symbol
        The symbolic object used in the expression.

    Returns
    -------
    Expr
        The symbolic expression after substitution.

    """
    r = sym.Symbol("r")
    core_expression = (1 - r) + r * s
    return core_expression.subs({r: ri})


def build_expression(syms, n):
    """
    Construct a cumulative product expression by combining individual symbolic expressions.

    Parameters
    ----------
    syms : iterable
        Iterable containing symbols to use in the expressions.
    n : int
        The number of terms to include in the cumulative product.

    Returns
    -------
    Expr
        The cumulative product of the expressions.

    """
    s = sym.Symbol("s")
    expression = 1
    for i in range(n):
        expression *= compute_core_expression(syms[i], s)
    return expression


def expression_subs(expression, n, predictions):
    """
    Substitute values into a symbolic expression based on a mapping from symbols to predictions.

    Parameters
    ----------
    expression : Expr
        The symbolic expression to perform substitution on.
    n : int
        Number of symbols and corresponding predictions.
    predictions : list
        List of numerical predictions to substitute.

    Returns
    -------
    Expr
        The expression after performing the substitution.

    """
    syms = create_symbols(n)
    substitution = dict(zip(syms, predictions))
    return expression.subs(substitution)


def return_coeff(expression, i):
    """
    Extract the coefficient of a specified power from an expanded symbolic expression.

    Parameters
    ----------
    expression : Expr
        The expression to expand and extract from.
    i : int
        The power of the term whose coefficient is to be extracted.

    Returns
    -------
    number
        The coefficient of the specified power in the expression.

    """
    s = sym.Symbol("s")
    return expand(expression).coeff(s, i)


def model_input_to_pred_proba(model_input, model):
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
    DataFrame
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
    predictions_proba, weights=None, normal_approx_threshold=30
):
    """
    Convert individual probability predictions into aggregate predicted probability distribution using optional weights.
    Uses a Normal approximation for large datasets (> normal_approx_threshold) for better performance.

    Parameters
    ----------
    predictions_proba : DataFrame
        A DataFrame containing the probability predictions; must have a single column named 'pred_proba'.
    weights : array-like, optional
        An array of weights, of the same length as the DataFrame rows, to apply to each prediction.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in predictions_proba exceeds this threshold, use a Normal distribution approximation.
        Set to None or a very large number to always use the exact symbolic computation.

    Returns
    -------
    DataFrame
        A DataFrame with a single column 'agg_proba' showing the aggregated probability,
        indexed from 0 to n, where n is the number of predictions.
    """
    n = len(predictions_proba)

    if n == 0:
        agg_predicted_dict = {0: 1}
    elif normal_approx_threshold is not None and n > normal_approx_threshold:
        # Apply a normal approximation for large datasets
        import numpy as np
        from scipy.stats import norm

        # Apply weights if provided
        if weights is not None:
            probs = predictions_proba["pred_proba"].values * weights
        else:
            probs = predictions_proba["pred_proba"].values

        # Calculate mean and variance for the normal approximation
        # For a sum of Bernoulli variables, mean = sum of probabilities
        mean = probs.sum()
        # Variance = sum of p_i * (1-p_i)
        variance = (probs * (1 - probs)).sum()

        # Handle the case where variance is zero (all probabilities are 0 or 1)
        if variance == 0:
            # If variance is zero, all probabilities are the same (either all 0 or all 1)
            # The distribution is deterministic - all probability mass is at the mean
            agg_predicted_dict = {int(round(mean)): 1.0}
        else:
            # Generate probabilities for each possible count using normal approximation
            counts = np.arange(n + 1)
            agg_predicted_dict = {}

            for i in counts:
                # Probability that count = i is the probability that a normal RV falls between i-0.5 and i+0.5
                if i == 0:
                    p = norm.cdf(0.5, loc=mean, scale=np.sqrt(variance))
                elif i == n:
                    p = 1 - norm.cdf(n - 0.5, loc=mean, scale=np.sqrt(variance))
                else:
                    p = norm.cdf(i + 0.5, loc=mean, scale=np.sqrt(variance)) - norm.cdf(
                        i - 0.5, loc=mean, scale=np.sqrt(variance)
                    )
                agg_predicted_dict[i] = p

            # Normalize to ensure the probabilities sum to 1
            total = sum(agg_predicted_dict.values())
            if total > 0:
                for i in agg_predicted_dict:
                    agg_predicted_dict[i] /= total
            else:
                # If all probabilities are zero, set a uniform distribution
                n = len(agg_predicted_dict)
                for i in agg_predicted_dict:
                    agg_predicted_dict[i] = 1.0 / n
    else:
        # Use the original symbolic computation for smaller datasets
        local_proba = predictions_proba.copy()
        if weights is not None:
            local_proba["pred_proba"] *= weights

        syms = create_symbols(n)
        expression = build_expression(syms, n)
        expression = expression_subs(expression, n, local_proba["pred_proba"])
        agg_predicted_dict = {i: return_coeff(expression, i) for i in range(n + 1)}

    agg_predicted = pd.DataFrame.from_dict(
        agg_predicted_dict, orient="index", columns=["agg_proba"]
    )
    return agg_predicted


def get_prob_dist_for_prediction_moment(
    X_test,
    model,
    weights=None,
    inference_time=False,
    y_test=None,
    category_filter=None,
    normal_approx_threshold=30,
):
    """
    Calculate both predicted distributions and observed values for a given date using test data.

    Parameters
    ----------
    X_test : array-like
        Test features for a specific snapshot date.
    model : object or TrainedClassifier
        Either a predictive model which provides a `predict_proba` method,
        or a TrainedClassifier object containing a pipeline.
    weights : array-like, optional
        Weights to apply to the predictions for aggregate calculation.
    inference_time : bool, optional (default=False)
        If True, do not calculate or return actual aggregate.
    y_test : array-like, optional
        Actual outcomes corresponding to the test features. Required if inference_time is False.
    category_filter : array-like, optional
        Boolean mask indicating which samples belong to the specific outcome category being analyzed.
        Should be the same length as y_test.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in X_test exceeds this threshold, use a Normal distribution approximation.
        Set to None or a very large number to always use the exact symbolic computation.

    Returns
    -------
    dict
        A dictionary with keys 'agg_predicted' and, if inference_time is False, 'agg_observed'.

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
            # Apply category filter when calculating observed sum
            if category_filter is None:
                prediction_moment_dict["agg_observed"] = sum(y_test)
            else:
                prediction_moment_dict["agg_observed"] = sum(y_test & category_filter)
    else:
        prediction_moment_dict["agg_predicted"] = pd.DataFrame(
            {"agg_proba": [1]}, index=[0]
        )
        if not inference_time:
            prediction_moment_dict["agg_observed"] = 0

    return prediction_moment_dict


def get_prob_dist(
    snapshots_dict,
    X_test,
    y_test,
    model,
    weights=None,
    verbose=False,
    category_filter=None,
    normal_approx_threshold=30,
):
    """
    Calculate probability distributions for each snapshot date based on given model predictions.

    Parameters
    ----------
    snapshots_dict : dict
        A dictionary mapping snapshot dates to indices in `X_test` and `y_test`.
        Must have datetime.date objects as keys and lists of indices as values.
    X_test : DataFrame or array-like
        Input test data to be passed to the model.
    y_test : array-like
        Observed target values.
    model : object or TrainedClassifier
        Either a predictive model which provides a `predict_proba` method,
        or a TrainedClassifier object containing a pipeline.
    weights : pandas.Series, optional
        A Series containing weights for the test data points, which may influence the prediction,
        by default None. If provided, the weights should be indexed similarly to `X_test` and `y_test`.
    verbose : bool, optional (default=False)
        If True, print progress information.
    category_filter : array-like, optional
        Boolean mask indicating which samples belong to the specific outcome category being analyzed.
        Should be the same length as y_test.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in a snapshot exceeds this threshold, use a Normal distribution approximation.
        Set to None or a very large number to always use the exact symbolic computation.

    Returns
    -------
    dict
        A dictionary mapping snapshot dates to probability distributions.

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
            print("This may take a minute or more")

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

            # Pass the normal_approx_threshold to get_prob_dist_for_prediction_moment
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
    verbose=False,
):
    """
    Calculate probability distributions for each snapshot date using an EmpiricalIncomingAdmissionPredictor.

    Parameters
    ----------
    snapshot_dates : array-like
        Array of dates for which to calculate probability distributions.
    test_visits : pandas.DataFrame
        DataFrame containing test visit data. Must have either:
        - start_time_col as a column and end_time_col as a column, or
        - start_time_col as the index and end_time_col as a column
    category : str
        Category to use for predictions (e.g., 'medical', 'surgical')
    prediction_time : tuple
        Tuple of (hour, minute) representing the time of day for predictions
    prediction_window : timedelta
        The prediction window duration
    start_time_col : str
        Name of the column containing start times (or index name if using index)
    end_time_col : str
        Name of the column containing end times
    model : EmpiricalSurvivalPredictor
        A fitted instance of EmpiricalSurvivalPredictor
    verbose : bool, optional (default=False)
        If True, print progress information

    Returns
    -------
    dict
        A dictionary mapping snapshot dates to probability distributions.

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
        raise ValueError("Model must be fitted before calling get_prob_dist_empirical")

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
