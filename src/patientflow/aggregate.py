"""
Aggregate Prediction From Patient-Level Probabilities

This submodule provides functions to aggregate patient-level predicted probabilities into a probability distribution.
The module uses symbolic mathematics to generate and manipulate expressions, enabling the computation of aggregate probabilities based on individual patient-level predictions.

Dependencies:
    - numpy: For array operations and numerical calculations.
    - pandas: To handle and manipulate tabular data (DataFrames) for analysis.
    - sympy: A symbolic mathematics library for building and manipulating symbolic expressions, particularly for calculating probabilities.

Functions
---------
create_symbols(n):
    Generates a list of symbolic variables to represent probability terms.

    Parameters
    ----------
    n : int
        Number of symbolic variables to generate.

    Returns
    -------
    list of sympy.Symbol
        A list containing n symbolic variables.

compute_core_expression(ri, s):
    Computes a symbolic expression using symbolic variables and constants.

    Parameters
    ----------
    ri : float
        A constant value (often a probability).
    s : sympy.Symbol
        A symbolic variable.

    Returns
    -------
    sympy.Mul
        A symbolic expression representing the product of `ri` and `s`.

build_expression(syms, n):
    Constructs a cumulative product of symbolic expressions using symbolic variables.

    Parameters
    ----------
    syms : list of sympy.Symbol
        A list of symbolic variables.
    n : int
        The number of terms to include in the cumulative product.

    Returns
    -------
    sympy.Expr
        A symbolic expression representing the cumulative product of `syms`.

expression_subs(expression, n, predictions):
    Substitutes numeric values into a symbolic expression.

    Parameters
    ----------
    expression : sympy.Expr
        A symbolic expression to perform substitution on.
    n : int
        The number of variables to substitute.
    predictions : array-like
        Numeric values (e.g., predicted probabilities) to substitute into the expression.

    Returns
    -------
    sympy.Expr
        The symbolic expression after substitution.

return_coeff(expression, i):
    Extracts the coefficient corresponding to a specific term in an expanded symbolic expression.

    Parameters
    ----------
    expression : sympy.Expr
        A symbolic expression that has been expanded.
    i : int
        The index of the term for which the coefficient is to be extracted.

    Returns
    -------
    float
        The coefficient for the i-th term.

model_input_to_pred_proba(model_input, model):
    Converts input data into predicted probabilities using the provided model.

    Parameters
    ----------
    model_input : array-like
        The input data to feed into the model.
    model : object
        A predictive model object that implements a `predict_proba` method.

    Returns
    -------
    array-like
        The predicted probabilities output by the model.

pred_proba_to_agg_predicted(predictions_proba, weights=None, normal_approx_threshold=30):
    Aggregates individual predicted probabilities into an overall prediction using provided weights.
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

get_prob_dist_for_prediction_moment(X_test, model, weights=None, inference_time=False, y_test=None, category_filter=None, normal_approx_threshold=30):
    Computes predicted and observed probabilities for a specific prediction date.

    Parameters
    ----------
    X_test : DataFrame or array-like
        Input test data to be passed to the model for prediction.
    model : object or TrainedClassifier
        Either a predictive model which provides a `predict_proba` method,
        or a TrainedClassifier object containing a pipeline.
    weights : array-like, optional
        Weights for aggregating the predicted probabilities.
    inference_time : bool
        Indicates whether the function is used in inference mode (i.e., whether observed data is available).
    y_test : array-like
        Observed target values corresponding to the test data (optional for inference).
    category_filter : array-like, optional
        Boolean mask indicating which samples belong to the specific outcome category being analyzed.
        Should be the same length as y_test.
    normal_approx_threshold : int, optional (default=30)
        If the number of rows in X_test exceeds this threshold, use a Normal distribution approximation.
        Set to None or a very large number to always use the exact symbolic computation.

    Returns
    -------
    dict
        A dictionary containing the predicted and, if applicable, observed probability distributions.

get_prob_dist(snapshots_dict, X_test, y_test, model, weights=None, verbose=False, category_filter=None, normal_approx_threshold=30):
    Computes probability distributions for multiple snapshot dates.

    Parameters
    ----------
    snapshots_dict : dict
        A dictionary where keys are snapshot dates and values are associated metadata (e.g., test data).
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
        A dictionary where each key is a snapshot date and the value is the corresponding probability distribution.

    Raises
    ------
    ValueError
        If snapshots_dict is not properly formatted or empty.
        If model has no predict_proba method and is not a TrainedClassifier.

    Example Usage
    -------------
    # Assuming a predictive model and test data are available
    snapshot_dates = ['2023-01-01', '2023-01-02']
    predicted_distribution = get_prob_dist(snapshot_dates, dataset, X_test, y_test, model)
    print(predicted_distribution)

"""

import pandas as pd
import sympy as sym
from sympy import expand, symbols
from datetime import date, datetime, time, timedelta, timezone
from patientflow.calculate.admission_in_prediction_window import calculate_probability
from typing import Dict, List, Tuple, Union
from scipy.stats import rv_discrete
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor


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
        A Series containing weights for the test data points.
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


def get_prob_dist_without_patient_snapshots(
    prediction_time: Tuple[int, int],
    categories: List[str],
    model: Union[WeightedPoissonPredictor, rv_discrete],
    test_df: pd.DataFrame,
    prediction_window: int,
    snapshot_dates: List[date],
    x1: float = 0.0,  # Default float value instead of None
    y1: float = 0.0,  # Default float value instead of None
    x2: float = 0.0,  # Default float value instead of None
    y2: float = 0.0,  # Default float value instead of None
    datetime_col: str = "arrival_datetime",
    max_range: int = 20,
) -> Dict[str, Dict[date, Dict[str, Union[pd.DataFrame, float]]]]:
    """
    Calculate probability distributions for yet-to-arrive patients for each category at a specific prediction time.

    Args:
        prediction_time: Tuple of (hour, minute) representing the prediction time
        categories: List of categories to analyze
        model: Prediction model (can be WeightedPoissonPredictor or a statistical distribution)
        test_df: DataFrame containing test set inpatient arrivals
        prediction_window: Time window for predictions in minutes
        snapshot_dates: List[date]
            List of dates to analyze
        x1: float
            First x-coordinate for curve parameter (required for WeightedPoissonPredictor)
        y1: float
            First y-coordinate for curve parameter (required for WeightedPoissonPredictor)
        x2: float
            Second x-coordinate for curve parameter (required for WeightedPoissonPredictor)
        y2: float
            Second y-coordinate for curve parameter (required for WeightedPoissonPredictor)
        datetime_col: Name of the column containing arrival datetimes (default: 'arrival_datetime')
        max_range: Maximum number of arrivals to consider in probability distribution (default: 20)

    Returns:
        Dictionary containing probability distributions for each category
    """
    # Validate prediction_time format
    if (
        not isinstance(prediction_time, (list, tuple))
        or len(prediction_time) != 2
        or not isinstance(prediction_time[0], int)
        or not isinstance(prediction_time[1], int)
    ):
        raise ValueError("prediction_time must be a (hour, minute) tuple")

    # Check if model is WeightedPoissonPredictor
    is_weighted_poisson = (
        hasattr(model, "__class__")
        and model.__class__.__name__ == "WeightedPoissonPredictor"
    )

    # Additional validation for WeightedPoissonPredictor
    if is_weighted_poisson:
        # Validate curve parameters are provided
        if x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0:
            raise ValueError(
                "Meaningful curve parameters (x1, y1, x2, y2) are required for WeightedPoissonPredictor"
            )

        # Validate prediction window
        if (
            not hasattr(model, "prediction_window")
            or model.prediction_window != prediction_window
        ):
            raise ValueError(
                f"model.prediction_window ({model.prediction_window}) does not match provided prediction_window ({prediction_window})"
            )

        # Validate categories are subset of model weights keys
        if not hasattr(model, "weights"):
            raise ValueError("WeightedPoissonPredictor must have 'weights' attribute")
        valid_categories = set(model.weights.keys())
        invalid_categories = set(categories) - valid_categories
        if invalid_categories:
            raise ValueError(
                f"Categories {invalid_categories} not found in model weights. Valid categories are {valid_categories}"
            )

    # Validate datetime_col exists in test_df
    if datetime_col not in test_df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in test_df")

    # Initialize dictionary to store probability distributions
    prob_dist_dict: Dict[str, Dict[date, Dict[str, Union[pd.DataFrame, float]]]] = {}
    hour, minute = prediction_time

    # Loop through each category
    for category in categories:
        prob_dist_dict[category] = {}

        # Get predicted distribution
        if is_weighted_poisson:
            prediction_context = {category: {"prediction_time": prediction_time}}
            agg_predicted_for_prediction_time = model.predict(
                prediction_context, x1, y1, x2, y2
            )[category]
        else:
            # Assume model is a statistical distribution (e.g., from scipy.stats)
            agg_predicted_for_prediction_time = pd.DataFrame(
                {"agg_proba": [model.pmf(k) for k in range(max_range)]},
                index=range(max_range),
            )

        # Calculate distributions for each date
        for date_val in snapshot_dates:
            snapshot_datetime = datetime.combine(
                date_val, time(hour=hour, minute=minute), tzinfo=timezone.utc
            )
            prob_dist_dict[category][date_val] = {}

            # Store predicted distribution
            prob_dist_dict[category][date_val]["agg_predicted"] = (
                agg_predicted_for_prediction_time
            )

            # Calculate observed count of patients who arrived during the prediction window
            observed_patients = test_df[
                (test_df[datetime_col] > snapshot_datetime)
                & (
                    test_df[datetime_col]
                    <= snapshot_datetime + timedelta(minutes=prediction_window)
                )
                & (test_df.specialty == category)
            ]

            if is_weighted_poisson:
                # Apply weighting for WeightedPoissonPredictor
                hours_til_arrival = (
                    observed_patients[datetime_col] - snapshot_datetime
                ).dt.total_seconds() / 3600
                remaining_hours_in_window = prediction_window / 60 - hours_til_arrival

                prob_dist_dict[category][date_val]["agg_observed"] = (
                    remaining_hours_in_window.apply(
                        lambda x: calculate_probability(
                            elapsed_los_td_hrs=0,
                            prediction_window_hrs=x,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    ).sum()
                )
            else:
                # Simple count for other model types
                prob_dist_dict[category][date_val]["agg_observed"] = len(
                    observed_patients
                )

    return prob_dist_dict
