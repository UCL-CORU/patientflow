"""Backward-compatible scalar helper functions.

This module provides the established ``calculate_results`` and ``calc_mae_mpe``
helpers that existing notebooks import directly.  The implementations are
self-contained so that the evaluation package has no dependency on its
predecessor module.
"""

import re
from typing import Any, Dict, List, Union

import numpy as np


def calculate_results(
    expected_values: List[Union[int, float]], observed_values: List[float]
) -> Dict[str, Union[List[Union[int, float]], float]]:
    """Calculate evaluation metrics based on expected and observed values.

    Parameters
    ----------
    expected_values : List[Union[int, float]]
        List of expected values.
    observed_values : List[float]
        List of observed values.

    Returns
    -------
    Dict[str, Union[List[Union[int, float]], float]]
        Dictionary containing:
        - expected : List[Union[int, float]]
            Original expected values
        - observed : List[float]
            Original observed values
        - mae : float
            Mean Absolute Error
        - mpe : float
            Mean Percentage Error
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
    """Calculate MAE and MPE for all prediction times in the given probability distribution dictionary.

    Parameters
    ----------
    prob_dist_dict_all : Dict[Any, Dict[Any, Dict[str, Any]]]
        Nested dictionary containing probability distributions.
    use_most_probable : bool, optional
        Whether to use the most probable value or mathematical expectation of
        the distribution.  Default is False.

    Returns
    -------
    Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]
        Dictionary of results sorted by prediction time, containing:
        - expected : List[Union[int, float]]
            Expected values for each prediction
        - observed : List[float]
            Observed values for each prediction
        - mae : float
            Mean Absolute Error
        - mpe : float
            Mean Percentage Error
    """
    unsorted_results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

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

    def get_time_value(key: str) -> int:
        match = re.search(r"(\d{3,4})$", key)
        if match:
            return int(match.group(1))
        return 99999

    sorted_results = dict(
        sorted(unsorted_results.items(), key=lambda x: get_time_value(x[0]))
    )

    return sorted_results
