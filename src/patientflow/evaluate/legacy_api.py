"""Legacy scalar helpers preserved from the former `evaluate` module.

Import from `patientflow.evaluate.legacy_api`, for example::

    from patientflow.evaluate.legacy_api import calculate_results, calc_mae_mpe
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np


def calculate_results(
    expected_values: List[Union[int, float]], observed_values: List[float]
) -> Dict[str, Union[List[Union[int, float]], float]]:
    """Calculate evaluation metrics based on expected and observed values.

    Parameters
    ----------
    expected_values : list of int or float
        Predicted or model-implied values aligned with `observed_values`.
    observed_values : list of float
        Realised values in the same order as `expected_values`.

    Returns
    -------
    dict[str, list of int or float or float]
        Keys `expected`, `observed` (echo inputs), `mae` (mean absolute
        error), and `mpe` (mean percentage error over non-zero observed only).

    Notes
    -----
    If either list is empty, returns zero MAE and MPE with the original lists
    preserved.
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
    """Calculate MAE and MPE for each prediction-time model key in the nested dict.

    Parameters
    ----------
    prob_dist_dict_all : dict
        Outer keys are model keys (for example `admissions_1030`). Values are
        per-date dicts; each date maps to inner dicts with `agg_predicted`
        (pandas `Series`) and `agg_observed` (scalar).
    use_most_probable : bool, optional
        If `True`, expected value is the mode of `agg_predicted`; if `False`,
        the expectation `sum(k * p(k))`. Default is `False`.

    Returns
    -------
    dict
        Same keys as the outer dict of `prob_dist_dict_all`, sorted by suffix
        time in the key name; values are the dict returned by
        `calculate_results`.

    Notes
    -----
    Sorting assumes keys look like `prefix_HHMM` (time in the last segment).
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
        time_str = key.split("_")[1]
        return int(time_str)

    sorted_results = dict(
        sorted(unsorted_results.items(), key=lambda x: get_time_value(str(x[0])))
    )

    return sorted_results
