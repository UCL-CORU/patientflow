"""Compatibility wrappers for existing evaluation helper functions.

These wrappers expose established scalar helper behaviour from the legacy
evaluation module while keeping the typed evaluation package self-contained.
"""

from typing import Any, Dict, List, Union

from patientflow.evaluate import calc_mae_mpe as _legacy_calc_mae_mpe
from patientflow.evaluate import calculate_results as _legacy_calculate_results


def calculate_results(
    expected_values: List[Union[int, float]], observed_values: List[float]
) -> Dict[str, Union[List[Union[int, float]], float]]:
    """Compute MAE/MPE using the established helper implementation.

    Parameters
    ----------
    expected_values
        Expected counts.
    observed_values
        Observed counts.

    Returns
    -------
    Dict[str, Union[List[Union[int, float]], float]]
        Metrics dictionary from the canonical implementation.
    """
    return _legacy_calculate_results(expected_values, observed_values)


def calc_mae_mpe(
    prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]],
    use_most_probable: bool = False,
) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """Compute per-time MAE/MPE on plotting-ready distribution dictionaries.

    Parameters
    ----------
    prob_dist_dict_all
        Distribution dictionary keyed by model key and snapshot date.
    use_most_probable
        Whether to use modal value instead of expectation for expected counts.

    Returns
    -------
    Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]
        Per-time metric dictionary from the canonical implementation.
    """
    return _legacy_calc_mae_mpe(
        prob_dist_dict_all=prob_dist_dict_all,
        use_most_probable=use_most_probable,
    )
