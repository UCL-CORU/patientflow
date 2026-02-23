"""
Legacy compatibility functions for backward compatibility with existing code.

This module contains functions that were previously in other modules but have been
moved here to break circular import dependencies. These functions maintain the
original API for backward compatibility.

Functions
---------
create_special_category_objects
    Legacy function for creating special category objects (deprecated)
get_age
    Helper function to get patient age from row data
"""

import warnings
from typing import Dict, Any, Union, List
import pandas as pd


def get_age(row: Union[pd.Series, dict]) -> int:
    """Get patient age, defaulting to representative values for age groups."""
    if "age_on_arrival" in row and pd.notna(row["age_on_arrival"]):
        return int(row["age_on_arrival"])

    age_group = str(row.get("age_group", "")).lower()
    if "0-17" in age_group:
        return 10
    elif "65" in age_group:
        return 70
    else:
        return 40


def create_special_category_objects(
    columns: Union[List[str], pd.Index],
) -> Dict[str, Any]:
    """
    Legacy function - returns ONLY original paediatric/adult logic.

    This function is deprecated and maintained only for backward compatibility.
    Use create_subgroup_system() from subgroup_predictor.py instead.

    Parameters
    ----------
    columns : list or pd.Index
        Available columns (for validation, not currently used)

    Returns
    -------
    dict
        Dictionary containing special category functions and mappings
    """
    warnings.warn(
        "create_special_category_objects() is deprecated. Use create_subgroup_system() instead.",
        DeprecationWarning,
    )

    from patientflow.predictors.subgroup_definitions import (
        _is_paediatric,
        _is_adult,
    )

    # Return ONLY the original legacy structure
    return {
        "special_category_func": _is_paediatric,
        "special_category_dict": {
            "medical": 0.0,
            "surgical": 0.0,
            "haem/onc": 0.0,
            "paediatric": 1.0,
        },
        "special_func_map": {
            "paediatric": _is_paediatric,
            "default": _is_adult,
            # NO adult subgroup functions here!
        },
    }


def create_yta_filters(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Create filters for yet-to-arrive predictions (backward compatibility).

    Includes all specialties present in the legacy special category dict, e.g.
    'medical', 'surgical', 'haem/onc', and 'paediatric'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data

    Returns
    -------
    dict
        Dictionary mapping specialty names to filter dictionaries
    """
    special_params = create_special_category_objects(df.columns)
    special_category_dict = special_params["special_category_dict"]

    filters: Dict[str, Dict[str, Any]] = {}
    for specialty, is_paediatric_flag in special_category_dict.items():
        if is_paediatric_flag == 1.0:
            filters[specialty] = {"is_child": True}
        else:
            filters[specialty] = {"specialty": specialty, "is_child": False}

    return filters
