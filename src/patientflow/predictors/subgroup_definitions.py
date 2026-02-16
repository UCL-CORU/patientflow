"""
Shared subgroup definitions for patientflow predictors.

This module provides common subgroup identification functions and utilities
used across different predictors in the patientflow ecosystem.
"""

from typing import Dict, Callable, Union
import pandas as pd
from patientflow.predictors.legacy_compatibility import get_age


def _is_paediatric(row):
    """Return True if the patient is paediatric (age < 18)."""
    return get_age(row) < 18


def _is_adult(row):
    """Return True if the patient is an adult (age >= 18)."""
    return get_age(row) >= 18


def _is_adult_male_young(row):
    """Return True if the patient is a young adult male (18 <= age < 65)."""
    age = get_age(row)
    return 18 <= age < 65 and row.get("sex") == "M"


def _is_adult_female_young(row):
    """Return True if the patient is a young adult female (18 <= age < 65)."""
    age = get_age(row)
    return 18 <= age < 65 and row.get("sex") == "F"


def _is_adult_male_senior(row):
    """Return True if the patient is a senior male (age >= 65)."""
    age = get_age(row)
    return age >= 65 and row.get("sex") == "M"


def _is_adult_female_senior(row):
    """Return True if the patient is a senior female (age >= 65)."""
    age = get_age(row)
    return age >= 65 and row.get("sex") == "F"


def create_paediatric_adult_subgroup_functions() -> (
    Dict[str, Callable[[Union[pd.Series, dict]], bool]]
):
    """Create a simple paediatric/adult subgroup split.

    Uses [get_age][patientflow.predictors.legacy_compatibility.get_age] so
    the functions work with both ``age_on_arrival`` (numeric) and
    ``age_group`` (categorical) columns.

    Returns
    -------
    dict
        ``{"paediatric": <func>, "adult": <func>}``
    """
    return {
        "paediatric": _is_paediatric,
        "adult": _is_adult,
    }


def create_subgroup_functions() -> Dict[str, Callable[[Union[pd.Series, dict]], bool]]:
    """Create the 5 standard subgroup identification functions."""
    return {
        "paediatric": _is_paediatric,
        "adult_male_young": _is_adult_male_young,
        "adult_female_young": _is_adult_female_young,
        "adult_male_senior": _is_adult_male_senior,
        "adult_female_senior": _is_adult_female_senior,
    }
