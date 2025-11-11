"""
Shared subgroup definitions for patientflow predictors.

This module provides common subgroup identification functions and utilities
used across different predictors in the patientflow ecosystem.
"""

from typing import Dict, Callable, Union
import pandas as pd
from patientflow.predictors.legacy_compatibility import get_age


def create_subgroup_functions() -> Dict[str, Callable[[Union[pd.Series, dict]], bool]]:
    """Create the 5 standard subgroup identification functions."""

    def is_paediatric(row):
        return get_age(row) < 18

    def is_adult_male_young(row):
        age = get_age(row)
        return 18 <= age < 65 and row.get("sex") == "M"

    def is_adult_female_young(row):
        age = get_age(row)
        return 18 <= age < 65 and row.get("sex") == "F"

    def is_adult_male_senior(row):
        age = get_age(row)
        return age >= 65 and row.get("sex") == "M"

    def is_adult_female_senior(row):
        age = get_age(row)
        return age >= 65 and row.get("sex") == "F"

    return {
        "paediatric": is_paediatric,
        "adult_male_young": is_adult_male_young,
        "adult_female_young": is_adult_female_young,
        "adult_male_senior": is_adult_male_senior,
        "adult_female_senior": is_adult_female_senior,
    }
