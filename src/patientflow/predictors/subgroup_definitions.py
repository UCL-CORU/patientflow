"""
Shared subgroup definitions for patientflow predictors.

This module provides common subgroup identification functions and utilities
used across different predictors in the patientflow ecosystem.
"""

from typing import Dict, Callable, Optional, Union
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


def resolve_patient_subgroup(
    row: Union[pd.Series, dict],
    subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]],
) -> Optional[str]:
    """Resolve the subgroup a single patient row belongs to.

    Applies each subgroup predicate in ``subgroup_functions`` to ``row`` and
    returns the name of the single matching subgroup. This is the shared
    resolution rule used by both transfer routing
    ([compute_transfer_arrivals][patientflow.predict.transfers.compute_transfer_arrivals])
    and ED specialty routing
    ([MultiSubgroupPredictor][patientflow.predictors.subgroup_predictor.MultiSubgroupPredictor]).

    Parameters
    ----------
    row : pandas.Series or dict
        A single patient row exposing the columns used by the subgroup
        predicates (typically ``age_on_arrival``/``age_group`` and ``sex``).
    subgroup_functions : dict of str to callable
        Mapping of subgroup name to a boolean predicate over a row. The
        standard set comes from
        [create_subgroup_functions][patientflow.predictors.subgroup_definitions.create_subgroup_functions].

    Returns
    -------
    str or None
        The name of the matching subgroup, or ``None`` when no predicate
        matches. ``None`` typically arises for an adult with missing, null, or
        non-``M``/``F`` ``sex`` — such rows are excluded from subgroup routing
        by callers rather than pooled.

    Raises
    ------
    ValueError
        If more than one predicate matches the row. Subgroup predicates must be
        mutually exclusive (same contract as
        [MultiSubgroupPredictor.predict_dataframe][patientflow.predictors.subgroup_predictor.MultiSubgroupPredictor.predict_dataframe]).
    """
    matched = [name for name, func in subgroup_functions.items() if func(row)]
    if len(matched) > 1:
        raise ValueError(
            f"Patient row matches multiple subgroups {sorted(matched)}; "
            "subgroup functions must be mutually exclusive."
        )
    return matched[0] if matched else None


def assign_patient_subgroups(
    df: pd.DataFrame,
    subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]],
) -> pd.Series:
    """Resolve subgroups for every row of a frame (vectorised over predicates).

    Equivalent to applying
    [resolve_patient_subgroup][patientflow.predictors.subgroup_definitions.resolve_patient_subgroup]
    to each row, but computes membership masks once per predicate and checks
    mutual exclusivity across the whole frame in one pass.

    Parameters
    ----------
    df : pandas.DataFrame
        Patient rows exposing the columns used by the subgroup predicates.
    subgroup_functions : dict of str to callable
        Mapping of subgroup name to a boolean predicate over a row.

    Returns
    -------
    pandas.Series
        Series aligned to ``df.index`` whose values are the matching subgroup
        name, or ``None`` for rows that match no predicate.

    Raises
    ------
    ValueError
        If any row matches more than one predicate.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype=object)

    masks: Dict[str, pd.Series] = {
        name: df.apply(func, axis=1).fillna(False).astype(bool)
        for name, func in subgroup_functions.items()
    }

    if masks:
        mask_df = pd.DataFrame(masks, index=df.index)
        if (mask_df.sum(axis=1) > 1).any():
            raise ValueError(
                "Subgroup functions overlap for some rows; ensure they are "
                "mutually exclusive."
            )

    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    for name, mask in masks.items():
        result[mask] = name
    return result
