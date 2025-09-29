"""
Simple subgroup system for managing multiple patient prediction models.

This module provides an approach to training and using separate
SequenceToOutcomePredictor models for different patient subgroups.

Functions
---------
create_subgroup_system
    Main entry point - creates subgroup functions and multi-model predictor
create_special_category_objects
    Backward compatibility function (deprecated)
"""

import warnings
from typing import Dict, Any, Union, List, Type, Callable, Optional, Tuple
import pandas as pd
import numpy as np
from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
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


class MultiSubgroupPredictor:
    """Manages multiple SequenceToOutcomePredictor models, one per subgroup.

    Parameters
    ----------
    subgroup_functions : Dict[str, Callable]
        Dictionary mapping subgroup names to functions that identify patients in each subgroup
    base_predictor_class : Type[SequenceToOutcomePredictor], optional
        The SequenceToOutcomePredictor class to instantiate for each subgroup
    input_var : str, default='consultation_sequence'
        Name of the input sequence column
    grouping_var : str, default='final_sequence'
        Name of the grouping sequence column
    outcome_var : str, default='observed_specialty'
        Name of the outcome variable column
    min_samples : int, default=50
        Minimum number of samples required to train a model for a subgroup
    """

    def __init__(
        self,
        subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]],
        base_predictor_class: Optional[Type[SequenceToOutcomePredictor]],
        input_var: str = "consultation_sequence",
        grouping_var: str = "final_sequence",
        outcome_var: str = "observed_specialty",
        min_samples: int = 50,
    ):
        self.subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]] = (
            subgroup_functions
        )
        self.base_predictor_class: Optional[Type[SequenceToOutcomePredictor]] = (
            base_predictor_class
        )
        self.input_var: str = input_var
        self.grouping_var: str = grouping_var
        self.outcome_var: str = outcome_var
        self.min_samples: int = min_samples
        self.models: Dict[str, Any] = {}
        self.special_params: Optional[Dict[str, Any]] = None
        # Mapping from specialty -> list of subgroup names observed in training data
        self.specialty_to_subgroups: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame) -> "MultiSubgroupPredictor":
        """Train models for each subgroup that has sufficient data."""
        if self.base_predictor_class is None:
            raise ValueError("base_predictor_class must be provided to fit models")
        for name, func in self.subgroup_functions.items():
            # Filter to subgroup
            subgroup_data = X[X.apply(func, axis=1)]

            if len(subgroup_data) >= self.min_samples:
                # Train model
                model = self.base_predictor_class(
                    input_var=self.input_var,
                    grouping_var=self.grouping_var,
                    outcome_var=self.outcome_var,
                    apply_special_category_filtering=False,  # We handle subgroups ourselves
                )
                model.fit(subgroup_data)
                self.models[name] = model
            else:
                warnings.warn(f"Skipping {name}: only {len(subgroup_data)} samples")

        # Create backward compatibility params
        self.special_params = self._create_legacy_params()

        # Infer and store specialty->subgroups mapping from the provided training data
        try:
            self.specialty_to_subgroups = infer_specialty_to_subgroups(
                X, self.subgroup_functions, outcome_var=self.outcome_var
            )
        except Exception:
            # Be robust to unexpected data shapes; leave mapping empty if inference fails
            self.specialty_to_subgroups = {}
        return self

    def predict(
        self, input_data: Union[Tuple[Any, ...], pd.Series]
    ) -> Dict[str, float]:
        """Predict using appropriate subgroup model."""
        # Handle legacy tuple input (just sequence) - not supported without subgroup context
        if isinstance(input_data, tuple):
            raise RuntimeError(
                "Tuple input is not supported in MultiSubgroupPredictor; provide a full row (pd.Series) so the subgroup can be determined."
            )

        # Handle new Series input (full patient row)
        for name, func in self.subgroup_functions.items():
            if func(input_data) and name in self.models:
                sequence = input_data.get(self.input_var, ())
                return self.models[name].predict(sequence)

        # No subgroup model matched or model not trained for matched subgroup
        raise RuntimeError(
            "No trained subgroup model is available for this input; ensure fit() has trained models for applicable subgroups."
        )

    def predict_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized per-row prediction returning a Series of dictionaries.

        Each element is a dict of outcome -> probability for that row's subgroup/model.
        Rows with no applicable subgroup/model (or missing input sequence) return None.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the columns used by subgroup functions and
            the input sequence column specified by `self.input_var`.

        Returns
        -------
        pd.Series
            Series of dictionaries (or None) aligned to df.index.
        """
        input_col = self.input_var

        # Validate expected input column exists
        if input_col not in df.columns:
            raise ValueError(
                f"Input column '{input_col}' not found in DataFrame. "
                "Ensure the DataFrame contains the configured input_var."
            )

        # Precompute subgroup membership masks
        subgroup_masks: Dict[str, pd.Series] = {
            name: df.apply(func, axis=1)
            for name, func in self.subgroup_functions.items()
        }

        # Validate masks are mutually exclusive
        if len(subgroup_masks) > 0:
            mask_df = pd.DataFrame(subgroup_masks).fillna(False)
            overlaps = mask_df.sum(axis=1) > 1
            if overlaps.any():
                raise ValueError(
                    "Subgroup functions overlap for some rows; ensure they are mutually exclusive."
                )

        # Initialize result series with NaN (conventional missing in pandas)
        result_series = pd.Series(index=df.index, dtype=object)

        # Normalize sequences to tuples; preserve None/NaN as None
        def _as_tuple(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            if isinstance(x, (list, pd.Series)):
                return tuple(x)
            return x

        # For each subgroup with a trained base model, compute predictions per unique sequence
        for name, mask in subgroup_masks.items():
            if name not in self.models:
                continue

            sub_idx = mask.fillna(False)
            if not sub_idx.any():
                continue

            base_model: SequenceToOutcomePredictor = self.models[name]
            seq_series = df.loc[sub_idx, input_col].map(_as_tuple)

            unique_seqs = pd.unique(seq_series.dropna())
            seq_to_probs: Dict[Any, Any] = {}
            for seq in unique_seqs:
                pred = base_model.predict(seq)
                # Treat empty/zero-sum predictions as missing (NaN) for consistency
                if not pred or (isinstance(pred, dict) and sum(pred.values()) == 0):
                    seq_to_probs[seq] = np.nan
                else:
                    seq_to_probs[seq] = pred

            # Map dicts back to rows; missing/NaN sequences become NaN
            sub_probs_series = seq_series.map(
                lambda s: seq_to_probs.get(s) if s in seq_to_probs else np.nan
            )

            # Direct assignment is safe because masks must be non-overlapping
            result_series.loc[sub_idx] = sub_probs_series

        return result_series

    def _create_legacy_params(self) -> Dict[str, Any]:
        """Create special_params for backward compatibility."""
        # Prefer an explicitly provided 'paediatric' subgroup function; otherwise fall back to no-op
        if "paediatric" in self.subgroup_functions:
            paediatric_func = self.subgroup_functions["paediatric"]
        else:
            # Fallback: no paediatric indicator available; use a no-op (always False)
            paediatric_func = lambda row: False

        # Create function map for all subgroups plus legacy keys
        func_map = {
            "paediatric": paediatric_func,
            "default": lambda row: not paediatric_func(row),
        }
        func_map.update(self.subgroup_functions)

        return {
            "special_category_func": paediatric_func,
            "special_category_dict": {
                "medical": 0.0,
                "surgical": 0.0,
                "haem/onc": 0.0,
                "paediatric": 1.0,
            },
            "special_func_map": func_map,
        }


def create_subgroup_system(
    columns: Union[List[str], pd.Index],
    base_predictor_class: Optional[Type[SequenceToOutcomePredictor]] = None,
    input_var: str = "consultation_sequence",
    grouping_var: str = "final_sequence",
    outcome_var: str = "observed_specialty",
) -> Dict[str, Any]:
    """
    Create subgroup system with 5 standard patient categories.

    Parameters
    ----------
    columns : list
        Available columns (for validation)
    base_predictor_class : Type[SequenceToOutcomePredictor], optional
        SequenceToOutcomePredictor class
    input_var : str, default='consultation_sequence'
        Name of the input sequence column
    grouping_var : str, default='final_sequence'
        Name of the grouping sequence column
    outcome_var : str, default='observed_specialty'
        Name of the outcome variable column

    Returns
    -------
    dict
        Contains 'predictor' and legacy compatibility fields
    """
    subgroup_functions = create_subgroup_functions()

    if base_predictor_class is not None:
        predictor = MultiSubgroupPredictor(
            subgroup_functions,
            base_predictor_class,
            input_var=input_var,
            grouping_var=grouping_var,
            outcome_var=outcome_var,
        )
    else:
        predictor = None

    # Create legacy compatibility layer
    paediatric_func = subgroup_functions["paediatric"]
    legacy_params = {
        "special_category_func": paediatric_func,
        "special_category_dict": {
            "medical": 0.0,
            "surgical": 0.0,
            "haem/onc": 0.0,
            "paediatric": 1.0,
        },
        "special_func_map": {
            "paediatric": paediatric_func,
            "default": lambda row: not paediatric_func(row),
            **subgroup_functions,
        },
    }

    return {
        "predictor": predictor,
        "subgroup_functions": subgroup_functions,
        # Legacy fields for backward compatibility
        "special_category_func": legacy_params["special_category_func"],
        "special_category_dict": legacy_params["special_category_dict"],
        "special_func_map": legacy_params["special_func_map"],
    }


def infer_specialty_to_subgroups(
    df: pd.DataFrame,
    subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]],
    outcome_var: str = "observed_specialty",
) -> Dict[str, List[str]]:
    """Infer mapping from specialty to contributing subgroups from data.

    For each specialty present in ``df[outcome_var]``, this computes how many
    rows belong to each subgroup (based on ``subgroup_functions``) and includes
    every subgroup with at least one occurrence for that specialty.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with at least ``outcome_var`` and columns needed by
        ``subgroup_functions``.
    subgroup_functions : Dict[str, Callable]
        Mapping of subgroup name -> boolean row predicate.
    outcome_var : str, default 'observed_specialty'
        Column containing the realized specialty.
        
    Returns
    -------
    Dict[str, List[str]]
        Mapping of specialty -> list of subgroup names to include. The list
        may be empty if no subgroup appears for that specialty.
    """
    if outcome_var not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_var}' not found in DataFrame."
        )

    # Compute subgroup membership masks (allowing for potential overlaps)
    subgroup_masks: Dict[str, pd.Series] = {
        name: df.apply(func, axis=1).fillna(False) for name, func in subgroup_functions.items()
    }

    # Build counts per (specialty, subgroup)
    result: Dict[str, List[str]] = {}
    specialties = df[outcome_var].dropna().unique().tolist()

    for specialty in specialties:
        spec_mask = df[outcome_var] == specialty
        spec_total = int(spec_mask.sum())
        if spec_total == 0:
            result[specialty] = []
            continue

        counts: Dict[str, int] = {
            subgroup: int((spec_mask & mask).sum()) for subgroup, mask in subgroup_masks.items()
        }

        # Include all subgroups observed at least once
        eligible = [subgroup for subgroup, cnt in counts.items() if cnt > 0]

        result[specialty] = eligible

    return result
