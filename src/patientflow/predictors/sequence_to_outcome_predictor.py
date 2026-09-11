"""
This module implements a `SequenceToOutcomePredictor` class that models and predicts the probability distribution
of sequences in categorical data. The class builds a model based on training data, where input sequences
are mapped to specific outcome categories. It provides methods to fit the model, compute sequence-based
probabilities, and make predictions on an unseen dataset of input sequences.

Classes
-------
SequenceToOutcomePredictor : sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
    A model that predicts the probability of ending in different outcome categories based on input sequences.

Notes
-----
All sequence inputs are expected to be tuples. Lists are converted to tuples automatically, and ``None``
values are converted to empty tuples.
"""

from typing import Dict
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

from patientflow.predictors.legacy_compatibility import create_special_category_objects


class SequenceToOutcomePredictor(BaseEstimator, TransformerMixin):
    """
    A class to model sequence-based predictions for categorical data using input and grouping sequences.
    This class implements both the `fit` and `predict` methods from the parent sklearn classes.

    Parameters
    ----------
    input_var : str
        Name of the column representing the input sequence in the DataFrame.
    grouping_var : str
        Name of the column representing the grouping sequence in the DataFrame.
    outcome_var : str
        Name of the column representing the outcome category in the DataFrame.
    apply_special_category_filtering : bool, default=True
        Whether to filter out special categories of patients before fitting the model.
    admit_col : str, default='is_admitted'
        Name of the column indicating whether a patient was admitted.

    Attributes
    ----------
    weights : dict
        A dictionary storing the probabilities of different input sequences leading to specific outcome categories.
    input_to_grouping_probs : pd.DataFrame
        A DataFrame that stores the computed probabilities of input sequences being associated with different grouping sequences.
    special_params : dict, optional
        The special category parameters used for filtering, only populated if apply_special_category_filtering=True.
    metrics : dict
        A dictionary to store metrics related to the training process.
    """

    def __init__(
        self,
        input_var,
        grouping_var,
        outcome_var,
        apply_special_category_filtering=True,
        admit_col="is_admitted",
    ):
        self.input_var = input_var
        self.grouping_var = grouping_var
        self.outcome_var = outcome_var
        self.apply_special_category_filtering = apply_special_category_filtering
        self.admit_col = admit_col
        self.weights = None
        self.special_params = None
        self.metrics = {}

    def __repr__(self):
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n"
            f"    input_var='{self.input_var}',\n"
            f"    grouping_var='{self.grouping_var}',\n"
            f"    outcome_var='{self.outcome_var}',\n"
            f"    apply_special_category_filtering={self.apply_special_category_filtering},\n"
            f"    admit_col='{self.admit_col}'\n"
            f")"
        )

    def _ensure_tuple(self, sequence):
        """
        Convert a sequence to tuple if it's not already a tuple.
        Handles string cleaning to avoid double-quoting issues.

        Parameters
        ----------
        sequence : tuple, list, or None
            The sequence to convert

        Returns
        -------
        tuple
            The input sequence as a tuple, or an empty tuple if input was None
        """
        if sequence is None:
            return ()
        if isinstance(sequence, (list, pd.Series)):
            # Clean any quoted strings in the sequence
            cleaned_sequence = [
                ast.literal_eval(item)
                if isinstance(item, str) and item.startswith("'") and item.endswith("'")
                else item
                for item in sequence
            ]
            return tuple(cleaned_sequence) if cleaned_sequence else ()
        if isinstance(sequence, tuple):
            # Clean any quoted strings in the tuple
            return tuple(
                ast.literal_eval(item)
                if isinstance(item, str) and item.startswith("'") and item.endswith("'")
                else item
                for item in sequence
            )
        return sequence

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data before fitting the model.

        Steps include:
        1. Selecting only admitted patients with a non-null specialty
        2. Optionally filtering out special categories
        3. Converting sequence columns to tuple format if they aren't already

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing patient data.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame ready for model fitting.
        """
        # Make a copy to avoid modifying the original
        df = X.copy()

        # Step 1: Select only admitted patients with a non-null specialty
        if self.admit_col in df.columns:
            df = df[df[self.admit_col] & ~df[self.outcome_var].isnull()]

        # Step 2: Optionally apply filtering for special categories
        if self.apply_special_category_filtering:
            # Get configuration for categorizing patients based on columns
            self.special_params = create_special_category_objects(df.columns)

            # Extract function that identifies non-special category patients
            opposite_special_category_func = self.special_params["special_func_map"][
                "default"
            ]

            # Determine which category is the special category
            special_category_key = next(
                key
                for key, value in self.special_params["special_category_dict"].items()
                if value == 1.0
            )

            # Filter out special category patients
            df = df[
                df.apply(opposite_special_category_func, axis=1)
                & (df[self.outcome_var] != special_category_key)
            ]

        # Step 3: Convert sequence columns to tuple format
        if self.input_var in df.columns:
            df[self.input_var] = df[self.input_var].apply(self._ensure_tuple)

        if self.grouping_var in df.columns:
            df[self.grouping_var] = df[self.grouping_var].apply(self._ensure_tuple)

        return df

    def fit(self, X: pd.DataFrame) -> "SequenceToOutcomePredictor":
        """
        Fits the predictor by composing P(grouping sequence | input sequence) with
        P(outcome | grouping sequence), giving P(outcome | input sequence) for each
        observed input sequence.

        Automatically preprocesses the data before fitting.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing at least the columns specified by `input_var`, `grouping_var`, and `outcome_var`.

        Returns
        -------
        self : SequenceToOutcomePredictor
            The fitted SequenceToOutcomePredictor model with calculated probabilities for each sequence.
        """
        # Store metrics about the training data
        self.metrics["train_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.metrics["train_set_no"] = len(X)
        if not X.empty:
            self.metrics["start_date"] = X["snapshot_date"].min()
            self.metrics["end_date"] = X["snapshot_date"].max()

        # Preprocess the data
        X = self._preprocess_data(X)

        if X.empty:
            self.weights = {}
            self.input_to_grouping_probs = pd.DataFrame()
            return self

        # For each grouping sequence, count observed outcome categories
        X_grouped = (
            X.groupby(self.grouping_var)[self.outcome_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # P(specialty | grouping sequence)
        row_totals = X_grouped.sum(axis=1)
        proportions = X_grouped.div(row_totals, axis=0).fillna(0)

        def clean_tuple_key(key):
            if isinstance(key, tuple):
                return tuple(
                    ast.literal_eval(item)
                    if isinstance(item, str)
                    and item.startswith("'")
                    and item.endswith("'")
                    else item
                    for item in key
                )
            return key

        # P(specialty | input_seq) = sum over grouping sequences of
        #     P(grouping_seq | input_seq) * P(specialty | grouping_seq)
        input_to_specialty_probs = {}
        for input_seq in X[self.input_var].unique():
            prob_grouping_given_input = X.loc[
                X[self.input_var] == input_seq, self.grouping_var
            ].value_counts(normalize=True)
            prob_specialty_given_grouping = proportions.reindex(
                prob_grouping_given_input.index
            ).fillna(0)
            input_to_specialty_probs[clean_tuple_key(input_seq)] = (
                prob_specialty_given_grouping.mul(prob_grouping_given_input, axis=0)
                .sum()
                .to_dict()
            )

        self.weights = input_to_specialty_probs
        self.input_to_grouping_probs = self._probability_of_input_to_grouping_sequence(
            X
        )

        return self

    def _probability_of_input_to_grouping_sequence(self, X):
        """
        Computes the probabilities of different input sequences leading to specific grouping sequences.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing at least the columns specified by `input_var` and `grouping_var`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the probabilities of input sequences leading to grouping sequences.
        """
        # For each input sequence count the number of grouping sequences
        X_grouped = (
            X.groupby(self.input_var)[self.grouping_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # # Calculate the total number of times each input sequence occurred
        row_totals = X_grouped.sum(axis=1)

        # # Calculate for each grouping sequence, the proportion of ending with each grouping sequence
        proportions = X_grouped.div(row_totals, axis=0)

        # # Calculate the probability of each input sequence occurring in the original data
        proportions["probability_of_input_value"] = row_totals / row_totals.sum()

        return proportions

    def predict(self, input_sequence: tuple[str, ...]) -> Dict[str, float]:
        """
        Predicts the probabilities of ending in various outcome categories for a given input sequence.

        Parameters
        ----------
        input_sequence : tuple[str, ...]
            A tuple containing the categories that have been observed for an entity in the order they
            have been encountered. An empty tuple represents an entity with no observed categories.

        Returns
        -------
        dict
            A dictionary of categories and the probabilities that the input sequence will end in them.
        """
        input_sequence = self._ensure_tuple(input_sequence)

        if input_sequence is None or pd.isna(input_sequence):
            return self.weights.get(tuple(), {})

        # Return a direct lookup of probabilities if possible.
        if input_sequence in self.weights:
            return self.weights[input_sequence]

        # Otherwise, if the sequence has multiple elements, work back looking for a match
        while len(input_sequence) > 1:
            input_sequence_list = list(input_sequence)
            input_sequence = tuple(input_sequence_list[:-1])  # remove last element

            if input_sequence in self.weights:
                return self.weights[input_sequence]

        # If no relevant data is found:
        return self.weights.get(tuple(), {})
