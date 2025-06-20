"""
This module implements a `SequenceToOutcomePredictor` class that models and predicts the probability distribution
of sequences in categorical data. The class builds a model based on training data, where input sequences
are mapped to specific outcome categories. It provides methods to fit the model, compute sequence-based
probabilities, and make predictions on an unseen datatset of input sequences.

Classes
-------
SequenceToOutcomePredictor : sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
    A model that predicts the probability of ending in different outcome categories based on input sequences.
    Note: All sequence inputs are expected to be tuples. Lists will be automatically converted to tuples,
    and None values will be converted to empty tuples.
"""

from typing import Dict
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

from patientflow.prepare import create_special_category_objects


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
        """Return a string representation of the estimator."""
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
        Fits the predictor based on training data by computing the proportion of each input variable sequence
        ending in specific outcome variable categories.

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

        # derive the names of the observed outcome variables from the data
        prop_keys = X[self.outcome_var].unique()

        # For each sequence count the number of observed categories
        X_grouped = (
            X.groupby(self.grouping_var)[self.outcome_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Calculate the total number of times each grouping sequence occurred
        row_totals = X_grouped.sum(axis=1)

        # Calculate for each grouping sequence, the proportion of ending with each observed specialty
        proportions = X_grouped.div(row_totals, axis=0)

        # Calculate the probability of each grouping sequence occurring in the original data
        proportions["probability_of_grouping_sequence"] = row_totals / row_totals.sum()

        # Reweight probabilities of ending with each observed specialty
        # by the likelihood of each grouping sequence occurring
        for col in proportions.columns[
            :-1
        ]:  # Avoid the last column which is the 'probability_of_grouping_sequence'
            proportions[col] *= proportions["probability_of_grouping_sequence"]

        # Convert final sequence to a string in order to conduct string searches on it
        proportions["grouping_sequence_to_string"] = proportions.index.map(
            lambda x: "-".join(map(str, x))
        )

        # Row-wise function to return, for each input sequence,
        # the proportion that end up in each final sequence and thereby
        # the probability of it ending in any observed category
        proportions["prob_input_var_ends_in_observed_specialty"] = proportions[
            "grouping_sequence_to_string"
        ].apply(lambda x: self._string_match_input_var(x, proportions, prop_keys))

        # Convert the prob_input_var_ends_in_observed_specialty column to a dictionary
        result_dict = proportions["prob_input_var_ends_in_observed_specialty"].to_dict()

        # Clean the key to remove excess strint quotes
        def clean_tuple_key(key):
            if isinstance(key, tuple):
                return tuple(
                    ast.literal_eval(item)
                    if item.startswith("'") and item.endswith("'")
                    else item
                    for item in key
                )
            return key

        cleaned_dict = {clean_tuple_key(k): v for k, v in result_dict.items()}

        # save prob_input_var_ends_in_observed_specialty as weights within the model
        self.weights = cleaned_dict

        # save the input to grouping probabilities for use as a reference
        self.input_to_grouping_probs = self._probability_of_input_to_grouping_sequence(
            X
        )

        return self

    def _string_match_input_var(self, input_var_string, proportions, prop_keys):
        """
        Matches a given input sequence string with grouped sequences (expressed as strings) in the dataset and aggregates
        their probabilities for each outcome category. This function filters the data to
        match only those rows where the *beginning* of the grouped sequence string
        matches the given input sequence string, allowing for partial matches.
        For instance, the sequence 'medical' will match 'medical, elderly' and 'medical, surgical'
        as well as 'medical' on its own. It computes the total probabilities of any input sequence ending
        in each outcome category, and normalizes these totals if possible.

        Parameters
        ----------
        input_var_string : str
            The sequence of inputs represented as a string, used to match against sequences in the proportions DataFrame.
        proportions : pd.DataFrame
            DataFrame containing proportions data with an additional column 'grouping_sequence_to_string'
            which includes string representations of sequences.
        prop_keys : np.array
            Array of unique outcomes to consider in calculations.

        Returns
        -------
        dict
            A dictionary where keys are outcome names and values are the aggregated and normalized probabilities
            of an input sequence ending in those outcomes.

        """
        # Filter rows where the grouped sequence string starts with the input sequence string
        props = proportions[
            proportions["grouping_sequence_to_string"].str.match("^" + input_var_string)
        ][prop_keys].sum()

        # Sum of all probabilities to normalize them
        props_total = props.sum()

        # Handle cases where the total probability is zero to avoid division by zero
        if props_total > 0:
            normalized_props = props / props_total
        else:
            normalized_props = (
                props * 0
            )  # Returns zero probabilities if no matches found

        return dict(zip(prop_keys, normalized_props))

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
        proportions["probability_of_grouping_sequence"] = row_totals / row_totals.sum()

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
