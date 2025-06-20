"""
This module implements a `ValueToOutcomePredictor` class that models and predicts the probability distribution
of outcomes based on a single categorical input. The class builds a model based on training data, where
input values are mapped to specific outcome categories through an intermediate grouping variable. It provides
methods to fit the model, compute probabilities, and make predictions on unseen data.

Classes
-------
ValueToOutcomePredictor : sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
    A model that predicts the probability of ending in different outcome categories based on a single input value.
    Note: All inputs are expected to be strings. None values will be converted to empty strings during preprocessing.
"""

from typing import Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

from patientflow.prepare import create_special_category_objects


class ValueToOutcomePredictor(BaseEstimator, TransformerMixin):
    """
    A class to model predictions for categorical data using a single input value and grouping variable.
    This class implements both the `fit` and `predict` methods from the parent sklearn classes.

    Parameters
    ----------
    input_var : str
        Name of the column representing the input value in the DataFrame.
    grouping_var : str
        Name of the column representing the grouping value in the DataFrame.
    outcome_var : str
        Name of the column representing the outcome category in the DataFrame.
    apply_special_category_filtering : bool, default=True
        Whether to filter out special categories of patients before fitting the model.
    admit_col : str, default='is_admitted'
        Name of the column indicating whether a patient was admitted.

    Attributes
    ----------
    weights : dict
        A dictionary storing the probabilities of different input values leading to specific outcome categories.
    input_to_grouping_probs : pd.DataFrame
        A DataFrame that stores the computed probabilities of input values being associated with different grouping values.
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

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data before fitting the model.

        Steps include:
        1. Selecting only admitted patients with a non-null specialty
        2. Optionally filtering out special categories
        3. Converting input values to strings and handling nulls

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

        # Step 3: Convert input values to strings and handle nulls
        if self.input_var in df.columns:
            df[self.input_var] = df[self.input_var].fillna("").astype(str)

        if self.grouping_var in df.columns:
            df[self.grouping_var] = df[self.grouping_var].fillna("").astype(str)

        return df

    def fit(self, X: pd.DataFrame) -> "ValueToOutcomePredictor":
        """
        Fits the predictor based on training data by computing the proportion of each input value
        ending in specific outcome variable categories.

        Automatically preprocesses the data before fitting. During preprocessing, any null values in the
        input and grouping variables are converted to empty strings. These empty strings are then used
        as keys in the model's weights dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing at least the columns specified by `input_var`, `grouping_var`, and `outcome_var`.

        Returns
        -------
        self : ValueToOutcomePredictor
            The fitted ValueToOutcomePredictor model with calculated probabilities for each input value.
            The weights dictionary will contain an empty string key ('') for any null values from the input data.
        """

        # Store metrics about the training data
        self.metrics["train_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.metrics["train_set_no"] = len(X)
        if not X.empty:
            self.metrics["start_date"] = X["snapshot_date"].min()
            self.metrics["end_date"] = X["snapshot_date"].max()

        # Preprocess the data
        X = self._preprocess_data(X)

        # For each grouping value count the number of observed categories
        X_grouped = (
            X.groupby(self.grouping_var)[self.outcome_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Calculate the total number of times each grouping value occurred
        row_totals = X_grouped.sum(axis=1)

        # Calculate for each grouping value, the proportion of ending with each observed specialty
        proportions = X_grouped.div(row_totals, axis=0).fillna(0)

        # Calculate probabilities for each input value
        input_probs = {}
        for input_val in X[self.input_var].unique():
            # Get all grouping values associated with this input value
            grouping_vals = X[X[self.input_var] == input_val][
                self.grouping_var
            ].unique()

            # Calculate probability distribution of grouping values for this input value
            input_to_group_probs = X[X[self.input_var] == input_val][
                self.grouping_var
            ].value_counts(normalize=True)

            # Get the probability distribution of outcomes for all relevant grouping values
            # This includes all rows in proportions where the grouping value appears for this input
            group_to_outcome_probs = proportions.loc[grouping_vals]

            # Ensure the rows are aligned by reindexing group_to_outcome_probs
            aligned_group_to_outcome = group_to_outcome_probs.reindex(
                input_to_group_probs.index
            )

            # Create outer product matrix of probabilities:
            # - Rows represent grouping values
            # - Columns represent outcome categories
            # Each cell contains the joint probability of the grouping value and outcome
            input_to_outcome_probs = pd.DataFrame(
                input_to_group_probs.values.reshape(-1, 1)
                * aligned_group_to_outcome.values,
                index=input_to_group_probs.index,
                columns=group_to_outcome_probs.columns,
            )

            # Sum across grouping values to get final probability distribution for this input value
            input_probs[input_val] = input_to_outcome_probs.sum().to_dict()

        # Clean the keys to remove excess string quotes
        def clean_key(key):
            if isinstance(key, str):
                # Remove surrounding quotes if they exist
                if key.startswith("'") and key.endswith("'"):
                    return key[1:-1]
            return key

        # Note: cleaned_dict will contain an empty string key ('') for any null values from the input data
        # This is because null values are converted to empty strings during preprocessing
        cleaned_dict = {clean_key(k): v for k, v in input_probs.items()}

        # save probabilities as weights within the model
        self.weights = cleaned_dict

        # save the input to grouping probabilities for use as a reference
        self.input_to_grouping_probs = self._probability_of_input_to_grouping_value(X)

        return self

    def _probability_of_input_to_grouping_value(self, X):
        """
        Computes the probabilities of different input values leading to specific grouping values.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing at least the columns specified by `input_var` and `grouping_var`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the probabilities of input values leading to grouping values.
        """
        # For each input value count the number of grouping values
        X_grouped = (
            X.groupby(self.input_var)[self.grouping_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Calculate the total number of times each input value occurred
        row_totals = X_grouped.sum(axis=1)

        # Calculate for each grouping value, the proportion of ending with each grouping value
        proportions = X_grouped.div(row_totals, axis=0)

        # Calculate the probability of each input value occurring in the original data
        proportions["probability_of_input_value"] = row_totals / row_totals.sum()

        return proportions

    def predict(self, input_value: str) -> Dict[str, float]:
        """
        Predicts the probabilities of ending in various outcome categories for a given input value.

        Parameters
        ----------
        input_value : str
            The input value to predict outcomes for. None values will be handled appropriately.

        Returns
        -------
        dict
            A dictionary of categories and the probabilities that the input value will end in them.
        """
        if input_value is None or pd.isna(input_value):
            return self.weights.get("", {})

        # Convert input to string if it isn't already
        input_value = str(input_value)

        # Return a direct lookup of probabilities if possible
        if input_value in self.weights:
            return self.weights[input_value]

        # If no relevant data is found, return null probabilities
        return self.weights.get(None, {})
