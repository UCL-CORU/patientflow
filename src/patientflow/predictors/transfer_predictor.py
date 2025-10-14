"""
Transfer probability predictor for patient movements between subspecialties.

This module provides a predictor for modelling patient transfers between hospital
subspecialties based on historical movement patterns. It computes the probability
that a departing patient will transfer to another subspecialty versus being
discharged, and the distribution of destination subspecialties for transfers.

The predictor follows sklearn conventions with fit() and predict() methods, allowing
it to be saved and loaded as a model artifact alongside other predictors in the
patientflow ecosystem.

Notes
-----
This predictor is designed to work in conjunction with inpatient departure classifiers
to provide a complete picture of subspecialty bed demand by accounting for both
external discharges and internal transfers.

The predictor stores transition probabilities between subspecialties, which can be
used at prediction time to model internal patient flows and compute arrival
distributions for each subspecialty from transfers.
"""

import pandas as pd
from typing import Dict, Set, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class TransferProbabilityEstimator(BaseEstimator, TransformerMixin):
    """Estimate patient transfer probabilities from subspecialty movement patterns.

    This estimator computes and stores transfer probabilities between subspecialties
    based on historical patient movement data. For each source subspecialty, it
    calculates:
    1. The probability that a departure results in a transfer (vs discharge)
    2. The distribution of destinations given that a transfer occurs

    The estimator follows sklearn conventions with fit() and predict() methods,
    allowing it to be saved and loaded as a model artifact.

    Parameters
    ----------
    source_col : str, default='current_subspecialty'
        Name of the column containing the source subspecialty (where patient is leaving from)
    destination_col : str, default='next_subspecialty'
        Name of the column containing the destination subspecialty (where patient is going to).
        None/NaN values indicate discharge rather than transfer.

    Attributes
    ----------
    source_col : str
        Name of the source subspecialty column
    destination_col : str
        Name of the destination subspecialty column
    transfer_probabilities : dict
        Nested dictionary containing transfer statistics for each subspecialty:
        {
            'source_subspecialty': {
                'prob_transfer': float,
                    Probability that a departure from this subspecialty results
                    in a transfer to another subspecialty (vs discharge)
                'destination_distribution': dict,
                    Distribution of destinations given a transfer occurs.
                    Maps destination subspecialty names to probabilities that
                    sum to 1.0
            }
        }
    subspecialties : set
        Set of all subspecialties in the system
    is_fitted_ : bool
        Whether the predictor has been fitted

    Examples
    --------
    >>> # Prepare training data with subspecialty movements
    >>> X = pd.DataFrame({
    ...     'current_subspecialty': ['cardiology', 'cardiology', 'surgery'],
    ...     'next_subspecialty': ['surgery', None, None]  # None = discharge
    ... })
    >>> subspecialties = {'cardiology', 'surgery', 'medicine'}
    >>>
    >>> # Train the estimator with default column names
    >>> estimator = TransferProbabilityEstimator()
    >>> estimator.fit(X, subspecialties)
    >>>
    >>> # Or with custom column names
    >>> estimator = TransferProbabilityEstimator(
    ...     source_col='from_ward',
    ...     destination_col='to_ward'
    ... )
    >>>
    >>> # Get transfer probabilities
    >>> prob_transfer = estimator.get_transfer_prob('cardiology')
    >>> destinations = estimator.get_destination_distribution('cardiology')

    Notes
    -----
    The input DataFrame should contain patient movement events where each row
    represents a departure from a subspecialty. The destination column should be
    None/NaN for discharges and contain the destination subspecialty name for
    transfers.

    All subspecialties in the system should be provided to ensure complete
    probability distributions, even for subspecialties with no observed transfers.
    """

    def __init__(
        self,
        source_col: str = "current_subspecialty",
        destination_col: str = "next_subspecialty",
    ):
        self.source_col = source_col
        self.destination_col = destination_col
        self.transfer_probabilities: Optional[Dict[str, Dict]] = None
        self.subspecialties: Optional[Set[str]] = None
        self.is_fitted_ = False

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if not self.is_fitted_:
            return (
                f"{class_name}(\n"
                f"    source_col='{self.source_col}',\n"
                f"    destination_col='{self.destination_col}',\n"
                f"    fitted=False\n"
                f")"
            )

        n_subspecialties = (
            len(self.subspecialties) if self.subspecialties is not None else 0
        )
        n_with_transfers = 0
        if self.transfer_probabilities is not None:
            n_with_transfers = sum(
                1
                for stats in self.transfer_probabilities.values()
                if stats["prob_transfer"] > 0
            )

        return (
            f"{class_name}(\n"
            f"    source_col='{self.source_col}',\n"
            f"    destination_col='{self.destination_col}',\n"
            f"    fitted=True,\n"
            f"    n_subspecialties={n_subspecialties},\n"
            f"    n_with_transfers={n_with_transfers}\n"
            f")"
        )

    def fit(
        self, X: pd.DataFrame, subspecialties: Set[str]
    ) -> "TransferProbabilityEstimator":
        """Fit the transfer probability estimator from patient movement data.

        This method computes transfer probabilities from historical patient
        movements between subspecialties. For each source subspecialty, it
        calculates what proportion of departures result in transfers and
        where those transfers go.

        Parameters
        ----------
        X : pandas.DataFrame
            Training DataFrame containing patient movement data with columns as
            specified by source_col and destination_col parameters.
        subspecialties : set of str
            Set of all subspecialty names in the system

        Returns
        -------
        self : TransferProbabilityEstimator
            The fitted estimator

        Raises
        ------
        ValueError
            If required columns are missing from X
        TypeError
            If subspecialties is not a set or collection of strings
        """
        # Validate inputs
        required_columns = {self.source_col, self.destination_col}
        if not required_columns.issubset(X.columns):
            missing = required_columns - set(X.columns)
            raise ValueError(
                f"Input DataFrame missing required columns: {missing}. "
                f"Required columns are: {required_columns}"
            )

        if not isinstance(subspecialties, (set, list, tuple)):
            raise TypeError(
                f"subspecialties must be a set, list, or tuple, got {type(subspecialties)}"
            )

        # Convert to set if needed
        self.subspecialties = set(subspecialties)

        # Validate that all destinations are in subspecialties
        destinations = X[self.destination_col].dropna().unique()
        unknown_destinations = set(destinations) - self.subspecialties
        if unknown_destinations:
            raise ValueError(
                f"Found destination subspecialties in data that are not in the "
                f"subspecialties set: {sorted(unknown_destinations)}. "
                f"Please ensure all destination subspecialties are included in the "
                f"subspecialties parameter."
            )

        # Compute transfer probabilities
        self.transfer_probabilities = self._prepare_transfer_probabilities(
            self.subspecialties, X
        )

        self.is_fitted_ = True
        return self

    def _prepare_transfer_probabilities(
        self, subspecialties: Set[str], X: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Prepare transfer probabilities dictionary from patient movement data.

        This is the core computation method that calculates transfer statistics
        for each subspecialty.

        Parameters
        ----------
        subspecialties : set of str
            Set of all subspecialty names
        X : pandas.DataFrame
            Training DataFrame with source_col and destination_col columns

        Returns
        -------
        dict
            Transfer probabilities dictionary with structure:
            {
                'source_subspecialty': {
                    'prob_transfer': float,
                    'destination_distribution': {
                        'destination': float, ...
                    }
                }
            }

        Notes
        -----
        Subspecialties with no observed movements will have prob_transfer=0.0
        and an empty destination_distribution.
        """
        transfer_probabilities = {}

        for source_subspecialty in subspecialties:
            # Filter to movements from this source
            source_movements = X[X[self.source_col] == source_subspecialty].copy()

            if len(source_movements) == 0:
                # No data for this subspecialty - set prob_transfer to 0
                transfer_probabilities[source_subspecialty] = {
                    "prob_transfer": 0.0,
                    "destination_distribution": {},
                }
                continue

            # Calculate prob_transfer: proportion that go to another subspecialty
            total_departures = len(source_movements)
            transfers = source_movements[source_movements[self.destination_col].notna()]
            num_transfers = len(transfers)

            prob_transfer = num_transfers / total_departures

            # If no transfers, include with empty destination distribution
            if num_transfers == 0:
                transfer_probabilities[source_subspecialty] = {
                    "prob_transfer": 0.0,
                    "destination_distribution": {},
                }
                continue

            # Calculate destination distribution among transfers
            destination_counts = transfers[self.destination_col].value_counts()
            destination_distribution = (destination_counts / num_transfers).to_dict()

            # Store results
            transfer_probabilities[source_subspecialty] = {
                "prob_transfer": prob_transfer,
                "destination_distribution": destination_distribution,
            }

        return transfer_probabilities

    def get_transfer_prob(self, source_subspecialty: str) -> float:
        """Get the probability that a departure from a subspecialty is a transfer.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty

        Returns
        -------
        float
            Probability that a departure from this subspecialty results in a
            transfer to another subspecialty (vs being discharged). Returns 0.0
            if no transfers observed.

        Raises
        ------
        ValueError
            If the predictor has not been fitted
        KeyError
            If source_subspecialty is not in the trained subspecialties
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if source_subspecialty not in self.transfer_probabilities:
            raise KeyError(
                f"Subspecialty '{source_subspecialty}' not found in trained model. "
                f"Available subspecialties: {sorted(self.transfer_probabilities.keys())}"
            )

        return self.transfer_probabilities[source_subspecialty]["prob_transfer"]

    def get_destination_distribution(
        self, source_subspecialty: str
    ) -> Dict[str, float]:
        """Get the distribution of destinations given that a transfer occurs.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty

        Returns
        -------
        dict
            Dictionary mapping destination subspecialty names to probabilities.
            Probabilities sum to 1.0. Returns empty dict if no transfers observed.

        Raises
        ------
        ValueError
            If the predictor has not been fitted
        KeyError
            If source_subspecialty is not in the trained subspecialties

        Notes
        -----
        This distribution is conditional on a transfer occurring. To get the
        unconditional probability of transferring to a specific destination,
        multiply by get_transfer_prob().
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if source_subspecialty not in self.transfer_probabilities:
            raise KeyError(
                f"Subspecialty '{source_subspecialty}' not found in trained model. "
                f"Available subspecialties: {sorted(self.transfer_probabilities.keys())}"
            )

        return self.transfer_probabilities[source_subspecialty][
            "destination_distribution"
        ]

    def predict(self, source_subspecialty: str) -> Dict[str, float]:
        """Get full transfer statistics for a source subspecialty.

        This is a convenience method that returns both the transfer probability
        and destination distribution in a single call.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty

        Returns
        -------
        dict
            Dictionary containing:
            - 'prob_transfer': Probability of transfer vs discharge
            - 'destination_distribution': Distribution of destinations

        Raises
        ------
        ValueError
            If the predictor has not been fitted
        KeyError
            If source_subspecialty is not in the trained subspecialties
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if source_subspecialty not in self.transfer_probabilities:
            raise KeyError(
                f"Subspecialty '{source_subspecialty}' not found in trained model. "
                f"Available subspecialties: {sorted(self.transfer_probabilities.keys())}"
            )

        return self.transfer_probabilities[source_subspecialty].copy()

    def get_all_transfer_probabilities(self) -> Dict[str, Dict]:
        """Get the complete transfer probabilities dictionary.

        Returns
        -------
        dict
            Complete nested dictionary of transfer probabilities for all
            subspecialties

        Raises
        ------
        ValueError
            If the predictor has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        return self.transfer_probabilities.copy()

    def get_transition_matrix(self) -> pd.DataFrame:
        """Format transition probabilities as a matrix.

        Creates a DataFrame with source subspecialties as rows and target
        subspecialties as columns, plus a 'Discharge' column. Each cell
        contains the probability of transitioning from the source (row) to
        the target (column).

        Returns
        -------
        pandas.DataFrame
            Transition probability matrix where:
            - Index: source subspecialties
            - Columns: target subspecialties + 'Discharge'
            - Values: transition probabilities (sum to 1.0 across each row)

        Raises
        ------
        ValueError
            If the predictor has not been fitted

        Examples
        --------
        >>> estimator = TransferProbabilityEstimator()
        >>> estimator.fit(X, subspecialties={'cardiology', 'surgery'})
        >>> matrix = estimator.get_transition_matrix()
        >>> print(matrix)
                      cardiology  surgery  Discharge
        cardiology          0.0      0.3        0.7
        surgery             0.1      0.0        0.9

        Notes
        -----
        Each row represents a source subspecialty and sums to 1.0. The
        'Discharge' column contains the probability of being discharged from
        that subspecialty. Other columns contain the probability of transferring
        to the target subspecialty (unconditional probabilities, not conditional
        on a transfer occurring).
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_
        assert self.subspecialties is not None  # Guaranteed by is_fitted_

        # Sort subspecialties for consistent ordering
        sorted_subspecialties = sorted(self.subspecialties)

        # Initialize matrix with zeros
        matrix = pd.DataFrame(
            0.0,
            index=sorted_subspecialties,
            columns=sorted_subspecialties + ["Discharge"],
        )

        # Fill in the matrix
        for source in sorted_subspecialties:
            prob_transfer = self.transfer_probabilities[source]["prob_transfer"]
            destination_dist = self.transfer_probabilities[source][
                "destination_distribution"
            ]

            # Probability of discharge (1 - prob_transfer)
            matrix.loc[source, "Discharge"] = 1.0 - prob_transfer

            # Probabilities of transferring to each destination
            # Only include destinations that are in the subspecialties set
            for destination, conditional_prob in destination_dist.items():
                if destination in self.subspecialties:
                    # Unconditional probability = prob_transfer * conditional_prob
                    matrix.loc[source, destination] = prob_transfer * conditional_prob

        return matrix
