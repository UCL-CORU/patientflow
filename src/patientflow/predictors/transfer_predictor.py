"""
Transfer probability predictor for patient movements between subspecialties.

This module provides a predictor for modelling patient transfers between hospital
subspecialties based on historical movement patterns. It computes the probability
that a departing patient will transfer to another subspecialty versus being
discharged, and the distribution of destination subspecialties for transfers.

The predictor supports cohort-based analysis, allowing separate transfer probability
models to be trained for different patient groups (e.g., elective vs emergency
admissions). This enables more accurate modeling of patient flows when transfer
patterns differ significantly between cohorts.

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

When cohort_col is specified, separate probability models are trained for each
cohort. When cohort_col is None, all data is used together to train a single model.
"""

import pandas as pd
from typing import Dict, Set, Optional, Callable, Union
from sklearn.base import BaseEstimator, TransformerMixin
from patientflow.predictors.subgroup_definitions import create_subgroup_functions


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
    visit_col : str, optional
        Name of the column identifying a visit/encounter. If provided, the
        training data `X` will be de-duplicated to a single row per unique
        combination of `(visit_col, source_col, destination_col)` to avoid
        multiple rows representing the same transition for a visit.
    cohort_col : str, optional
        Name of the column containing cohort information (e.g., 'admission_type').
        If None, all data is used together. If specified, separate models are trained
        for each cohort.
    cohort_values : list, optional
        List of expected cohort values for validation. If provided, the fit method
        will validate that only these values appear in the cohort_col.
    subgroup_functions : dict, optional
        Dictionary mapping subgroup names to functions that identify patients in each subgroup.
        If None, uses the standard 5 subgroup functions (paediatric, adult_male_young, etc.).

    Attributes
    ----------
    source_col : str
        Name of the source subspecialty column
    destination_col : str
        Name of the destination subspecialty column
    cohort_col : str or None
        Name of the cohort column, or None if not using cohorts
    cohort_values : list or None
        List of expected cohort values for validation, or None
    transfer_probabilities : dict
        Nested dictionary containing transfer statistics. Structure depends on
        whether cohorts are used:

        If cohort_col is None:
        {
            'all': {
                'source_subspecialty': {
                    'prob_transfer': float,
                    'destination_distribution': dict
                }
            }
        }

        If cohort_col is specified:
        {
            'cohort_name': {
                'source_subspecialty': {
                    'prob_transfer': float,
                    'destination_distribution': dict
                },
                'subgroup_name': {
                    'source_subspecialty': {
                        'prob_transfer': float,
                        'destination_distribution': dict
                    }
                }
            }
        }
    subspecialties : set
        Set of all subspecialties in the system
    cohorts : set or None
        Set of all cohorts in the system, or None if not using cohorts
    subgroups : set or None
        Set of all subgroups in the system, or None if not using subgroups
    is_fitted_ : bool
        Whether the predictor has been fitted

    Examples
    --------
    >>> # Prepare training data with subspecialty movements
    >>> X = pd.DataFrame({
    ...     'current_subspecialty': ['cardiology', 'cardiology', 'surgery'],
    ...     'next_subspecialty': ['surgery', None, None],  # None = discharge
    ...     'admission_type': ['elective', 'emergency', 'elective']
    ... })
    >>> subspecialties = {'cardiology', 'surgery', 'medicine'}
    >>>
    >>> # Train the estimator without cohorts (uses all data)
    >>> estimator = TransferProbabilityEstimator()
    >>> estimator.fit(X, subspecialties)
    >>> prob_transfer = estimator.get_transfer_prob('cardiology')
    >>>
    >>> # Train the estimator with cohorts (separate models for each cohort)
    >>> estimator = TransferProbabilityEstimator(cohort_col='admission_type')
    >>> estimator.fit(X, subspecialties)
    >>> prob_transfer_elective = estimator.get_transfer_prob('cardiology', 'elective')
    >>> prob_transfer_emergency = estimator.get_transfer_prob('cardiology', 'emergency')
    >>>
    >>> # Get all available cohorts
    >>> cohorts = estimator.get_available_cohorts()  # {'elective', 'emergency'}

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
        visit_col: Optional[str] = None,
        cohort_col: Optional[str] = None,
        cohort_values: Optional[list] = None,
        subgroup_functions: Optional[
            Dict[str, Callable[[Union[pd.Series, dict]], bool]]
        ] = None,
    ):
        self.source_col = source_col
        self.destination_col = destination_col
        self.visit_col = visit_col
        self.cohort_col = cohort_col
        self.cohort_values = cohort_values
        self.subgroup_functions = subgroup_functions or create_subgroup_functions()
        self.transfer_probabilities: Optional[Dict[str, Dict[str, Dict[str, Dict]]]] = (
            None
        )
        self.subspecialties: Optional[Set[str]] = None
        self.cohorts: Optional[Set[str]] = None
        self.subgroups: Optional[Set[str]] = None
        self.patient_counts: Optional[Dict[str, Dict[str, int]]] = None
        self.is_fitted_ = False

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if not self.is_fitted_:
            cohort_info = (
                f",\n    cohort_col='{self.cohort_col}'" if self.cohort_col else ""
            )
            return (
                f"{class_name}(\n"
                f"    source_col='{self.source_col}',\n"
                f"    destination_col='{self.destination_col}'{cohort_info},\n"
                f"    fitted=False\n"
                f")"
            )

        n_subspecialties = (
            len(self.subspecialties) if self.subspecialties is not None else 0
        )
        n_cohorts = len(self.cohorts) if self.cohorts is not None else 0
        n_subgroups = len(self.subgroups) if self.subgroups is not None else 0

        n_with_transfers = 0
        cohort_subspecialty_counts: Dict[str, Dict[str, int]] = {}
        subspecialties_with_transfers: Set[str] = set()

        if self.transfer_probabilities is not None:
            # Process each cohort separately to get cohort-specific counts
            for cohort_name, cohort_data in self.transfer_probabilities.items():
                cohort_subspecialty_counts[cohort_name] = {}
                for key, value in cohort_data.items():
                    # Check if this is subspecialty data (has prob_transfer at top level)
                    if isinstance(value, dict) and "prob_transfer" in value:
                        prob_transfer = value["prob_transfer"]
                        if (
                            isinstance(prob_transfer, (int, float))
                            and prob_transfer > 0
                        ):
                            subspecialties_with_transfers.add(
                                key
                            )  # Track unique subspecialties with transfers
                    # Check if this is subgroup data (contains nested subspecialty data)
                    elif isinstance(value, dict) and any(
                        isinstance(v, dict) and "prob_transfer" in v
                        for v in value.values()
                    ):
                        # Count subspecialties in this subgroup for this specific cohort
                        # Only count subspecialties that have actual data (prob_transfer > 0)
                        subgroup_total_subspecialties = sum(
                            1
                            for v in value.values()
                            if isinstance(v, dict)
                            and "prob_transfer" in v
                            and v["prob_transfer"] > 0
                        )
                        cohort_subspecialty_counts[cohort_name][key] = (
                            subgroup_total_subspecialties
                        )

            # Set final count of unique subspecialties with transfers
            n_with_transfers = len(subspecialties_with_transfers)

        # Collect patient counts for subgroups by cohort
        cohort_subgroup_counts = {}
        if self.patient_counts is not None:
            for cohort_name, cohort_counts in self.patient_counts.items():
                cohort_subgroup_counts[cohort_name] = cohort_counts.copy()

        # Build single-line core info (source, destination, cohort)
        core_lines = [
            f"    source_col='{self.source_col}', destination_col='{self.destination_col}'"
        ]
        if self.visit_col:
            core_lines.append(f"    visit_col='{self.visit_col}'")
        if self.cohort_col:
            core_lines.append(
                f"    cohort_col='{self.cohort_col}', n_cohorts={n_cohorts}"
            )
        core_info = "\n".join(core_lines)

        # Build subgroup info lines (each cohort on its own line)
        subgroup_info = ""
        if n_subgroups > 0:
            if cohort_subgroup_counts and cohort_subspecialty_counts:
                lines = [f"    n_subgroups={n_subgroups}"]
                for cohort_name in sorted(cohort_subgroup_counts.keys()):
                    entries = []
                    for subgroup_name in sorted(
                        cohort_subgroup_counts[cohort_name].keys()
                    ):
                        patient_count = cohort_subgroup_counts[cohort_name][
                            subgroup_name
                        ]
                        subspec_count = cohort_subspecialty_counts.get(
                            cohort_name, {}
                        ).get(subgroup_name, 0)
                        entries.append(
                            f"{subgroup_name} ({subspec_count}/{patient_count})"
                        )
                    lines.append(
                        f"    {cohort_name} (subspecialties/snapshots): "
                        + "; ".join(entries)
                    )
                subgroup_info = "\n".join(lines)
            else:
                lines = [f"    n_subgroups={n_subgroups}"]
                subgroup_details = []
                for (
                    cohort_name,
                    cohort_spec_counts,
                ) in cohort_subspecialty_counts.items():
                    for subgroup_name in sorted(cohort_spec_counts.keys()):
                        subspec_count = cohort_spec_counts[subgroup_name]
                        subgroup_details.append(f"{subgroup_name} ({subspec_count})")
                if subgroup_details:
                    lines.append("    " + "; ".join(subgroup_details))
                subgroup_info = "\n".join(lines)

        # Avoid backslash in f-string expression by precomputing the optional newline
        subgroup_info_line = subgroup_info if subgroup_info else ""

        return (
            f"{class_name}(\n"
            f"{core_info}\n"
            f"{subgroup_info_line}\n"
            f"    fitted=True,\n"
            f"    n_subspecialties={n_subspecialties}, n_with_transfers={n_with_transfers}\n"
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

        If cohort_col is specified, separate transfer probabilities are computed
        for each cohort. If cohort_col is None, all data is used together.

        Parameters
        ----------
        X : pandas.DataFrame
            Training DataFrame containing patient movement data with columns as
            specified by source_col and destination_col parameters. If cohort_col
            is specified, must also contain that column.
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
        if self.visit_col is not None:
            required_columns.add(self.visit_col)
        if self.cohort_col is not None:
            required_columns.add(self.cohort_col)

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

        # If visit_col is provided, drop duplicate transitions per visit
        if self.visit_col is not None:
            X = X.drop_duplicates(
                subset=[self.visit_col, self.source_col, self.destination_col]
            )

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

        # Handle cohort processing
        if self.cohort_col is not None:
            # Extract cohorts from data
            self.cohorts = set(X[self.cohort_col].dropna().unique())

            # Validate cohort values if provided
            if self.cohort_values is not None:
                unknown_cohorts = self.cohorts - set(self.cohort_values)
                if unknown_cohorts:
                    raise ValueError(
                        f"Found cohort values in data that are not in the "
                        f"cohort_values parameter: {sorted(unknown_cohorts)}. "
                        f"Expected cohorts: {sorted(self.cohort_values)}"
                    )

            # Compute transfer probabilities for each cohort
            self.transfer_probabilities = {}
            self.patient_counts = {}
            for cohort in self.cohorts:
                cohort_data = X[X[self.cohort_col] == cohort]

                # Train global model for this cohort (all data) - store directly at cohort level
                self.transfer_probabilities[cohort] = (
                    self._prepare_transfer_probabilities(
                        self.subspecialties, cohort_data
                    )
                )

                # Train subgroup models for this cohort
                self.patient_counts[cohort] = {}
                for subgroup_name, subgroup_func in self.subgroup_functions.items():
                    subgroup_data = cohort_data[
                        cohort_data.apply(subgroup_func, axis=1)
                    ]
                    if len(subgroup_data) > 0:  # Only train if subgroup has data
                        self.transfer_probabilities[cohort][subgroup_name] = (
                            self._prepare_transfer_probabilities(
                                self.subspecialties, subgroup_data
                            )
                        )
                        # Store patient count for this subgroup
                        self.patient_counts[cohort][subgroup_name] = len(subgroup_data)

            # Extract all subgroups that were actually trained
            self.subgroups = set()
            for cohort_data in self.transfer_probabilities.values():
                # The global model is stored directly at cohort level, subgroups are nested
                for key, value in cohort_data.items():
                    if isinstance(value, dict) and any(
                        "prob_transfer" in str(v)
                        for v in value.values()
                        if isinstance(v, dict)
                    ):
                        # This is a subgroup (contains subspecialty data)
                        self.subgroups.add(key)
        else:
            # No cohort processing - use all data
            self.cohorts = None
            self.transfer_probabilities = {
                "all": self._prepare_transfer_probabilities(self.subspecialties, X)
            }
            self.patient_counts = {"all": {}}

            # Train subgroup models for all data
            for subgroup_name, subgroup_func in self.subgroup_functions.items():
                subgroup_data = X[X.apply(subgroup_func, axis=1)]
                if len(subgroup_data) > 0:  # Only train if subgroup has data
                    self.transfer_probabilities["all"][subgroup_name] = (
                        self._prepare_transfer_probabilities(
                            self.subspecialties, subgroup_data
                        )
                    )
                    # Store patient count for this subgroup
                    self.patient_counts["all"][subgroup_name] = len(subgroup_data)

            # Extract all subgroups that were actually trained
            self.subgroups = set()
            for cohort_data in self.transfer_probabilities.values():
                # The global model is stored directly at cohort level, subgroups are nested
                for key, value in cohort_data.items():
                    if isinstance(value, dict) and any(
                        "prob_transfer" in str(v)
                        for v in value.values()
                        if isinstance(v, dict)
                    ):
                        # This is a subgroup (contains subspecialty data)
                        self.subgroups.add(key)

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

    def get_transfer_prob(
        self,
        source_subspecialty: str,
        cohort: Optional[str] = None,
        subgroup: Optional[str] = None,
    ) -> float:
        """Get the probability that a departure from a subspecialty is a transfer.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty
        cohort : str, optional
            Name of the cohort to get probabilities for. If None and multiple
            cohorts exist, uses the first available cohort. If None and only
            one cohort exists, uses that cohort.

        Returns
        -------
        float
            Probability that a departure from this subspecialty results in a
            transfer to another subspecialty (vs being discharged). Returns 0.0
            if no transfers observed.

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
        KeyError
            If source_subspecialty is not in the trained subspecialties
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        # Determine which cohort to use
        if cohort is None:
            if len(self.transfer_probabilities) == 1:
                cohort = list(self.transfer_probabilities.keys())[0]
            else:
                raise ValueError(
                    f"Multiple cohorts available: {sorted(self.transfer_probabilities.keys())}. "
                    f"Please specify the cohort parameter."
                )

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        # Determine which subgroup to use
        cohort_data = self.transfer_probabilities[cohort]

        if subgroup is None:
            # Use global model (stored directly at cohort level)
            if source_subspecialty not in cohort_data:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}'. "
                    f"Available subspecialties: {sorted(cohort_data.keys())}"
                )
            prob_transfer = cohort_data[source_subspecialty]["prob_transfer"]
            if isinstance(prob_transfer, (int, float)):
                return prob_transfer
            else:
                raise ValueError(f"Invalid prob_transfer type: {type(prob_transfer)}")
        else:
            # Use specific subgroup
            if subgroup not in cohort_data:
                raise ValueError(
                    f"Subgroup '{subgroup}' not found in cohort '{cohort}'. "
                    f"Available subgroups: {sorted([k for k in cohort_data.keys() if isinstance(cohort_data[k], dict) and any('prob_transfer' in str(v) for v in cohort_data[k].values() if isinstance(v, dict))])}"
                )

            subgroup_probabilities = cohort_data[subgroup]
            if source_subspecialty not in subgroup_probabilities:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}', subgroup '{subgroup}'. "
                    f"Available subspecialties: {sorted(subgroup_probabilities.keys())}"
                )

            prob_transfer = subgroup_probabilities[source_subspecialty]["prob_transfer"]
            if isinstance(prob_transfer, (int, float)):
                return prob_transfer
            else:
                raise ValueError(f"Invalid prob_transfer type: {type(prob_transfer)}")

    def get_destination_distribution(
        self,
        source_subspecialty: str,
        cohort: Optional[str] = None,
        subgroup: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get the distribution of destinations given that a transfer occurs.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty
        cohort : str, optional
            Name of the cohort to get probabilities for. If None and multiple
            cohorts exist, uses the first available cohort. If None and only
            one cohort exists, uses that cohort.

        Returns
        -------
        dict
            Dictionary mapping destination subspecialty names to probabilities.
            Probabilities sum to 1.0. Returns empty dict if no transfers observed.

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
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

        # Determine which cohort to use
        if cohort is None:
            if len(self.transfer_probabilities) == 1:
                cohort = list(self.transfer_probabilities.keys())[0]
            else:
                raise ValueError(
                    f"Multiple cohorts available: {sorted(self.transfer_probabilities.keys())}. "
                    f"Please specify the cohort parameter."
                )

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        # Determine which subgroup to use
        cohort_data = self.transfer_probabilities[cohort]

        if subgroup is None:
            # Use global model (stored directly at cohort level)
            if source_subspecialty not in cohort_data:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}'. "
                    f"Available subspecialties: {sorted(cohort_data.keys())}"
                )
            return cohort_data[source_subspecialty]["destination_distribution"]
        else:
            # Use specific subgroup
            if subgroup not in cohort_data:
                raise ValueError(
                    f"Subgroup '{subgroup}' not found in cohort '{cohort}'. "
                    f"Available subgroups: {sorted([k for k in cohort_data.keys() if isinstance(cohort_data[k], dict) and any('prob_transfer' in str(v) for v in cohort_data[k].values() if isinstance(v, dict))])}"
                )

            subgroup_probabilities = cohort_data[subgroup]
            if source_subspecialty not in subgroup_probabilities:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}', subgroup '{subgroup}'. "
                    f"Available subspecialties: {sorted(subgroup_probabilities.keys())}"
                )

            return subgroup_probabilities[source_subspecialty][
                "destination_distribution"
            ]

    def predict(
        self,
        source_subspecialty: str,
        cohort: Optional[str] = None,
        subgroup: Optional[str] = None,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Get full transfer statistics for a source subspecialty.

        This is a convenience method that returns both the transfer probability
        and destination distribution in a single call.

        Parameters
        ----------
        source_subspecialty : str
            Name of the source subspecialty
        cohort : str, optional
            Name of the cohort to get probabilities for. If None and multiple
            cohorts exist, uses the first available cohort. If None and only
            one cohort exists, uses that cohort.

        Returns
        -------
        dict
            Dictionary containing:
            - 'prob_transfer': Probability of transfer vs discharge
            - 'destination_distribution': Distribution of destinations

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
        KeyError
            If source_subspecialty is not in the trained subspecialties
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        # Determine which cohort to use
        if cohort is None:
            if len(self.transfer_probabilities) == 1:
                cohort = list(self.transfer_probabilities.keys())[0]
            else:
                raise ValueError(
                    f"Multiple cohorts available: {sorted(self.transfer_probabilities.keys())}. "
                    f"Please specify the cohort parameter."
                )

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        # Determine which subgroup to use
        cohort_data = self.transfer_probabilities[cohort]

        if subgroup is None:
            # Use global model (stored directly at cohort level)
            if source_subspecialty not in cohort_data:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}'. "
                    f"Available subspecialties: {sorted(cohort_data.keys())}"
                )
            subspecialty_data = cohort_data[source_subspecialty]
            return {
                "prob_transfer": subspecialty_data["prob_transfer"],
                "destination_distribution": subspecialty_data[
                    "destination_distribution"
                ],
            }
        else:
            # Use specific subgroup
            if subgroup not in cohort_data:
                raise ValueError(
                    f"Subgroup '{subgroup}' not found in cohort '{cohort}'. "
                    f"Available subgroups: {sorted([k for k in cohort_data.keys() if isinstance(cohort_data[k], dict) and any('prob_transfer' in str(v) for v in cohort_data[k].values() if isinstance(v, dict))])}"
                )

            subgroup_probabilities = cohort_data[subgroup]
            if source_subspecialty not in subgroup_probabilities:
                raise KeyError(
                    f"Subspecialty '{source_subspecialty}' not found in cohort '{cohort}', subgroup '{subgroup}'. "
                    f"Available subspecialties: {sorted(subgroup_probabilities.keys())}"
                )

            subspecialty_data = subgroup_probabilities[source_subspecialty]
            return {
                "prob_transfer": subspecialty_data["prob_transfer"],
                "destination_distribution": subspecialty_data[
                    "destination_distribution"
                ],
            }

    def get_all_transfer_probabilities(
        self, cohort: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Get the complete transfer probabilities dictionary.

        Parameters
        ----------
        cohort : str, optional
            Name of the cohort to get probabilities for. If None, returns all
            cohorts. If specified, returns only that cohort's probabilities.

        Returns
        -------
        dict
            Complete nested dictionary of transfer probabilities. If cohort is
            specified, returns only that cohort's data. If cohort is None,
            returns all cohorts' data.

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if cohort is None:
            return self.transfer_probabilities.copy()

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        return {cohort: self.transfer_probabilities[cohort].copy()}

    def get_transition_matrix(self, cohort: Optional[str] = None) -> pd.DataFrame:
        """Format transition probabilities as a matrix.

        Creates a DataFrame with source subspecialties as rows and target
        subspecialties as columns, plus a 'Discharge' column. Each cell
        contains the probability of transitioning from the source (row) to
        the target (column).

        Parameters
        ----------
        cohort : str, optional
            Name of the cohort to get transition matrix for. If None and multiple
            cohorts exist, uses the first available cohort. If None and only
            one cohort exists, uses that cohort.

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
            If the predictor has not been fitted or cohort not found

        Examples
        --------
        >>> estimator = TransferProbabilityEstimator(cohort_col='admission_type')
        >>> estimator.fit(X, subspecialties={'cardiology', 'surgery'})
        >>> matrix = estimator.get_transition_matrix('elective')
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

        # Determine which cohort to use
        if cohort is None:
            if len(self.transfer_probabilities) == 1:
                cohort = list(self.transfer_probabilities.keys())[0]
            else:
                raise ValueError(
                    f"Multiple cohorts available: {sorted(self.transfer_probabilities.keys())}. "
                    f"Please specify the cohort parameter."
                )

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        # Sort subspecialties for consistent ordering
        sorted_subspecialties = sorted(self.subspecialties)

        # Initialize matrix with zeros
        matrix = pd.DataFrame(
            0.0,
            index=sorted_subspecialties,
            columns=sorted_subspecialties + ["Discharge"],
        )

        # Fill in the matrix using the specified cohort's global probabilities
        cohort_data = self.transfer_probabilities[cohort]
        # Global model is stored directly at cohort level
        for source in sorted_subspecialties:
            if source in cohort_data:
                prob_transfer = cohort_data[source]["prob_transfer"]
                destination_dist = cohort_data[source]["destination_distribution"]

                # Probability of discharge (1 - prob_transfer)
                if isinstance(prob_transfer, (int, float)):
                    matrix.loc[source, "Discharge"] = 1.0 - prob_transfer
                else:
                    matrix.loc[source, "Discharge"] = 1.0

                # Probabilities of transferring to each destination
                # Only include destinations that are in the subspecialties set
                for destination, conditional_prob in destination_dist.items():
                    if destination in self.subspecialties:
                        # Unconditional probability = prob_transfer * conditional_prob
                        matrix.loc[source, destination] = (
                            prob_transfer * conditional_prob
                        )

        return matrix

    def get_available_cohorts(self) -> Set[str]:
        """Get the cohorts this model was trained on.

        Returns
        -------
        set of str
            Set of cohort names that were used during training

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
        return set(self.transfer_probabilities.keys())

    def get_cohort_transfer_probabilities(self, cohort: str) -> Dict[str, Dict]:
        """Get all transfer probabilities for a specific cohort.

        Parameters
        ----------
        cohort : str
            Name of the cohort to get probabilities for

        Returns
        -------
        dict
            Dictionary containing transfer probabilities for all subspecialties
            in the specified cohort

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        return self.transfer_probabilities[cohort].copy()

    def get_available_subgroups(self, cohort: Optional[str] = None) -> Set[str]:
        """Get the subgroups this model was trained on.

        Parameters
        ----------
        cohort : str, optional
            Name of the cohort to get subgroups for. If None, returns all
            subgroups across all cohorts.

        Returns
        -------
        set of str
            Set of subgroup names that were used during training

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort not found
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if cohort is None:
            # Return all subgroups across all cohorts
            all_subgroups = set()
            for cohort_data in self.transfer_probabilities.values():
                # The global model is stored directly at cohort level, subgroups are nested
                for key, value in cohort_data.items():
                    if isinstance(value, dict) and any(
                        "prob_transfer" in str(v)
                        for v in value.values()
                        if isinstance(v, dict)
                    ):
                        # This is a subgroup (contains subspecialty data)
                        all_subgroups.add(key)
            return all_subgroups

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        # Return subgroups for the specific cohort
        cohort_data = self.transfer_probabilities[cohort]
        cohort_subgroups = set()
        for key, value in cohort_data.items():
            if isinstance(value, dict) and any(
                "prob_transfer" in str(v) for v in value.values() if isinstance(v, dict)
            ):
                # This is a subgroup (contains subspecialty data)
                cohort_subgroups.add(key)
        return cohort_subgroups

    def get_subgroup_transfer_probabilities(
        self, cohort: str, subgroup: str
    ) -> Dict[str, Dict]:
        """Get all transfer probabilities for a specific cohort and subgroup.

        Parameters
        ----------
        cohort : str
            Name of the cohort to get probabilities for
        subgroup : str
            Name of the subgroup to get probabilities for

        Returns
        -------
        dict
            Dictionary containing transfer probabilities for all subspecialties
            in the specified cohort and subgroup

        Raises
        ------
        ValueError
            If the predictor has not been fitted or cohort/subgroup not found
        """
        if not self.is_fitted_:
            raise ValueError(
                "This TransferProbabilityEstimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        assert self.transfer_probabilities is not None  # Guaranteed by is_fitted_

        if cohort not in self.transfer_probabilities:
            raise ValueError(
                f"Cohort '{cohort}' not found in trained model. "
                f"Available cohorts: {sorted(self.transfer_probabilities.keys())}"
            )

        cohort_data = self.transfer_probabilities[cohort]
        if subgroup not in cohort_data:
            raise ValueError(
                f"Subgroup '{subgroup}' not found in cohort '{cohort}'. "
                f"Available subgroups: {sorted(cohort_data.keys())}"
            )

        return cohort_data[subgroup].copy()
