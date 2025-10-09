"""Demand preparation utilities for later consolidation.

This module prepares per-subspecialty inputs that can be consumed by any
downstream roll-up or forecasting workflow. It converts trained model
outputs and current patient snapshots into:

- A probability mass function (PMF) for admissions from current ED patients
- A probability mass function (PMF) for departures from current inpatients
- Poisson means for yet-to-arrive admissions (ED, non-ED emergency, elective)
- Transfer arrival distributions from internal subspecialty movements

The outputs are independent per subspecialty and do not presuppose any
particular consolidation hierarchy. They can be used directly for single-
specialty analyses or fed into any combination scheme (including hierarchical
schemes) implemented elsewhere.

Notes
-----
This module integrates with the broader patientflow ecosystem by:

1. Using trained classifiers and specialty models from the train module
2. Processing patient snapshots from the prepare module
3. Computing admission probabilities using the calculate module
4. Modelling internal patient transfers using the transfer predictor

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from patientflow.predict.emergency_demand import (
    add_missing_columns,
    get_specialty_probs,
)
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
    EmpiricalIncomingAdmissionPredictor,
    DirectAdmissionPredictor,
)
from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.value_to_outcome_predictor import (
    ValueToOutcomePredictor,
)
from patientflow.predictors.subgroup_predictor import (
    MultiSubgroupPredictor,
)
from patientflow.aggregate import (
    model_input_to_pred_proba,
    pred_proba_to_agg_predicted,
)
from patientflow.calculate.admission_in_prediction_window import (
    calculate_probability,
    calculate_admission_probability_from_survival_curve,
)
from patientflow.model_artifacts import TrainedClassifier


@dataclass(frozen=True)
class SubspecialtyPredictionInputs:
    """Input parameters for subspecialty demand prediction.

    These inputs represent the probability distributions and parameters
    needed to predict demand for a single subspecialty. This dataclass packages
    the outputs from build_subspecialty_data for use in hierarchical prediction.

    Attributes
    ----------
    pmf_ed_current_within_window : np.ndarray
        Probability mass function for current ED admissions within the prediction window.
        Array where pmf[k] represents P(k ED patients admitted to this subspecialty).
    pmf_inpatient_departures_within_window : np.ndarray
        Probability mass function for current inpatient departures within the prediction window.
        Array where pmf[k] represents P(k current inpatients depart from this subspecialty).
    lambda_ed_yta_within_window : float
        Poisson parameter (rate) for ED yet-to-arrive admissions within the prediction window.
        Represents the expected number of new ED arrivals who will be admitted to this subspecialty.
    lambda_non_ed_yta_within_window : float
        Poisson parameter (rate) for non-ED emergency admissions within the prediction window.
        Represents the expected number of direct emergency admissions to this subspecialty.
    lambda_elective_yta_within_window : float
        Poisson parameter (rate) for elective admissions within the prediction window.
        Represents the expected number of planned/elective admissions to this subspecialty.

    Notes
    -----
    This dataclass is immutable (frozen=True) to prevent accidental modification after creation.
    All arrays and parameters should represent distributions/rates for the same prediction window.
    """

    pmf_ed_current_within_window: np.ndarray
    pmf_inpatient_departures_within_window: np.ndarray
    lambda_ed_yta_within_window: float
    lambda_non_ed_yta_within_window: float
    lambda_elective_yta_within_window: float

    def __repr__(self) -> str:
        """Return a clean, readable representation showing PMF values and lambdas."""

        def format_pmf(arr: np.ndarray, max_display: int = 10) -> str:
            """Format PMF array, automatically showing the most informative range."""
            if len(arr) <= max_display:
                values = ", ".join(f"{v:.3f}" for v in arr)
                return f"[{values}]"

            # Find where the probability mass is concentrated
            mode_idx = int(np.argmax(arr))
            
            # Determine display window centered on mode
            half_window = max_display // 2
            start_idx = max(0, mode_idx - half_window)
            end_idx = min(len(arr), start_idx + max_display)
            
            # Adjust if we're near the end
            if end_idx - start_idx < max_display:
                start_idx = max(0, end_idx - max_display)
            
            # Format the displayed portion
            display_values = ", ".join(f"{v:.3f}" for v in arr[start_idx:end_idx])
            
            # Build output with index information
            if start_idx == 0 and end_idx == len(arr):
                return f"[{display_values}]"
            elif start_idx == 0:
                remaining = len(arr) - end_idx
                return f"[{display_values}, ... +{remaining} more] (sum={arr.sum():.3f})"
            elif end_idx == len(arr):
                return f"[... {start_idx} before, {display_values}] (sum={arr.sum():.3f})"
            else:
                before = start_idx
                after = len(arr) - end_idx
                return f"[... {before} before, {display_values}, +{after} more] (mode@{mode_idx}, sum={arr.sum():.3f})"

        ed_pmf_str = format_pmf(self.pmf_ed_current_within_window)
        inpt_pmf_str = format_pmf(self.pmf_inpatient_departures_within_window)

        return (
            f"SubspecialtyPredictionInputs(\n"
            f"  PMF ED current:        {ed_pmf_str}\n"
            f"  PMF inpatient depart:  {inpt_pmf_str}\n"
            f"  λ ED yet-to-arrive:    {self.lambda_ed_yta_within_window:.3f}\n"
            f"  λ non-ED emergency:    {self.lambda_non_ed_yta_within_window:.3f}\n"
            f"  λ elective:            {self.lambda_elective_yta_within_window:.3f}\n"
            f")"
        )


def build_subspecialty_data(
    models: Tuple[
        TrainedClassifier,
        TrainedClassifier,
        Union[
            SequenceToOutcomePredictor, ValueToOutcomePredictor, MultiSubgroupPredictor
        ],
        Union[
            ParametricIncomingAdmissionPredictor,
            EmpiricalIncomingAdmissionPredictor,
        ],
        DirectAdmissionPredictor,
        DirectAdmissionPredictor,
    ],
    prediction_time: Tuple[int, int],
    ed_snapshots: pd.DataFrame,
    inpatient_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: Optional[List[float]] = None,
    use_admission_in_window_prob: bool = True,
) -> Dict[str, SubspecialtyPredictionInputs]:
    """Build per-subspecialty inputs for downstream roll-up.

    This function processes current patient snapshots through trained models and
    computes, for each subspecialty, the probability distribution of admissions
    from current ED patients, departures from current inpatients, and the expected
    means of yet-to-arrive admissions.

    Parameters
    ----------
    models : tuple
        Tuple of six trained models:

        - ed_classifier: TrainedClassifier for ED admission probability prediction
        - inpatient_classifier: TrainedClassifier for inpatient departure probability prediction
        - spec_model: SequenceToOutcomePredictor | ValueToOutcomePredictor | MultiSubgroupPredictor
          for specialty assignment probabilities
        - ed_yta_model: ParametricIncomingAdmissionPredictor | EmpiricalIncomingAdmissionPredictor
          for ED yet-to-arrive predictions
        - non_ed_yta_model: DirectAdmissionPredictor for non-ED emergency predictions
        - elective_yta_model: DirectAdmissionPredictor for elective predictions
    prediction_time : tuple of (int, int)
        Hour and minute for inference time
    ed_snapshots : pandas.DataFrame
        DataFrame of current ED patients. Must include 'elapsed_los' column as timedelta.
        Each row represents a patient currently in the ED.
    inpatient_snapshots : pandas.DataFrame
        DataFrame of current inpatients. Must include 'elapsed_los' column as timedelta.
        Each row represents a patient currently in a subspecialty ward.
    specialties : list of str
        List of subspecialties to prepare inputs for
    prediction_window : datetime.timedelta
        Time window over which to predict admissions
    x1, y1, x2, y2 : float
        Parameters for the parametric admission-in-window curve. Used when
        ed_yta_model is parametric and for computing in-ED window probabilities.
    cdf_cut_points : list of float, optional
        Ignored in this function; present for API compatibility. If provided,
        has no effect on output.
    use_admission_in_window_prob : bool, default=True
        Whether to weight current ED admissions by their probability of being
        admitted within the prediction window.

    Returns
    -------
    dict of str to SubspecialtyPredictionInputs
        Dictionary mapping subspecialty_id to SubspecialtyPredictionInputs dataclass.
        See SubspecialtyPredictionInputs for field details.

    Raises
    ------
    TypeError
        If any model is not of the expected type
    ValueError
        If required columns are missing, models are not fitted, or parameters
        don't match between models and requested parameters

    Notes
    -----
    The function combines five sources of demand:

    1. Current ED patients (converted to probability mass function)
    2. Current inpatients (converted to probability mass function for departures)
    3. Yet-to-arrive ED patients (converted to Poisson parameters)
    4. Yet-to-arrive non-ED emergency patients (converted to Poisson parameters)
    5. Yet-to-arrive elective patients (converted to Poisson parameters)

    """
    (
        ed_classifier,
        inpatient_classifier,
        spec_model,
        yet_to_arrive_model,
        non_ed_yta_model,
        elective_yta_model,
    ) = models

    # Validate model types
    if not isinstance(ed_classifier, TrainedClassifier):
        raise TypeError("First model must be of type TrainedClassifier (ED classifier)")
    if not isinstance(inpatient_classifier, TrainedClassifier):
        raise TypeError(
            "Second model must be of type TrainedClassifier (inpatient classifier)"
        )
    if not isinstance(
        spec_model,
        (SequenceToOutcomePredictor, ValueToOutcomePredictor, MultiSubgroupPredictor),
    ):
        raise TypeError(
            "Third model must be of type SequenceToOutcomePredictor or ValueToOutcomePredictor or MultiSubgroupPredictor"
        )
    yet_to_arrive_class_name = type(yet_to_arrive_model).__name__
    expected_types = (
        "ParametricIncomingAdmissionPredictor",
        "EmpiricalIncomingAdmissionPredictor",
    )
    if yet_to_arrive_class_name not in expected_types:
        actual_module = type(yet_to_arrive_model).__module__
        raise TypeError(
            "Fourth model must be of type ParametricIncomingAdmissionPredictor or "
            "EmpiricalIncomingAdmissionPredictor, "
            f"but got {actual_module}.{yet_to_arrive_class_name}. "
            "If you're using Jupyter, try restarting the kernel."
        )

    # Validate that non-ED and elective models are DirectAdmissionPredictor
    if not isinstance(non_ed_yta_model, DirectAdmissionPredictor):
        raise TypeError(
            "Fifth model must be of type DirectAdmissionPredictor (non-ED emergency)"
        )
    if not isinstance(elective_yta_model, DirectAdmissionPredictor):
        raise TypeError(
            "Sixth model must be of type DirectAdmissionPredictor (elective)"
        )

    # Validate elapsed_los column presence and dtype for ED snapshots
    if "elapsed_los" not in ed_snapshots.columns:
        raise ValueError("Column 'elapsed_los' not found in ed_snapshots")
    if not pd.api.types.is_timedelta64_dtype(ed_snapshots["elapsed_los"]):
        actual_type = ed_snapshots["elapsed_los"].dtype
        raise ValueError(
            "Column 'elapsed_los' must be a timedelta column in ed_snapshots, but found type: "
            f"{actual_type}"
        )

    # Validate elapsed_los column presence and dtype for inpatient snapshots
    if "elapsed_los" not in inpatient_snapshots.columns:
        raise ValueError("Column 'elapsed_los' not found in inpatient_snapshots")
    if not pd.api.types.is_timedelta64_dtype(inpatient_snapshots["elapsed_los"]):
        actual_type = inpatient_snapshots["elapsed_los"].dtype
        raise ValueError(
            "Column 'elapsed_los' must be a timedelta column in inpatient_snapshots, but found type: "
            f"{actual_type}"
        )

    # Check that all models have been fit
    if not hasattr(ed_classifier, "pipeline") or ed_classifier.pipeline is None:
        raise ValueError("ED classifier model has not been fit")
    if (
        not hasattr(inpatient_classifier, "pipeline")
        or inpatient_classifier.pipeline is None
    ):
        raise ValueError("Inpatient classifier model has not been fit")
    if isinstance(spec_model, (SequenceToOutcomePredictor, ValueToOutcomePredictor)):
        if not hasattr(spec_model, "weights") or spec_model.weights is None:
            raise ValueError("Specialty model has not been fit")
    else:
        if not hasattr(spec_model, "specialty_to_subgroups"):
            raise ValueError("Specialty model has not been fit")
    if (
        not hasattr(yet_to_arrive_model, "prediction_window")
        or yet_to_arrive_model.prediction_window is None
    ):
        raise ValueError("Yet-to-arrive model has not been fit")

    # Validate prediction_time and prediction_window compatibility
    if not ed_classifier.training_results.prediction_time == prediction_time:
        raise ValueError(
            "Requested prediction time {pt} does not match the prediction time of the "
            "trained ED classifier {ct}".format(
                pt=prediction_time, ct=ed_classifier.training_results.prediction_time
            )
        )
    if not inpatient_classifier.training_results.prediction_time == prediction_time:
        raise ValueError(
            "Requested prediction time {pt} does not match the prediction time of the "
            "trained inpatient classifier {ct}".format(
                pt=prediction_time,
                ct=inpatient_classifier.training_results.prediction_time,
            )
        )
    if prediction_window != yet_to_arrive_model.prediction_window:
        raise ValueError(
            "Requested prediction window {pw} does not match the prediction window of "
            "the trained yet-to-arrive model {mw}".format(
                pw=prediction_window, mw=yet_to_arrive_model.prediction_window
            )
        )

    # Ensure DirectAdmissionPredictors are fit and aligned to the requested prediction window
    for name, model in (("non-ED", non_ed_yta_model), ("elective", elective_yta_model)):
        if not hasattr(model, "prediction_window") or model.prediction_window is None:
            raise ValueError(
                f"{name} DirectAdmissionPredictor has not been fit (missing prediction_window)"
            )
        if model.prediction_window != prediction_window:
            raise ValueError(
                f"Requested prediction window {prediction_window} does not match the prediction window of the trained {name} model {model.prediction_window}"
            )

    # Validate specialties alignment
    if hasattr(yet_to_arrive_model, "filters"):
        if not set(yet_to_arrive_model.filters.keys()) == set(specialties):
            raise ValueError(
                f"Requested specialties {set(specialties)} do not match the specialties of the trained yet-to-arrive model {set(yet_to_arrive_model.filters.keys())}"
            )

    special_params = spec_model.special_params

    if special_params:
        special_category_func = special_params["special_category_func"]
        special_category_dict = special_params["special_category_dict"]
        special_func_map = special_params["special_func_map"]
    else:
        special_category_func = special_category_dict = special_func_map = None

    if special_category_dict is not None and not set(specialties) == set(
        special_category_dict.keys()
    ):
        # Only enforce the legacy check if there is no subgroup mapping available
        has_mapping = (
            hasattr(spec_model, "specialty_to_subgroups")
            and isinstance(getattr(spec_model, "specialty_to_subgroups"), dict)
            and len(getattr(spec_model, "specialty_to_subgroups")) > 0
        )
        if not has_mapping:
            raise ValueError(
                "Requested specialties do not match the specialty dictionary defined in special_params"
            )
    # Use calibrated pipeline if available for ED classifier
    ed_pipeline = (
        ed_classifier.calibrated_pipeline
        if hasattr(ed_classifier, "calibrated_pipeline")
        and ed_classifier.calibrated_pipeline is not None
        else ed_classifier.pipeline
    )

    # Use calibrated pipeline if available for inpatient classifier
    inpatient_pipeline = (
        inpatient_classifier.calibrated_pipeline
        if hasattr(inpatient_classifier, "calibrated_pipeline")
        and inpatient_classifier.calibrated_pipeline is not None
        else inpatient_classifier.pipeline
    )

    # Ensure model expects columns exist
    ed_snapshots = add_missing_columns(ed_pipeline, ed_snapshots.copy())
    inpatient_snapshots = add_missing_columns(
        inpatient_pipeline, inpatient_snapshots.copy()
    )

    # Convert elapsed_los to seconds for the ED classifier pipeline
    ed_snapshots_temp = ed_snapshots.copy()
    ed_snapshots_temp["elapsed_los"] = ed_snapshots_temp[
        "elapsed_los"
    ].dt.total_seconds()

    # Admission probability for current ED patients (per row)
    prob_admission_after_ed = model_input_to_pred_proba(ed_snapshots_temp, ed_pipeline)

    # Convert elapsed_los to seconds for the inpatient classifier pipeline
    inpatient_snapshots_temp = inpatient_snapshots.copy()
    inpatient_snapshots_temp["elapsed_los"] = inpatient_snapshots_temp[
        "elapsed_los"
    ].dt.total_seconds()

    # Departure probability for current inpatients (per row)
    prob_departure_after_inpatient = model_input_to_pred_proba(
        inpatient_snapshots_temp, inpatient_pipeline
    )

    # Specialty probabilities per row for ED patients
    if hasattr(spec_model, "predict_dataframe"):
        ed_snapshots.loc[:, "specialty_prob"] = spec_model.predict_dataframe(
            ed_snapshots
        )
    else:
        ed_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
            specialties,
            spec_model,
            ed_snapshots,
            special_category_func=special_category_func,
            special_category_dict=special_category_dict,
        )

    # No specialty probability calculation needed for inpatients (no weighting required)

    # Probability of being admitted within window (per row) for ED patients
    if use_admission_in_window_prob:
        if isinstance(yet_to_arrive_model, EmpiricalIncomingAdmissionPredictor):
            prob_admission_in_window = ed_snapshots.apply(
                lambda row: calculate_admission_probability_from_survival_curve(
                    row["elapsed_los"],
                    prediction_window,
                    yet_to_arrive_model.survival_df,
                ),
                axis=1,
            )
        else:
            prob_admission_in_window = ed_snapshots.apply(
                lambda row: calculate_probability(
                    row["elapsed_los"], prediction_window, x1, y1, x2, y2
                ),
                axis=1,
            )
    else:
        prob_admission_in_window = pd.Series(1.0, index=ed_snapshots.index)

    # No window filtering needed for inpatients (no weighting required)

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}

    # Resolve specialty_to_subgroups directly from the model attribute
    specialty_to_subgroups: Dict[str, List[str]] = getattr(
        spec_model, "specialty_to_subgroups", {}
    )

    # Precompute subgroup/function masks once for ED patients
    ed_masks_by_func: Dict[str, pd.Series] = {
        name: ed_snapshots.apply(func, axis=1)
        for name, func in special_func_map.items()
    }

    # No subgroup processing needed for inpatients (no weighting required)

    subspecialty_data: Dict[str, SubspecialtyPredictionInputs] = {}

    for spec in specialties:
        if specialty_to_subgroups and spec in specialty_to_subgroups:
            func_keys = specialty_to_subgroups[spec]
        else:
            func_keys = ["default"]

        # Process ED patients
        ed_combined_mask = pd.Series(False, index=ed_snapshots.index)
        for key in func_keys:
            ed_combined_mask = ed_combined_mask | ed_masks_by_func.get(
                key, pd.Series(False, index=ed_snapshots.index)
            )

        ed_non_zero_indices = ed_snapshots[ed_combined_mask].index
        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[
            ed_non_zero_indices
        ]

        filtered_prob_admission_to_specialty = (
            ed_snapshots["specialty_prob"]
            .loc[ed_non_zero_indices]
            .apply(lambda d: d.get(spec, 0.0) if isinstance(d, dict) else 0.0)
        )
        filtered_prob_admission_in_window = prob_admission_in_window.loc[
            ed_non_zero_indices
        ]
        filtered_weights = (
            filtered_prob_admission_to_specialty * filtered_prob_admission_in_window
        )

        agg_predicted_in_ed = pred_proba_to_agg_predicted(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )

        # Process inpatients (no weighting required)
        inpatient_mask = inpatient_snapshots["current_subspecialty"] == spec
        inpatient_indices = inpatient_snapshots[inpatient_mask].index

        if len(inpatient_indices) > 0:
            filtered_prob_departure_after_inpatient = (
                prob_departure_after_inpatient.loc[inpatient_indices]
            )
            agg_predicted_inpatient_departures = pred_proba_to_agg_predicted(
                filtered_prob_departure_after_inpatient
            )
        else:
            # No inpatients in this specialty, create zero PMF
            agg_predicted_inpatient_departures = {"agg_proba": np.array([1.0, 0.0])}

        prediction_context = {spec: {"prediction_time": prediction_time}}

        subspecialty_data[spec] = SubspecialtyPredictionInputs(
            pmf_ed_current_within_window=np.array(agg_predicted_in_ed["agg_proba"]),
            pmf_inpatient_departures_within_window=np.array(
                agg_predicted_inpatient_departures["agg_proba"]
            ),
            lambda_ed_yta_within_window=float(
                yet_to_arrive_model.predict_mean(
                    prediction_context, x1=x1, y1=y1, x2=x2, y2=y2
                )
            ),
            lambda_non_ed_yta_within_window=float(
                non_ed_yta_model.predict_mean(prediction_context)
                if non_ed_yta_model is not None
                else 0.0
            ),
            lambda_elective_yta_within_window=float(
                elective_yta_model.predict_mean(prediction_context)
                if elective_yta_model is not None
                else 0.0
            ),
        )

    return subspecialty_data


def scale_pmf_by_probability(pmf: np.ndarray, compound_prob: float) -> np.ndarray:
    """Scale a departure PMF by compound probability for transfer destination.

    This function creates a mixture distribution representing: "given these
    departures, how many actually transfer to our destination?" It accounts
    for the fact that most departures don't go to any specific destination.

    The scaling works as follows:
    - P(0 transfers) = P(0 departed) + P(n>0 departed but none went to destination)
    - P(k transfers, k>0) = compound_prob × P(k departed)

    This is effectively a Binomial thinning of each possible departure count.

    Parameters
    ----------
    pmf : numpy.ndarray
        Probability mass function for number of departures. pmf[k] = P(k departures)
    compound_prob : float
        Combined probability: prob_transfer × prob_destination.
        Represents P(departure goes to this specific destination)

    Returns
    -------
    numpy.ndarray
        Scaled PMF representing the distribution of transfers to the destination

    Examples
    --------
    >>> # 50% chance of 0 departures, 50% chance of 2 departures
    >>> pmf = np.array([0.5, 0.0, 0.5])
    >>> # 20% of departures go to our destination
    >>> scaled = scale_pmf_by_probability(pmf, 0.2)
    >>> # Most weight on 0 (no departures OR departures went elsewhere)
    >>> # Some weight on 1 or 2 transfers

    Notes
    -----
    For compound_prob = 0.0, returns [1.0, 0.0, ...] (no transfers possible).
    For compound_prob = 1.0, returns the original pmf (all departures transfer here).
    """
    if compound_prob == 0.0:
        # No transfers to this destination
        return np.array([1.0, 0.0])

    if compound_prob == 1.0:
        # All departures go to this destination
        return pmf.copy()

    n_max = len(pmf) - 1
    scaled_pmf = np.zeros(n_max + 1)

    # P(0 transfers to destination) includes:
    # 1. P(0 departures) - these definitely don't transfer
    # 2. P(k>0 departures) * P(none go to destination)
    #    = P(k departures) * (1 - compound_prob)^k
    scaled_pmf[0] = pmf[0]  # Start with P(0 departures)

    for k in range(1, n_max + 1):
        if pmf[k] > 0:
            # P(none of k departures go to destination)
            prob_none_transfer = (1 - compound_prob) ** k
            scaled_pmf[0] += pmf[k] * prob_none_transfer

            # P(exactly j of k departures go to destination)
            # Using binomial: C(k,j) * p^j * (1-p)^(k-j)
            from scipy.stats import binom

            for j in range(1, k + 1):
                prob_j_transfers = binom.pmf(j, k, compound_prob)
                scaled_pmf[j] += pmf[k] * prob_j_transfers

    return scaled_pmf


def convolve_pmfs(pmf1: np.ndarray, pmf2: np.ndarray) -> np.ndarray:
    """Convolve two probability mass functions.

    Convolution of PMFs gives the distribution of the sum of two independent
    random variables. This is used to aggregate arrivals from multiple source
    subspecialties.

    Mathematically: P(total = k) = sum over all i,j where i+j=k of P1(i) × P2(j)

    Parameters
    ----------
    pmf1 : numpy.ndarray
        First probability mass function
    pmf2 : numpy.ndarray
        Second probability mass function

    Returns
    -------
    numpy.ndarray
        Convolved PMF representing the distribution of the sum

    Examples
    --------
    >>> # Two independent sources, each with 50% chance of 0 or 1 arrival
    >>> pmf1 = np.array([0.5, 0.5])
    >>> pmf2 = np.array([0.5, 0.5])
    >>> result = convolve_pmfs(pmf1, pmf2)
    >>> # Result: 25% chance of 0, 50% chance of 1, 25% chance of 2
    >>> np.allclose(result, [0.25, 0.5, 0.25])
    True

    Notes
    -----
    This function uses numpy.convolve which implements discrete convolution
    efficiently using FFT for large arrays.
    """
    return np.convolve(pmf1, pmf2)


def compute_transfer_arrivals(
    subspecialty_data: Dict[str, Dict[str, Any]],
    transfer_model: Any,
    subspecialties: List[str],
) -> Dict[str, np.ndarray]:
    """Compute arrival PMFs from internal transfers for each subspecialty.

    This function uses departure PMFs from subspecialty_data and transfer
    probabilities from transfer_model to calculate how many patients arrive
    at each subspecialty from transfers within other subspecialties.

    Parameters
    ----------
    subspecialty_data : dict
        Output from build_subspecialty_data, containing departure PMFs.
        Must have 'pmf_inpatient_departures_within_window' for each subspecialty.
    transfer_model : TransferProbabilityEstimator
        Trained transfer probability estimator with methods:
        - get_transfer_prob(source) -> float
        - get_destination_distribution(source) -> dict
    subspecialties : list of str
        List of all subspecialties in the system

    Returns
    -------
    dict
        Mapping of subspecialty_id to PMF of arrivals from transfers.
        {
            'subspecialty_name': numpy.ndarray (PMF of transfer arrivals)
        }

    Raises
    ------
    KeyError
        If subspecialty_data is missing required 'pmf_inpatient_departures_within_window'
    ValueError
        If transfer_model has not been fitted

    Examples
    --------
    >>> # After computing subspecialty_data with departure PMFs
    >>> transfer_arrivals = compute_transfer_arrivals(
    ...     subspecialty_data,
    ...     transfer_model,
    ...     subspecialties=['cardiology', 'surgery', 'medicine']
    ... )
    >>> # Access arrival PMF for a specific subspecialty
    >>> cardiology_arrivals = transfer_arrivals['cardiology']

    Notes
    -----
    Algorithm:

    For each target subspecialty, the function:

    1. Initializes with zero arrivals (PMF = [1.0, 0.0])
    2. Iterates over each potential source subspecialty
    3. Gets the departure PMF from the source
    4. Gets transfer probabilities from the transfer model
    5. If the source sends patients to the target:

       - Calculates compound_prob = prob_transfer × prob_destination
       - Scales the departure PMF by compound_prob
       - Convolves with the accumulating arrival PMF

    6. Stores the final aggregated arrival PMF

    Assumptions:

    - Transfers from different source subspecialties are independent
    - Transfer probabilities are constant across patients
    - The departure PMF already accounts for the timing window
    - Self-transfers (source == target) are excluded

    The function handles zero probabilities naturally without requiring a threshold
    parameter. Convolution operations are numerically stable even with small
    probabilities.
    """
    predicted_arrivals = {}

    for target_subspecialty in subspecialties:
        # Initialize with zero arrivals: P(0 arrivals) = 1.0
        arrival_pmf = np.array([1.0, 0.0])

        for source_subspecialty in subspecialties:
            # Skip self-transfers
            if source_subspecialty == target_subspecialty:
                continue

            # Get departure PMF for source
            if (
                "pmf_inpatient_departures_within_window"
                not in subspecialty_data[source_subspecialty]
            ):
                raise KeyError(
                    f"Missing 'pmf_inpatient_departures_within_window' for "
                    f"subspecialty '{source_subspecialty}'"
                )

            departure_pmf = subspecialty_data[source_subspecialty][
                "pmf_inpatient_departures_within_window"
            ]

            # Get transfer probabilities from model
            try:
                prob_transfer = transfer_model.get_transfer_prob(source_subspecialty)
                dest_dist = transfer_model.get_destination_distribution(
                    source_subspecialty
                )
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Error getting transfer probabilities for '{source_subspecialty}': {e}"
                )

            # Skip if no transfers from this source
            if prob_transfer == 0:
                continue

            # Check if this source sends to our target
            if target_subspecialty not in dest_dist:
                continue

            prob_this_dest = dest_dist[target_subspecialty]

            # Calculate compound probability
            compound_prob = prob_transfer * prob_this_dest

            # Scale the departure PMF by compound probability
            scaled_pmf = scale_pmf_by_probability(departure_pmf, compound_prob)

            # Accumulate by convolving with existing arrival PMF
            arrival_pmf = convolve_pmfs(arrival_pmf, scaled_pmf)

        # Store the final arrival PMF for this target
        predicted_arrivals[target_subspecialty] = arrival_pmf

    return predicted_arrivals
