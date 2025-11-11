"""Demand preparation utilities for later consolidation.

This module prepares per-subspecialty demand inputs using a flexible
architecture. It converts trained model outputs and current patient snapshots
into structured representations organised by direction (inflows/outflows).
The flow-based structure allows flexible selection of which flows to include
in predictions and easy extension to new flow types in the future.

The outputs are independent per subspecialty and do not presuppose any
particular consolidation hierarchy. They can be used directly for single-
specialty analyses or fed into any combination scheme (including hierarchical
schemes) implemented elsewhere.

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
from patientflow.predictors.transfer_predictor import (
    TransferProbabilityEstimator,
)
from patientflow.aggregate import (
    model_input_to_pred_proba,
    pred_proba_to_agg_predicted,
)
from patientflow.predict.distribution import Distribution
from patientflow.calculate.admission_in_prediction_window import (
    calculate_probability,
    calculate_admission_probability_from_survival_curve,
)
from patientflow.model_artifacts import TrainedClassifier


@dataclass(frozen=True)
class FlowInputs:
    """Represents a single source of patient flow.

    This class encapsulates a flow of patients (either arriving or departing)
    with its distribution type and parameters. It provides a uniform interface
    for both probability mass functions and Poisson-distributed flows.

    Attributes
    ----------
    flow_id : str
        Unique identifier for this flow (e.g., "ed_current", "transfers_in")
    flow_type : str
        Type of distribution: "pmf" for probability mass function or "poisson" for Poisson
    distribution : np.ndarray or float
        For "pmf": numpy array where distribution[k] = P(k patients)
        For "poisson": float representing the Poisson rate parameter (lambda)
    display_name : str, optional
        Human-readable name for display purposes. If not provided, flow_id will be
        formatted automatically (underscores replaced with spaces, title cased).

    Examples
    --------
    >>> # PMF flow (e.g., current ED patients)
    >>> ed_flow = FlowInputs(
    ...     flow_id="ed_current",
    ...     flow_type="pmf",
    ...     distribution=np.array([0.5, 0.3, 0.2]),
    ...     display_name="Admissions from current ED"
    ... )

    >>> # Poisson flow (e.g., yet-to-arrive patients)
    >>> yta_flow = FlowInputs(
    ...     flow_id="ed_yta",
    ...     flow_type="poisson",
    ...     distribution=2.5,
    ...     display_name="ED yet-to-arrive admissions"
    ... )
    """

    flow_id: str
    flow_type: str
    distribution: Union[np.ndarray, float]
    display_name: Optional[str] = None

    def get_display_name(self) -> str:
        """Get human-readable display name.

        Returns
        -------
        str
            Display name if provided, otherwise formatted flow_id.
        """
        if self.display_name:
            return self.display_name
        return self.flow_id.replace("_", " ").title()


@dataclass(frozen=True)
class SubspecialtyPredictionInputs:
    """Input parameters for subspecialty demand prediction.

    These inputs represent the probability distributions and parameters
    needed to predict demand for a single subspecialty. This dataclass packages
    the outputs from build_subspecialty_data for use in hierarchical prediction.

    The inputs are organized into inflows (patient arrivals) and outflows (patient
    departures), with each flow represented as a FlowInputs object containing its
    distribution type and parameters.

    Attributes
    ----------
    subspecialty_id : str
        Unique identifier for the subspecialty
    prediction_window : Any
        Time window over which predictions are made (typically a timedelta)
    inflows : Dict[str, FlowInputs]
        Dictionary mapping flow identifiers to FlowInputs objects for arrivals.
        Standard keys include:

        - "ed_current": Current ED patients who will be admitted (PMF)
        - "ed_yta": Yet-to-arrive ED patients who will be admitted (Poisson)
        - "non_ed_yta": Yet-to-arrive non-ED emergency admissions (Poisson)
        - "elective_yta": Yet-to-arrive elective admissions (Poisson)
        - "elective_transfers": Elective patients transferring from other subspecialties (PMF)
        - "emergency_transfers": Emergency patients transferring from other subspecialties (PMF)
    outflows : Dict[str, FlowInputs]
        Dictionary mapping flow identifiers to FlowInputs objects for departures.
        Standard keys include:

        - "elective_departures": Current elective inpatients who will depart (PMF)
        - "emergency_departures": Current emergency inpatients who will depart (PMF)

        Future extensions may include "transfers_out", "deaths", etc.

    Notes
    -----
    This dataclass is immutable (frozen=True) to prevent accidental modification after creation.
    All flows should represent distributions/rates for the same prediction window.

    The dictionary-based structure allows flexible inclusion/exclusion of flow types
    and easy extension to new flow types in the future.
    """

    subspecialty_id: str
    prediction_window: Any
    inflows: Dict[str, FlowInputs]
    outflows: Dict[str, FlowInputs]

    def __repr__(self) -> str:
        def format_pmf(
            arr: np.ndarray,
            max_display: int = 10,
            total_count: Optional[int] = None,
            custom_bracket_text: Optional[str] = None,
        ) -> str:
            expectation = np.sum(np.arange(len(arr)) * arr)

            # Use custom bracket text if provided, otherwise use total_count
            if custom_bracket_text is not None:
                if custom_bracket_text == "":
                    total_str = ""
                else:
                    total_str = f" {custom_bracket_text}"
            elif total_count is not None:
                total_str = f" of {total_count}"
            else:
                total_str = ""

            if len(arr) <= max_display:
                values = ", ".join(f"{v:.3f}" for v in arr)
                return f"PMF[0:{len(arr)}]: [{values}] (E={expectation:.1f}{total_str})"

            # Determine display window centered on expectation
            center_idx = int(np.round(expectation))
            half_window = max_display // 2
            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(arr), start_idx + max_display)

            # Adjust if we're near the end
            if end_idx - start_idx < max_display:
                start_idx = max(0, end_idx - max_display)

            # Format the displayed portion
            display_values = ", ".join(f"{v:.3f}" for v in arr[start_idx:end_idx])

            # Show with index range
            return f"PMF[{start_idx}:{end_idx}]: [{display_values}] (E={expectation:.1f}{total_str})"

        def format_flow(flow: FlowInputs) -> str:
            if flow.flow_type == "pmf":
                assert isinstance(flow.distribution, np.ndarray)
                total_count = len(flow.distribution) - 1

                # Customize bracket text based on flow type
                custom_bracket_text = None
                if flow.flow_id == "ed_current":
                    custom_bracket_text = f"of {total_count} patients in ED"
                elif flow.flow_id in ["elective_transfers", "emergency_transfers"]:
                    # Remove 'of N' for transfers - just show expectation
                    custom_bracket_text = ""
                elif flow.flow_id == "emergency_departures":
                    custom_bracket_text = (
                        f"of {total_count} emergency patients in subspec"
                    )
                elif flow.flow_id == "elective_departures":
                    custom_bracket_text = (
                        f"of {total_count} elective patients in subspec"
                    )

                return format_pmf(
                    flow.distribution,
                    total_count=total_count,
                    custom_bracket_text=custom_bracket_text,
                )
            elif flow.flow_type == "poisson":
                return f"λ = {flow.distribution:.3f}"
            else:
                return f"{flow.flow_type}: {flow.distribution}"

        # Build output dynamically
        lines = [f"SubspecialtyPredictionInputs(subspecialty='{self.subspecialty_id}')"]

        # INFLOWS section
        if self.inflows:
            lines.append("  INFLOWS:")
            for flow in self.inflows.values():
                flow_str = format_flow(flow)
                lines.append(f"    {flow.get_display_name():<40} {flow_str}")

        # OUTFLOWS section
        if self.outflows:
            lines.append("  OUTFLOWS:")
            for flow in self.outflows.values():
                flow_str = format_flow(flow)
                lines.append(f"    {flow.get_display_name():<40} {flow_str}")

        return "\n".join(lines)


def _validate_models_and_data(
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
        TransferProbabilityEstimator,
    ],
    prediction_time: Tuple[int, int],
    ed_snapshots: pd.DataFrame,
    inpatient_snapshots: pd.DataFrame,
    prediction_window,
    specialties: List[str],
) -> None:
    """Validate all models and input data.

    Raises
    ------
    TypeError
        If any model is not of the expected type
    ValueError
        If required columns are missing, models are not fitted, or parameters
        don't match between models and requested parameters
    """
    (
        ed_classifier,
        inpatient_classifier,
        spec_model,
        yet_to_arrive_model,
        non_ed_yta_model,
        elective_yta_model,
        transfer_model,
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
    if not isinstance(transfer_model, TransferProbabilityEstimator):
        raise TypeError(
            "Seventh model must be of type TransferProbabilityEstimator (transfer)"
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

    # Ensure TransferProbabilityEstimator has been fitted
    if not hasattr(transfer_model, "is_fitted_") or not transfer_model.is_fitted_:
        raise ValueError("Transfer model has not been fit")

    # Validate specialties alignment
    if hasattr(yet_to_arrive_model, "filters"):
        if not set(yet_to_arrive_model.filters.keys()) == set(specialties):
            raise ValueError(
                f"Requested specialties {set(specialties)} do not match the specialties of the trained yet-to-arrive model {set(yet_to_arrive_model.filters.keys())}"
            )

    special_params = spec_model.special_params

    if special_params:
        special_category_dict = special_params["special_category_dict"]
    else:
        special_category_dict = None

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


def _prepare_base_probabilities(
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
        TransferProbabilityEstimator,
    ],
    ed_snapshots: pd.DataFrame,
    inpatient_snapshots: pd.DataFrame,
    prediction_window,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    use_admission_in_window_prob: bool,
) -> Dict[str, Any]:
    """Prepare base probability calculations for all patients.

    Returns
    -------
    dict
        Dictionary containing prepared probabilities and other computed values
    """
    (
        ed_classifier,
        inpatient_classifier,
        spec_model,
        yet_to_arrive_model,
        non_ed_yta_model,
        elective_yta_model,
        transfer_model,
    ) = models

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

    # Split inpatient snapshots into elective and emergency
    elective_snapshots = inpatient_snapshots_temp[
        inpatient_snapshots_temp["admission_type"] == "elective"
    ]
    emergency_snapshots = inpatient_snapshots_temp[
        inpatient_snapshots_temp["admission_type"] == "emergency"
    ]

    # Departure probability for current inpatients (per row)
    prob_departure_after_elective = (
        model_input_to_pred_proba(elective_snapshots, inpatient_pipeline)
        if not elective_snapshots.empty
        else pd.Series(dtype=float)
    )

    prob_departure_after_emergency = (
        model_input_to_pred_proba(emergency_snapshots, inpatient_pipeline)
        if not emergency_snapshots.empty
        else pd.Series(dtype=float)
    )

    # Specialty probabilities per row for ED patients
    if hasattr(spec_model, "predict_dataframe"):
        ed_snapshots.loc[:, "specialty_prob"] = spec_model.predict_dataframe(
            ed_snapshots
        )
    else:
        special_params = spec_model.special_params
        if special_params:
            special_category_func = special_params["special_category_func"]
            special_category_dict = special_params["special_category_dict"]
        else:
            special_category_func = special_category_dict = None

        ed_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
            [],  # specialties will be determined from the model
            spec_model,
            ed_snapshots,
            special_category_func=special_category_func,
            special_category_dict=special_category_dict,
        )

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

    # Prepare subgroup masks if using MultiSubgroupPredictor
    special_params = spec_model.special_params
    if special_params:
        special_func_map = special_params["special_func_map"]
    else:
        special_func_map = None

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
    if "default" not in ed_masks_by_func:
        ed_masks_by_func["default"] = pd.Series(True, index=ed_snapshots.index)

    return {
        "ed_snapshots": ed_snapshots,
        "inpatient_snapshots": inpatient_snapshots,
        "prob_admission_after_ed": prob_admission_after_ed,
        "prob_departure_after_elective": prob_departure_after_elective,
        "prob_departure_after_emergency": prob_departure_after_emergency,
        "prob_admission_in_window": prob_admission_in_window,
        "specialty_to_subgroups": specialty_to_subgroups,
        "ed_masks_by_func": ed_masks_by_func,
        "special_func_map": special_func_map,
    }


def _process_ed_patients_for_specialty(
    spec: str,
    ed_snapshots: pd.DataFrame,
    specialty_to_subgroups: Dict[str, List[str]],
    ed_masks_by_func: Dict[str, pd.Series],
    prob_admission_after_ed: pd.Series,
    prob_admission_in_window: pd.Series,
) -> Dict[str, Any]:
    """Process ED patients for a specific specialty.

    Returns
    -------
    dict
        Dictionary containing processed ED data for the specialty
    """
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
    filtered_prob_admission_after_ed = prob_admission_after_ed.loc[ed_non_zero_indices]

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

    return {
        "agg_predicted_in_ed": agg_predicted_in_ed,
    }


def _process_inpatients_for_specialty_by_admission_type(
    spec: str,
    inpatient_snapshots: pd.DataFrame,
    prob_departure_series: pd.Series,
    admission_type: str,
) -> Dict[str, Any]:
    """Process inpatients for a specific specialty and admission type.

    Parameters
    ----------
    spec : str
        The subspecialty to process
    inpatient_snapshots : pd.DataFrame
        DataFrame containing inpatient snapshot data
    prob_departure_series : pd.Series
        Series containing departure probabilities for the admission type
    admission_type : str
        The admission type to process ("elective" or "emergency")

    Returns
    -------
    dict
        Dictionary containing processed inpatient data for the specialty and admission type
    """
    # Process inpatients for the specific admission type (no weighting required)
    admission_type_mask = (inpatient_snapshots["current_subspecialty"] == spec) & (
        inpatient_snapshots["admission_type"] == admission_type
    )
    admission_type_indices = inpatient_snapshots[admission_type_mask].index

    if len(admission_type_indices) > 0:
        filtered_prob_departure = prob_departure_series.loc[admission_type_indices]
        agg_predicted_departures = pred_proba_to_agg_predicted(filtered_prob_departure)
    else:
        # No inpatients of this type in this specialty, create zero PMF
        agg_predicted_departures = {"agg_proba": np.array([1.0, 0.0])}

    return {
        f"agg_predicted_{admission_type}_departures": agg_predicted_departures,
    }


def _create_flow_inputs(
    spec: str,
    agg_predicted_in_ed: Dict[str, np.ndarray],
    agg_predicted_elective_departures: Dict[str, np.ndarray],
    agg_predicted_emergency_departures: Dict[str, np.ndarray],
    yet_to_arrive_model: Union[
        ParametricIncomingAdmissionPredictor, EmpiricalIncomingAdmissionPredictor
    ],
    non_ed_yta_model: DirectAdmissionPredictor,
    elective_yta_model: DirectAdmissionPredictor,
    prediction_time: Tuple[int, int],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Dict[str, Dict[str, FlowInputs]]:
    """Create FlowInputs objects for inflows and outflows.

    Returns
    -------
    dict
        Dictionary with 'inflows' and 'outflows' keys containing FlowInputs objects
    """
    prediction_context = {spec: {"prediction_time": prediction_time}}

    # Build FlowInputs objects for inflows and outflows
    # INFLOWS: All sources of patient arrivals to this subspecialty
    inflows_dict = {
        "ed_current": FlowInputs(
            flow_id="ed_current",
            flow_type="pmf",
            distribution=np.array(agg_predicted_in_ed["agg_proba"]),
            display_name="Admissions from current ED",
        ),
        "ed_yta": FlowInputs(
            flow_id="ed_yta",
            flow_type="poisson",
            distribution=float(
                yet_to_arrive_model.predict_mean(
                    prediction_context, x1=x1, y1=y1, x2=x2, y2=y2
                )
            ),
            display_name="ED yet-to-arrive admissions",
        ),
        "non_ed_yta": FlowInputs(
            flow_id="non_ed_yta",
            flow_type="poisson",
            distribution=float(
                non_ed_yta_model.predict_mean(prediction_context)
                if non_ed_yta_model is not None
                else 0.0
            ),
            display_name="Non-ED emergency admissions",
        ),
        "elective_yta": FlowInputs(
            flow_id="elective_yta",
            flow_type="poisson",
            distribution=float(
                elective_yta_model.predict_mean(prediction_context)
                if elective_yta_model is not None
                else 0.0
            ),
            display_name="Elective admissions",
        ),
        # Note: "transfers_in" will be added later after compute_transfer_arrivals()
    }

    # OUTFLOWS: All sources of patient departures from this subspecialty
    outflows_dict = {
        "emergency_departures": FlowInputs(
            flow_id="emergency_departures",
            flow_type="pmf",
            distribution=np.array(agg_predicted_emergency_departures["agg_proba"]),
            display_name="Emergency inpatient departures",
        ),
        "elective_departures": FlowInputs(
            flow_id="elective_departures",
            flow_type="pmf",
            distribution=np.array(agg_predicted_elective_departures["agg_proba"]),
            display_name="Elective inpatient departures",
        ),
    }

    return {
        "inflows": inflows_dict,
        "outflows": outflows_dict,
    }


def _build_legacy_flows(
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
        TransferProbabilityEstimator,
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
    base_probs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build flows for all specialties using legacy processing logic.

    Returns
    -------
    dict
        Dictionary mapping specialty to temporary flow data
    """
    (
        ed_classifier,
        inpatient_classifier,
        spec_model,
        yet_to_arrive_model,
        non_ed_yta_model,
        elective_yta_model,
        transfer_model,
    ) = models

    # Extract prepared data
    ed_snapshots = base_probs["ed_snapshots"]
    inpatient_snapshots = base_probs["inpatient_snapshots"]
    prob_admission_after_ed = base_probs["prob_admission_after_ed"]
    prob_departure_after_elective = base_probs["prob_departure_after_elective"]
    prob_departure_after_emergency = base_probs["prob_departure_after_emergency"]
    prob_admission_in_window = base_probs["prob_admission_in_window"]
    specialty_to_subgroups = base_probs["specialty_to_subgroups"]
    ed_masks_by_func = base_probs["ed_masks_by_func"]

    # First pass: gather computed data in temporary structure
    temp_subspecialty_data: Dict[str, Dict[str, Any]] = {}

    for spec in specialties:
        # Process ED patients
        ed_data = _process_ed_patients_for_specialty(
            spec,
            ed_snapshots,
            specialty_to_subgroups,
            ed_masks_by_func,
            prob_admission_after_ed,
            prob_admission_in_window,
        )

        # Process inpatients
        elective_data = _process_inpatients_for_specialty_by_admission_type(
            spec, inpatient_snapshots, prob_departure_after_elective, "elective"
        )
        emergency_data = _process_inpatients_for_specialty_by_admission_type(
            spec, inpatient_snapshots, prob_departure_after_emergency, "emergency"
        )

        # Create flow inputs
        flow_data = _create_flow_inputs(
            spec,
            ed_data["agg_predicted_in_ed"],
            elective_data["agg_predicted_elective_departures"],
            emergency_data["agg_predicted_emergency_departures"],
            yet_to_arrive_model,
            non_ed_yta_model,
            elective_yta_model,
            prediction_time,
            x1,
            y1,
            x2,
            y2,
        )

        # Store in temporary dictionary structure
        temp_subspecialty_data[spec] = flow_data

    return temp_subspecialty_data


def _finalise_subspecialty_data(
    temp_subspecialty_data: Dict[str, Dict[str, Any]],
    transfer_model: TransferProbabilityEstimator,
    specialties: List[str],
    prediction_window,
) -> Dict[str, SubspecialtyPredictionInputs]:
    """Add transfers and create final SubspecialtyPredictionInputs objects.

    Returns
    -------
    dict
        Dictionary mapping subspecialty_id to SubspecialtyPredictionInputs
    """
    # Compute transfer arrivals using the departure PMFs from temporary data
    transfer_arrivals = compute_transfer_arrivals(
        temp_subspecialty_data, transfer_model, specialties
    )

    # Second pass: Add transfer arrivals to inflows and create final immutable dataclass objects
    subspecialty_data: Dict[str, SubspecialtyPredictionInputs] = {}
    for spec in specialties:
        # Add elective and emergency transfers to the inflows dictionary
        temp_subspecialty_data[spec]["inflows"]["elective_transfers"] = FlowInputs(
            flow_id="elective_transfers",
            flow_type="pmf",
            distribution=transfer_arrivals["elective"][spec],
            display_name="Elective transfers from other subspecialties",
        )

        temp_subspecialty_data[spec]["inflows"]["emergency_transfers"] = FlowInputs(
            flow_id="emergency_transfers",
            flow_type="pmf",
            distribution=transfer_arrivals["emergency"][spec],
            display_name="Emergency transfers from other subspecialties",
        )

        # Create final immutable dataclass with complete inflows and outflows
        subspecialty_data[spec] = SubspecialtyPredictionInputs(
            subspecialty_id=spec,
            prediction_window=prediction_window,
            inflows=temp_subspecialty_data[spec]["inflows"],
            outflows=temp_subspecialty_data[spec]["outflows"],
        )

    return subspecialty_data


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
        TransferProbabilityEstimator,
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
    from current ED patients, departures from current inpatients, the expected
    means of yet-to-arrive admissions, and transfer arrival distributions.

    Parameters
    ----------
    models : tuple
        Tuple of seven trained models:

        - ed_classifier: TrainedClassifier for ED admission probability prediction
        - inpatient_classifier: TrainedClassifier for inpatient departure probability prediction
        - spec_model: SequenceToOutcomePredictor | ValueToOutcomePredictor | MultiSubgroupPredictor
          for specialty assignment probabilities
        - ed_yta_model: ParametricIncomingAdmissionPredictor | EmpiricalIncomingAdmissionPredictor
          for ED yet-to-arrive predictions
        - non_ed_yta_model: DirectAdmissionPredictor for non-ED emergency predictions
        - elective_yta_model: DirectAdmissionPredictor for elective predictions
        - transfer_model: TransferProbabilityEstimator for internal transfer predictions
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
    The function combines six sources of demand:

    1. Current ED patients (converted to probability mass function)
    2. Current inpatients (converted to probability mass function for departures)
    3. Yet-to-arrive ED patients (converted to Poisson parameters)
    4. Yet-to-arrive non-ED emergency patients (converted to Poisson parameters)
    5. Yet-to-arrive elective patients (converted to Poisson parameters)
    6. Transfer arrivals from other subspecialties (converted to probability mass function)

    """
    # 1. Validate inputs
    _validate_models_and_data(
        models,
        prediction_time,
        ed_snapshots,
        inpatient_snapshots,
        prediction_window,
        specialties,
    )

    # 2. Prepare base probabilities
    base_probs = _prepare_base_probabilities(
        models,
        ed_snapshots,
        inpatient_snapshots,
        prediction_window,
        x1,
        y1,
        x2,
        y2,
        use_admission_in_window_prob,
    )

    # 3. Build flows using legacy processing logic
    temp_subspecialty_data = _build_legacy_flows(
        models,
        prediction_time,
        ed_snapshots,
        inpatient_snapshots,
        specialties,
        prediction_window,
        x1,
        y1,
        x2,
        y2,
        base_probs,
    )

    # 4. Finalize with transfers
    return _finalise_subspecialty_data(
        temp_subspecialty_data, models[6], specialties, prediction_window
    )


def compute_transfer_arrivals(
    subspecialty_data: Union[
        Dict[str, Dict[str, Any]], Dict[str, SubspecialtyPredictionInputs]
    ],
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
        Either a dict of dicts with nested structure containing departure FlowInputs,
        or a dict of SubspecialtyPredictionInputs objects. For dict format, expects:
        {'subspecialty': {'outflows': {'departures': FlowInputs(...)}}}
    transfer_model : TransferProbabilityEstimator
        Trained transfer probability estimator with methods:
        - get_transfer_prob(source) -> float
        - get_destination_distribution(source) -> dict
    subspecialties : list of str
        List of all subspecialties in the system

    Returns
    -------
    dict
        Nested dictionary mapping admission_type to subspecialty_id to PMF of arrivals from transfers.
        {
            'elective': {
                'subspecialty_name': numpy.ndarray (PMF of elective transfer arrivals)
            },
            'emergency': {
                'subspecialty_name': numpy.ndarray (PMF of emergency transfer arrivals)
            }
        }

    Raises
    ------
    KeyError
        If subspecialty_data is missing required departure flow information
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
    >>> # Access arrival PMF for a specific subspecialty and admission type
    >>> cardiology_elective_arrivals = transfer_arrivals['elective']['cardiology']
    >>> cardiology_emergency_arrivals = transfer_arrivals['emergency']['cardiology']

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
       - Scales the departure PMF by compound_prob using Distribution.thin()
       - Convolves with the accumulating arrival PMF using Distribution.convolve()

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
    predicted_arrivals: Dict[str, Dict[str, np.ndarray]] = {
        "elective": {},
        "emergency": {},
    }

    for admission_type in ["elective", "emergency"]:
        for target_subspecialty in subspecialties:
            # Initialize with zero arrivals: P(0 arrivals) = 1.0
            arrival_dist = Distribution.from_pmf(np.array([1.0, 0.0]))

            for source_subspecialty in subspecialties:
                # Skip self-transfers
                if source_subspecialty == target_subspecialty:
                    continue

                # Get departure PMF for source (handle both dict and dataclass)
                source_data = subspecialty_data[source_subspecialty]
                if isinstance(source_data, SubspecialtyPredictionInputs):
                    # Access through new structure: outflows dict -> "departures" -> distribution
                    if f"{admission_type}_departures" not in source_data.outflows:
                        raise KeyError(
                            f"Missing '{admission_type}_departures' outflow for subspecialty '{source_subspecialty}'"
                        )
                    departure_pmf = source_data.outflows[
                        f"{admission_type}_departures"
                    ].distribution
                elif isinstance(source_data, dict):
                    # Temporary data structure during build (dict with "inflows" and "outflows" keys)
                    if (
                        "outflows" not in source_data
                        or f"{admission_type}_departures" not in source_data["outflows"]
                    ):
                        raise KeyError(
                            f"Missing '{admission_type}_departures' in 'outflows' for subspecialty '{source_subspecialty}'"
                        )
                    departure_pmf = source_data["outflows"][
                        f"{admission_type}_departures"
                    ].distribution
                else:
                    raise TypeError(
                        f"subspecialty_data values must be dict or SubspecialtyPredictionInputs, "
                        f"got {type(source_data)}"
                    )

                # Get transfer probabilities from model
                try:
                    prob_transfer = transfer_model.get_transfer_prob(
                        source_subspecialty, admission_type
                    )
                    dest_dist = transfer_model.get_destination_distribution(
                        source_subspecialty, admission_type
                    )
                except ValueError as e:
                    # Handle case where cohort doesn't exist in transfer model
                    if "not found in trained model" in str(e):
                        # Skip this admission type if not trained for it
                        continue
                    else:
                        raise ValueError(
                            f"Error getting transfer probabilities for '{source_subspecialty}' and admission type '{admission_type}': {e}"
                        )
                except KeyError as e:
                    raise ValueError(
                        f"Error getting transfer probabilities for '{source_subspecialty}' and admission type '{admission_type}': {e}"
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

                # Create Distribution from departure PMF and apply thinning
                departure_dist = Distribution.from_pmf(departure_pmf)
                scaled_dist = departure_dist.thin(compound_prob)

                # Accumulate by convolving with existing arrival distribution
                arrival_dist = arrival_dist.convolve(scaled_dist)

            # Store the final arrival PMF for this target
            predicted_arrivals[admission_type][target_subspecialty] = (
                arrival_dist.probabilities
            )

    return predicted_arrivals
