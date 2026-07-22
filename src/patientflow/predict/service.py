"""Demand preparation utilities for later consolidation.

This module prepares per-service demand inputs using a flexible
architecture. It converts trained model outputs and current patient snapshots
into structured representations organised by direction (inflows/outflows).
The flow-based structure allows flexible selection of which flows to include
in predictions and easy extension to new flow types in the future.

The outputs are independent per service and do not presuppose any
particular consolidation hierarchy. They can be used directly for single-
service analyses or fed into any combination scheme (including hierarchical
schemes) implemented elsewhere.

"""

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from patientflow.predict.emergency_demand import (
    add_missing_columns,
    dataframe_for_classifier_predict_proba,
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
from patientflow.predict.transfers import compute_transfer_arrivals
from patientflow.calculate.admission_in_prediction_window import (
    calculate_probability,
    calculate_admission_probability_from_survival_curve,
)
from patientflow.model_artifacts import TrainedClassifier, ServiceModels
from patientflow.predict.flow_selection_checks import (
    assert_model_types_for_flow,
    validate_admission_curve_params,
    validate_ed_classifier,
    validate_ed_snapshots_present,
    validate_elective_yta,
    validate_inpatient_classifier,
    validate_inpatient_snapshots_present,
    validate_non_ed_yta,
    validate_spec_model_for_ed,
    validate_transfer_model,
    validate_yta_ed,
)
from patientflow.predict.types import FlowSelection


def warn_specialty_mismatch(
    requested: set,
    trained: set,
    source_label: str,
    *,
    stacklevel: int = 3,
) -> None:
    """Emit warnings when requested and trained specialty sets diverge.

    Parameters
    ----------
    requested : set
        Specialties coming from the current request (e.g. Clarity).
    trained : set
        Specialties the model was trained on.
    source_label : str
        Human-readable name for the trained artefact, used in messages
        (e.g. `"yet-to-arrive model"` or `"special_category_dict"`).
    stacklevel : int, optional
        Passed to `warnings.warn()` so the warning points to the
        caller rather than this helper.  Default is 3 (caller's caller).
    """
    new_in_request = requested - trained
    missing_from_request = trained - requested
    if new_in_request:
        warnings.warn(
            f"{len(new_in_request)} specialties found in the request but absent "
            f"from the trained {source_label} (models may need retraining).",
            stacklevel=stacklevel,
        )
    if missing_from_request:
        warnings.warn(
            f"{len(missing_from_request)} specialties present in the trained "
            f"{source_label} but absent from the request.",
            stacklevel=stacklevel,
        )


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
    aspirational: bool = False

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
class ServicePredictionInputs:
    """Input parameters for service demand prediction.

    These inputs represent the probability distributions and parameters
    needed to predict demand for a single service (e.g., subspecialty). This dataclass packages
    the outputs from build_service_data for use in prediction.

    The inputs are organized into inflows (patient arrivals) and outflows (patient
    departures), with each flow represented as a FlowInputs object containing its
    distribution type and parameters.

    Attributes
    ----------
    service_id : str
        Unique identifier for the service
    prediction_window : Any
        Time window over which predictions are made (typically a timedelta)
    inflows : Dict[str, FlowInputs]
        Dictionary mapping flow identifiers to FlowInputs objects for arrivals.
        Standard keys include:

        - "ed_current": Current ED patients who will be admitted (PMF)
        - "ed_yta": Yet-to-arrive ED patients who will be admitted (Poisson)
        - "non_ed_yta": Yet-to-arrive non-ED emergency admissions (Poisson)
        - "elective_yta": Yet-to-arrive elective admissions (Poisson)
        - "elective_transfers": Elective patients transferring from other services (PMF)
        - "emergency_transfers": Emergency patients transferring from other services (PMF)
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

    service_id: str
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
                        f"of {total_count} emergency patients in service"
                    )
                elif flow.flow_id == "elective_departures":
                    custom_bracket_text = (
                        f"of {total_count} elective patients in service"
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
        lines = [f"ServicePredictionInputs(service='{self.service_id}')"]

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


def _normalize_to_service_models(
    models: Union[
        ServiceModels,
        Tuple[
            Optional[TrainedClassifier],
            Optional[TrainedClassifier],
            Optional[
                Union[
                    SequenceToOutcomePredictor,
                    ValueToOutcomePredictor,
                    MultiSubgroupPredictor,
                ]
            ],
            Optional[
                Union[
                    ParametricIncomingAdmissionPredictor,
                    EmpiricalIncomingAdmissionPredictor,
                ]
            ],
            Optional[DirectAdmissionPredictor],
            Optional[DirectAdmissionPredictor],
            Optional[TransferProbabilityEstimator],
        ],
    ],
    prediction_time: Tuple[int, int],
    prediction_window,
) -> ServiceModels:
    """Normalise `models` to a `ServiceModels` instance (`patientflow.model_artifacts`).

    Parameters
    ----------
    models : ServiceModels or tuple of length 7
        Either a `ServiceModels` bundle or the legacy seven-tuple of optional
        model slots `(ed_classifier, inpatient_classifier, spec_model,
        ed_yta_model, non_ed_yta_model, elective_yta_model, transfer_model)`.
    prediction_time : tuple of (int, int)
        Hour and minute; must match `ServiceModels.prediction_time` when *models*
        is already a `ServiceModels` instance.
    prediction_window : datetime.timedelta
        Horizon; must match `ServiceModels.prediction_window` when *models* is
        a `ServiceModels` instance.

    Returns
    -------
    ServiceModels
        Normalised model bundle.

    Raises
    ------
    TypeError
        If *models* is a tuple with length other than seven.
    ValueError
        If *models* is `ServiceModels` but `prediction_time` or
        `prediction_window` disagree with the dataclass fields.
    """
    if isinstance(models, ServiceModels):
        sm = models
        if sm.prediction_time != prediction_time:
            raise ValueError(
                f"ServiceModels.prediction_time {sm.prediction_time} does not match "
                f"prediction_time argument {prediction_time}"
            )
        if sm.prediction_window != prediction_window:
            raise ValueError(
                "ServiceModels.prediction_window does not match prediction_window argument"
            )
        return sm
    if not isinstance(models, tuple) or len(models) != 7:
        raise TypeError("models must be a ServiceModels instance or a 7-tuple")
    return ServiceModels(
        prediction_time=prediction_time,
        prediction_window=prediction_window,
        ed_classifier=models[0],
        inpatient_classifier=models[1],
        spec_model=models[2],
        ed_yta_model=models[3],
        non_ed_yta_model=models[4],
        elective_yta_model=models[5],
        transfer_model=models[6],
    )


def _validate_models_and_data(
    service_models: ServiceModels,
    flow_selection: FlowSelection,
    ed_snapshots: Optional[pd.DataFrame],
    inpatient_snapshots: Optional[pd.DataFrame],
    specialties: List[str],
    *,
    use_admission_in_window_prob: bool,
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
) -> None:
    """Validate model slots, snapshots, and curve parameters for *flow_selection*.

    Parameters
    ----------
    service_models : ServiceModels
        Named model bundle including `prediction_time` and `prediction_window`.
    flow_selection : FlowSelection
        Which flows are active; drives which slots and inputs are required.
    ed_snapshots : pandas.DataFrame or None
        ED patient snapshot at the prediction moment.
    inpatient_snapshots : pandas.DataFrame or None
        Inpatient snapshot at the prediction moment.
    specialties : list of str
        Service identifiers to prepare.
    use_admission_in_window_prob : bool
        Whether current-ED rows use in-window admission weighting.
    x1, y1, x2, y2 : float or None
        Parametric curve parameters when required.

    Raises
    ------
    ValueError
        From flow-selection checks, missing columns, unfitted models, or
        mismatched `prediction_time` on classifiers.
    TypeError
        From `assert_model_types_for_flow` in `patientflow.predict.flow_selection_checks`
        when a non-`None` model has the wrong type.

    See Also
    --------
    patientflow.predict.flow_selection_checks
    """
    flow_selection.validate()
    prediction_time = service_models.prediction_time
    ed_classifier = service_models.ed_classifier
    inpatient_classifier = service_models.inpatient_classifier
    spec_model = service_models.spec_model
    yet_to_arrive_model = service_models.ed_yta_model
    non_ed_yta_model = service_models.non_ed_yta_model
    elective_yta_model = service_models.elective_yta_model
    transfer_model = service_models.transfer_model

    validate_ed_classifier(flow_selection, ed_classifier)
    validate_spec_model_for_ed(flow_selection, spec_model)
    validate_inpatient_classifier(flow_selection, inpatient_classifier)
    validate_yta_ed(flow_selection, yet_to_arrive_model)
    validate_non_ed_yta(flow_selection, non_ed_yta_model)
    validate_elective_yta(flow_selection, elective_yta_model)
    validate_transfer_model(flow_selection, transfer_model)
    validate_ed_snapshots_present(flow_selection, ed_snapshots)
    validate_inpatient_snapshots_present(
        flow_selection, transfer_model, inpatient_snapshots
    )
    has_ed_snapshots = ed_snapshots is not None
    validate_admission_curve_params(
        flow_selection,
        yet_to_arrive_model,
        use_admission_in_window_prob=use_admission_in_window_prob,
        has_ed_snapshots=has_ed_snapshots,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    assert_model_types_for_flow(
        flow_selection,
        ed_classifier=ed_classifier,
        inpatient_classifier=inpatient_classifier,
        spec_model=spec_model,
        yet_to_arrive_model=yet_to_arrive_model,
        non_ed_yta_model=non_ed_yta_model,
        elective_yta_model=elective_yta_model,
        transfer_model=transfer_model,
    )

    if ed_snapshots is not None:
        if "elapsed_los" not in ed_snapshots.columns:
            raise ValueError("Column 'elapsed_los' not found in ed_snapshots")
        if not pd.api.types.is_timedelta64_dtype(ed_snapshots["elapsed_los"]):
            actual_type = ed_snapshots["elapsed_los"].dtype
            raise ValueError(
                "Column 'elapsed_los' must be a timedelta column in ed_snapshots, but found type: "
                f"{actual_type}"
            )

    if inpatient_snapshots is not None:
        if "elapsed_los" not in inpatient_snapshots.columns:
            raise ValueError("Column 'elapsed_los' not found in inpatient_snapshots")
        if not pd.api.types.is_timedelta64_dtype(inpatient_snapshots["elapsed_los"]):
            actual_type = inpatient_snapshots["elapsed_los"].dtype
            raise ValueError(
                "Column 'elapsed_los' must be a timedelta column in inpatient_snapshots, but found type: "
                f"{actual_type}"
            )

    if ed_classifier is not None and (
        not hasattr(ed_classifier, "pipeline") or ed_classifier.pipeline is None
    ):
        raise ValueError("ED classifier model has not been fit")
    if inpatient_classifier is not None and (
        not hasattr(inpatient_classifier, "pipeline")
        or inpatient_classifier.pipeline is None
    ):
        raise ValueError("Inpatient classifier model has not been fit")

    if spec_model is not None:
        if isinstance(
            spec_model, (SequenceToOutcomePredictor, ValueToOutcomePredictor)
        ):
            if not hasattr(spec_model, "weights") or spec_model.weights is None:
                raise ValueError("Specialty model has not been fit")
        else:
            if not hasattr(spec_model, "specialty_to_subgroups"):
                raise ValueError("Specialty model has not been fit")
    if yet_to_arrive_model is not None and (
        not hasattr(yet_to_arrive_model, "weights") or not yet_to_arrive_model.weights
    ):
        raise ValueError("Yet-to-arrive (ED YTA) model has not been fit")

    if (
        ed_classifier is not None
        and ed_classifier.training_results.prediction_time != prediction_time
    ):
        raise ValueError(
            "Requested prediction time {pt} does not match the prediction time of the "
            "trained ED classifier {ct}".format(
                pt=prediction_time, ct=ed_classifier.training_results.prediction_time
            )
        )
    if (
        inpatient_classifier is not None
        and inpatient_classifier.training_results.prediction_time != prediction_time
    ):
        raise ValueError(
            "Requested prediction time {pt} does not match the prediction time of the "
            "trained inpatient classifier {ct}".format(
                pt=prediction_time,
                ct=inpatient_classifier.training_results.prediction_time,
            )
        )
    for name, model in (("non-ED", non_ed_yta_model), ("elective", elective_yta_model)):
        if model is None:
            continue
        if not hasattr(model, "weights") or not model.weights:
            raise ValueError(f"{name} DirectAdmissionPredictor has not been fit")

    if transfer_model is not None and (
        not hasattr(transfer_model, "is_fitted_") or not transfer_model.is_fitted_
    ):
        raise ValueError("Transfer model has not been fit")

    if yet_to_arrive_model is not None and hasattr(yet_to_arrive_model, "filters"):
        warn_specialty_mismatch(
            set(specialties),
            set(yet_to_arrive_model.filters.keys()),
            "yet-to-arrive model",
        )

    special_params = spec_model.special_params if spec_model is not None else None

    if special_params:
        special_category_dict = special_params["special_category_dict"]
    else:
        special_category_dict = None

    if special_category_dict is not None and not set(specialties) == set(
        special_category_dict.keys()
    ):
        has_mapping = (
            spec_model is not None
            and hasattr(spec_model, "specialty_to_subgroups")
            and isinstance(getattr(spec_model, "specialty_to_subgroups"), dict)
            and len(getattr(spec_model, "specialty_to_subgroups")) > 0
        )
        if not has_mapping:
            warn_specialty_mismatch(
                set(specialties),
                set(special_category_dict.keys()),
                "special_category_dict",
            )


def _prepare_base_probabilities(
    service_models: ServiceModels,
    ed_snapshots: Optional[pd.DataFrame],
    inpatient_snapshots: Optional[pd.DataFrame],
    prediction_window,
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
    use_admission_in_window_prob: bool,
) -> Dict[str, Any]:
    """Prepare base probability calculations for all patients.

    Returns
    -------
    dict
        Dictionary containing prepared probabilities and other computed values
    """
    ed_classifier = service_models.ed_classifier
    inpatient_classifier = service_models.inpatient_classifier
    spec_model = service_models.spec_model
    yet_to_arrive_model = service_models.ed_yta_model

    # Use calibrated pipeline if available for ED classifier
    if ed_classifier is not None:
        ed_pipeline = (
            ed_classifier.calibrated_pipeline
            if hasattr(ed_classifier, "calibrated_pipeline")
            and ed_classifier.calibrated_pipeline is not None
            else ed_classifier.pipeline
        )
    else:
        ed_pipeline = None

    # Use calibrated pipeline if available for inpatient classifier
    if inpatient_classifier is not None:
        inpatient_pipeline = (
            inpatient_classifier.calibrated_pipeline
            if hasattr(inpatient_classifier, "calibrated_pipeline")
            and inpatient_classifier.calibrated_pipeline is not None
            else inpatient_classifier.pipeline
        )
    else:
        inpatient_pipeline = None

    # Legacy only: add missing columns before the ColumnTransformer step.
    if (
        ed_pipeline is not None
        and ed_snapshots is not None
        and "feature_columns" not in ed_pipeline.named_steps
    ):
        ed_snapshots = add_missing_columns(ed_pipeline, ed_snapshots.copy())
    if (
        inpatient_pipeline is not None
        and inpatient_snapshots is not None
        and "feature_columns" not in inpatient_pipeline.named_steps
    ):
        inpatient_snapshots = add_missing_columns(
            inpatient_pipeline, inpatient_snapshots.copy()
        )

    if ed_snapshots is not None and ed_pipeline is not None:
        ed_snapshots_temp = dataframe_for_classifier_predict_proba(
            ed_pipeline, ed_snapshots
        )
    elif ed_snapshots is not None:
        ed_snapshots_temp = ed_snapshots.copy()
    else:
        ed_snapshots_temp = None

    # Admission probability for current ED patients (per row)
    if ed_pipeline is not None and ed_snapshots_temp is not None:
        prob_admission_after_ed = model_input_to_pred_proba(
            ed_snapshots_temp, ed_pipeline
        )
    elif ed_snapshots is not None:
        prob_admission_after_ed = pd.Series(0.0, index=ed_snapshots.index)
    else:
        prob_admission_after_ed = pd.Series(dtype=float)

    if inpatient_snapshots is not None and inpatient_pipeline is not None:
        inpatient_snapshots_temp = dataframe_for_classifier_predict_proba(
            inpatient_pipeline, inpatient_snapshots
        )
        elective_snapshots = inpatient_snapshots_temp[
            inpatient_snapshots_temp["admission_type"] == "elective"
        ]
        emergency_snapshots = inpatient_snapshots_temp[
            inpatient_snapshots_temp["admission_type"] == "emergency"
        ]
    elif inpatient_snapshots is not None:
        inpatient_snapshots_temp = inpatient_snapshots.copy()
        elective_snapshots = inpatient_snapshots_temp[
            inpatient_snapshots_temp["admission_type"] == "elective"
        ]
        emergency_snapshots = inpatient_snapshots_temp[
            inpatient_snapshots_temp["admission_type"] == "emergency"
        ]
    else:
        inpatient_snapshots_temp = None
        elective_snapshots = pd.DataFrame()
        emergency_snapshots = pd.DataFrame()

    # Departure probability for current inpatients (per row)
    if inpatient_pipeline is not None:
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
    else:
        prob_departure_after_elective = (
            pd.Series(0.0, index=elective_snapshots.index)
            if not elective_snapshots.empty
            else pd.Series(dtype=float)
        )
        prob_departure_after_emergency = (
            pd.Series(0.0, index=emergency_snapshots.index)
            if not emergency_snapshots.empty
            else pd.Series(dtype=float)
        )

    # Specialty probabilities per row for ED patients
    if (
        spec_model is not None
        and hasattr(spec_model, "predict_dataframe")
        and ed_snapshots is not None
    ):
        ed_snapshots.loc[:, "specialty_prob"] = spec_model.predict_dataframe(
            ed_snapshots
        )
    elif ed_snapshots is not None:
        special_params = spec_model.special_params if spec_model is not None else None
        if special_params:
            special_category_func = special_params["special_category_func"]
            special_category_dict = special_params["special_category_dict"]
        else:
            special_category_func = special_category_dict = None

        if spec_model is not None:
            ed_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
                [],  # specialties will be determined from the model
                spec_model,
                ed_snapshots,
                special_category_func=special_category_func,
                special_category_dict=special_category_dict,
            )
        else:
            raise ValueError(
                "Cannot compute specialty probabilities: spec_model is None "
                "but ED snapshots are present. This should have been caught "
                "by validation — ensure _validate_models_and_data is called first."
            )

    # Probability of being admitted within window (per row) for ED patients
    if (
        use_admission_in_window_prob
        and yet_to_arrive_model is not None
        and ed_snapshots is not None
    ):
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
            if x1 is None or y1 is None or x2 is None or y2 is None:
                raise ValueError(
                    "x1, y1, x2, y2 are required for parametric admission-in-window probabilities"
                )
            prob_admission_in_window = ed_snapshots.apply(
                lambda row: calculate_probability(
                    row["elapsed_los"],
                    prediction_window,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ),
                axis=1,
            )
    elif ed_snapshots is not None:
        prob_admission_in_window = pd.Series(1.0, index=ed_snapshots.index)
    else:
        prob_admission_in_window = pd.Series(dtype=float)

    # Prepare subgroup masks if using MultiSubgroupPredictor
    special_params = spec_model.special_params if spec_model is not None else None
    if special_params:
        special_func_map = special_params["special_func_map"]
    else:
        special_func_map = None

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}

    # Resolve specialty_to_subgroups directly from the model attribute
    specialty_to_subgroups: Dict[str, List[str]] = (
        getattr(spec_model, "specialty_to_subgroups", {})
        if spec_model is not None
        else {}
    )

    # Precompute subgroup/function masks once for ED patients
    if ed_snapshots is not None:
        ed_masks_by_func: Dict[str, pd.Series] = {
            name: ed_snapshots.apply(func, axis=1)
            for name, func in special_func_map.items()
        }
        if "default" not in ed_masks_by_func:
            ed_masks_by_func["default"] = pd.Series(True, index=ed_snapshots.index)
    else:
        ed_masks_by_func = {}

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
    service_col: str = "current_subspecialty",
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
    service_col : str, default='current_subspecialty'
        Column naming the patient's current service unit.

    Returns
    -------
    dict
        Dictionary containing processed inpatient data for the specialty and admission type
    """
    # Process inpatients for the specific admission type (no weighting required)
    admission_type_mask = (inpatient_snapshots[service_col] == spec) & (
        inpatient_snapshots["admission_type"] == admission_type
    )
    admission_type_indices = inpatient_snapshots[admission_type_mask].index

    if len(admission_type_indices) > 0:
        filtered_prob_departure = prob_departure_series.loc[admission_type_indices]
        agg_predicted_departures = pred_proba_to_agg_predicted(filtered_prob_departure)
    else:
        # No inpatients of this type in this specialty, create zero PMF
        # For 0 patients, PMF should be [1.0] (P(0 departures) = 1.0)
        agg_predicted_departures = {"agg_proba": np.array([1.0])}

    return {
        f"agg_predicted_{admission_type}_departures": agg_predicted_departures,
    }


def _create_flow_inputs(
    spec: str,
    agg_predicted_in_ed: Dict[str, np.ndarray],
    agg_predicted_elective_departures: Dict[str, np.ndarray],
    agg_predicted_emergency_departures: Dict[str, np.ndarray],
    yet_to_arrive_model: Optional[
        Union[ParametricIncomingAdmissionPredictor, EmpiricalIncomingAdmissionPredictor]
    ],
    non_ed_yta_model: Optional[DirectAdmissionPredictor],
    elective_yta_model: Optional[DirectAdmissionPredictor],
    prediction_time: Tuple[int, int],
    prediction_window,
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
    prediction_date: Optional[date] = None,
) -> Dict[str, Dict[str, FlowInputs]]:
    """Create FlowInputs objects for inflows and outflows.

    Returns
    -------
    dict
        Dictionary with 'inflows' and 'outflows' keys containing FlowInputs objects
    """

    def _safe_predict_mean(model, **kwargs) -> float:
        # Return 0.0 when the model is None or doesn't recognise the filter key
        if model is None:
            return 0.0
        if hasattr(model, "weights") and spec not in model.weights:
            return 0.0
        return float(
            model.predict_mean(
                prediction_time=prediction_time,
                prediction_window=prediction_window,
                filter_key=spec,
                prediction_date=prediction_date,
                **kwargs,
            )
        )

    # Parametric YTA models need x1/y1/x2/y2; empirical and direct do not.
    if isinstance(yet_to_arrive_model, ParametricIncomingAdmissionPredictor):
        if x1 is None or y1 is None or x2 is None or y2 is None:
            raise ValueError(
                "x1, y1, x2, y2 are required when ed_yta_model is parametric"
            )
        ed_yta_mean = _safe_predict_mean(
            yet_to_arrive_model,
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        )
    else:
        ed_yta_mean = _safe_predict_mean(yet_to_arrive_model)

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
            distribution=ed_yta_mean,
            display_name="ED yet-to-arrive admissions",
            aspirational=isinstance(
                yet_to_arrive_model, ParametricIncomingAdmissionPredictor
            ),
        ),
        "non_ed_yta": FlowInputs(
            flow_id="non_ed_yta",
            flow_type="poisson",
            distribution=_safe_predict_mean(non_ed_yta_model),
            display_name="Non-ED emergency admissions",
        ),
        "elective_yta": FlowInputs(
            flow_id="elective_yta",
            flow_type="poisson",
            distribution=_safe_predict_mean(elective_yta_model),
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
    service_models: ServiceModels,
    prediction_time: Tuple[int, int],
    ed_snapshots: Optional[pd.DataFrame],
    inpatient_snapshots: Optional[pd.DataFrame],
    specialties: List[str],
    prediction_window,
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
    base_probs: Dict[str, Any],
    prediction_date: Optional[date] = None,
    inpatient_service_col: str = "current_subspecialty",
) -> Dict[str, Dict[str, Any]]:
    """Build flows for all specialties using processing logic.

    Returns
    -------
    dict
        Dictionary mapping specialty to temporary flow data
    """
    yet_to_arrive_model = service_models.ed_yta_model
    non_ed_yta_model = service_models.non_ed_yta_model
    elective_yta_model = service_models.elective_yta_model

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
    temp_service_data: Dict[str, Dict[str, Any]] = {}

    for spec in specialties:
        if ed_snapshots is not None:
            ed_data = _process_ed_patients_for_specialty(
                spec,
                ed_snapshots,
                specialty_to_subgroups,
                ed_masks_by_func,
                prob_admission_after_ed,
                prob_admission_in_window,
            )
        else:
            ed_data = {"agg_predicted_in_ed": {"agg_proba": np.array([1.0])}}

        # Process inpatients
        if inpatient_snapshots is not None:
            elective_data = _process_inpatients_for_specialty_by_admission_type(
                spec,
                inpatient_snapshots,
                prob_departure_after_elective,
                "elective",
                service_col=inpatient_service_col,
            )
            emergency_data = _process_inpatients_for_specialty_by_admission_type(
                spec,
                inpatient_snapshots,
                prob_departure_after_emergency,
                "emergency",
                service_col=inpatient_service_col,
            )
        else:
            elective_data = {
                "agg_predicted_elective_departures": {"agg_proba": np.array([1.0])}
            }
            emergency_data = {
                "agg_predicted_emergency_departures": {"agg_proba": np.array([1.0])}
            }

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
            prediction_window,
            x1,
            y1,
            x2,
            y2,
            prediction_date=prediction_date,
        )

        # Store in temporary dictionary structure
        temp_service_data[spec] = flow_data

    return temp_service_data


def _finalise_service_data(
    temp_service_data: Dict[str, Dict[str, Any]],
    transfer_model: Optional[TransferProbabilityEstimator],
    specialties: List[str],
    prediction_window,
    inpatient_snapshots: Optional[pd.DataFrame] = None,
    prob_departure_after_elective: Optional[pd.DataFrame] = None,
    prob_departure_after_emergency: Optional[pd.DataFrame] = None,
    inpatient_service_col: str = "current_subspecialty",
) -> Dict[str, ServicePredictionInputs]:
    """Add transfers and create final ServicePredictionInputs objects.

    Transfer arrivals use per-patient subgroup routing computed directly from
    *inpatient_snapshots* and the per-patient departure probabilities (the same
    ``p_depart_i`` used for inpatient departure outflows), rather than scalar
    thinning of pre-aggregated departure PMFs. See
    [compute_transfer_arrivals][patientflow.predict.transfers.compute_transfer_arrivals].

    Returns
    -------
    dict
        Dictionary mapping service_id to ServicePredictionInputs
    """
    # Compute transfer arrivals using per-patient subgroup routing.
    if transfer_model is not None:
        transfer_arrivals = compute_transfer_arrivals(
            inpatient_snapshots,
            transfer_model,
            specialties,
            prob_departure_after_elective=prob_departure_after_elective,
            prob_departure_after_emergency=prob_departure_after_emergency,
            source_col=inpatient_service_col,
        )
    else:
        # If no transfer model, assume 0 transfers
        transfer_arrivals = {
            "elective": {spec: np.array([1.0]) for spec in specialties},
            "emergency": {spec: np.array([1.0]) for spec in specialties},
        }

    # Second pass: Add transfer arrivals to inflows and create final immutable dataclass objects
    service_data: Dict[str, ServicePredictionInputs] = {}
    for spec in specialties:
        # Add elective and emergency transfers to the inflows dictionary
        temp_service_data[spec]["inflows"]["elective_transfers"] = FlowInputs(
            flow_id="elective_transfers",
            flow_type="pmf",
            distribution=transfer_arrivals["elective"][spec],
            display_name="Elective transfers from other services",
        )

        temp_service_data[spec]["inflows"]["emergency_transfers"] = FlowInputs(
            flow_id="emergency_transfers",
            flow_type="pmf",
            distribution=transfer_arrivals["emergency"][spec],
            display_name="Emergency transfers from other services",
        )

        # Create final immutable dataclass with complete inflows and outflows
        service_data[spec] = ServicePredictionInputs(
            service_id=spec,
            prediction_window=prediction_window,
            inflows=temp_service_data[spec]["inflows"],
            outflows=temp_service_data[spec]["outflows"],
        )

    return service_data


def build_service_data(
    models: Union[
        ServiceModels,
        Tuple[
            Optional[TrainedClassifier],
            Optional[TrainedClassifier],
            Optional[
                Union[
                    SequenceToOutcomePredictor,
                    ValueToOutcomePredictor,
                    MultiSubgroupPredictor,
                ]
            ],
            Optional[
                Union[
                    ParametricIncomingAdmissionPredictor,
                    EmpiricalIncomingAdmissionPredictor,
                ]
            ],
            Optional[DirectAdmissionPredictor],
            Optional[DirectAdmissionPredictor],
            Optional[TransferProbabilityEstimator],
        ],
    ],
    prediction_time: Tuple[int, int],
    ed_snapshots: Optional[pd.DataFrame],
    inpatient_snapshots: Optional[pd.DataFrame],
    specialties: List[str],
    prediction_window,
    flow_selection: Optional[FlowSelection] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
    cdf_cut_points: Optional[List[float]] = None,
    use_admission_in_window_prob: bool = True,
    prediction_date: Optional[date] = None,
    inpatient_service_col: str = "current_subspecialty",
) -> Dict[str, ServicePredictionInputs]:
    """Build per-service inputs for downstream roll-up.

    This function processes current patient snapshots through trained models and
    computes, for each service, the probability distribution of admissions
    from current ED patients, departures from current inpatients, the expected
    means of yet-to-arrive admissions, and transfer arrival distributions.

    Parameters
    ----------
    models : ServiceModels or tuple of length 7
        Either a `ServiceModels` instance (`patientflow.model_artifacts`) or
        the legacy seven-tuple of optional trained objects (`None` allowed in
        unused slots):

        - `ed_classifier`: `TrainedClassifier` for ED admission probability
        - `inpatient_classifier`: `TrainedClassifier` for inpatient departures
        - `spec_model`: `SequenceToOutcomePredictor`,
          `ValueToOutcomePredictor`, or `MultiSubgroupPredictor`
        - `ed_yta_model`: `ParametricIncomingAdmissionPredictor` or
          `EmpiricalIncomingAdmissionPredictor`
        - `non_ed_yta_model`, `elective_yta_model`: `DirectAdmissionPredictor`
        - `transfer_model`: `TransferProbabilityEstimator`
    prediction_time : tuple of (int, int)
        Hour and minute for inference time
    ed_snapshots : pandas.DataFrame or None
        DataFrame of current ED patients. When provided, must include an
        `elapsed_los` column as `timedelta`. May be `None` when
        *flow_selection* does not include current ED patients.
    inpatient_snapshots : pandas.DataFrame or None
        DataFrame of current inpatients. When provided, must include
        `elapsed_los` as `timedelta`. May be `None` when *flow_selection*
        does not require inpatient-derived flows.
    specialties : list of str
        List of services/specialties to prepare inputs for
    prediction_window : datetime.timedelta
        Time window over which to predict admissions
    flow_selection : FlowSelection, optional
        Which flows are included; drives validation of models and snapshots.
        When omitted, defaults to `FlowSelection.default()` for 1.6.2-style callers.
    x1, y1, x2, y2 : float, optional
        Parameters for the parametric admission-in-window curve. Required when
        the selected flows use a parametric ED YTA model or parametric
        admission-in-window weighting for current ED patients.
    cdf_cut_points : list of float, optional
        Ignored in this function; present for API compatibility. If provided,
        has no effect on output.
    use_admission_in_window_prob : bool, default=True
        Whether to weight current ED admissions by their probability of being
        admitted within the prediction window.
    prediction_date : datetime.date, optional
        Calendar date at the inference `prediction_time`. When provided, YTA
        Poisson means use per-weekday arrival profiles for models fitted with
        weekday stratification (the default for incoming admission predictors).
        When omitted (default), behaviour matches previous releases: pooled
        profiles are used and legacy callers stay warning-free.
    inpatient_service_col : str, default='current_subspecialty'
        Column on *inpatient_snapshots* naming the patient's current service
        unit. Used to filter departures by service and as the source column
        for transfer routing.

    Returns
    -------
    dict of str to ServicePredictionInputs
        Dictionary mapping service_id to ServicePredictionInputs dataclass.
        See ServicePredictionInputs for field details.

    Raises
    ------
    TypeError
        If *models* is neither `ServiceModels` nor a seven-tuple, or if a
        supplied model has an unexpected concrete type.
    ValueError
        If *flow_selection* requires a model or snapshot that was not supplied,
        if required columns are missing, if models are not fitted, or if
        parameters disagree with training metadata.

    See Also
    --------
    patientflow.predict.flow_selection_checks
    patientflow.predict.demand.DemandPredictor.predict_service

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
    if flow_selection is None:
        flow_selection = FlowSelection.default()

    service_models = _normalize_to_service_models(
        models, prediction_time, prediction_window
    )
    _validate_models_and_data(
        service_models,
        flow_selection,
        ed_snapshots,
        inpatient_snapshots,
        specialties,
        use_admission_in_window_prob=use_admission_in_window_prob,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    base_probs = _prepare_base_probabilities(
        service_models,
        ed_snapshots,
        inpatient_snapshots,
        prediction_window,
        x1,
        y1,
        x2,
        y2,
        use_admission_in_window_prob,
    )

    temp_service_data = _build_legacy_flows(
        service_models,
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
        prediction_date=prediction_date,
        inpatient_service_col=inpatient_service_col,
    )

    return _finalise_service_data(
        temp_service_data,
        service_models.transfer_model,
        specialties,
        prediction_window,
        inpatient_snapshots=base_probs["inpatient_snapshots"],
        prob_departure_after_elective=base_probs["prob_departure_after_elective"],
        prob_departure_after_emergency=base_probs["prob_departure_after_emergency"],
        inpatient_service_col=inpatient_service_col,
    )
