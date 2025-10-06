"""Demand preparation utilities for later consolidation.

This module prepares per-subspecialty inputs that can be consumed by any
downstream roll-up or forecasting workflow. It converts trained model
outputs and current patient snapshots into:

- A probability mass function (PMF) for admissions from current ED patients
- Poisson means for yet-to-arrive admissions (ED, non-ED emergency, elective)

The outputs are independent per subspecialty and do not presuppose any
particular consolidation hierarchy. They can be used directly for single-
specialty analyses or fed into any combination scheme (including hierarchical
schemes) implemented elsewhere.

Functions
---------
build_subspecialty_data
    Prepare subspecialty-level prediction inputs from trained models and snapshots

Notes
-----
This module integrates with the broader patientflow ecosystem by:
1. Using trained classifiers and specialty models from the train module
2. Processing patient snapshots from the prepare module  
3. Computing admission probabilities using the calculate module

"""

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


def build_subspecialty_data(
    models: Tuple[
        TrainedClassifier,
        Union[SequenceToOutcomePredictor, ValueToOutcomePredictor, MultiSubgroupPredictor],
        Union[
            ParametricIncomingAdmissionPredictor,
            EmpiricalIncomingAdmissionPredictor,
        ],
        DirectAdmissionPredictor,
        DirectAdmissionPredictor,
    ],
    prediction_time: Tuple[int, int],
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: Optional[List[float]] = None,
    use_admission_in_window_prob: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Build per-subspecialty inputs for downstream roll-up.

    This function processes current patient snapshots through trained models and
    computes, for each subspecialty, the probability distribution of admissions
    from current ED patients and the expected means of yet-to-arrive admissions.
    
    The function combines three sources of demand:
    1. Current ED patients (converted to probability mass function)
    2. Yet-to-arrive ED patients (converted to Poisson parameters)
    3. Yet-to-arrive non-ED emergency patients (converted to Poisson parameters)
    4. Yet-to-arrive elective patients (converted to Poisson parameters)
    
    Parameters
    ----------
    models : tuple
        Tuple of five trained models:
        - classifier: TrainedClassifier for admission probability prediction
        - spec_model: SequenceToOutcomePredictor | ValueToOutcomePredictor | MultiSubgroupPredictor
          for specialty assignment probabilities
        - ed_yta_model: ParametricIncomingAdmissionPredictor | EmpiricalIncomingAdmissionPredictor
          for ED yet-to-arrive predictions
        - non_ed_yta_model: DirectAdmissionPredictor for non-ED emergency predictions
        - elective_yta_model: DirectAdmissionPredictor for elective predictions
    prediction_time : tuple[int, int]
        Hour and minute for inference time
    prediction_snapshots : pandas.DataFrame
        DataFrame of current ED patients. Must include 'elapsed_los' column as timedelta.
        Each row represents a patient currently in the ED.
    specialties : list[str]
        List of subspecialties to prepare inputs for
    prediction_window : datetime.timedelta
        Time window over which to predict admissions
    x1, y1, x2, y2 : float
        Parameters for the parametric admission-in-window curve. Used when
        ed_yta_model is parametric and for computing in-ED window probabilities.
    cdf_cut_points : list[float], optional
        Ignored in this function; present for API compatibility. If provided,
        has no effect on output.
    use_admission_in_window_prob : bool, default=True
        Whether to weight current ED admissions by their probability of being
        admitted within the prediction window.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping subspecialty_id to prediction parameters:
        - 'prob_admission_pats_in_ed': numpy.ndarray
          Probability mass function for current ED admissions within window
        - 'lambda_ed_yta': float
          Poisson parameter for ED yet-to-arrive admissions
        - 'lambda_non_ed_yta': float
          Poisson parameter for non-ED emergency admissions
        - 'lambda_elective_yta': float
          Poisson parameter for elective admissions

    Raises
    ------
    TypeError
        If any model is not of the expected type
    ValueError
        If required columns are missing, models are not fitted, or parameters
        don't match between models and requested parameters

    """
    classifier, spec_model, yet_to_arrive_model, non_ed_yta_model, elective_yta_model = models

    # Validate model types
    if not isinstance(classifier, TrainedClassifier):
        raise TypeError("First model must be of type TrainedClassifier")
    if not isinstance(
        spec_model,
        (SequenceToOutcomePredictor, ValueToOutcomePredictor, MultiSubgroupPredictor),
    ):
        raise TypeError(
            "Second model must be of type SequenceToOutcomePredictor or ValueToOutcomePredictor or MultiSubgroupPredictor"
        )
    yet_to_arrive_class_name = type(yet_to_arrive_model).__name__
    expected_types = (
        "ParametricIncomingAdmissionPredictor",
        "EmpiricalIncomingAdmissionPredictor",
    )
    if yet_to_arrive_class_name not in expected_types:
        actual_module = type(yet_to_arrive_model).__module__
        raise TypeError(
            "Third model must be of type ParametricIncomingAdmissionPredictor or "
            "EmpiricalIncomingAdmissionPredictor, "
            f"but got {actual_module}.{yet_to_arrive_class_name}. "
            "If you're using Jupyter, try restarting the kernel."
        )

    # Validate that non-ED and elective models are DirectAdmissionPredictor
    if not isinstance(non_ed_yta_model, DirectAdmissionPredictor):
        raise TypeError("Fourth model must be of type DirectAdmissionPredictor (non-ED emergency)")
    if not isinstance(elective_yta_model, DirectAdmissionPredictor):
        raise TypeError("Fifth model must be of type DirectAdmissionPredictor (elective)")

    # Validate elapsed_los column presence and dtype
    if "elapsed_los" not in prediction_snapshots.columns:
        raise ValueError("Column 'elapsed_los' not found in prediction_snapshots")
    if not pd.api.types.is_timedelta64_dtype(prediction_snapshots["elapsed_los"]):
        actual_type = prediction_snapshots["elapsed_los"].dtype
        raise ValueError(
            "Column 'elapsed_los' must be a timedelta column, but found type: "
            f"{actual_type}"
        )

    # Check that all models have been fit
    if not hasattr(classifier, "pipeline") or classifier.pipeline is None:
        raise ValueError("Classifier model has not been fit")
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
    if not classifier.training_results.prediction_time == prediction_time:
        raise ValueError(
            "Requested prediction time {pt} does not match the prediction time of the "
            "trained classifier {ct}".format(
                pt=prediction_time, ct=classifier.training_results.prediction_time
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
            raise ValueError(f"{name} DirectAdmissionPredictor has not been fit (missing prediction_window)")
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
    # Use calibrated pipeline if available
    pipeline = (
        classifier.calibrated_pipeline
        if hasattr(classifier, "calibrated_pipeline")
        and classifier.calibrated_pipeline is not None
        else classifier.pipeline
    )

    # Ensure model expects columns exist
    prediction_snapshots = add_missing_columns(pipeline, prediction_snapshots.copy())

    # Convert elapsed_los to seconds for the classifier pipeline
    prediction_snapshots_temp = prediction_snapshots.copy()
    prediction_snapshots_temp["elapsed_los"] = prediction_snapshots_temp[
        "elapsed_los"
    ].dt.total_seconds()

    # Admission probability for current ED patients (per row)
    prob_admission_after_ed = model_input_to_pred_proba(
        prediction_snapshots_temp, pipeline
    )

    # Specialty probabilities per row
    if hasattr(spec_model, "predict_dataframe"):
        prediction_snapshots.loc[:, "specialty_prob"] = spec_model.predict_dataframe(
            prediction_snapshots
        )
    else:
        prediction_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
            specialties,
            spec_model,
            prediction_snapshots,
            special_category_func=special_category_func,
            special_category_dict=special_category_dict,
        )

    # Probability of being admitted within window (per row)
    if use_admission_in_window_prob:
        if isinstance(yet_to_arrive_model, EmpiricalIncomingAdmissionPredictor):
            prob_admission_in_window = prediction_snapshots.apply(
                lambda row: calculate_admission_probability_from_survival_curve(
                    row["elapsed_los"], prediction_window, yet_to_arrive_model.survival_df
                ),
                axis=1,
            )
        else:
            prob_admission_in_window = prediction_snapshots.apply(
                lambda row: calculate_probability(
                    row["elapsed_los"], prediction_window, x1, y1, x2, y2
                ),
                axis=1,
            )
    else:
        prob_admission_in_window = pd.Series(1.0, index=prediction_snapshots.index)

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}
        
    # Resolve specialty_to_subgroups directly from the model attribute
    specialty_to_subgroups: Dict[str, List[str]] = getattr(
        spec_model, "specialty_to_subgroups", {}
    )

    # Precompute subgroup/function masks once
    masks_by_func: Dict[str, pd.Series] = {
        name: prediction_snapshots.apply(func, axis=1)
        for name, func in special_func_map.items()
    }

    subspecialty_data: Dict[str, Dict[str, Any]] = {spec: {} for spec in specialties}

    for spec in specialties:
        if specialty_to_subgroups and spec in specialty_to_subgroups:
            func_keys = specialty_to_subgroups[spec]
        else:
            func_keys = ["default"]

        combined_mask = pd.Series(False, index=prediction_snapshots.index)
        for key in func_keys:
            combined_mask = combined_mask | masks_by_func.get(
                key, pd.Series(False, index=prediction_snapshots.index)
            )

        non_zero_indices = prediction_snapshots[combined_mask].index
        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]

        filtered_prob_admission_to_specialty = (
            prediction_snapshots["specialty_prob"]
            .loc[non_zero_indices]
            .apply(lambda d: d.get(spec, 0.0) if isinstance(d, dict) else 0.0)
        )
        filtered_prob_admission_in_window = prob_admission_in_window.loc[
            non_zero_indices
        ]
        filtered_weights = (
            filtered_prob_admission_to_specialty * filtered_prob_admission_in_window
        )

        agg_predicted_in_ed = pred_proba_to_agg_predicted(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )

        subspecialty_data[spec]["prob_admission_pats_in_ed"] = np.array(agg_predicted_in_ed["agg_proba"])

        prediction_context = {spec: {"prediction_time": prediction_time}}

        lambda_ed = float(yet_to_arrive_model.predict_mean(prediction_context, x1=x1, y1=y1, x2=x2, y2=y2))
        lambda_non_ed = float(
            non_ed_yta_model.predict_mean(prediction_context) if non_ed_yta_model is not None else 0.0
        )
        lambda_elective = float(
            elective_yta_model.predict_mean(prediction_context)
            if elective_yta_model is not None
            else 0.0
        )

        subspecialty_data[spec]["lambda_ed_yta"] = lambda_ed
        subspecialty_data[spec]["lambda_non_ed_yta"] = lambda_non_ed
        subspecialty_data[spec]["lambda_elective_yta"] = lambda_elective

    return subspecialty_data
