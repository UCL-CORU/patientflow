"""Emergency demand prediction module.

This module provides functionality for predicting emergency department demand,
including specialty-specific predictions for both current patients and yet-to-arrive patients.
It handles probability calculations, model predictions, and threshold-based resource estimation.

The module integrates multiple prediction models:
- Admission prediction classifier
- Specialty sequence predictor
- Yet-to-arrive weighted Poisson predictor

Functions
---------
add_missing_columns : function
    Add missing columns required by the prediction pipeline
find_probability_threshold_index : function
    Find index where cumulative probability exceeds threshold
get_specialty_probs : function
    Calculate specialty probability distributions
create_predictions : function
    Create predictions for emergency demand

"""

from typing import List, Dict, Tuple, Union
from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


from patientflow.calculate.admission_in_prediction_window import (
    calculate_probability,
)

from patientflow.aggregate import (
    model_input_to_pred_proba,
    pred_proba_to_agg_predicted,
)


import warnings

from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.value_to_outcome_predictor import ValueToOutcomePredictor
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
)
from patientflow.model_artifacts import TrainedClassifier

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def add_missing_columns(pipeline, df):
    """Add missing columns required by the prediction pipeline from the training data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The trained pipeline containing the feature transformer
    df : pandas.DataFrame
        Input dataframe that may be missing required columns

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing columns added and filled with appropriate default values

    Notes
    -----
    Adds columns with default values based on column name patterns:
    - lab_orders_, visited_, has_ : False
    - num_, total_ : 0
    - latest_ : pd.NA
    - arrival_method : "None"
    - others : pd.NA
    """
    # check input data for missing columns
    column_transformer = pipeline.named_steps["feature_transformer"]

    # Function to get feature names before one-hot encoding
    def get_feature_names_before_encoding(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers:
            if isinstance(transformer, OneHotEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, OrdinalEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, StandardScaler):
                feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        return feature_names

    feature_names_before_encoding = get_feature_names_before_encoding(
        column_transformer
    )

    added_columns = []
    for missing_col in set(feature_names_before_encoding).difference(set(df.columns)):
        if missing_col.startswith(("lab_orders_", "visited_", "has_")):
            df[missing_col] = False
        elif missing_col.startswith(("num_", "total_")):
            df[missing_col] = 0
        elif missing_col.startswith("latest_"):
            df[missing_col] = pd.NA
        elif missing_col == "arrival_method":
            df[missing_col] = "None"
        else:
            df[missing_col] = pd.NA
        added_columns.append(missing_col)

    if added_columns:
        print(
            f"Warning: The following columns were used in training, but not found in the real-time data. These have been added to the dataframe: {', '.join(added_columns)}"
        )

    return df


def find_probability_threshold_index(sequence: List[float], threshold: float) -> int:
    """Find index where cumulative probability exceeds threshold.

    Parameters
    ----------
    sequence : List[float]
        The probability mass function (PMF) of resource needs
    threshold : float
        The probability threshold (e.g., 0.9 for 90%)

    Returns
    -------
    int
        The index where the cumulative probability exceeds 1 - threshold,
        indicating the number of resources needed with the specified probability

    Examples
    --------
    >>> pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    >>> find_probability_threshold_index(pmf, 0.9)
    5
    # This means there is a 90% probability of needing at least 5 beds
    """
    cumulative_sum = 0.0
    for i, value in enumerate(sequence):
        cumulative_sum += value
        if cumulative_sum >= 1 - threshold:
            return i
    return len(sequence) - 1  # Return the last index if the threshold isn't reached


def get_specialty_probs(
    specialties,
    specialty_model,
    snapshots_df,
    special_category_func=None,
    special_category_dict=None,
):
    """Calculate specialty probability distributions for patient visits.

    Parameters
    ----------
    specialties : str
        List of specialty names for which predictions are required
    specialty_model : object
        Trained model for making specialty predictions
    snapshots_df : pandas.DataFrame
        DataFrame containing the data on which predictions are to be made. Must include
        the input_var column if no special_category_func is applied
    special_category_func : callable, optional
        A function that takes a DataFrame row (Series) as input and returns True if the row
        belongs to a special category that requires a fixed probability distribution
    special_category_dict : dict, optional
        A dictionary containing the fixed probability distribution for special category cases.
        Required if special_category_func is provided

    Returns
    -------
    pandas.Series
        A Series containing dictionaries as values. Each dictionary represents the probability
        distribution of specialties for each patient visit

    Raises
    ------
    ValueError
        If special_category_func is provided but special_category_dict is None

    """

    # Convert input_var to tuple if not already a tuple
    if len(snapshots_df[specialty_model.input_var]) > 0 and not isinstance(
        snapshots_df[specialty_model.input_var].iloc[0], tuple
    ):
        snapshots_df.loc[:, specialty_model.input_var] = snapshots_df[
            specialty_model.input_var
        ].apply(lambda x: tuple(x) if x else ())

    if special_category_func and not special_category_dict:
        raise ValueError(
            "special_category_dict must be provided if special_category_func is specified."
        )

    # Function to determine the specialty probabilities
    def determine_specialty(row):
        if special_category_func and special_category_func(row):
            return special_category_dict
        else:
            return specialty_model.predict(row[specialty_model.input_var])

    # Apply the determine_specialty function to each row
    specialty_prob_series = snapshots_df.apply(determine_specialty, axis=1)

    # Find all unique keys used in any dictionary within the series
    all_keys = set().union(
        *(d.keys() for d in specialty_prob_series if isinstance(d, dict))
    )

    # Combine all_keys with the specialties requested
    all_keys = set(all_keys).union(set(specialties))

    # Ensure each dictionary contains all keys found, with default values of 0 for missing keys
    specialty_prob_series = specialty_prob_series.apply(
        lambda d: (
            {key: d.get(key, 0) for key in all_keys} if isinstance(d, dict) else d
        )
    )

    return specialty_prob_series


def create_predictions(
    models: Tuple[
        TrainedClassifier,
        Union[SequenceToOutcomePredictor, ValueToOutcomePredictor],
        ParametricIncomingAdmissionPredictor,
    ],
    prediction_time: Tuple,
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window: timedelta,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: List[float],
    use_admission_in_window_prob: bool = True,
) -> Dict[str, Dict[str, List[int]]]:
    """Create predictions for emergency demand for a single prediction moment.

    Parameters
    ----------
    models : Tuple[TrainedClassifier, Union[SequenceToOutcomePredictor, ValueToOutcomePredictor], ParametricIncomingAdmissionPredictor]
        Tuple containing:
        - classifier: TrainedClassifier containing admission predictions
        - spec_model: SequenceToOutcomePredictor or ValueToOutcomePredictor for specialty predictions
        - yet_to_arrive_model: ParametricIncomingAdmissionPredictor for yet-to-arrive predictions
    prediction_time : Tuple
        Hour and minute of time for model inference
    prediction_snapshots : pandas.DataFrame
        DataFrame containing prediction snapshots. Must have an 'elapsed_los' column of type timedelta.
    specialties : List[str]
        List of specialty names for predictions (e.g., ['surgical', 'medical'])
    prediction_window : timedelta
        Prediction window as a timedelta object
    x1 : float
        X-coordinate of first point for probability curve
    y1 : float
        Y-coordinate of first point for probability curve
    x2 : float
        X-coordinate of second point for probability curve
    y2 : float
        Y-coordinate of second point for probability curve
    cdf_cut_points : List[float]
        List of cumulative distribution function cut points (e.g., [0.9, 0.7])
    use_admission_in_window_prob : bool, optional
        Whether to use probability calculation for admission within prediction window for patients
        already in the ED. If False, probability is set to 1.0 for all current ED patients.
        This parameter does not affect the yet-to-arrive predictions. By default True

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        Nested dictionary containing predictions for each specialty:
        {
            'specialty_name': {
                'in_ed': [pred1, pred2, ...],
                'yet_to_arrive': [pred1, pred2, ...]
            }
        }

    Raises
    ------
    TypeError
        If any of the models are not of the expected type or if prediction_window is not a timedelta
    ValueError
        If models have not been fit or if prediction parameters don't match training parameters
        If 'elapsed_los' column is missing or not of type timedelta

    Notes
    -----
    The models in the models dictionary must be ModelResults objects
    that contain either a 'pipeline' or 'calibrated_pipeline' attribute. The pipeline
    will be used for making predictions, with calibrated_pipeline taking precedence
    if both exist.
    """
    # Validate model types
    classifier, spec_model, yet_to_arrive_model = models

    if not isinstance(classifier, TrainedClassifier):
        raise TypeError("First model must be of type TrainedClassifier")
    if not isinstance(
        spec_model, (SequenceToOutcomePredictor, ValueToOutcomePredictor)
    ):
        raise TypeError(
            "Second model must be of type SequenceToOutcomePredictor or ValueToOutcomePredictor"
        )
    if not isinstance(yet_to_arrive_model, ParametricIncomingAdmissionPredictor):
        raise TypeError(
            "Third model must be of type ParametricIncomingAdmissionPredictor"
        )
    if "elapsed_los" not in prediction_snapshots.columns:
        raise ValueError("Column 'elapsed_los' not found in prediction_snapshots")
    if not pd.api.types.is_timedelta64_dtype(prediction_snapshots["elapsed_los"]):
        actual_type = prediction_snapshots["elapsed_los"].dtype
        raise ValueError(
            f"Column 'elapsed_los' must be a timedelta column, but found type: {actual_type}"
        )

    # Check that all models have been fit
    if not hasattr(classifier, "pipeline") or classifier.pipeline is None:
        raise ValueError("Classifier model has not been fit")
    if not hasattr(spec_model, "weights") or spec_model.weights is None:
        raise ValueError("Specialty model has not been fit")
    if (
        not hasattr(yet_to_arrive_model, "prediction_window")
        or yet_to_arrive_model.prediction_window is None
    ):
        raise ValueError("Yet-to-arrive model has not been fit")

    # Validate that the correct models have been passed for the requested prediction time and prediction window
    if not classifier.training_results.prediction_time == prediction_time:
        raise ValueError(
            f"Requested prediction time {prediction_time} does not match the prediction time of the trained classifier {classifier.training_results.prediction_time}"
        )

    # Compare prediction windows directly
    if prediction_window != yet_to_arrive_model.prediction_window:
        raise ValueError(
            f"Requested prediction window {prediction_window} does not match the prediction window of the trained yet-to-arrive model {yet_to_arrive_model.prediction_window}"
        )

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
        raise ValueError(
            "Requested specialties do not match the specialty dictionary defined in special_params"
        )

    predictions: Dict[str, Dict[str, List[int]]] = {
        specialty: {"in_ed": [], "yet_to_arrive": []} for specialty in specialties
    }

    # Use calibrated pipeline if available, otherwise use regular pipeline
    if (
        hasattr(classifier, "calibrated_pipeline")
        and classifier.calibrated_pipeline is not None
    ):
        pipeline = classifier.calibrated_pipeline
    else:
        pipeline = classifier.pipeline

    # Add missing columns expected by the model
    prediction_snapshots = add_missing_columns(pipeline, prediction_snapshots)

    # Before we get predictions, we need to create a temp copy with the elapsed_los column in seconds
    prediction_snapshots_temp = prediction_snapshots.copy()
    prediction_snapshots_temp["elapsed_los"] = prediction_snapshots_temp[
        "elapsed_los"
    ].dt.total_seconds()

    # Get predictions of admissions for ED patients
    prob_admission_after_ed = model_input_to_pred_proba(
        prediction_snapshots_temp, pipeline
    )

    # Get predictions of admission to specialty
    prediction_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
        specialties,
        spec_model,
        prediction_snapshots,
        special_category_func=special_category_func,
        special_category_dict=special_category_dict,
    )

    # Get probability of admission within prediction window for current ED patients
    if use_admission_in_window_prob:
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

    for specialty in specialties:
        func = special_func_map.get(specialty, special_func_map["default"])
        non_zero_indices = prediction_snapshots[
            prediction_snapshots.apply(func, axis=1)
        ].index

        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]
        prob_admission_to_specialty = prediction_snapshots["specialty_prob"].apply(
            lambda x: x[specialty]
        )

        filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[
            non_zero_indices
        ]
        filtered_prob_admission_in_window = prob_admission_in_window.loc[
            non_zero_indices
        ]

        filtered_weights = (
            filtered_prob_admission_to_specialty * filtered_prob_admission_in_window
        )

        agg_predicted_in_ed = pred_proba_to_agg_predicted(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )

        prediction_context = {specialty: {"prediction_time": prediction_time}}
        agg_predicted_yta = yet_to_arrive_model.predict(
            prediction_context, x1=x1, y1=y1, x2=x2, y2=y2
        )

        predictions[specialty]["in_ed"] = [
            find_probability_threshold_index(
                agg_predicted_in_ed["agg_proba"].values.cumsum(), cut_point
            )
            for cut_point in cdf_cut_points
        ]
        predictions[specialty]["yet_to_arrive"] = [
            find_probability_threshold_index(
                agg_predicted_yta[specialty]["agg_proba"].values.cumsum(), cut_point
            )
            for cut_point in cdf_cut_points
        ]

    return predictions
