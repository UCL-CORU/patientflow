"""
Emergency demand prediction training module.

This module provides functionality that is specific to the implementation of the
patientflow package at University College London Hospital (ULCH). It trains models
to predict emergency bed demand.

The module trains three model types:
1. Admission prediction models (multiple classifiers, one for each prediction time)
2. Specialty prediction models (sequence-based)
3. Yet-to-arrive prediction models (aspirational)

Functions
---------
test_real_time_predictions : Test real-time prediction functionality
    Selects random test cases and validates that the trained models can
    generate predictions as if it where making a real-time prediction.

train_all_models : Complete training pipeline
    Trains all three model types (admissions, specialty, yet-to-arrive)
    with proper validation and optional model saving.

main : Entry point for training pipeline
    Loads configuration, data, and runs the complete training process.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta
import sys

from patientflow.prepare import (
    create_temporal_splits,
)
from patientflow.load import (
    load_config_file,
    set_file_paths,
    load_data,
    parse_args,
    set_project_root,
)

from patientflow.train.utils import save_model
from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
)
from patientflow.predict.emergency_demand import create_predictions

from patientflow.train.classifiers import train_multiple_classifiers
from patientflow.train.sequence_predictor import train_sequence_predictor
from patientflow.train.incoming_admission_predictor import (
    train_parametric_admission_predictor,
)
from patientflow.model_artifacts import TrainedClassifier


def test_real_time_predictions(
    visits,
    models: Tuple[
        Dict[str, TrainedClassifier],
        SequenceToOutcomePredictor,
        ParametricIncomingAdmissionPredictor,
    ],
    prediction_window,
    specialties,
    cdf_cut_points,
    curve_params,
    random_seed,
):
    """
    Test real-time predictions by selecting a random sample from the visits dataset
    and generating predictions using the trained models.

    Parameters
    ----------
    visits : pd.DataFrame
        DataFrame containing visit data with columns including 'prediction_time',
        'snapshot_date', and other required features for predictions.
    models : Tuple[Dict[str, TrainedClassifier], SequenceToOutcomePredictor, ParametricIncomingAdmissionPredictor]
        Tuple containing:
        - trained_classifiers: TrainedClassifier containing admission predictions
        - spec_model: SequenceToOutcomePredictor for specialty predictions
        - yet_to_arrive_model: ParametricIncomingAdmissionPredictor for yet-to-arrive predictions
    prediction_window : int
        Size of the prediction window in minutes for which to generate forecasts.
    specialties : list[str]
        List of specialty names to generate predictions for (e.g., ['surgical',
        'medical', 'paediatric']).
    cdf_cut_points : list[float]
        List of probability thresholds for cumulative distribution function
        cut points (e.g., [0.9, 0.7]).
    curve_params : tuple[float, float, float, float]
        Parameters (x1, y1, x2, y2) defining the curve used for predictions.
    random_seed : int
        Random seed for reproducible sampling of test cases.

    Returns
    -------
    dict
        Dictionary containing:
        - 'prediction_time': str, The time point for which predictions were made
        - 'prediction_date': str, The date for which predictions were made
        - 'realtime_preds': dict, The generated predictions for the sample

    Raises
    ------
    Exception
        If real-time inference fails, with detailed error message printed before
        system exit.

    Notes
    -----
    The function selects a single random row from the visits DataFrame and
    generates predictions for that specific time point using all provided models.
    The predictions are made using the create_predictions() function with the
    specified parameters.
    """
    # Select random test set row
    random_row = visits.sample(n=1, random_state=random_seed)
    prediction_time = random_row.prediction_time.values[0]
    prediction_date = random_row.snapshot_date.values[0]

    # Get prediction snapshots
    prediction_snapshots = visits[
        (visits.prediction_time == prediction_time)
        & (visits.snapshot_date == prediction_date)
    ]

    trained_classifiers, spec_model, yet_to_arrive_model = models

    # Find the model matching the required prediction time
    classifier = None
    for model_key, trained_model in trained_classifiers.items():
        if trained_model.training_results.prediction_time == prediction_time:
            classifier = trained_model
            break

    if classifier is None:
        raise ValueError(f"No model found for prediction time {prediction_time}")

    try:
        x1, y1, x2, y2 = curve_params
        _ = create_predictions(
            models=(classifier, spec_model, yet_to_arrive_model),
            prediction_time=prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=specialties,
            prediction_window=prediction_window,
            cdf_cut_points=cdf_cut_points,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        print("Real-time inference ran correctly")
    except Exception as e:
        print(f"Real-time inference failed due to this error: {str(e)}")
        sys.exit(1)

    return


def train_all_models(
    visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    yta,
    prediction_times,
    prediction_window: timedelta,
    yta_time_interval: timedelta,
    epsilon,
    grid_params,
    exclude_columns,
    ordinal_mappings,
    random_seed,
    visit_col="visit_number",
    specialties=None,
    cdf_cut_points=None,
    curve_params=None,
    model_file_path=None,
    save_models=True,
    test_realtime=True,
):
    """
    Train and evaluate patient flow models.

    Parameters
    ----------
    visits : pd.DataFrame
        DataFrame containing visit data.
    yta : pd.DataFrame
        DataFrame containing yet-to-arrive data.
    prediction_times : list
        List of times for making predictions.
    prediction_window : int
        Prediction window size in minutes.
    yta_time_interval : int
        Interval size for yet-to-arrive predictions in minutes.
    epsilon : float
        Epsilon parameter for model training.
    grid_params : dict
        Hyperparameter grid for model training.
    exclude_columns : list
        Columns to exclude during training.
    ordinal_mappings : dict
        Ordinal variable mappings for categorical features.
    random_seed : int
        Random seed for reproducibility.
    visit_col : str, optional
        Name of column in dataset that is used to identify a hospital visit (eg visit_number, csn).
    specialties : list, optional
        List of specialties to consider. Required if test_realtime is True.
    cdf_cut_points : list, optional
        CDF cut points for predictions. Required if test_realtime is True.
    curve_params : tuple, optional
        Curve parameters (x1, y1, x2, y2). Required if test_realtime is True.
    model_file_path : Path, optional
        Path to save trained models. Required if save_models is True.
    save_models : bool, optional
        Whether to save the trained models to disk. Defaults to True.
    test_realtime : bool, optional
        Whether to run real-time prediction tests. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If save_models is True but model_file_path is not provided,
        or if test_realtime is True but any of specialties, cdf_cut_points, or curve_params are not provided.

    Notes
    -----
    The function generates model names internally:
    - "admissions": "admissions"
    - "specialty": "ed_specialty"
    - "yet_to_arrive": f"yet_to_arrive_{int(prediction_window.total_seconds()/3600)}_hours"
    """
    # Validate parameters
    if save_models and model_file_path is None:
        raise ValueError("model_file_path must be provided when save_models is True")

    if test_realtime:
        if specialties is None:
            raise ValueError("specialties must be provided when test_realtime is True")
        if cdf_cut_points is None:
            raise ValueError(
                "cdf_cut_points must be provided when test_realtime is True"
            )
        if curve_params is None:
            raise ValueError("curve_params must be provided when test_realtime is True")

    # Set random seed
    np.random.seed(random_seed)

    # Define model names internally
    model_names = {
        "admissions": "admissions",
        "specialty": "ed_specialty",
        "yet_to_arrive": f"yet_to_arrive_{int(prediction_window.total_seconds()/3600)}_hours",
    }

    if "arrival_datetime" in visits.columns:
        col_name = "arrival_datetime"
    else:
        col_name = "snapshot_date"

    train_visits, valid_visits, test_visits = create_temporal_splits(
        visits,
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        col_name=col_name,
    )

    train_yta, _, _ = create_temporal_splits(
        yta[(~yta.specialty.isnull())],
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        col_name="arrival_datetime",
    )

    # Use predicted_times from visits if not explicitly provided
    if prediction_times is None:
        prediction_times = list(visits.prediction_time.unique())

    # Train admission models
    admission_models = train_multiple_classifiers(
        train_visits=train_visits,
        valid_visits=valid_visits,
        test_visits=test_visits,
        grid=grid_params,
        exclude_from_training_data=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        prediction_times=prediction_times,
        model_name=model_names["admissions"],
        visit_col=visit_col,
    )

    # Save admission models if requested

    if save_models:
        save_model(admission_models, model_names["admissions"], model_file_path)

    # Train specialty model
    specialty_model = train_sequence_predictor(
        train_visits=train_visits,
        model_name=model_names["specialty"],
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
        visit_col=visit_col,
    )

    # Save specialty model if requested
    if save_models:
        save_model(specialty_model, model_names["specialty"], model_file_path)

    # Train yet-to-arrive model
    yta_model_name = model_names["yet_to_arrive"]

    num_days = (start_validation_set - start_training_set).days

    yta_model = train_parametric_admission_predictor(
        train_visits=train_visits,
        train_yta=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        num_days=num_days,
    )

    # Save yet-to-arrive model if requested
    if save_models:
        save_model(yta_model, yta_model_name, model_file_path)
        print(f"Models have been saved to {model_file_path}")

    # Test real-time predictions if requested
    if test_realtime:
        visits["elapsed_los"] = visits["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        test_real_time_predictions(
            visits=visits,
            models=(admission_models, specialty_model, yta_model),
            prediction_window=prediction_window,
            specialties=specialties,
            cdf_cut_points=cdf_cut_points,
            curve_params=curve_params,
            random_seed=random_seed,
        )

    return


def main(data_folder_name=None):
    """
    Main entry point for training patient flow models.

    This function orchestrates the complete training pipeline for emergency demand
    prediction models. It loads configuration, data, and trains all three model
    types: admission prediction models, specialty prediction models, and
    yet-to-arrive prediction models.

    Parameters
    ----------
    data_folder_name : str, optional
        Name of the data folder containing the training datasets.
        If None, will be extracted from command line arguments.

    Returns
    -------
    None
        The function trains and optionally saves models but does not return
        any values.

    Notes
    -----
    The function performs the following steps:
    1. Loads configuration from config.yaml
    2. Loads ED visits and inpatient arrivals data
    3. Sets up model parameters and hyperparameters
    4. Trains admission prediction classifiers
    5. Trains specialty prediction sequence model
    6. Trains yet-to-arrive prediction model
    7. Optionally saves trained models
    8. Optionally tests real-time prediction functionality

    """
    # Parse arguments if not provided
    if data_folder_name is None:
        args = parse_args()
        data_folder_name = (
            data_folder_name if data_folder_name is not None else args.data_folder_name
        )
    print(f"Loading data from folder: {data_folder_name}")

    project_root = set_project_root()

    # Set file locations
    data_file_path, _, model_file_path, config_path = set_file_paths(
        project_root=project_root,
        inference_time=False,
        train_dttm=None,
        data_folder_name=data_folder_name,
        config_file="config.yaml",
    )

    # Load parameters
    config = load_config_file(config_path)

    # Extract parameters
    prediction_times = config["prediction_times"]
    start_training_set = config["start_training_set"]
    start_validation_set = config["start_validation_set"]
    start_test_set = config["start_test_set"]
    end_test_set = config["end_test_set"]
    prediction_window = timedelta(minutes=config["prediction_window"])
    epsilon = float(config["epsilon"])
    yta_time_interval = timedelta(minutes=config["yta_time_interval"])
    x1, y1, x2, y2 = config["x1"], config["y1"], config["x2"], config["y2"]

    # Load data
    ed_visits = load_data(
        data_file_path=data_file_path,
        file_name="ed_visits.csv",
        index_column="snapshot_id",
        sort_columns=["visit_number", "snapshot_date", "prediction_time"],
        eval_columns=["prediction_time", "consultation_sequence", "final_sequence"],
    )
    inpatient_arrivals = load_data(
        data_file_path=data_file_path, file_name="inpatient_arrivals.csv"
    )

    # Create snapshot date
    ed_visits["snapshot_date"] = pd.to_datetime(
        ed_visits["snapshot_date"], dayfirst=True
    ).dt.date

    # Set up model parameters
    grid_params = {"n_estimators": [30], "subsample": [0.7], "colsample_bytree": [0.7]}

    exclude_columns = [
        "visit_number",
        "snapshot_date",
        "prediction_time",
        "specialty",
        "consultation_sequence",
        "final_sequence",
    ]

    ordinal_mappings = {
        "age_group": [
            "0-17",
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65-74",
            "75-115",
        ],
        "latest_acvpu": ["A", "C", "V", "P", "U"],
        "latest_obs_manchester_triage_acuity": [
            "Blue",
            "Green",
            "Yellow",
            "Orange",
            "Red",
        ],
        "latest_obs_objective_pain_score": [
            "Nil",
            "Mild",
            "Moderate",
            "Severe\\E\\Very Severe",
        ],
        "latest_obs_level_of_consciousness": ["A", "C", "V", "P", "U"],
    }

    specialties = ["surgical", "haem/onc", "medical", "paediatric"]
    cdf_cut_points = [0.9, 0.7]
    curve_params = (x1, y1, x2, y2)
    random_seed = 42

    # Call train_all_models with prepared parameters
    train_all_models(
        visits=ed_visits,
        start_training_set=start_training_set,
        start_validation_set=start_validation_set,
        start_test_set=start_test_set,
        end_test_set=end_test_set,
        yta=inpatient_arrivals,
        model_file_path=model_file_path,
        prediction_times=prediction_times,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        epsilon=epsilon,
        curve_params=curve_params,
        grid_params=grid_params,
        exclude_columns=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        specialties=specialties,
        cdf_cut_points=cdf_cut_points,
        random_seed=random_seed,
    )

    return


if __name__ == "__main__":
    main()
