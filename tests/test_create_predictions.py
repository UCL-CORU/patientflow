import unittest
import pandas as pd
import numpy as np
import os
from datetime import timedelta

from pathlib import Path

from patientflow.predict.emergency_demand import create_predictions
from patientflow.load import get_model_key
from patientflow.model_artifacts import TrainedClassifier, TrainingResults

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.value_to_outcome_predictor import ValueToOutcomePredictor
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
)
from patientflow.prepare import create_yta_filters


def create_random_df(n=1000, include_consults=False):
    # Generate random data
    np.random.seed(0)
    # Generate single random snapshot date for all rows
    snapshot_date = pd.Timestamp("2023-01-01") + pd.Timedelta(
        minutes=np.random.randint(0, 60 * 24 * 7)
    )
    snapshot_date = [snapshot_date] * n
    age_on_arrival = np.random.randint(1, 100, size=n)
    elapsed_los = np.random.randint(0, 3 * 24 * 3600, size=n)
    arrival_method = np.random.choice(
        ["ambulance", "public_transport", "walk-in"], size=n
    )
    sex = np.random.choice(["M", "F"], size=n)
    is_admitted = np.random.choice([0, 1], size=n)

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "age_on_arrival": age_on_arrival,
            "elapsed_los": elapsed_los,
            "arrival_method": arrival_method,
            "sex": sex,
            "is_admitted": is_admitted,
        }
    )

    if include_consults:
        # Generate random consultation sequence
        consultations = ["medical", "surgical", "haem/onc", "paediatric"]
        df["final_sequence"] = [
            [
                str(x)
                for x in np.random.choice(consultations, size=np.random.randint(1, 4))
            ]
            for _ in range(n)
        ]
        # Create consultation_sequence by removing random number of items from final_sequence
        df["consultation_sequence"] = [
            seq[: -np.random.randint(0, len(seq))] if len(seq) > 0 else []
            for seq in df["final_sequence"]
        ]
        # Set specialty as a random item from final_sequence, ensuring we always have a valid string
        df["specialty"] = [
            str(np.random.choice(seq))
            if len(seq) > 0
            else consultations[0]  # default to first consultation type
            for seq in df["final_sequence"]
        ]

    return df


def create_random_arrivals(n=1000):
    """Create random arrival data with arrival times and specialties.

    Parameters
    ----------
    n : int
        Number of arrivals to generate

    Returns
    -------
    pd.DataFrame
        DataFrame with arrival_datetime and specialty columns
    """
    # Use same seed as create_random_df for consistency
    np.random.seed(0)

    # Generate random arrival times over a week
    base_date = pd.Timestamp("2023-01-01")
    arrival_datetime = [
        base_date + pd.Timedelta(minutes=np.random.randint(0, 60 * 24 * 7))
        for _ in range(n)
    ]

    # Use same specialty list as create_random_df
    specialties = ["medical", "surgical", "haem/onc", "paediatric"]
    specialty = np.random.choice(specialties, size=n)

    # Generate random is_child values
    is_child = np.random.choice([True, False], size=n)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "arrival_datetime": arrival_datetime,
            "specialty": specialty,
            "is_child": is_child,
        }
    )

    return df


def create_admissions_model(prediction_time, n):
    """Create a test admissions model with TrainedClassifier structure.

    Parameters
    ----------
    prediction_time : float
        The prediction time point to create the model for

    Returns
    -------
    tuple
        (TrainedClassifier object, model_name string)
    """
    # Define the feature columns and target
    feature_columns = ["elapsed_los", "sex", "age_on_arrival", "arrival_method"]
    target_column = "is_admitted"

    df = create_random_df(include_consults=True, n=n)
    # Split the data into features and target
    X = df[feature_columns]
    y = df[target_column]

    # Define the model
    model = XGBClassifier(eval_metric="logloss")
    column_transformer = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                ["sex", "arrival_method"],
            ),
            ("passthrough", "passthrough", ["elapsed_los", "age_on_arrival"]),
        ]
    )

    # Create a pipeline with the feature transformer and the model
    pipeline = Pipeline(
        [("feature_transformer", column_transformer), ("classifier", model)]
    )

    # Fit the pipeline to the data
    pipeline.fit(X, y)

    # Create TrainingResults object
    training_results = TrainingResults(
        prediction_time=prediction_time,
    )

    # Create ModelResults object
    model_results = TrainedClassifier(
        pipeline=pipeline,
        training_results=training_results,
        calibrated_pipeline=None,  # No calibration for test
    )

    model_name = get_model_key("admissions", prediction_time)
    return (model_results, model_name, df)


def create_spec_model(df, apply_special_category_filtering):
    model = SequenceToOutcomePredictor(
        input_var="consultation_sequence",  # Column containing input sequences
        grouping_var="final_sequence",  # Column containing grouping sequences
        outcome_var="specialty",  # Column containing outcome categories
        apply_special_category_filtering=apply_special_category_filtering,
        admit_col="is_admitted",
    )
    model.fit(df)

    return model


def create_value_to_outcome_spec_model(df, apply_special_category_filtering):
    """Create a test specialty model using ValueToOutcomePredictor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training data
    apply_special_category_filtering : bool
        Whether to apply special category filtering

    Returns
    -------
    ValueToOutcomePredictor
        Fitted ValueToOutcomePredictor model
    """
    model = ValueToOutcomePredictor(
        input_var="consultation_sequence",  # Column containing input sequences
        grouping_var="final_sequence",  # Column containing grouping sequences
        outcome_var="specialty",  # Column containing outcome categories
        apply_special_category_filtering=apply_special_category_filtering,
        admit_col="is_admitted",
    )
    model.fit(df)

    return model


def create_yta_model(prediction_window, df, arrivals_df, yta_time_interval=60):
    """Create a test yet-to-arrive model using ParametricIncomingAdmissionPredictor.

    Parameters
    ----------
    prediction_window : timedelta
        The prediction window as a timedelta
    arrivals_df : pd.DataFrame
        DataFrame containing historical arrival data with arrival_datetime and specialty columns
    yta_time_interval : int or timedelta, optional
        The time interval for predictions in minutes or as timedelta. Default is 60 minutes.

    Returns
    -------
    tuple
        (model, model_name)
    """

    filters = create_yta_filters(df)

    # Convert yta_time_interval to timedelta if it's an int
    if isinstance(yta_time_interval, int):
        yta_time_interval = timedelta(minutes=yta_time_interval)

    model = ParametricIncomingAdmissionPredictor(filters=filters)

    # Convert timedelta to hours for model name
    hours = prediction_window.total_seconds() / 3600
    model_name = f"ed_yet_to_arrive_by_spec_{str(int(hours))}_hours"

    # Fit the model
    prediction_times = [(7, 0)]  # 7am predictions
    num_days = 7  # One week of data

    model.fit(
        train_df=arrivals_df.set_index("arrival_datetime"),
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        num_days=num_days,
    )

    return (model, model_name)


class TestCreatePredictions(unittest.TestCase):
    def setUp(self):
        self.model_file_path = Path("tmp")
        os.makedirs(self.model_file_path, exist_ok=True)
        self.prediction_time = (7, 0)
        self.prediction_window = timedelta(hours=8)
        self.x1, self.y1, self.x2, self.y2 = 4.0, 0.76, 12.0, 0.99
        self.cdf_cut_points = [0.7, 0.9]
        self.specialties = ["paediatric", "surgical", "haem/onc", "medical"]

        # Create and save models
        admissions_model, admissions_name, self.df = create_admissions_model(
            self.prediction_time, n=1000
        )

        self.arrivals_df = create_random_arrivals(n=1000)

        spec_model = create_spec_model(self.df, apply_special_category_filtering=False)

        yta_model, yta_name = create_yta_model(
            self.prediction_window, self.df, self.arrivals_df
        )
        self.models = (admissions_model, spec_model, yta_model)

    def test_basic_functionality(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )

        predictions = create_predictions(
            models=self.models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        self.assertEqual(predictions["paediatric"]["in_ed"], [1, 0])
        self.assertEqual(predictions["medical"]["yet_to_arrive"], [2, 1])

    def test_basic_functionality_with_special_category(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )

        # print("\nWithout special category")
        # print(self.df[self.df.is_admitted == 1])

        predictions_without_special_category = create_predictions(
            models=self.models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        # print("\nWith special category")
        admission_model, _, yta_model = self.models
        spec_model = create_spec_model(self.df, apply_special_category_filtering=True)
        models = (admission_model, spec_model, yta_model)

        predictions_with_special_category = create_predictions(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIn("paediatric", predictions_without_special_category)
        self.assertIn("paediatric", predictions_with_special_category)

        self.assertEqual(
            predictions_without_special_category["paediatric"]["in_ed"], [1, 0]
        )
        self.assertEqual(
            predictions_with_special_category["paediatric"]["in_ed"], [2, 1]
        )

    def test_single_row_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=1, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        predictions = create_predictions(
            models=self.models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(predictions, dict)
        for specialty in self.specialties:
            self.assertEqual(predictions[specialty]["in_ed"], [0, 0])

    def test_model_not_found(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        non_existing_window_hrs = timedelta(hours=10)

        # Replace YTA model with None while keeping admission and specialty models
        admission_model, spec_model, yta_model = self.models
        models = (admission_model, spec_model, None)

        with self.assertRaises(TypeError):
            create_predictions(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window=non_existing_window_hrs,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_model_not_fit(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        pipeline = Pipeline([("classifier", XGBClassifier())])
        _, spec_model, yta_model = self.models
        models = (pipeline, spec_model, yta_model)

        with self.assertRaises(TypeError):
            create_predictions(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

        unfitted_spec_model = SequenceToOutcomePredictor(
            input_var="consultation_sequence",
            grouping_var="final_sequence",
            outcome_var="specialty",
            admit_col="is_admitted",
        )
        models = (pipeline, unfitted_spec_model, yta_model)

        with self.assertRaises(TypeError):
            create_predictions(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )
        unfitted_yta_model = ParametricIncomingAdmissionPredictor()
        models = (pipeline, spec_model, unfitted_yta_model)

        with self.assertRaises(TypeError):
            create_predictions(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_prediction_window_extremes(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        short_window_hrs = timedelta(minutes=6)  # 0.1 hours
        long_window_hrs = timedelta(hours=100)

        # Test that an error is raised for where prediction_window is less than yta_time_interval
        with self.assertRaises(ValueError):
            short_yta_model, _ = create_yta_model(
                short_window_hrs, self.df, self.arrivals_df
            )

        short_window_hrs = timedelta(minutes=15)  # 0.25 hours
        yta_time_interval = timedelta(minutes=15)

        short_yta_model, _ = create_yta_model(
            short_window_hrs, self.df, self.arrivals_df, yta_time_interval
        )

        admission_model, spec_model, _ = self.models
        models = (admission_model, spec_model, short_yta_model)

        short_window_predictions = create_predictions(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=short_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        long_yta_model, _ = create_yta_model(long_window_hrs, self.df, self.arrivals_df)
        models = (admission_model, spec_model, long_yta_model)

        # Test that a warning is raised for long prediction windows
        with self.assertWarns(UserWarning) as warning:
            long_window_predictions = create_predictions(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window=long_window_hrs,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

        self.assertIn(
            "prediction_window appears to be longer than 72 hours", str(warning.warning)
        )

        self.assertIsInstance(short_window_predictions, dict)
        self.assertIsInstance(long_window_predictions, dict)

    def test_missing_key_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )
        admission_model, _, yta_model = self.models
        spec_model = create_spec_model(
            self.df[~self.df.final_sequence.apply(lambda x: "paediatric" in x)],
            apply_special_category_filtering=False,
        )
        models = (admission_model, spec_model, yta_model)

        predictions = create_predictions(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIn("paediatric", predictions)
        self.assertNotIn("paediatric", spec_model.weights.keys())

    def test_value_to_outcome_predictor_integration(self):
        """Test that ValueToOutcomePredictor works with create_predictions function."""
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )

        # Create models with ValueToOutcomePredictor
        admission_model, _, yta_model = self.models
        value_to_outcome_spec_model = create_value_to_outcome_spec_model(
            self.df, apply_special_category_filtering=False
        )
        models = (admission_model, value_to_outcome_spec_model, yta_model)

        predictions = create_predictions(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        # Verify that predictions work correctly with ValueToOutcomePredictor
        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        # Check that we get reasonable predictions
        self.assertIsInstance(predictions["paediatric"]["in_ed"], list)
        self.assertIsInstance(predictions["medical"]["yet_to_arrive"], list)
        self.assertEqual(len(predictions["paediatric"]["in_ed"]), 2)
        self.assertEqual(len(predictions["medical"]["yet_to_arrive"]), 2)

    def test_value_to_outcome_predictor_with_special_categories(self):
        """Test ValueToOutcomePredictor with special category filtering."""
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        prediction_snapshots["elapsed_los"] = prediction_snapshots["elapsed_los"].apply(
            lambda x: timedelta(seconds=x)
        )

        # Create models with ValueToOutcomePredictor and special category filtering
        admission_model, _, yta_model = self.models
        value_to_outcome_spec_model = create_value_to_outcome_spec_model(
            self.df, apply_special_category_filtering=True
        )
        models = (admission_model, value_to_outcome_spec_model, yta_model)

        predictions = create_predictions(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        # Verify that predictions work correctly with special category filtering
        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        # Check that we get reasonable predictions
        self.assertIsInstance(predictions["paediatric"]["in_ed"], list)
        self.assertIsInstance(predictions["medical"]["yet_to_arrive"], list)
        self.assertEqual(len(predictions["paediatric"]["in_ed"]), 2)
        self.assertEqual(len(predictions["medical"]["yet_to_arrive"]), 2)


if __name__ == "__main__":
    unittest.main()
