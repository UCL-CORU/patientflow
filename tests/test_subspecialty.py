import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from patientflow.predict.subspecialty import build_subspecialty_data
from patientflow.model_artifacts import TrainedClassifier, TrainingResults
from patientflow.load import get_model_key

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.value_to_outcome_predictor import (
    ValueToOutcomePredictor,
)
from patientflow.predictors.subgroup_predictor import (
    create_subgroup_functions,
    MultiSubgroupPredictor,
)
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
    EmpiricalIncomingAdmissionPredictor,
    DirectAdmissionPredictor,
)
from patientflow.prepare import create_yta_filters


def _create_random_df(n=1000, include_consults=False):
    np.random.seed(0)
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
        consultations = ["medical", "surgical", "haem/onc", "paediatric"]
        df["final_sequence"] = [
            [
                str(x)
                for x in np.random.choice(consultations, size=np.random.randint(1, 4))
            ]
            for _ in range(n)
        ]
        df["consultation_sequence"] = [
            seq[: -np.random.randint(0, len(seq))] if len(seq) > 0 else []
            for seq in df["final_sequence"]
        ]
        df["specialty"] = [
            str(np.random.choice(seq)) if len(seq) > 0 else consultations[0]
            for seq in df["final_sequence"]
        ]

    return df


def _create_random_arrivals(n=1000):
    np.random.seed(0)
    base_date = pd.Timestamp("2023-01-01")
    arrival_datetime = [
        base_date + pd.Timedelta(minutes=np.random.randint(0, 60 * 24 * 7))
        for _ in range(n)
    ]
    specialties = ["medical", "surgical", "haem/onc", "paediatric"]
    specialty = np.random.choice(specialties, size=n)
    is_child = np.random.choice([True, False], size=n)
    df = pd.DataFrame(
        {
            "arrival_datetime": arrival_datetime,
            "specialty": specialty,
            "is_child": is_child,
        }
    )
    return df


def _create_random_arrivals_with_departures(n=1000):
    np.random.seed(0)
    base_date = pd.Timestamp("2023-01-01")
    arrival_datetime = [
        base_date + pd.Timedelta(minutes=np.random.randint(0, 60 * 24 * 7))
        for _ in range(n)
    ]
    length_of_stay_minutes = np.random.randint(60, 48 * 60, size=n)
    departure_datetime = [
        arrival + pd.Timedelta(minutes=los)
        for arrival, los in zip(arrival_datetime, length_of_stay_minutes)
    ]
    specialties = ["medical", "surgical", "haem/onc", "paediatric"]
    specialty = np.random.choice(specialties, size=n)
    is_child = np.random.choice([True, False], size=n)
    df = pd.DataFrame(
        {
            "arrival_datetime": arrival_datetime,
            "departure_datetime": departure_datetime,
            "specialty": specialty,
            "is_child": is_child,
        }
    )
    return df


def _create_admissions_model(prediction_time, n):
    feature_columns = ["elapsed_los", "sex", "age_on_arrival", "arrival_method"]
    target_column = "is_admitted"

    df = _create_random_df(include_consults=True, n=n)
    X = df[feature_columns]
    y = df[target_column]

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

    pipeline = Pipeline(
        [("feature_transformer", column_transformer), ("classifier", model)]
    )
    pipeline.fit(X, y)

    training_results = TrainingResults(
        prediction_time=prediction_time,
    )

    model_results = TrainedClassifier(
        pipeline=pipeline,
        training_results=training_results,
        calibrated_pipeline=None,
    )

    model_name = get_model_key("admissions", prediction_time)
    return (model_results, model_name, df)


def _create_spec_model(df, apply_special_category_filtering):
    model = SequenceToOutcomePredictor(
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
        apply_special_category_filtering=apply_special_category_filtering,
        admit_col="is_admitted",
    )
    model.fit(df)
    return model


def _create_value_to_outcome_spec_model(df, apply_special_category_filtering):
    model = ValueToOutcomePredictor(
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
        apply_special_category_filtering=apply_special_category_filtering,
        admit_col="is_admitted",
    )
    model.fit(df)
    return model


def _create_parametric_yta_model(
    prediction_window, df, arrivals_df, yta_time_interval=60
):
    filters = create_yta_filters(df)
    if isinstance(yta_time_interval, int):
        yta_time_interval = timedelta(minutes=yta_time_interval)
    model = ParametricIncomingAdmissionPredictor(filters=filters)
    prediction_times = [(7, 0)]
    num_days = 7
    model.fit(
        train_df=arrivals_df.set_index("arrival_datetime"),
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        num_days=num_days,
    )
    return model


def _create_empirical_yta_model(
    prediction_window, df, arrivals_df, yta_time_interval=60
):
    filters = create_yta_filters(df)
    if isinstance(yta_time_interval, int):
        yta_time_interval = timedelta(minutes=yta_time_interval)
    model = EmpiricalIncomingAdmissionPredictor(filters=filters)
    prediction_times = [(7, 0)]
    num_days = 7
    model.fit(
        train_df=arrivals_df,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        num_days=num_days,
    )
    return model


def _create_direct_predictor(prediction_window, df, arrivals_df, yta_time_interval=60):
    filters = create_yta_filters(df)
    if isinstance(yta_time_interval, int):
        yta_time_interval = timedelta(minutes=yta_time_interval)
    model = DirectAdmissionPredictor(filters=filters)
    prediction_times = [(7, 0)]
    num_days = 7
    model.fit(
        train_df=arrivals_df.set_index("arrival_datetime"),
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        num_days=num_days,
    )
    return model


class TestBuildSubspecialtyData(unittest.TestCase):
    def setUp(self):
        self.prediction_time = (7, 0)
        self.prediction_window = timedelta(hours=8)
        self.x1, self.y1, self.x2, self.y2 = 4.0, 0.76, 12.0, 0.99
        self.specialties = ["paediatric", "surgical", "haem/onc", "medical"]

        admissions_model, _, self.train_df = _create_admissions_model(
            self.prediction_time, n=1000
        )
        self.admissions_model = admissions_model
        self.arrivals_df = _create_random_arrivals(n=1000)

        self.spec_model = _create_spec_model(self.train_df, False)
        self.param_yta_model = _create_parametric_yta_model(
            self.prediction_window, self.train_df, self.arrivals_df
        )
        # Direct predictors for non-ED and elective flows
        self.direct_non_ed = _create_direct_predictor(
            self.prediction_window, self.train_df, self.arrivals_df
        )
        self.direct_elective = _create_direct_predictor(
            self.prediction_window, self.train_df, self.arrivals_df
        )

        self.models = (
            self.admissions_model,
            self.spec_model,
            self.param_yta_model,
            self.direct_non_ed,
            self.direct_elective,
        )

    def _make_snapshots(self, n=50):
        df = _create_random_df(n=n, include_consults=True)
        df["elapsed_los"] = df["elapsed_los"].apply(lambda x: timedelta(seconds=x))
        return df

    def test_basic_functionality_returns_expected_keys(self):
        snapshots = self._make_snapshots(50)
        result = build_subspecialty_data(
            models=self.models,
            prediction_time=self.prediction_time,
            prediction_snapshots=snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(result, dict)
        for spec in self.specialties:
            self.assertIn(spec, result)
            spec_data = result[spec]
            self.assertIn("pmf_ed_current_within_window", spec_data)
            self.assertIn("lambda_ed_yta_within_window", spec_data)
            self.assertIn("lambda_non_ed_yta_within_window", spec_data)
            self.assertIn("lambda_elective_yta_within_window", spec_data)
            pmf = np.asarray(spec_data["pmf_ed_current_within_window"])
            self.assertGreater(len(pmf), 0)
            self.assertIsInstance(spec_data["lambda_ed_yta_within_window"], float)
            self.assertIsInstance(spec_data["lambda_non_ed_yta_within_window"], float)
            self.assertIsInstance(spec_data["lambda_elective_yta_within_window"], float)

    def test_empirical_yta_integration(self):
        empirical_arrivals = _create_random_arrivals_with_departures(n=1000)
        empirical_yta = _create_empirical_yta_model(
            self.prediction_window, self.train_df, empirical_arrivals
        )
        models = (
            self.admissions_model,
            self.spec_model,
            empirical_yta,
            self.direct_non_ed,
            self.direct_elective,
        )
        snapshots = self._make_snapshots(40)
        result = build_subspecialty_data(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )
        self.assertIn("medical", result)
        self.assertGreater(
            len(np.asarray(result["medical"]["pmf_ed_current_within_window"])), 0
        )

    def test_prediction_time_and_window_mismatch_errors(self):
        snapshots = self._make_snapshots(10)
        # Wrong prediction time
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=(8, 0),
                prediction_snapshots=snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

        # Mismatched elective window
        other_window = timedelta(hours=10)
        bad_elective = _create_direct_predictor(
            other_window, self.train_df, self.arrivals_df
        )
        models = (
            self.admissions_model,
            self.spec_model,
            self.param_yta_model,
            self.direct_non_ed,
            bad_elective,
        )
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=models,
                prediction_time=self.prediction_time,
                prediction_snapshots=snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_missing_or_invalid_elapsed_los(self):
        snapshots = self._make_snapshots(5)
        # Missing column
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=self.prediction_time,
                prediction_snapshots=snapshots.drop(columns=["elapsed_los"]),
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )
        # Wrong dtype
        snapshots_bad = snapshots.copy()
        snapshots_bad["elapsed_los"] = snapshots_bad["elapsed_los"].dt.total_seconds()
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=self.prediction_time,
                prediction_snapshots=snapshots_bad,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_multisubgroup_predictor_path(self):
        subgroup_functions = create_subgroup_functions()
        msp = MultiSubgroupPredictor(
            subgroup_functions=subgroup_functions,
            base_predictor_class=SequenceToOutcomePredictor,
            input_var="consultation_sequence",
            grouping_var="final_sequence",
            outcome_var="specialty",
            min_samples=1,
        )
        msp.fit(self.train_df)
        models = (
            self.admissions_model,
            msp,
            self.param_yta_model,
            self.direct_non_ed,
            self.direct_elective,
        )
        snapshots = self._make_snapshots(60)
        result = build_subspecialty_data(
            models=models,
            prediction_time=self.prediction_time,
            prediction_snapshots=snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )
        self.assertIn("paediatric", result)


if __name__ == "__main__":
    unittest.main()
