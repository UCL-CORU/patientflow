import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from patientflow.predict.subspecialty import (
    build_subspecialty_data,
    SubspecialtyPredictionInputs,
    FlowInputs,
    scale_pmf_by_probability,
    convolve_pmfs,
    compute_transfer_arrivals,
)
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator
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


def _create_transfer_model(specialties):
    """Create a simple transfer model for testing."""
    # Create some basic transfer data
    transfers = []
    for _ in range(100):
        source = np.random.choice(specialties)
        # 50% chance of transfer vs discharge
        if np.random.rand() < 0.5:
            dest = np.random.choice([s for s in specialties if s != source])
        else:
            dest = None  # Discharge
        transfers.append({"current_subspecialty": source, "next_subspecialty": dest})
    
    X = pd.DataFrame(transfers)
    model = TransferProbabilityEstimator()
    model.fit(X, set(specialties))
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

        # Create inpatient discharge classifier (same structure as ED classifier but for discharge prediction)
        inpatient_discharge_model, _, _ = _create_admissions_model(
            self.prediction_time, n=1000
        )
        self.inpatient_discharge_model = inpatient_discharge_model

        # Create transfer model
        self.transfer_model = _create_transfer_model(self.specialties)

        self.models = (
            self.admissions_model,  # ED classifier
            self.inpatient_discharge_model,  # Inpatient discharge classifier
            self.spec_model,
            self.param_yta_model,
            self.direct_non_ed,
            self.direct_elective,
            self.transfer_model,  # Transfer model
        )

    def _make_snapshots(self, n=50):
        df = _create_random_df(n=n, include_consults=True)
        df["elapsed_los"] = df["elapsed_los"].apply(lambda x: timedelta(seconds=x))
        return df

    def _make_inpatient_snapshots(self, n=50):
        """Create inpatient snapshots with current_subspecialty column"""
        df = _create_random_df(n=n, include_consults=False)
        df["elapsed_los"] = df["elapsed_los"].apply(lambda x: timedelta(seconds=x))
        # Add current_subspecialty column - assign random specialties
        df["current_subspecialty"] = np.random.choice(self.specialties, size=n)
        return df

    def test_basic_functionality_returns_expected_keys(self):
        ed_snapshots = self._make_snapshots(50)
        inpatient_snapshots = self._make_inpatient_snapshots(30)
        result = build_subspecialty_data(
            models=self.models,
            prediction_time=self.prediction_time,
            ed_snapshots=ed_snapshots,
            inpatient_snapshots=inpatient_snapshots,
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
            self.assertIsInstance(spec_data, SubspecialtyPredictionInputs)
            # Check new structure attributes exist
            self.assertTrue(hasattr(spec_data, "subspecialty_id"))
            self.assertTrue(hasattr(spec_data, "prediction_window"))
            self.assertTrue(hasattr(spec_data, "inflows"))
            self.assertTrue(hasattr(spec_data, "outflows"))
            # Check inflows
            self.assertIn("ed_current", spec_data.inflows)
            self.assertIn("ed_yta", spec_data.inflows)
            self.assertIn("non_ed_yta", spec_data.inflows)
            self.assertIn("elective_yta", spec_data.inflows)
            self.assertIn("transfers_in", spec_data.inflows)
            # Check outflows
            self.assertIn("departures", spec_data.outflows)
            # Check values
            ed_pmf = np.asarray(spec_data.inflows["ed_current"].distribution)
            inpatient_pmf = np.asarray(spec_data.outflows["departures"].distribution)
            transfer_pmf = np.asarray(spec_data.inflows["transfers_in"].distribution)
            self.assertGreater(len(ed_pmf), 0)
            self.assertGreater(len(inpatient_pmf), 0)
            self.assertGreater(len(transfer_pmf), 0)
            self.assertIsInstance(spec_data.inflows["ed_yta"].distribution, float)
            self.assertIsInstance(spec_data.inflows["non_ed_yta"].distribution, float)
            self.assertIsInstance(spec_data.inflows["elective_yta"].distribution, float)

    def test_empirical_yta_integration(self):
        empirical_arrivals = _create_random_arrivals_with_departures(n=1000)
        empirical_yta = _create_empirical_yta_model(
            self.prediction_window, self.train_df, empirical_arrivals
        )
        models = (
            self.admissions_model,  # ED classifier
            self.inpatient_discharge_model,  # Inpatient discharge classifier
            self.spec_model,
            empirical_yta,
            self.direct_non_ed,
            self.direct_elective,
            self.transfer_model,  # Transfer model
        )
        ed_snapshots = self._make_snapshots(40)
        inpatient_snapshots = self._make_inpatient_snapshots(25)
        result = build_subspecialty_data(
            models=models,
            prediction_time=self.prediction_time,
            ed_snapshots=ed_snapshots,
            inpatient_snapshots=inpatient_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )
        self.assertIn("medical", result)
        self.assertGreater(
            len(np.asarray(result["medical"].inflows["ed_current"].distribution)), 0
        )

    def test_prediction_time_and_window_mismatch_errors(self):
        ed_snapshots = self._make_snapshots(10)
        inpatient_snapshots = self._make_inpatient_snapshots(5)
        # Wrong prediction time
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=(8, 0),
                ed_snapshots=ed_snapshots,
                inpatient_snapshots=inpatient_snapshots,
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
            self.admissions_model,  # ED classifier
            self.inpatient_discharge_model,  # Inpatient discharge classifier
            self.spec_model,
            self.param_yta_model,
            self.direct_non_ed,
            bad_elective,
            self.transfer_model,  # Transfer model
        )
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=models,
                prediction_time=self.prediction_time,
                ed_snapshots=ed_snapshots,
                inpatient_snapshots=inpatient_snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_missing_or_invalid_elapsed_los(self):
        ed_snapshots = self._make_snapshots(5)
        inpatient_snapshots = self._make_inpatient_snapshots(3)
        # Missing column in ED snapshots
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=self.prediction_time,
                ed_snapshots=ed_snapshots.drop(columns=["elapsed_los"]),
                inpatient_snapshots=inpatient_snapshots,
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )
        # Missing column in inpatient snapshots
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=self.prediction_time,
                ed_snapshots=ed_snapshots,
                inpatient_snapshots=inpatient_snapshots.drop(columns=["elapsed_los"]),
                specialties=self.specialties,
                prediction_window=self.prediction_window,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )
        # Wrong dtype for ED snapshots
        ed_snapshots_bad = ed_snapshots.copy()
        ed_snapshots_bad["elapsed_los"] = ed_snapshots_bad[
            "elapsed_los"
        ].dt.total_seconds()
        with self.assertRaises(ValueError):
            build_subspecialty_data(
                models=self.models,
                prediction_time=self.prediction_time,
                ed_snapshots=ed_snapshots_bad,
                inpatient_snapshots=inpatient_snapshots,
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
            self.admissions_model,  # ED classifier
            self.inpatient_discharge_model,  # Inpatient discharge classifier
            msp,
            self.param_yta_model,
            self.direct_non_ed,
            self.direct_elective,
            self.transfer_model,  # Transfer model
        )
        ed_snapshots = self._make_snapshots(60)
        inpatient_snapshots = self._make_inpatient_snapshots(35)
        result = build_subspecialty_data(
            models=models,
            prediction_time=self.prediction_time,
            ed_snapshots=ed_snapshots,
            inpatient_snapshots=inpatient_snapshots,
            specialties=self.specialties,
            prediction_window=self.prediction_window,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )
        self.assertIn("paediatric", result)


class TestScalePMFByProbability(unittest.TestCase):
    """Test suite for scale_pmf_by_probability function."""

    def test_edge_cases(self):
        """Test edge cases: zero and unit probability."""
        pmf = np.array([0.5, 0.3, 0.2])

        # Zero probability should return [1.0, 0.0]
        result_zero = scale_pmf_by_probability(pmf, 0.0)
        np.testing.assert_array_almost_equal(result_zero, np.array([1.0, 0.0]))

        # Unit probability should return original PMF
        result_one = scale_pmf_by_probability(pmf, 1.0)
        np.testing.assert_array_almost_equal(result_one, pmf)

    def test_binomial_thinning_calculation(self):
        """Test the binomial thinning calculation with a simple case."""
        # 50% chance of 0 departures, 50% chance of 2 departures
        pmf = np.array([0.5, 0.0, 0.5])
        compound_prob = 0.5

        result = scale_pmf_by_probability(pmf, compound_prob)

        # P(0 transfers) = P(0 departures) + P(2 depart, none transfer)
        #                = 0.5 + 0.5 * (1-0.5)^2 = 0.5 + 0.125 = 0.625
        self.assertAlmostEqual(result[0], 0.625)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_expected_value_scaling(self):
        """Test that expected value scales linearly with compound probability."""
        pmf = np.array([0.0, 0.0, 1.0])  # Certain to have 2 departures

        for compound_prob in [0.1, 0.5, 0.9]:
            result = scale_pmf_by_probability(pmf, compound_prob)
            expected_transfers = np.sum(result * np.arange(len(result)))
            self.assertAlmostEqual(expected_transfers, 2 * compound_prob, places=6)

    def test_probability_conservation(self):
        """Test that probabilities sum to 1.0 for various inputs."""
        np.random.seed(42)
        raw_pmf = np.random.rand(10)
        pmf = raw_pmf / raw_pmf.sum()

        for compound_prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = scale_pmf_by_probability(pmf, compound_prob)
            self.assertAlmostEqual(np.sum(result), 1.0, places=10)
            self.assertTrue(np.all(result >= 0))


class TestConvolvePMFs(unittest.TestCase):
    """Test suite for convolve_pmfs function."""

    def test_basic_convolution_calculation(self):
        """Test basic convolution calculation."""
        # Each has 50% chance of 0 or 1
        pmf1 = np.array([0.5, 0.5])
        pmf2 = np.array([0.5, 0.5])

        result = convolve_pmfs(pmf1, pmf2)

        # Result should be: 25% chance of 0, 50% chance of 1, 25% chance of 2
        expected = np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(result, expected)

    def test_expected_value_additivity(self):
        """Test that E[X+Y] = E[X] + E[Y]."""
        pmf1 = np.array([0.2, 0.5, 0.3])
        pmf2 = np.array([0.4, 0.4, 0.2])

        result = convolve_pmfs(pmf1, pmf2)

        ev1 = np.sum(pmf1 * np.arange(len(pmf1)))
        ev2 = np.sum(pmf2 * np.arange(len(pmf2)))
        ev_result = np.sum(result * np.arange(len(result)))

        self.assertAlmostEqual(ev_result, ev1 + ev2, places=10)

    def test_probability_conservation(self):
        """Test that convolution preserves probability mass."""
        np.random.seed(42)
        for _ in range(5):
            raw1 = np.random.rand(5)
            raw2 = np.random.rand(5)
            pmf1 = raw1 / raw1.sum()
            pmf2 = raw2 / raw2.sum()

            result = convolve_pmfs(pmf1, pmf2)
            self.assertAlmostEqual(np.sum(result), 1.0, places=10)


class TestComputeTransferArrivals(unittest.TestCase):
    """Test suite for compute_transfer_arrivals function."""

    def setUp(self):
        """Set up test fixtures."""
        self.subspecialties = ["cardiology", "surgery", "medicine"]

    def test_simple_transfer_calculation(self):
        """Test basic transfer calculation: cardiology -> surgery."""
        subspecialty_data = {
            "cardiology": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.0, 0.0, 1.0])  # 2 departures certain
                    )
                }
            },
            "surgery": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
            "medicine": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
        }

        # Cardiology transfers 100% to surgery
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", "surgery"],
            }
        )
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        result = compute_transfer_arrivals(
            subspecialty_data, transfer_model, self.subspecialties
        )

        # Surgery should receive 2 arrivals with certainty
        self.assertAlmostEqual(result["surgery"][2], 1.0, places=5)
        # Medicine should have no arrivals
        self.assertGreater(result["medicine"][0], 0.99)

    def test_mixed_transfers_and_discharges(self):
        """Test calculation when some patients transfer and some are discharged (None)."""
        subspecialty_data = {
            "cardiology": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # 4 departures certain
                    )
                }
            },
            "surgery": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
            "medicine": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
        }

        # Of 4 departures: 1 -> surgery, 1 -> medicine, 2 -> discharge (None)
        # This means: 50% transfer, 50% discharge
        # Of the 50% that transfer: 50% to surgery, 50% to medicine
        X = pd.DataFrame(
            {
                "current_subspecialty": [
                    "cardiology",
                    "cardiology",
                    "cardiology",
                    "cardiology",
                ],
                "next_subspecialty": ["surgery", "medicine", None, None],
            }
        )
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        result = compute_transfer_arrivals(
            subspecialty_data, transfer_model, self.subspecialties
        )

        # Check cardiology stats
        prob_transfer = transfer_model.get_transfer_prob("cardiology")
        self.assertAlmostEqual(prob_transfer, 0.5)  # 2 out of 4 transfer

        dest_dist = transfer_model.get_destination_distribution("cardiology")
        self.assertAlmostEqual(
            dest_dist["surgery"], 0.5
        )  # Of transfers, 50% to surgery
        self.assertAlmostEqual(
            dest_dist["medicine"], 0.5
        )  # Of transfers, 50% to medicine

        # With 4 departures and 50% transfer probability:
        # Expected arrivals to surgery: 4 * 0.5 * 0.5 = 1.0
        # Expected arrivals to medicine: 4 * 0.5 * 0.5 = 1.0
        surgery_arrivals = result["surgery"]
        medicine_arrivals = result["medicine"]

        # Expected value should be ~1 for each
        ev_surgery = np.sum(surgery_arrivals * np.arange(len(surgery_arrivals)))
        ev_medicine = np.sum(medicine_arrivals * np.arange(len(medicine_arrivals)))
        self.assertAlmostEqual(ev_surgery, 1.0, places=5)
        self.assertAlmostEqual(ev_medicine, 1.0, places=5)

        # PMFs should sum to 1
        self.assertAlmostEqual(np.sum(surgery_arrivals), 1.0)
        self.assertAlmostEqual(np.sum(medicine_arrivals), 1.0)

    def test_multiple_sources_aggregation(self):
        """Test aggregation when multiple sources transfer to one destination."""
        subspecialty_data = {
            "cardiology": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.0, 1.0])
                    )
                }
            },
            "surgery": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.0, 1.0])
                    )
                }
            },
            "medicine": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
        }

        # Both cardiology and surgery transfer to medicine
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "surgery"],
                "next_subspecialty": ["medicine", "medicine"],
            }
        )
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        result = compute_transfer_arrivals(
            subspecialty_data, transfer_model, self.subspecialties
        )

        # Medicine should receive 2 arrivals (convolution of two Bernoulli)
        self.assertAlmostEqual(result["medicine"][2], 1.0, places=5)

    def test_complex_transfer_network(self):
        """Test realistic complex network with circular transfers."""
        subspecialty_data = {
            "cardiology": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.5, 0.5])
                    )
                }
            },
            "surgery": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.5, 0.5])
                    )
                }
            },
            "medicine": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([0.5, 0.5])
                    )
                }
            },
        }

        # Network: cardiology -> surgery, surgery -> medicine, medicine -> cardiology
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "surgery", "medicine"],
                "next_subspecialty": ["surgery", "medicine", "cardiology"],
            }
        )
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        result = compute_transfer_arrivals(
            subspecialty_data, transfer_model, self.subspecialties
        )

        # All subspecialties should receive arrivals and sum to 1
        for subspecialty in self.subspecialties:
            self.assertAlmostEqual(np.sum(result[subspecialty]), 1.0)
            self.assertTrue(np.all(result[subspecialty] >= 0))

    def test_probability_validity(self):
        """Test that arrival PMFs are valid probability distributions."""
        np.random.seed(42)
        subspecialty_data = {
            subspecialty: {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.random.dirichlet(np.ones(4))
                    )
                }
            }
            for subspecialty in self.subspecialties
        }

        # Create random transfer network
        transfers = [
            {
                "current_subspecialty": np.random.choice(self.subspecialties),
                "next_subspecialty": np.random.choice([None] + self.subspecialties),
            }
            for _ in range(10)
        ]
        X = pd.DataFrame(transfers)
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        result = compute_transfer_arrivals(
            subspecialty_data, transfer_model, self.subspecialties
        )

        for subspecialty in self.subspecialties:
            arrivals = result[subspecialty]
            self.assertAlmostEqual(np.sum(arrivals), 1.0, places=10)
            self.assertTrue(np.all(arrivals >= 0))

    def test_error_handling(self):
        """Test essential error conditions."""
        # Missing departure PMF
        subspecialty_data = {
            "cardiology": {},
            "surgery": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
            "medicine": {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            },
        }
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": ["surgery"],
            }
        )
        transfer_model = TransferProbabilityEstimator()
        transfer_model.fit(X, set(self.subspecialties))

        with self.assertRaises(KeyError):
            compute_transfer_arrivals(
                subspecialty_data, transfer_model, self.subspecialties
            )

        # Unfitted transfer model
        subspecialty_data_valid = {
            spec: {
                "outflows": {
                    "departures": FlowInputs(
                        flow_id="departures",
                        flow_type="pmf",
                        distribution=np.array([1.0, 0.0])
                    )
                }
            }
            for spec in self.subspecialties
        }
        unfitted_model = TransferProbabilityEstimator()

        with self.assertRaises(ValueError):
            compute_transfer_arrivals(
                subspecialty_data_valid, unfitted_model, self.subspecialties
            )


if __name__ == "__main__":
    unittest.main()
