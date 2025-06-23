import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the predictors to test
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
    EmpiricalIncomingAdmissionPredictor,
    weighted_poisson_binomial,
    aggregate_probabilities,
    convolute_distributions,
    poisson_binom_generating_function,
    find_nearest_previous_prediction_time,
)


class TestIncomingAdmissionPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        # Set random seed for reproducible tests
        np.random.seed(42)

        # Generate synthetic hospital admission data
        n_patients = 1000
        start_date = datetime(2024, 1, 1)

        # Generate random arrival times
        arrival_times = []
        departure_times = []
        specialties = []

        for i in range(n_patients):
            # Random arrival time within the date range
            days_offset = np.random.randint(0, 31)
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)

            arrival_time = start_date + timedelta(
                days=days_offset, hours=hours_offset, minutes=minutes_offset
            )

            # Random time to admission (0.5 to 48 hours)
            time_to_admission = np.random.exponential(6)  # Mean 6 hours
            time_to_admission = max(0.5, min(48, time_to_admission))

            departure_time = arrival_time + timedelta(hours=time_to_admission)

            arrival_times.append(arrival_time)
            departure_times.append(departure_time)

            # Random specialty
            specialty = np.random.choice(
                ["medical", "surgical", "haem/onc", "paediatric"]
            )
            specialties.append(specialty)

        # Create DataFrame
        cls.test_df = pd.DataFrame(
            {
                "arrival_datetime": arrival_times,
                "departure_datetime": departure_times,
                "specialty": specialties,
            }
        ).set_index("arrival_datetime")

        # Set up common parameters
        cls.prediction_window = timedelta(hours=8)
        cls.yta_time_interval = timedelta(minutes=30)
        cls.prediction_times = [(8, 0), (12, 0), (16, 0), (20, 0)]
        cls.num_days = 31

        # Set up filters
        cls.filters = {
            "medical": {"specialty": "medical"},
            "surgical": {"specialty": "surgical"},
            "haem_onc": {"specialty": "haem/onc"},
            "paediatric": {"specialty": "paediatric"},
        }

    def test_weighted_poisson_binomial(self):
        """Test the weighted_poisson_binomial function."""
        # Test normal case
        result = weighted_poisson_binomial(5, 3.0, 0.7)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 6)  # 0 to 5 inclusive
        self.assertTrue(np.all(result >= 0))

        # Test edge cases
        result_zero = weighted_poisson_binomial(0, 1.0, 0.5)
        self.assertEqual(len(result_zero), 1)
        # For i=0, the result should be poisson.pmf(0, 1.0) * binom.pmf(0, 0, 0.5) = 0.36787944117144233 * 1.0
        expected_value = 0.36787944117144233  # poisson.pmf(0, 1.0)
        self.assertAlmostEqual(result_zero[0], expected_value, places=10)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            weighted_poisson_binomial(-1, 1.0, 0.5)
        with self.assertRaises(ValueError):
            weighted_poisson_binomial(5, -1.0, 0.5)
        with self.assertRaises(ValueError):
            weighted_poisson_binomial(5, 1.0, 1.5)

    def test_aggregate_probabilities(self):
        """Test the aggregate_probabilities function."""
        lam = np.array([1.0, 2.0, 3.0])
        theta = np.array([0.5, 0.6, 0.7])
        kmax = 5
        time_index = 1

        result = aggregate_probabilities(lam, kmax, theta, time_index)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), kmax + 1)
        self.assertTrue(np.all(result >= 0))

        # Test invalid inputs
        with self.assertRaises(ValueError):
            aggregate_probabilities(lam, -1, theta, time_index)
        with self.assertRaises(ValueError):
            aggregate_probabilities(lam, kmax, theta, -1)
        with self.assertRaises(ValueError):
            aggregate_probabilities(lam, kmax, theta, 3)  # Index out of bounds

    def test_convolute_distributions(self):
        """Test the convolute_distributions function."""
        # Create test distributions
        dist_a = pd.DataFrame({"sum": [0, 1, 2], "prob": [0.3, 0.5, 0.2]})
        dist_b = pd.DataFrame({"sum": [0, 1], "prob": [0.6, 0.4]})

        result = convolute_distributions(dist_a, dist_b)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("sum" in result.columns)
        self.assertTrue("prob" in result.columns)
        self.assertTrue(np.allclose(result["prob"].sum(), 1.0, atol=1e-10))

        # Test invalid inputs
        invalid_dist = pd.DataFrame({"wrong_col": [1, 2], "prob": [0.5, 0.5]})
        with self.assertRaises(ValueError):
            convolute_distributions(invalid_dist, dist_b)

    def test_poisson_binom_generating_function(self):
        """Test the poisson_binom_generating_function."""
        NTimes = 3
        arrival_rates = np.array([1.0, 2.0, 1.5])
        theta = np.array([0.5, 0.6, 0.7])
        epsilon = 1e-6

        result = poisson_binom_generating_function(
            NTimes, arrival_rates, theta, epsilon
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("agg_proba" in result.columns)
        self.assertTrue(np.allclose(result["agg_proba"].sum(), 1.0, atol=1e-10))

        # Test invalid inputs
        with self.assertRaises(ValueError):
            poisson_binom_generating_function(0, arrival_rates, theta, epsilon)
        with self.assertRaises(ValueError):
            poisson_binom_generating_function(NTimes, arrival_rates, theta, 0)
        with self.assertRaises(ValueError):
            poisson_binom_generating_function(NTimes, arrival_rates, theta, 1.0)

    def test_find_nearest_previous_prediction_time(self):
        """Test the find_nearest_previous_prediction_time function."""
        prediction_times = [(8, 0), (12, 0), (16, 0), (20, 0)]

        # Test exact match
        result = find_nearest_previous_prediction_time((12, 0), prediction_times)
        self.assertEqual(result, (12, 0))

        # Test nearest previous - should warn and return (12, 0)
        with self.assertWarns(UserWarning) as warning_context:
            result = find_nearest_previous_prediction_time((14, 30), prediction_times)
        self.assertEqual(result, (12, 0))
        self.assertIn(
            "Time of day requested of (14, 30) was not in model training",
            str(warning_context.warning),
        )
        self.assertIn(
            "Reverting to predictions for (12, 0)", str(warning_context.warning)
        )

        # Test wrap around (before first time) - should warn and return (20, 0)
        with self.assertWarns(UserWarning) as warning_context:
            result = find_nearest_previous_prediction_time((6, 0), prediction_times)
        self.assertEqual(result, (20, 0))  # Should wrap to last time of previous day
        self.assertIn(
            "Time of day requested of (6, 0) was not in model training",
            str(warning_context.warning),
        )
        self.assertIn(
            "Reverting to predictions for (20, 0)", str(warning_context.warning)
        )

        # Test edge case - time exactly between two prediction times
        with self.assertWarns(UserWarning) as warning_context:
            result = find_nearest_previous_prediction_time((10, 0), prediction_times)
        self.assertEqual(result, (8, 0))  # Should return the earlier time
        self.assertIn(
            "Time of day requested of (10, 0) was not in model training",
            str(warning_context.warning),
        )
        self.assertIn(
            "Reverting to predictions for (8, 0)", str(warning_context.warning)
        )

    def test_incoming_admission_predictor_base_class(self):
        """Test the base IncomingAdmissionPredictor class."""
        # Test initialization - use a concrete subclass instead of abstract base class
        predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters, verbose=True
        )
        self.assertEqual(predictor.filters, self.filters)
        self.assertTrue(predictor.verbose)

        # Test filter_dataframe method
        filtered_df = predictor.filter_dataframe(self.test_df, {"specialty": "medical"})
        self.assertTrue(all(filtered_df["specialty"] == "medical"))

        # Test with function filter
        def is_medical(specialty):
            return specialty == "medical"

        filtered_df_func = predictor.filter_dataframe(
            self.test_df, {"specialty": is_medical}
        )
        self.assertTrue(all(filtered_df_func["specialty"] == "medical"))

    def test_parametric_predictor_initialization(self):
        """Test ParametricIncomingAdmissionPredictor initialization."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self.assertEqual(predictor.filters, self.filters)
        self.assertFalse(predictor.verbose)

    def test_parametric_predictor_fit(self):
        """Test ParametricIncomingAdmissionPredictor fit method."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)

        # Test successful fit
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Check that weights were calculated
        self.assertIsInstance(predictor.weights, dict)
        self.assertTrue(len(predictor.weights) > 0)

        # Check that all filter keys are present
        for key in self.filters.keys():
            self.assertIn(key, predictor.weights)

        # Check that prediction times are stored correctly
        self.assertEqual(predictor.prediction_times, self.prediction_times)

        # Test input validation
        with self.assertRaises(TypeError):
            predictor.fit(
                self.test_df,
                "8 hours",  # Should be timedelta
                self.yta_time_interval,
                self.prediction_times,
                self.num_days,
            )

        with self.assertRaises(ValueError):
            predictor.fit(
                self.test_df,
                timedelta(seconds=0),  # Zero prediction window
                self.yta_time_interval,
                self.prediction_times,
                self.num_days,
            )

    def test_parametric_predictor_predict(self):
        """Test ParametricIncomingAdmissionPredictor predict method."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Test prediction
        prediction_context = {
            "medical": {"prediction_time": (8, 0)},
            "surgical": {"prediction_time": (12, 0)},
        }

        predictions = predictor.predict(prediction_context, x1=2, y1=0.5, x2=4, y2=0.9)

        # Check output format
        self.assertIsInstance(predictions, dict)
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        # Check that predictions are DataFrames with expected columns
        for key, pred_df in predictions.items():
            self.assertIsInstance(pred_df, pd.DataFrame)
            self.assertIn("agg_proba", pred_df.columns)
            self.assertTrue(np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10))

        # Test missing parameters
        with self.assertRaises(ValueError):
            predictor.predict(prediction_context, x1=2, y1=0.5)  # Missing x2, y2

        # Test invalid filter key
        invalid_context = {"invalid_key": {"prediction_time": (8, 0)}}
        with self.assertRaises(ValueError):
            predictor.predict(invalid_context, x1=2, y1=0.5, x2=4, y2=0.9)

        # Test missing prediction_time
        invalid_context = {"medical": {"wrong_key": (8, 0)}}
        with self.assertRaises(ValueError):
            predictor.predict(invalid_context, x1=2, y1=0.5, x2=4, y2=0.9)

    def test_empirical_predictor_initialization(self):
        """Test EmpiricalIncomingAdmissionPredictor initialization."""
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        self.assertEqual(predictor.filters, self.filters)
        self.assertIsNone(predictor.survival_df)

    def test_empirical_predictor_fit(self):
        """Test EmpiricalIncomingAdmissionPredictor fit method."""
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        # Test successful fit
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Check that survival curve was calculated
        self.assertIsNotNone(predictor.survival_df)
        self.assertIsInstance(predictor.survival_df, pd.DataFrame)
        self.assertIn("time_hours", predictor.survival_df.columns)
        self.assertIn("survival_probability", predictor.survival_df.columns)

        # Check that weights were calculated
        self.assertIsInstance(predictor.weights, dict)
        self.assertTrue(len(predictor.weights) > 0)

    def test_empirical_predictor_get_survival_curve(self):
        """Test EmpiricalIncomingAdmissionPredictor get_survival_curve method."""
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        # Test before fitting
        with self.assertRaises(RuntimeError):
            predictor.get_survival_curve()

        # Test after fitting
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        survival_curve = predictor.get_survival_curve()
        self.assertIsInstance(survival_curve, pd.DataFrame)
        self.assertIn("time_hours", survival_curve.columns)
        self.assertIn("survival_probability", survival_curve.columns)

    def test_empirical_predictor_predict(self):
        """Test EmpiricalIncomingAdmissionPredictor predict method."""
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Test prediction
        prediction_context = {
            "medical": {"prediction_time": (8, 0)},
            "surgical": {"prediction_time": (12, 0)},
        }

        predictions = predictor.predict(prediction_context, max_value=25)

        # Check output format
        self.assertIsInstance(predictions, dict)
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        # Check that predictions are DataFrames with expected columns
        for key, pred_df in predictions.items():
            self.assertIsInstance(pred_df, pd.DataFrame)
            self.assertIn("agg_proba", pred_df.columns)
            self.assertTrue(np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10))

        # Test without survival data
        predictor_no_survival = EmpiricalIncomingAdmissionPredictor(
            filters=self.filters
        )
        with self.assertRaises(RuntimeError):
            predictor_no_survival.predict(prediction_context)

    def test_predictor_comparison(self):
        """Test that both predictors produce reasonable results."""
        # Fit both predictors
        parametric_predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters
        )
        empirical_predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        parametric_predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        empirical_predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Make predictions
        prediction_context = {"medical": {"prediction_time": (8, 0)}}

        parametric_pred = parametric_predictor.predict(
            prediction_context, x1=2, y1=0.5, x2=4, y2=0.9
        )
        empirical_pred = empirical_predictor.predict(prediction_context, max_value=25)

        # Both should produce valid probability distributions
        for pred_dict in [parametric_pred, empirical_pred]:
            for key, pred_df in pred_dict.items():
                self.assertTrue(np.all(pred_df["agg_proba"] >= 0))
                self.assertTrue(
                    np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10)
                )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty filters
        predictor = ParametricIncomingAdmissionPredictor(filters={})
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Should have 'unfiltered' key
        self.assertIn("unfiltered", predictor.weights)

        # Test with single prediction time
        single_time_predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters
        )
        single_time_predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            [(8, 0)],  # Single prediction time
            self.num_days,
        )

        # Test with very short prediction window
        short_window = timedelta(minutes=30)
        short_predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        short_predictor.fit(
            self.test_df,
            short_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

    def test_get_weights(self):
        """Test the get_weights method."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        weights = predictor.get_weights()
        self.assertIsInstance(weights, dict)
        self.assertEqual(weights, predictor.weights)

    def test_metrics_storage(self):
        """Test that metrics are properly stored during fitting."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Check that metrics were stored
        self.assertIsInstance(predictor.metrics, dict)
        self.assertIn("train_dttm", predictor.metrics)
        self.assertIn("train_set_no", predictor.metrics)
        self.assertIn("start_date", predictor.metrics)
        self.assertIn("end_date", predictor.metrics)
        self.assertIn("num_days", predictor.metrics)

        # Check that values are reasonable
        self.assertEqual(predictor.metrics["train_set_no"], len(self.test_df))
        self.assertEqual(predictor.metrics["num_days"], self.num_days)

    def test_predictor_warning_functionality(self):
        """Test that predictors properly warn when using non-trained prediction times."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        # Test prediction with a time not in training data - should warn and use nearest
        prediction_context = {
            "medical": {"prediction_time": (14, 30)},  # Not in training times
            "surgical": {"prediction_time": (6, 0)},  # Not in training times
        }

        with self.assertWarns(UserWarning) as warning_context:
            predictions = predictor.predict(
                prediction_context, x1=2, y1=0.5, x2=4, y2=0.9
            )

        # Should have warnings for both times
        warning_messages = [
            str(warning.message) for warning in warning_context.warnings
        ]
        self.assertTrue(any("(14, 30)" in msg for msg in warning_messages))
        self.assertTrue(any("(6, 0)" in msg for msg in warning_messages))

        # Predictions should still be generated successfully
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        # Test with empirical predictor as well
        empirical_predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        empirical_predictor.fit(
            self.test_df,
            self.prediction_window,
            self.yta_time_interval,
            self.prediction_times,
            self.num_days,
        )

        with self.assertWarns(UserWarning) as warning_context:
            empirical_predictions = empirical_predictor.predict(
                prediction_context, max_value=25
            )

        # Should have warnings for both times
        warning_messages = [
            str(warning.message) for warning in warning_context.warnings
        ]
        self.assertTrue(any("(14, 30)" in msg for msg in warning_messages))
        self.assertTrue(any("(6, 0)" in msg for msg in warning_messages))

        # Predictions should still be generated successfully
        self.assertIn("medical", empirical_predictions)
        self.assertIn("surgical", empirical_predictions)


if __name__ == "__main__":
    unittest.main()
