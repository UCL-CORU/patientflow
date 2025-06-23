"""
Test script for patientflow.aggregate module

This script tests the core functionality of the aggregate module, including:
- Symbol creation and manipulation
- Expression building and coefficient extraction
- Probability aggregation with both exact symbolic computation and normal approximation
- Probability distribution generation for prediction moments
"""

import unittest
import pandas as pd
import numpy as np
import sympy as sym
from datetime import date, datetime, timezone, timedelta
from scipy.stats import poisson

from patientflow.aggregate import (
    create_symbols,
    compute_core_expression,
    build_expression,
    expression_subs,
    return_coeff,
    pred_proba_to_agg_predicted,
    get_prob_dist_for_prediction_moment,
    get_prob_dist,
    get_prob_dist_using_survival_curve,
)
from patientflow.predictors.incoming_admission_predictors import (
    EmpiricalIncomingAdmissionPredictor,
)


# Mock model for testing
class MockModel:
    def predict_proba(self, X):
        # Return a simple probability for each row: [1-p, p]
        n_samples = len(X)
        # Generate probabilities based on a feature to ensure deterministic output
        if "feature1" in X.columns:
            probs = np.clip(X["feature1"].values * 0.1, 0.05, 0.95)
        else:
            probs = np.full(n_samples, 0.2)  # Default probability

        return np.column_stack((1 - probs, probs))


class ParametricIncomingAdmissionPredictor:
    def __init__(self, prediction_window):
        self.prediction_window = prediction_window
        self.weights = {"Category1": 1.0, "Category2": 2.0}

    def predict(self, prediction_context, x1, y1, x2, y2):
        # Return a mock distribution
        category = list(prediction_context.keys())[0]
        probs = [poisson.pmf(k, self.weights[category] * 3) for k in range(10)]
        return {category: pd.DataFrame({"agg_proba": probs}, index=range(10))}


class TestAggregate(unittest.TestCase):
    def setUp(self):
        # Create test data
        np.random.seed(42)
        self.n_samples = 50

        # Create test features
        self.X_test = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, self.n_samples),
                "feature2": np.random.normal(0, 1, self.n_samples),
            }
        )

        # Create test labels (binary outcomes)
        self.y_test = pd.Series(np.random.binomial(1, 0.2, self.n_samples))

        # Ensure X_test and y_test have the same index
        index = pd.RangeIndex(self.n_samples)
        self.X_test.index = index
        self.y_test.index = index

        # Create test weights
        self.weights = pd.Series(
            np.random.uniform(0.5, 1.5, self.n_samples), index=index
        )

        # Create mock model
        self.model = MockModel()

        # Create snapshots dictionary for testing get_prob_dist
        self.snapshots_dict = {
            date(2023, 1, 1): list(range(10)),
            date(2023, 1, 2): list(range(10, 20)),
            date(2023, 1, 3): list(range(20, 30)),
        }

    def test_create_symbols(self):
        """Test symbol creation"""
        n = 5
        symbols = create_symbols(n)
        self.assertEqual(len(symbols), n)
        self.assertTrue(all(isinstance(s, sym.Symbol) for s in symbols))
        self.assertEqual(str(symbols[0]), "r0")
        self.assertEqual(str(symbols[4]), "r4")

    def test_compute_core_expression(self):
        """Test core expression computation"""
        s = sym.Symbol("s")
        ri = 0.3

        # Manually compute expected result: (1 - ri) + ri * s = 0.7 + 0.3s
        expected = 0.7 + 0.3 * s

        result = compute_core_expression(ri, s)
        self.assertEqual(result, expected)

    def test_build_expression(self):
        """Test building expression from symbols"""
        n = 3
        syms = create_symbols(n)

        expression = build_expression(syms, n)

        # Expression should contain s and be a product of core expressions
        self.assertIn("s", str(expression))

        # Substitute some values and check result is reasonable
        s = sym.Symbol("s")
        test_expr = expression.subs({s: 1})
        self.assertEqual(test_expr, 1)  # When s=1, result should be 1

    def test_expression_subs(self):
        """Test substitution in expressions"""
        n = 3
        syms = create_symbols(n)
        expression = build_expression(syms, n)

        predictions = [0.1, 0.2, 0.3]
        result = expression_subs(expression, n, predictions)

        # Check the result contains the symbol 's'
        self.assertIn("s", str(result))

    def test_return_coeff(self):
        """Test coefficient extraction"""
        s = sym.Symbol("s")
        expression = 0.1 + 0.2 * s + 0.3 * s**2

        coeff_0 = return_coeff(expression, 0)
        coeff_1 = return_coeff(expression, 1)
        coeff_2 = return_coeff(expression, 2)

        self.assertEqual(coeff_0, 0.1)
        self.assertEqual(coeff_1, 0.2)
        self.assertEqual(coeff_2, 0.3)

    def test_pred_proba_to_agg_predicted_empty(self):
        """Test aggregation with empty predictions"""
        empty_predictions = pd.DataFrame(columns=["pred_proba"])
        result = pred_proba_to_agg_predicted(empty_predictions)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["agg_proba"], 1)
        self.assertEqual(result.index[0], 0)

    def test_pred_proba_to_agg_predicted_small(self):
        """Test aggregation with small dataset (exact symbolic computation)"""
        # Create a small set of predictions
        predictions = pd.DataFrame({"pred_proba": [0.1, 0.2, 0.3]})

        # Test without weights
        result = pred_proba_to_agg_predicted(predictions, normal_approx_threshold=10)

        # Check that result has expected shape
        self.assertEqual(len(result), 4)  # 0 to 3 possible counts

        # Sum of probabilities should be 1
        self.assertAlmostEqual(result["agg_proba"].sum(), 1.0, places=10)

        # Test with weights
        weights = np.array([0.5, 1.0, 1.5])
        result_weighted = pred_proba_to_agg_predicted(
            predictions, weights, normal_approx_threshold=10
        )

        # Sum of probabilities should still be 1
        self.assertAlmostEqual(result_weighted["agg_proba"].sum(), 1.0, places=10)

    def test_pred_proba_to_agg_predicted_normal_approx(self):
        """Test aggregation with normal approximation for larger dataset"""
        # Create a large set of predictions
        n = 50
        predictions = pd.DataFrame({"pred_proba": np.random.uniform(0.1, 0.3, n)})

        # Test with normal approximation
        result = pred_proba_to_agg_predicted(predictions, normal_approx_threshold=10)

        # Check that result has expected shape
        self.assertEqual(len(result), n + 1)  # 0 to n possible counts

        # Sum of probabilities should be close to 1
        self.assertAlmostEqual(result["agg_proba"].sum(), 1.0, places=10)

        # Test with weights
        weights = np.random.uniform(0.5, 1.5, n)
        result_weighted = pred_proba_to_agg_predicted(
            predictions, weights, normal_approx_threshold=10
        )

        # Sum of probabilities should still be close to 1
        self.assertAlmostEqual(result_weighted["agg_proba"].sum(), 1.0, places=10)

    def test_get_prob_dist_for_prediction_moment(self):
        """Test probability distribution calculation for a prediction moment"""
        # Use a subset of test data
        X_subset = self.X_test.iloc[:10]
        y_subset = self.y_test.iloc[:10]

        # Test in non-inference mode
        result = get_prob_dist_for_prediction_moment(
            X_test=X_subset, model=self.model, inference_time=False, y_test=y_subset
        )

        # Check that result contains expected keys
        self.assertIn("agg_predicted", result)
        self.assertIn("agg_observed", result)

        # Check that agg_predicted is a DataFrame with agg_proba column
        self.assertIsInstance(result["agg_predicted"], pd.DataFrame)
        self.assertIn("agg_proba", result["agg_predicted"].columns)

        # Check that agg_observed is a number
        self.assertIsInstance(result["agg_observed"], (int, float))

        # Test in inference mode
        result_inference = get_prob_dist_for_prediction_moment(
            X_test=X_subset, model=self.model, inference_time=True
        )

        # Check that result contains only agg_predicted
        self.assertIn("agg_predicted", result_inference)
        self.assertNotIn("agg_observed", result_inference)

    def test_get_prob_dist(self):
        """Test probability distribution calculation for multiple snapshots"""
        result = get_prob_dist(
            snapshots_dict=self.snapshots_dict,
            X_test=self.X_test,
            y_test=self.y_test,
            model=self.model,
            weights=self.weights,
            verbose=False,
        )

        # Check that result contains all snapshot dates
        for dt in self.snapshots_dict.keys():
            self.assertIn(dt, result)

            # Check that each entry contains agg_predicted and agg_observed
            self.assertIn("agg_predicted", result[dt])
            self.assertIn("agg_observed", result[dt])

            # Check that agg_predicted is a DataFrame with agg_proba column
            self.assertIsInstance(result[dt]["agg_predicted"], pd.DataFrame)
            self.assertIn("agg_proba", result[dt]["agg_predicted"].columns)

            # Check that agg_observed is a number
            self.assertIsInstance(result[dt]["agg_observed"], (int, float))

    def test_get_prob_dist_using_survival_curve(self):
        """Test probability distribution generation using survival predictor"""
        # Create test data for patients
        test_df = pd.DataFrame(
            {
                "arrival_datetime": [
                    datetime(2023, 1, 1, 10, 15, tzinfo=timezone.utc),
                    datetime(2023, 1, 1, 10, 45, tzinfo=timezone.utc),
                    datetime(2023, 1, 2, 10, 30, tzinfo=timezone.utc),
                    datetime(2023, 1, 3, 11, 0, tzinfo=timezone.utc),
                ],
                "departure_datetime": [
                    datetime(2023, 1, 1, 12, 15, tzinfo=timezone.utc),
                    datetime(2023, 1, 1, 13, 45, tzinfo=timezone.utc),
                    datetime(2023, 1, 2, 14, 30, tzinfo=timezone.utc),
                    datetime(2023, 1, 3, 15, 0, tzinfo=timezone.utc),
                ],
                "specialty": ["medical", "medical", "surgical", "medical"],
            }
        )

        # Create and fit the EmpiricalIncomingAdmissionPredictor
        model = EmpiricalIncomingAdmissionPredictor()
        model.fit(
            train_df=test_df,
            prediction_window=timedelta(hours=8),
            yta_time_interval=timedelta(minutes=15),
            prediction_times=[(10, 0)],
            num_days=3,
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
        )
        # Test the function
        result = get_prob_dist_using_survival_curve(
            snapshot_dates=[date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            test_visits=test_df.reset_index(),
            category="unfiltered",
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=8),
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
            model=model,
        )

        # Check results
        self.assertIn(date(2023, 1, 1), result)
        self.assertIn(date(2023, 1, 2), result)
        self.assertIn(date(2023, 1, 3), result)

        # Check structure of results
        for dt in result:
            self.assertIn("agg_predicted", result[dt])
            self.assertIn("agg_observed", result[dt])

            # Check agg_predicted is a DataFrame with agg_proba column
            self.assertIsInstance(result[dt]["agg_predicted"], pd.DataFrame)
            self.assertIn("agg_proba", result[dt]["agg_predicted"].columns)

            # Check agg_observed is an integer
            self.assertIsInstance(result[dt]["agg_observed"], int)

        # Test with start_time_col as index
        test_df_indexed = test_df.set_index("arrival_datetime")
        result_indexed = get_prob_dist_using_survival_curve(
            snapshot_dates=[date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            test_visits=test_df_indexed,
            category="unfiltered",
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=8),
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
            model=model,
        )

        # Check results are the same as before
        for date_key in result:
            self.assertTrue(
                result[date_key]["agg_predicted"].equals(
                    result_indexed[date_key]["agg_predicted"]
                )
            )
            self.assertEqual(
                result[date_key]["agg_observed"],
                result_indexed[date_key]["agg_observed"],
            )

        # Test error cases
        with self.assertRaises(ValueError):
            # Test with missing end_time_col
            get_prob_dist_using_survival_curve(
                snapshot_dates=[date(2023, 1, 1)],
                test_visits=test_df.drop(columns=["departure_datetime"]),
                category="medical",
                prediction_time=(10, 0),
                prediction_window=timedelta(hours=8),
                start_time_col="arrival_datetime",
                end_time_col="departure_datetime",
                model=model,
            )

        with self.assertRaises(ValueError):
            # Test with unfitted model
            unfitted_model = EmpiricalIncomingAdmissionPredictor()
            get_prob_dist_using_survival_curve(
                snapshot_dates=[date(2023, 1, 1)],
                test_visits=test_df,
                category="medical",
                prediction_time=(10, 0),
                prediction_window=timedelta(hours=8),
                start_time_col="arrival_datetime",
                end_time_col="departure_datetime",
                model=unfitted_model,
            )
