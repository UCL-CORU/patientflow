"""
Test script for patientflow.aggregate module

This script tests the refactored functionality of the aggregate module, including:
- BernoulliGeneratingFunction class with exact and approximate methods
- Probability aggregation with dynamic programming approach
- Probability distribution generation for prediction moments
- Performance improvements and backward compatibility

The tests have been updated to remove dependency on symbolic mathematics
while maintaining comprehensive coverage of the core functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone, timedelta

from patientflow.aggregate import (
    BernoulliGeneratingFunction,
    pred_proba_to_agg_predicted,
    get_prob_dist_for_prediction_moment,
    get_prob_dist,
    get_prob_dist_using_survival_curve,
    model_input_to_pred_proba,
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


class MockTrainedClassifier:
    """Mock trained classifier with pipeline attribute"""

    def __init__(self):
        self.pipeline = MockModel()
        self.calibrated_pipeline = None


class TestBernoulliGeneratingFunction(unittest.TestCase):
    """Test the core BernoulliGeneratingFunction class"""

    def test_init_no_weights(self):
        """Test initialization without weights"""
        probs = [0.1, 0.3, 0.7]
        gf = BernoulliGeneratingFunction(probs)

        np.testing.assert_array_equal(gf.probs, np.array(probs))
        self.assertEqual(gf.n, 3)

    def test_init_with_weights(self):
        """Test initialization with weights"""
        probs = [0.1, 0.3, 0.7]
        weights = [0.5, 1.0, 2.0]
        gf = BernoulliGeneratingFunction(probs, weights)

        expected = np.array([0.05, 0.3, 1.4])  # probs * weights
        np.testing.assert_array_equal(gf.probs, expected)
        self.assertEqual(gf.n, 3)

    def test_exact_distribution_empty(self):
        """Test exact distribution with empty input"""
        gf = BernoulliGeneratingFunction([])
        result = gf.exact_distribution()

        self.assertEqual(result, {0: 1.0})

    def test_exact_distribution_single(self):
        """Test exact distribution with single variable"""
        gf = BernoulliGeneratingFunction([0.3])
        result = gf.exact_distribution()

        expected = {0: 0.7, 1: 0.3}
        for k in expected:
            self.assertAlmostEqual(result[k], expected[k], places=10)

    def test_exact_distribution_multiple(self):
        """Test exact distribution with multiple variables"""
        gf = BernoulliGeneratingFunction([0.2, 0.5])
        result = gf.exact_distribution()

        # Manual calculation:
        # P(sum=0) = 0.8 * 0.5 = 0.4
        # P(sum=1) = 0.2 * 0.5 + 0.8 * 0.5 = 0.5
        # P(sum=2) = 0.2 * 0.5 = 0.1
        expected = {0: 0.4, 1: 0.5, 2: 0.1}

        for k in expected:
            self.assertAlmostEqual(result[k], expected[k], places=10)

    def test_exact_distribution_probabilities_sum_to_one(self):
        """Test that exact distribution probabilities sum to 1"""
        probs = [0.1, 0.3, 0.7, 0.2, 0.9]
        gf = BernoulliGeneratingFunction(probs)
        result = gf.exact_distribution()

        total_prob = sum(result.values())
        self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_normal_approximation_zero_variance(self):
        """Test normal approximation with zero variance (deterministic case)"""
        # All probabilities are 0 (deterministic sum = 0)
        gf = BernoulliGeneratingFunction([0.0, 0.0, 0.0])
        result = gf.normal_approximation()

        self.assertEqual(result, {0: 1.0})

        # All probabilities are 1 (deterministic sum = n)
        gf2 = BernoulliGeneratingFunction([1.0, 1.0, 1.0])
        result2 = gf2.normal_approximation()

        self.assertEqual(result2, {3: 1.0})

    def test_normal_approximation_nonzero_variance(self):
        """Test normal approximation with non-zero variance"""
        probs = np.random.beta(2, 5, 50).tolist()  # Random probabilities
        gf = BernoulliGeneratingFunction(probs)
        result = gf.normal_approximation()

        # Check probabilities sum to 1
        total_prob = sum(result.values())
        self.assertAlmostEqual(total_prob, 1.0, places=10)

        # Check all probabilities are non-negative
        for prob in result.values():
            self.assertGreaterEqual(prob, 0)

    def test_get_distribution_threshold_logic(self):
        """Test that get_distribution uses correct method based on threshold"""
        probs = [0.1, 0.2, 0.3]
        gf = BernoulliGeneratingFunction(probs)

        # Should use exact method when n <= threshold
        exact_result = gf.get_distribution(normal_approx_threshold=5)
        manual_exact = gf.exact_distribution()
        self.assertEqual(exact_result, manual_exact)

        # Should use normal approximation when n > threshold
        approx_result = gf.get_distribution(normal_approx_threshold=2)
        manual_approx = gf.normal_approximation()
        self.assertEqual(approx_result, manual_approx)


class TestAggregateRefactored(unittest.TestCase):
    """Test the refactored aggregate functions"""

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

    def test_model_input_to_pred_proba_empty(self):
        """Test model_input_to_pred_proba with empty input"""
        empty_df = pd.DataFrame(columns=["feature1", "feature2"])
        result = model_input_to_pred_proba(empty_df, self.model)

        self.assertTrue(result.empty)
        self.assertIn("pred_proba", result.columns)

    def test_model_input_to_pred_proba_normal(self):
        """Test model_input_to_pred_proba with normal input"""
        subset = self.X_test.iloc[:5]
        result = model_input_to_pred_proba(subset, self.model)

        self.assertEqual(len(result), 5)
        self.assertIn("pred_proba", result.columns)
        self.assertTrue(all(0 <= prob <= 1 for prob in result["pred_proba"]))

    def test_pred_proba_to_agg_predicted_empty(self):
        """Test aggregation with empty predictions"""
        empty_predictions = pd.DataFrame(columns=["pred_proba"])
        result = pred_proba_to_agg_predicted(empty_predictions)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["agg_proba"], 1)
        self.assertEqual(result.index[0], 0)

    def test_pred_proba_to_agg_predicted_small_exact(self):
        """Test aggregation with small dataset (exact computation)"""
        predictions = pd.DataFrame({"pred_proba": [0.1, 0.2, 0.3]})

        # Test without weights - force exact computation
        result = pred_proba_to_agg_predicted(predictions, normal_approx_threshold=10)

        # Check that result has expected shape
        self.assertEqual(len(result), 4)  # 0 to 3 possible counts

        # Sum of probabilities should be 1
        self.assertAlmostEqual(result["agg_proba"].sum(), 1.0, places=10)

        # Check specific probabilities using manual calculation
        # P(sum=0) = (1-0.1)*(1-0.2)*(1-0.3) = 0.9*0.8*0.7 = 0.504
        self.assertAlmostEqual(result.loc[0, "agg_proba"], 0.504, places=10)

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
        np.random.seed(123)  # For reproducibility
        predictions = pd.DataFrame({"pred_proba": np.random.uniform(0.1, 0.3, n)})

        # Test with normal approximation
        result = pred_proba_to_agg_predicted(predictions, normal_approx_threshold=10)

        # Check that result has expected shape
        self.assertEqual(len(result), n + 1)  # 0 to n possible counts

        # Sum of probabilities should be close to 1
        self.assertAlmostEqual(result["agg_proba"].sum(), 1.0, places=10)

        # Check that the distribution is reasonable (concentrated around the mean)
        mean_expected = predictions["pred_proba"].sum()
        max_prob_index = result["agg_proba"].idxmax()
        self.assertLess(
            abs(max_prob_index - mean_expected), 5
        )  # Should be close to mean

        # Test with weights
        weights = np.random.uniform(0.5, 1.5, n)
        result_weighted = pred_proba_to_agg_predicted(
            predictions, weights, normal_approx_threshold=10
        )

        # Sum of probabilities should still be close to 1
        self.assertAlmostEqual(result_weighted["agg_proba"].sum(), 1.0, places=10)

    def test_get_prob_dist_for_prediction_moment_inference(self):
        """Test probability distribution calculation in inference mode"""
        X_subset = self.X_test.iloc[:10]

        # Test in inference mode (no y_test needed)
        result = get_prob_dist_for_prediction_moment(
            X_test=X_subset, model=self.model, inference_time=True
        )

        # Check that result contains only agg_predicted
        self.assertIn("agg_predicted", result)
        self.assertNotIn("agg_observed", result)

        # Check that agg_predicted is a DataFrame with agg_proba column
        self.assertIsInstance(result["agg_predicted"], pd.DataFrame)
        self.assertIn("agg_proba", result["agg_predicted"].columns)

    def test_get_prob_dist_for_prediction_moment_training(self):
        """Test probability distribution calculation in training mode"""
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
        self.assertIsInstance(result["agg_observed"], int)

        # Check that observed count is reasonable
        self.assertGreaterEqual(result["agg_observed"], 0)
        self.assertLessEqual(result["agg_observed"], len(y_subset))

    def test_get_prob_dist_for_prediction_moment_with_trained_classifier(self):
        """Test with TrainedClassifier mock object"""
        X_subset = self.X_test.iloc[:5]
        y_subset = self.y_test.iloc[:5]

        trained_classifier = MockTrainedClassifier()

        result = get_prob_dist_for_prediction_moment(
            X_test=X_subset,
            model=trained_classifier,
            inference_time=False,
            y_test=y_subset,
        )

        self.assertIn("agg_predicted", result)
        self.assertIn("agg_observed", result)

    def test_get_prob_dist_for_prediction_moment_with_category_filter(self):
        """Test probability distribution with category filter"""
        X_subset = self.X_test.iloc[:10]
        y_subset = self.y_test.iloc[:10]

        # Create a category filter (only count half the positives)
        category_filter = pd.Series([True, False] * 5, index=y_subset.index)

        result = get_prob_dist_for_prediction_moment(
            X_test=X_subset,
            model=self.model,
            inference_time=False,
            y_test=y_subset,
            category_filter=category_filter,
        )

        # Observed count should be filtered
        expected_observed = int(sum(y_subset & category_filter))
        self.assertEqual(result["agg_observed"], expected_observed)

    def test_get_prob_dist_error_handling(self):
        """Test error handling in get_prob_dist"""
        # Test empty snapshots_dict
        with self.assertRaises(ValueError):
            get_prob_dist({}, self.X_test, self.y_test, self.model)

        # Test invalid date types
        with self.assertRaises(ValueError):
            get_prob_dist(
                {"2023-01-01": [0, 1, 2]}, self.X_test, self.y_test, self.model
            )

        # Test invalid indices format
        with self.assertRaises(ValueError):
            get_prob_dist(
                {date(2023, 1, 1): "not_a_list"}, self.X_test, self.y_test, self.model
            )

    def test_get_prob_dist_normal_functionality(self):
        """Test normal functionality of get_prob_dist"""
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
            self.assertIsInstance(result[dt]["agg_observed"], int)

            # Check probabilities sum to 1
            prob_sum = result[dt]["agg_predicted"]["agg_proba"].sum()
            self.assertAlmostEqual(prob_sum, 1.0, places=10)

    def test_get_prob_dist_empty_snapshot(self):
        """Test get_prob_dist with empty snapshot"""
        empty_snapshots = {date(2023, 1, 1): []}

        result = get_prob_dist(
            snapshots_dict=empty_snapshots,
            X_test=self.X_test,
            y_test=self.y_test,
            model=self.model,
        )

        # Should have deterministic result for empty snapshot
        self.assertEqual(result[date(2023, 1, 1)]["agg_observed"], 0)
        expected_pred = pd.DataFrame({"agg_proba": [1]}, index=[0])
        pd.testing.assert_frame_equal(
            result[date(2023, 1, 1)]["agg_predicted"], expected_pred
        )

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

        # Check results structure
        for dt in [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]:
            self.assertIn(dt, result)
            self.assertIn("agg_predicted", result[dt])
            self.assertIn("agg_observed", result[dt])

            # Check types
            self.assertIsInstance(result[dt]["agg_predicted"], pd.DataFrame)
            self.assertIn("agg_proba", result[dt]["agg_predicted"].columns)
            self.assertIsInstance(result[dt]["agg_observed"], int)
