import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from patientflow.train.classifiers import train_classifier
from patientflow.model_artifacts import TrainedClassifier


class TestClassifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        np.random.seed(42)

        # Create sample visit data
        n_samples = 1000
        cls.train_visits = pd.DataFrame(
            {
                "visit_number": range(n_samples),
                "age": np.random.randint(0, 100, n_samples),
                "sex": pd.Series(
                    np.random.choice(["M", "F"], n_samples), dtype="string"
                ),
                "arrival_method": pd.Series(
                    np.random.choice(["ambulance", "walk-in", "referral"], n_samples),
                    dtype="string",
                ),
                "is_admitted": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                "snapshot_time": pd.date_range(
                    start="2023-01-01", periods=n_samples, freq="h"
                ),
                "prediction_time": [(4, 0)] * n_samples,  # All snapshots at 4:00
            }
        )

        # Create validation and test sets with similar structure
        cls.valid_visits = cls.train_visits.copy()
        cls.test_visits = cls.train_visits.copy()

        # Define common parameters
        cls.prediction_time = (4, 0)  # 4 hours after arrival
        cls.exclude_from_training_data = [
            "snapshot_time",
            "visit_number",
            "prediction_time",
        ]
        cls.grid = {"max_depth": [3], "learning_rate": [0.1], "n_estimators": [100]}
        cls.ordinal_mappings = {"arrival_method": ["walk-in", "referral", "ambulance"]}

    def test_basic_training(self):
        """Test basic model training with default parameters."""
        model = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            evaluate_on_test=True,  # Explicitly enable test evaluation for this test
        )

        # Check that we got a TrainedClassifier object
        self.assertIsInstance(model, TrainedClassifier)

        # Check that the pipeline was created
        self.assertIsNotNone(model.pipeline)
        self.assertIsInstance(model.pipeline, Pipeline)

        # Check that we have training results
        self.assertIsNotNone(model.training_results)

        # Check that we have test results
        self.assertIsNotNone(model.training_results.test_results)
        self.assertIn("test_auc", model.training_results.test_results)
        self.assertIn("test_logloss", model.training_results.test_results)
        self.assertIn("test_auprc", model.training_results.test_results)

    def test_optional_test_evaluation(self):
        """Test that test evaluation is optional and defaults to False."""
        # Test with evaluate_on_test=False (default) and no test_visits
        model_no_test = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            visit_col="visit_number",
            evaluate_on_test=False,
        )

        # Check that test results are None when not evaluated
        self.assertIsNone(model_no_test.training_results.test_results)

        # Test with evaluate_on_test=True and test_visits provided
        model_with_test = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            evaluate_on_test=True,
        )

        # Check that test results are available when evaluated
        self.assertIsNotNone(model_with_test.training_results.test_results)
        self.assertIn("test_auc", model_with_test.training_results.test_results)
        self.assertIn("test_logloss", model_with_test.training_results.test_results)
        self.assertIn("test_auprc", model_with_test.training_results.test_results)

    def test_test_visits_required_when_evaluate_on_test_true(self):
        """Test that test_visits is required when evaluate_on_test=True."""
        with self.assertRaises(ValueError):
            train_classifier(
                train_visits=self.train_visits,
                valid_visits=self.valid_visits,
                prediction_time=self.prediction_time,
                exclude_from_training_data=self.exclude_from_training_data,
                grid=self.grid,
                ordinal_mappings=self.ordinal_mappings,
                visit_col="visit_number",
                evaluate_on_test=True,  # This should raise an error without test_visits
            )

    def test_balanced_training(self):
        """Test training with balanced data."""
        model = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            use_balanced_training=True,
            majority_to_minority_ratio=1.0,
            evaluate_on_test=True,  # Enable test evaluation for this test
        )

        # Check balance info
        balance_info = model.training_results.balance_info
        self.assertTrue(balance_info["is_balanced"])
        self.assertEqual(balance_info["majority_to_minority_ratio"], 1.0)

        # Check that balanced size is less than or equal to original size
        self.assertLessEqual(
            balance_info["balanced_size"], balance_info["original_size"]
        )

    def test_calibration(self):
        """Test model calibration."""
        model = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            calibrate_probabilities=True,
            calibration_method="sigmoid",
            evaluate_on_test=True,  # Enable test evaluation for this test
        )

        # Check that we have a calibrated pipeline
        self.assertIsNotNone(model.calibrated_pipeline)
        self.assertIsInstance(model.calibrated_pipeline, Pipeline)

        # Check calibration info
        self.assertIsNotNone(model.training_results.calibration_info)
        self.assertEqual(model.training_results.calibration_info["method"], "sigmoid")

    def test_custom_model_class(self):
        """Test training with a custom model class."""
        from sklearn.ensemble import RandomForestClassifier

        model = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid={"n_estimators": [100], "max_depth": [3]},
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            model_class=RandomForestClassifier,
            evaluate_on_test=True,  # Enable test evaluation for this test
        )

        # Check that we got a TrainedClassifier object
        self.assertIsInstance(model, TrainedClassifier)
        self.assertIsNotNone(model.pipeline)

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test missing visit_col when single_snapshot_per_visit is True
        with self.assertRaises(ValueError):
            train_classifier(
                train_visits=self.train_visits,
                valid_visits=self.valid_visits,
                prediction_time=self.prediction_time,
                exclude_from_training_data=self.exclude_from_training_data,
                grid=self.grid,
                ordinal_mappings=self.ordinal_mappings,
                single_snapshot_per_visit=True,
            )

    def test_feature_importance(self):
        """Test that feature importance is captured when available."""
        model = train_classifier(
            train_visits=self.train_visits,
            valid_visits=self.valid_visits,
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            test_visits=self.test_visits,
            visit_col="visit_number",
            evaluate_on_test=True,  # Enable test evaluation for this test
        )

        # Check that feature information is captured
        self.assertIsNotNone(model.training_results.training_info)
        self.assertIn("features", model.training_results.training_info)
        features_info = model.training_results.training_info["features"]

        # Check feature names and importances
        self.assertIn("names", features_info)
        self.assertIn("importances", features_info)
        self.assertIn("has_importance_values", features_info)

        # For XGBoost, we should have importance values
        self.assertTrue(features_info["has_importance_values"])
        self.assertEqual(len(features_info["names"]), len(features_info["importances"]))


if __name__ == "__main__":
    unittest.main()
