import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from patientflow.train.classifiers import (
    FeatureColumnTransformer,
    FeatureKind,
    create_column_transformer,
    infer_feature_kind,
    train_classifier,
)
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
                # Use object dtype for categoricals so they are handled by the
                # column transformer as intended (one-hot encoded where appropriate).
                "sex": pd.Series(
                    np.random.choice(["M", "F"], n_samples), dtype="object"
                ),
                "arrival_method": pd.Series(
                    np.random.choice(["ambulance", "walk-in", "referral"], n_samples),
                    dtype="object",
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

    def test_string_dtype_columns(self):
        """Test training when categorical columns use pandas StringDtype."""
        visits = self.train_visits.copy()
        visits["sex"] = visits["sex"].astype("string")
        visits["arrival_method"] = visits["arrival_method"].astype("string")

        model = train_classifier(
            train_visits=visits,
            valid_visits=visits.copy(),
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            visit_col="visit_number",
        )
        self.assertIsInstance(model, TrainedClassifier)
        self.assertIsNotNone(model.pipeline)

    def test_categorical_dtype_columns(self):
        """Test training when categorical columns use CategoricalDtype."""
        visits = self.train_visits.copy()
        visits["sex"] = visits["sex"].astype("category")
        visits["arrival_method"] = visits["arrival_method"].astype("category")

        model = train_classifier(
            train_visits=visits,
            valid_visits=visits.copy(),
            prediction_time=self.prediction_time,
            exclude_from_training_data=self.exclude_from_training_data,
            grid=self.grid,
            ordinal_mappings=self.ordinal_mappings,
            visit_col="visit_number",
        )
        self.assertIsInstance(model, TrainedClassifier)
        self.assertIsNotNone(model.pipeline)

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

    def test_timedelta_column_uses_standard_scaler_not_one_hot(self):
        """timedelta64 features must be scaled (via seconds), not one-hot encoded."""
        df = pd.DataFrame(
            {
                "elapsed_los": pd.to_timedelta(np.arange(10), unit="h"),
                "sex": pd.Series(["M", "F"] * 5, dtype="object"),
            }
        )
        ct = create_column_transformer(df)
        by_col = {cols[0]: trans for _, trans, cols in ct.transformers}
        self.assertIsInstance(by_col["elapsed_los"], Pipeline)
        self.assertIsInstance(
            by_col["elapsed_los"].named_steps["scale"], StandardScaler
        )
        self.assertIsInstance(by_col["sex"], OneHotEncoder)

    def test_timedelta_column_transform_matches_float_seconds(self):
        """After fit on timedelta, transform accepts the same durations as float seconds."""
        df_td = pd.DataFrame({"elapsed_los": pd.to_timedelta([1, 2, 3], unit="h")})
        df_sec = pd.DataFrame({"elapsed_los": [3600.0, 7200.0, 10800.0]})
        ct: ColumnTransformer = create_column_transformer(df_td)
        ct.fit(df_td)
        out_td = ct.transform(df_td)
        out_sec = ct.transform(df_sec)
        np.testing.assert_allclose(out_td, out_sec, rtol=1e-10, atol=1e-10)

    def test_timedelta_binary_column_still_scaled_seconds(self):
        """Durations are always seconds + scaler (no binary passthrough)."""
        df = pd.DataFrame(
            {
                "elapsed_los": pd.to_timedelta([1, 2] * 5, unit="h"),
            }
        )
        ct = create_column_transformer(df)
        _, trans, cols = ct.transformers[0]
        self.assertEqual(cols, ["elapsed_los"])
        self.assertIsInstance(trans, Pipeline)
        self.assertIsInstance(trans.named_steps["scale"], StandardScaler)

    def test_infer_feature_kind_categorical_with_numeric_categories(self):
        """Pandas categorical with numeric categories follows numeric routing."""
        s = pd.Series(pd.Categorical.from_codes([0, 1, 0], categories=[1, 2]))
        self.assertEqual(infer_feature_kind(s, "x", {}), FeatureKind.NUMERIC_BINARY)

    def test_infer_feature_kind_two_string_object_is_categorical(self):
        s = pd.Series(["a", "b"] * 3, dtype="object")
        self.assertEqual(infer_feature_kind(s, "x", {}), FeatureKind.CATEGORICAL)

    def test_feature_column_transformer_timedelta_default(self):
        """Missing timedelta columns are filled with pd.Timedelta(0)."""
        fit_df = pd.DataFrame({"elapsed_los": pd.to_timedelta([1, 2, 3], unit="h")})
        fct = FeatureColumnTransformer()
        fct.fit(fit_df)
        self.assertEqual(fct.column_defaults_["elapsed_los"], pd.Timedelta(0))

        out = fct.transform(pd.DataFrame({"other": [1, 2]}))
        self.assertIn("elapsed_los", out.columns)
        self.assertTrue(
            (out["elapsed_los"] == pd.Timedelta(0)).all(),
            msg="expected zero timedelta fill for missing column",
        )


if __name__ == "__main__":
    unittest.main()
