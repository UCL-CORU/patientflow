import unittest
import pandas as pd
import numpy as np

from patientflow.predictors.value_to_outcome_predictor import ValueToOutcomePredictor


class TestValueToOutcomePredictor(unittest.TestCase):
    """Test cases for ValueToOutcomePredictor class."""

    def setUp(self):
        """Set up test data."""
        # Create test data with categorical variables
        np.random.seed(42)
        n_samples = 1000

        # Generate test data
        self.test_data = pd.DataFrame(
            {
                "snapshot_date": pd.date_range(
                    "2023-01-01", periods=n_samples, freq="h"
                ),
                "input_var": np.random.choice(["A", "B", "C", "D"], size=n_samples),
                "grouping_var": np.random.choice(["X", "Y", "Z"], size=n_samples),
                "outcome_var": np.random.choice(
                    ["medical", "surgical", "paediatric"], size=n_samples
                ),
                "is_admitted": np.random.choice([True, False], size=n_samples),
                "age_on_arrival": np.random.randint(1, 100, size=n_samples),
                "sex": np.random.choice(["M", "F"], size=n_samples),
            }
        )

        # Create some null values for testing
        self.test_data.loc[10:20, "input_var"] = None
        self.test_data.loc[30:40, "grouping_var"] = None

    def test_initialization(self):
        """Test ValueToOutcomePredictor initialization."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        self.assertEqual(predictor.input_var, "input_var")
        self.assertEqual(predictor.grouping_var, "grouping_var")
        self.assertEqual(predictor.outcome_var, "outcome_var")
        self.assertFalse(predictor.apply_special_category_filtering)
        self.assertEqual(predictor.admit_col, "is_admitted")
        self.assertIsNone(predictor.weights)
        self.assertIsNone(predictor.special_params)
        self.assertEqual(predictor.metrics, {})

    def test_repr(self):
        """Test string representation of the predictor."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=True,
            admit_col="is_admitted",
        )

        repr_str = repr(predictor)
        self.assertIn("ValueToOutcomePredictor", repr_str)
        self.assertIn("input_var='input_var'", repr_str)
        self.assertIn("grouping_var='grouping_var'", repr_str)
        self.assertIn("outcome_var='outcome_var'", repr_str)
        self.assertIn("apply_special_category_filtering=True", repr_str)
        self.assertIn("admit_col='is_admitted'", repr_str)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        fitted_predictor = predictor.fit(self.test_data)

        # Check that the model was fitted
        self.assertIsNotNone(fitted_predictor.weights)
        self.assertIsInstance(fitted_predictor.weights, dict)
        self.assertIsNotNone(fitted_predictor.input_to_grouping_probs)
        self.assertIsInstance(fitted_predictor.input_to_grouping_probs, pd.DataFrame)

        # Check metrics
        self.assertIn("train_dttm", fitted_predictor.metrics)
        self.assertIn("train_set_no", fitted_predictor.metrics)
        self.assertEqual(fitted_predictor.metrics["train_set_no"], len(self.test_data))

    def test_fit_with_admission_filtering(self):
        """Test fitting with admission filtering."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        fitted_predictor = predictor.fit(self.test_data)

        # Check that weights contain expected keys
        expected_inputs = ["A", "B", "C", "D", ""]  # Including empty string for nulls
        for input_val in expected_inputs:
            if input_val in fitted_predictor.weights:
                self.assertIsInstance(fitted_predictor.weights[input_val], dict)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for known input
        prediction = predictor.predict("A")
        self.assertIsInstance(prediction, dict)

        # Check that prediction contains expected outcome categories
        expected_outcomes = ["medical", "surgical", "paediatric"]
        for outcome in expected_outcomes:
            if outcome in prediction:
                self.assertIsInstance(prediction[outcome], (int, float))
                self.assertGreaterEqual(prediction[outcome], 0.0)
                self.assertLessEqual(prediction[outcome], 1.0)

    def test_predict_null_input(self):
        """Test prediction with null input."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for null input
        prediction = predictor.predict(None)
        self.assertIsInstance(prediction, dict)

        # Should return empty dict or dict with empty string key
        if prediction:
            self.assertIn("", predictor.weights)

    def test_predict_unknown_input(self):
        """Test prediction with unknown input."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for unknown input
        prediction = predictor.predict("UNKNOWN_VALUE")
        self.assertIsInstance(prediction, dict)

        # Should return empty dict or dict with None key
        if prediction:
            self.assertIn(None, predictor.weights)

    def test_fit_with_special_category_filtering(self):
        """Test fitting with special category filtering enabled."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=True,
            admit_col="is_admitted",
        )

        # Fit the model
        fitted_predictor = predictor.fit(self.test_data)

        # Check that special_params was populated
        self.assertIsNotNone(fitted_predictor.special_params)
        self.assertIn("special_func_map", fitted_predictor.special_params)
        self.assertIn("special_category_dict", fitted_predictor.special_params)

    def test_probability_calculation(self):
        """Test that probabilities sum to approximately 1.0."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test predictions for each input value
        for input_val in ["A", "B", "C", "D"]:
            prediction = predictor.predict(input_val)
            if prediction:  # If prediction is not empty
                total_prob = sum(prediction.values())
                # Allow for small floating point errors
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        empty_df = pd.DataFrame(columns=self.test_data.columns)

        # Should not raise an error
        fitted_predictor = predictor.fit(empty_df)
        self.assertIsNotNone(fitted_predictor.metrics)
        self.assertEqual(fitted_predictor.metrics["train_set_no"], 0)

    def test_missing_columns(self):
        """Test handling of missing columns."""
        predictor = ValueToOutcomePredictor(
            input_var="nonexistent_input",
            grouping_var="nonexistent_grouping",
            outcome_var="nonexistent_outcome",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Should raise KeyError when columns don't exist
        with self.assertRaises(KeyError):
            predictor.fit(self.test_data)

    def test_sklearn_compatibility(self):
        """Test that the predictor is compatible with sklearn interfaces."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Test that it has sklearn base classes
        from sklearn.base import BaseEstimator, TransformerMixin

        self.assertIsInstance(predictor, BaseEstimator)
        self.assertIsInstance(predictor, TransformerMixin)

        # Test fit method returns self
        fitted_predictor = predictor.fit(self.test_data)
        self.assertIs(fitted_predictor, predictor)

    def test_input_to_grouping_probabilities(self):
        """Test the input_to_grouping_probs attribute."""
        predictor = ValueToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Check that input_to_grouping_probs is a DataFrame
        self.assertIsInstance(predictor.input_to_grouping_probs, pd.DataFrame)

        # Check that it contains expected columns
        expected_inputs = ["A", "B", "C", "D"]
        for input_val in expected_inputs:
            if input_val in predictor.input_to_grouping_probs.index:
                row = predictor.input_to_grouping_probs.loc[input_val]
                # Check that probabilities sum to 1.0
                prob_sum = row.drop("probability_of_input_value").sum()
                self.assertAlmostEqual(prob_sum, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
