import unittest
import pandas as pd
import numpy as np

from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)


class TestSequenceToOutcomePredictor(unittest.TestCase):
    """Test cases for SequenceToOutcomePredictor class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        options = ["acute", "surgical", "medical", "paediatric"]

        input_vars = []
        grouping_vars = []
        for _ in range(n_samples):
            # input_var: tuple of random length (including 0 for empty tuple)
            input_len = np.random.choice([0, 1, 2])
            input_seq = tuple(np.random.choice(options, size=input_len, replace=True))
            input_vars.append(input_seq)
            # grouping_var: must start with input_var, and can be longer
            group_len = np.random.randint(
                input_len, input_len + 3
            )  # allow up to 2 more
            if group_len == input_len:
                grouping_vars.append(input_seq)
            else:
                extension = tuple(
                    np.random.choice(options, size=group_len - input_len, replace=True)
                )
                grouping_vars.append(input_seq + extension)

        outcome_options = options
        self.test_data = pd.DataFrame(
            {
                "snapshot_date": pd.date_range(
                    "2023-01-01", periods=n_samples, freq="h"
                ),
                "input_var": input_vars,
                "grouping_var": grouping_vars,
                "outcome_var": np.random.choice(outcome_options, size=n_samples),
                "is_admitted": np.random.choice([True, False], size=n_samples),
                "age_on_arrival": np.random.randint(1, 100, size=n_samples),
                "sex": np.random.choice(["M", "F"], size=n_samples),
            }
        )

    def test_initialization(self):
        """Test SequenceToOutcomePredictor initialization."""
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=True,
            admit_col="is_admitted",
        )

        repr_str = repr(predictor)
        self.assertIn("SequenceToOutcomePredictor", repr_str)
        self.assertIn("input_var='input_var'", repr_str)
        self.assertIn("grouping_var='grouping_var'", repr_str)
        self.assertIn("outcome_var='outcome_var'", repr_str)
        self.assertIn("apply_special_category_filtering=True", repr_str)
        self.assertIn("admit_col='is_admitted'", repr_str)

    def test_ensure_tuple(self):
        """Test the _ensure_tuple method."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
        )

        # Test None input
        self.assertEqual(predictor._ensure_tuple(None), ())

        # Test list input
        self.assertEqual(
            predictor._ensure_tuple(["medical", "surgical"]), ("medical", "surgical")
        )

        # Test tuple input
        self.assertEqual(
            predictor._ensure_tuple(("medical", "surgical")), ("medical", "surgical")
        )

        # Test empty list
        self.assertEqual(predictor._ensure_tuple([]), ())

        # Test empty tuple
        self.assertEqual(predictor._ensure_tuple(()), ())

        # Test with quoted strings
        self.assertEqual(
            predictor._ensure_tuple(["'medical'", "'surgical'"]),
            ("medical", "surgical"),
        )

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        fitted_predictor = predictor.fit(self.test_data)

        # Check that weights contain expected keys (tuples)
        for key in fitted_predictor.weights.keys():
            self.assertIsInstance(key, tuple)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for known input sequence
        test_sequence = ("medical", "surgical")
        prediction = predictor.predict(test_sequence)
        self.assertIsInstance(prediction, dict)

        # Check that prediction contains expected outcome categories
        expected_outcomes = ["medical", "surgical", "paediatric"]
        for outcome in expected_outcomes:
            if outcome in prediction:
                self.assertIsInstance(prediction[outcome], (int, float))
                self.assertGreaterEqual(prediction[outcome], 0.0)
                self.assertLessEqual(prediction[outcome], 1.0)

    def test_predict_empty_sequence(self):
        """Test prediction with empty sequence."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for empty sequence
        prediction = predictor.predict(())
        self.assertIsInstance(prediction, dict)

    def test_predict_none_input(self):
        """Test prediction with None input."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for None input
        prediction = predictor.predict(None)
        self.assertIsInstance(prediction, dict)

    def test_predict_unknown_sequence(self):
        """Test prediction with unknown sequence."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction for unknown sequence
        prediction = predictor.predict(("unknown", "sequence"))
        self.assertIsInstance(prediction, dict)

    def test_predict_sequence_fallback(self):
        """Test prediction fallback behavior for partial sequence matches."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction with a sequence that might not exist but has a prefix that does
        long_sequence = ("medical", "surgical", "paediatric", "haem/onc")
        prediction = predictor.predict(long_sequence)
        self.assertIsInstance(prediction, dict)

    def test_fit_with_special_category_filtering(self):
        """Test fitting with special category filtering enabled."""
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test predictions for some input sequences
        test_sequences = [
            ("medical",),
            ("surgical",),
            ("paediatric",),
            ("medical", "surgical"),
        ]
        for sequence in test_sequences:
            prediction = predictor.predict(sequence)
            if prediction:  # If prediction is not empty
                total_prob = sum(prediction.values())
                # Allow for small floating point errors
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
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
        predictor = SequenceToOutcomePredictor(
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
        if not predictor.input_to_grouping_probs.empty:
            # Drop the probability column from the DataFrame
            probs = predictor.input_to_grouping_probs.drop(
                columns=["probability_of_grouping_sequence"]
            )
            # Check that probabilities sum to 1.0 for each input sequence
            for idx, row in probs.iterrows():
                prob_sum = row.sum()
                self.assertAlmostEqual(prob_sum, 1.0, places=10)

    def test_sequence_string_matching(self):
        """Test the string matching functionality for sequences."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test that the model can handle different sequence formats
        test_sequences = [
            ("medical",),
            ("medical", "surgical"),
            ("medical", "surgical", "paediatric"),
            (),
            None,
        ]

        for sequence in test_sequences:
            prediction = predictor.predict(sequence)
            self.assertIsInstance(prediction, dict)

    def test_list_to_tuple_conversion(self):
        """Test that lists are properly converted to tuples."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Fit the model
        predictor.fit(self.test_data)

        # Test prediction with list input (should be converted to tuple)
        list_sequence = ["medical", "surgical"]
        prediction = predictor.predict(list_sequence)
        self.assertIsInstance(prediction, dict)

    def test_quoted_string_handling(self):
        """Test handling of quoted strings in sequences."""
        predictor = SequenceToOutcomePredictor(
            input_var="input_var",
            grouping_var="grouping_var",
            outcome_var="outcome_var",
            apply_special_category_filtering=False,
            admit_col="is_admitted",
        )

        # Test _ensure_tuple with quoted strings
        quoted_sequence = ["'medical'", "'surgical'"]
        cleaned_sequence = predictor._ensure_tuple(quoted_sequence)
        self.assertEqual(cleaned_sequence, ("medical", "surgical"))


if __name__ == "__main__":
    unittest.main()
