import unittest
import numpy as np
import pandas as pd
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator


class TestTransferProbabilityEstimator(unittest.TestCase):
    """Test suite for TransferProbabilityEstimator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.subspecialties = {"cardiology", "surgery", "medicine", "oncology"}

    def test_fit_and_calculate_transfer_probabilities(self):
        """Test basic fitting and calculation of transfer probabilities."""
        X = pd.DataFrame(
            {
                "current_subspecialty": [
                    "cardiology",
                    "cardiology",
                    "surgery",
                    "surgery",
                    "medicine",
                ],
                "next_subspecialty": [
                    "surgery",
                    None,
                    "medicine",
                    None,
                    None,
                ],  # None = discharge
            }
        )
        predictor = TransferProbabilityEstimator()
        predictor.fit(X, self.subspecialties)

        # Check cardiology: 1 transfer out of 2 departures = 50%
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology"), 0.5)
        # Check surgery: 1 transfer out of 2 departures = 50%
        self.assertAlmostEqual(predictor.get_transfer_prob("surgery"), 0.5)
        # Check medicine: 0 transfers out of 1 departure = 0%
        self.assertAlmostEqual(predictor.get_transfer_prob("medicine"), 0.0)
        # Check oncology: no data = 0%
        self.assertAlmostEqual(predictor.get_transfer_prob("oncology"), 0.0)

    def test_destination_distribution_calculation(self):
        """Test calculation of destination distributions."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"] * 10,
                "next_subspecialty": ["surgery"] * 5
                + ["medicine"] * 3
                + ["oncology"] * 2,
            }
        )
        predictor = TransferProbabilityEstimator()
        predictor.fit(X, self.subspecialties)

        dests = predictor.get_destination_distribution("cardiology")
        self.assertAlmostEqual(dests["surgery"], 0.5)
        self.assertAlmostEqual(dests["medicine"], 0.3)
        self.assertAlmostEqual(dests["oncology"], 0.2)
        self.assertAlmostEqual(sum(dests.values()), 1.0)

    def test_all_transfers_calculation(self):
        """Test when all departures are transfers."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", "medicine"],
            }
        )
        predictor = TransferProbabilityEstimator()
        predictor.fit(X, self.subspecialties)

        self.assertEqual(predictor.get_transfer_prob("cardiology"), 1.0)
        dests = predictor.get_destination_distribution("cardiology")
        self.assertAlmostEqual(dests["surgery"], 0.5)
        self.assertAlmostEqual(dests["medicine"], 0.5)

    def test_no_transfers_calculation(self):
        """Test when all departures are discharges."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "surgery"],
                "next_subspecialty": [None, None],
            }
        )
        predictor = TransferProbabilityEstimator()
        predictor.fit(X, self.subspecialties)

        self.assertEqual(predictor.get_transfer_prob("cardiology"), 0.0)
        self.assertEqual(predictor.get_destination_distribution("cardiology"), {})

    def test_probability_validity(self):
        """Test that computed probabilities are valid (between 0 and 1, sum to 1)."""
        np.random.seed(42)
        n = 500
        sources = np.random.choice(list(self.subspecialties), size=n)
        destinations = np.random.choice(list(self.subspecialties) + [None] * 2, size=n)

        X = pd.DataFrame(
            {
                "current_subspecialty": sources,
                "next_subspecialty": destinations,
            }
        )
        predictor = TransferProbabilityEstimator()
        predictor.fit(X, self.subspecialties)

        for subspecialty in self.subspecialties:
            prob_transfer = predictor.get_transfer_prob(subspecialty)
            self.assertGreaterEqual(prob_transfer, 0.0)
            self.assertLessEqual(prob_transfer, 1.0)

            dest_dist = predictor.get_destination_distribution(subspecialty)
            if len(dest_dist) > 0:
                self.assertAlmostEqual(sum(dest_dist.values()), 1.0)
                for prob in dest_dist.values():
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)

    def test_custom_column_names(self):
        """Test that custom column names work correctly."""
        X = pd.DataFrame(
            {
                "from_ward": ["cardiology", "cardiology"],
                "to_ward": ["surgery", None],
            }
        )
        predictor = TransferProbabilityEstimator(
            source_col="from_ward", destination_col="to_ward"
        )
        predictor.fit(X, self.subspecialties)

        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology"), 0.5)

    def test_error_handling(self):
        """Test essential error conditions."""
        predictor = TransferProbabilityEstimator()

        # Test unfitted error
        with self.assertRaises(ValueError):
            predictor.get_transfer_prob("cardiology")

        # Test missing columns
        X_bad = pd.DataFrame({"wrong_column": ["cardiology"]})
        with self.assertRaises(ValueError):
            predictor.fit(X_bad, self.subspecialties)

        # Test invalid subspecialties type
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": [None],
            }
        )
        with self.assertRaises(TypeError):
            predictor.fit(X, "not_a_collection")

        # Test unknown destination subspecialty
        X_unknown = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", "unknown_subspecialty"],
            }
        )
        with self.assertRaises(ValueError) as context:
            predictor.fit(X_unknown, self.subspecialties)
        self.assertIn("unknown_subspecialty", str(context.exception))

    def test_cohort_functionality_basic(self):
        """Test basic cohort functionality with separate models for each cohort."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology", "surgery", "surgery"],
                "next_subspecialty": ["surgery", None, None, "cardiology"],
                "admission_type": ["elective", "elective", "emergency", "emergency"],
            }
        )
        predictor = TransferProbabilityEstimator(cohort_col="admission_type")
        predictor.fit(X, self.subspecialties)

        # Check that cohorts are properly identified
        cohorts = predictor.get_available_cohorts()
        self.assertEqual(cohorts, {"elective", "emergency"})

        # Check elective cohort: cardiology 1/2 transfers, surgery 0/0 transfers
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology", "elective"), 0.5)
        self.assertAlmostEqual(predictor.get_transfer_prob("surgery", "elective"), 0.0)

        # Check emergency cohort: cardiology 0/0 transfers, surgery 1/2 transfers
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology", "emergency"), 0.0)
        self.assertAlmostEqual(predictor.get_transfer_prob("surgery", "emergency"), 0.5)

    def test_cohort_functionality_no_cohorts(self):
        """Test that predictor works without cohorts (backward compatibility)."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", None],
            }
        )
        predictor = TransferProbabilityEstimator()  # No cohort_col
        predictor.fit(X, self.subspecialties)

        # Should create "all" cohort automatically
        cohorts = predictor.get_available_cohorts()
        self.assertEqual(cohorts, {"all"})

        # Should work without specifying cohort
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology"), 0.5)

    def test_cohort_functionality_single_cohort_auto_select(self):
        """Test automatic cohort selection when only one cohort exists."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", None],
                "admission_type": ["elective", "elective"],  # Only one cohort
            }
        )
        predictor = TransferProbabilityEstimator(cohort_col="admission_type")
        predictor.fit(X, self.subspecialties)

        # Should automatically use the single cohort
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology"), 0.5)
        self.assertAlmostEqual(predictor.get_transfer_prob("cardiology", "elective"), 0.5)

    def test_cohort_functionality_multiple_cohorts_require_specification(self):
        """Test that multiple cohorts require explicit specification."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology"],
                "next_subspecialty": ["surgery", None],
                "admission_type": ["elective", "emergency"],
            }
        )
        predictor = TransferProbabilityEstimator(cohort_col="admission_type")
        predictor.fit(X, self.subspecialties)

        # Should raise error when no cohort specified with multiple cohorts
        with self.assertRaises(ValueError) as context:
            predictor.get_transfer_prob("cardiology")
        self.assertIn("Multiple cohorts available", str(context.exception))

    def test_cohort_functionality_error_handling(self):
        """Test cohort-related error conditions."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": [None],
                "admission_type": ["elective"],
            }
        )
        predictor = TransferProbabilityEstimator(cohort_col="admission_type")
        predictor.fit(X, self.subspecialties)

        # Test invalid cohort
        with self.assertRaises(ValueError) as context:
            predictor.get_transfer_prob("cardiology", "invalid_cohort")
        self.assertIn("Cohort 'invalid_cohort' not found", str(context.exception))

        # Test missing cohort column
        X_no_cohort = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": [None],
            }
        )
        predictor_with_cohort = TransferProbabilityEstimator(cohort_col="admission_type")
        with self.assertRaises(ValueError) as context:
            predictor_with_cohort.fit(X_no_cohort, self.subspecialties)
        self.assertIn("missing required columns", str(context.exception))

    def test_cohort_functionality_validation(self):
        """Test cohort values validation."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": [None],
                "admission_type": ["elective"],
            }
        )
        
        # Test with valid cohort_values
        predictor = TransferProbabilityEstimator(
            cohort_col="admission_type", 
            cohort_values=["elective", "emergency"]
        )
        predictor.fit(X, self.subspecialties)  # Should not raise

        # Test with invalid cohort in data
        X_invalid = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "next_subspecialty": [None],
                "admission_type": ["invalid_type"],
            }
        )
        predictor_with_validation = TransferProbabilityEstimator(
            cohort_col="admission_type", 
            cohort_values=["elective", "emergency"]
        )
        with self.assertRaises(ValueError) as context:
            predictor_with_validation.fit(X_invalid, self.subspecialties)
        self.assertIn("Found cohort values in data that are not in the cohort_values parameter", str(context.exception))

    def test_cohort_functionality_transition_matrix(self):
        """Test transition matrix generation for cohorts."""
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology", "surgery"],
                "next_subspecialty": ["surgery", None, "cardiology"],
                "admission_type": ["elective", "elective", "emergency"],
            }
        )
        predictor = TransferProbabilityEstimator(cohort_col="admission_type")
        predictor.fit(X, self.subspecialties)

        # Test elective cohort matrix
        matrix_elective = predictor.get_transition_matrix("elective")
        self.assertIn("Discharge", matrix_elective.columns)
        self.assertIn("cardiology", matrix_elective.index)
        self.assertIn("surgery", matrix_elective.index)
        
        # Check that rows sum to 1.0
        for idx in matrix_elective.index:
            self.assertAlmostEqual(matrix_elective.loc[idx].sum(), 1.0, places=10)

        # Test emergency cohort matrix
        matrix_emergency = predictor.get_transition_matrix("emergency")
        self.assertIn("Discharge", matrix_emergency.columns)
        
        # Check that rows sum to 1.0
        for idx in matrix_emergency.index:
            self.assertAlmostEqual(matrix_emergency.loc[idx].sum(), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
