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
                "next_subspecialty": ["surgery"] * 5 + ["medicine"] * 3 + ["oncology"] * 2,
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
        X = pd.DataFrame({
            "current_subspecialty": ["cardiology"],
            "next_subspecialty": [None],
        })
        with self.assertRaises(TypeError):
            predictor.fit(X, "not_a_collection")


if __name__ == "__main__":
    unittest.main()
