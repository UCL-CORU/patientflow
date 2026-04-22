"""Tests for legacy API wrappers in refactored evaluation package."""

import unittest

import pandas as pd

from patientflow.evaluate.legacy_api import calc_mae_mpe, calculate_results


class TestEvaluationLegacyApi(unittest.TestCase):
    """Ensure legacy helper wrappers remain callable and compatible."""

    def test_calculate_results_wrapper(self) -> None:
        result = calculate_results([10, 20], [12.0, 18.0])
        self.assertIn("mae", result)
        self.assertIn("mpe", result)
        self.assertAlmostEqual(result["mae"], 2.0)

    def test_calc_mae_mpe_wrapper(self) -> None:
        prob_dist_dict_all = {
            "admissions_0930": {
                "2026-01-01": {
                    "agg_predicted": pd.DataFrame(
                        {"agg_proba": [0.0, 1.0]}, index=[0, 1]
                    ),
                    "agg_observed": 1,
                }
            }
        }
        result = calc_mae_mpe(prob_dist_dict_all)
        self.assertIn("admissions_0930", result)
        self.assertIn("mae", result["admissions_0930"])


if __name__ == "__main__":
    unittest.main()
