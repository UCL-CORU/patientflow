"""Tests for typed evaluation adapter helpers."""

from datetime import date
import unittest

import numpy as np
import pandas as pd

from patientflow.evaluate.adapters import (
    from_legacy_prob_dist_dict,
    to_legacy_prob_dist_dict_all,
)
from patientflow.evaluate.types import SnapshotResult
from patientflow.load import get_model_key


class TestEvaluationAdapters(unittest.TestCase):
    """Contract tests for typed <-> legacy adapter conversions."""

    def test_from_legacy_prob_dist_dict(self):
        dt = date(2026, 1, 1)
        legacy = {
            dt: {
                "agg_predicted": pd.DataFrame(
                    {"agg_proba": [0.2, 0.5, 0.3]},
                    index=[0, 1, 2],
                ),
                "agg_observed": 1,
            }
        }
        converted = from_legacy_prob_dist_dict(legacy)
        self.assertIn(dt, converted)
        self.assertEqual(converted[dt].observed, 1)
        self.assertEqual(converted[dt].offset, 0)
        np.testing.assert_allclose(converted[dt].predicted_pmf, np.array([0.2, 0.5, 0.3]))

    def test_to_legacy_prob_dist_dict_all_preserves_offset(self):
        pt = (9, 30)
        dt = date(2026, 1, 2)
        typed = {
            pt: {
                dt: SnapshotResult(
                    predicted_pmf=np.array([0.1, 0.8, 0.1]),
                    observed=0,
                    offset=-1,
                )
            }
        }
        legacy = to_legacy_prob_dist_dict_all(typed, model_name="combined_net_elective")
        key = get_model_key("combined_net_elective", pt)
        self.assertIn(key, legacy)
        entry = legacy[key][dt]
        self.assertEqual(entry["agg_observed"], 0)
        self.assertEqual(list(entry["agg_predicted"].index), [-1, 0, 1])
        np.testing.assert_allclose(
            entry["agg_predicted"]["agg_proba"].to_numpy(),
            np.array([0.1, 0.8, 0.1]),
        )


if __name__ == "__main__":
    unittest.main()
