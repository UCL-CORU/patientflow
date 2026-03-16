"""Tests for flat scalar collection helpers."""

import unittest

import pandas as pd

from patientflow.evaluation.scalars import ScalarsCollector, default_scalars_meta


class TestScalarsCollector(unittest.TestCase):
    """Tests for flat scalar record collection."""

    def test_record_and_payload_shape(self):
        collector = ScalarsCollector()
        collector.record(
            flow="ed_current_beds",
            service="medical",
            component="arrivals",
            prediction_time="0930",
            flow_type="pmf",
            aspirational=False,
            evaluated=True,
            metrics={"mae": 1.2, "mpe": 8.0},
        )
        payload = collector.to_payload(
            default_scalars_meta({"classifier_positive_cases": 50})
        )
        self.assertIn("_meta", payload)
        self.assertIn("results", payload)
        self.assertEqual(len(payload["results"]), 1)
        row = payload["results"][0]
        self.assertEqual(row["flow"], "ed_current_beds")
        self.assertAlmostEqual(row["mae"], 1.2)
        self.assertAlmostEqual(row["mpe"], 8.0)

    def test_dataframe_compatibility(self):
        collector = ScalarsCollector()
        collector.record(
            flow="ed_current_admission_classifier",
            service=None,
            component="classifier",
            prediction_time="0930",
            flow_type="classifier",
            aspirational=False,
            evaluated=True,
            metrics={"auroc": 0.88},
        )
        collector.record(
            flow="combined_net_emergency",
            service="medical",
            component="net_flow",
            prediction_time="0930",
            flow_type="pmf",
            aspirational=True,
            evaluated=False,
            reason="Aspirational flow: observed-vs-predicted diagnostics skipped",
        )
        payload = collector.to_payload(default_scalars_meta({"distribution_snapshots": 30}))
        df = pd.DataFrame(payload["results"])
        self.assertEqual(df.shape[0], 2)
        self.assertIn("flow", df.columns)
        self.assertIn("service", df.columns)
        self.assertIn("component", df.columns)
        self.assertIn("prediction_time", df.columns)
        self.assertIn("aspirational", df.columns)
        self.assertIn("evaluated", df.columns)
        self.assertIn("auroc", df.columns)
        self.assertTrue(df.loc[df["flow"] == "combined_net_emergency", "aspirational"].item())

    def test_default_meta_contains_expected_keys(self):
        meta = default_scalars_meta({"distribution_snapshots": 30}, schema_version=3)
        self.assertEqual(meta["schema_version"], 3)
        self.assertIn("generated_at_utc", meta)
        self.assertEqual(meta["reliability_thresholds"]["distribution_snapshots"], 30)


if __name__ == "__main__":
    unittest.main()
