"""Tests for evaluation mode handlers."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from patientflow.evaluation.handlers import (
    evaluate_aspirational_skip,
    evaluate_distribution,
)
from patientflow.evaluation.scalars import ScalarsCollector
from patientflow.evaluation.types import EvaluationTarget, SnapshotResult


class TestEvaluationHandlers(unittest.TestCase):
    """Behaviour tests for mode handlers."""

    @patch("patientflow.evaluation.handlers.plot_deltas", return_value=None)
    @patch("patientflow.evaluation.handlers.plot_epudd", return_value=None)
    def test_evaluate_distribution_records_metrics(
        self, _mock_epudd, _mock_deltas
    ) -> None:
        target = EvaluationTarget(
            name="ed_current_beds",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
        )
        collector = ScalarsCollector()
        prediction_times = [(9, 30)]
        snapshots_by_time = {
            (9, 30): {
                date(2026, 1, 1): SnapshotResult(
                    predicted_pmf=np.array([0.0, 1.0]),
                    observed=1,
                ),
                date(2026, 1, 2): SnapshotResult(
                    predicted_pmf=np.array([0.0, 1.0]),
                    observed=1,
                ),
            }
        }

        with TemporaryDirectory() as tmpdir:
            evaluate_distribution(
                service_name="medical",
                flow_name="ed_current_beds",
                target=target,
                prediction_times=prediction_times,
                collector=collector,
                output_root=Path(tmpdir),
                snapshots_by_time=snapshots_by_time,
            )

        self.assertEqual(len(collector.rows), 1)
        row = collector.rows[0]
        self.assertTrue(row["evaluated"])
        self.assertEqual(row["flow"], "ed_current_beds")
        self.assertEqual(row["service"], "medical")
        self.assertEqual(row["component"], "arrivals")
        self.assertEqual(row["prediction_time"], "0930")
        self.assertAlmostEqual(row["mae"], 0.0)
        self.assertAlmostEqual(row["mpe"], 0.0)
        self.assertEqual(row["n_snapshots"], 2)
        self.assertFalse(row["reliable"])

    def test_evaluate_aspirational_skip_records_not_evaluated(self) -> None:
        target = EvaluationTarget(
            name="combined_net_emergency",
            flow_type="pmf",
            evaluation_mode="aspirational_skip",
            component="net_flow",
        )
        collector = ScalarsCollector()
        evaluate_aspirational_skip(
            service_name="medical",
            flow_name="combined_net_emergency",
            target=target,
            prediction_times=[(9, 30), (12, 0)],
            collector=collector,
        )

        self.assertEqual(len(collector.rows), 2)
        for row in collector.rows:
            self.assertFalse(row["evaluated"])
            self.assertTrue(row["aspirational"])
            self.assertIn("Aspirational flow", row["reason"])


if __name__ == "__main__":
    unittest.main()
