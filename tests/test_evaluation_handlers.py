"""Tests for evaluation mode handlers."""

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from patientflow.evaluate.handlers import (
    evaluate_aspirational_skip,
    evaluate_arrival_deltas,
    evaluate_distribution,
)
from patientflow.evaluate.scalars import ScalarsCollector
from patientflow.evaluate.types import (
    ArrivalDeltaPayload,
    EvaluationTarget,
    SnapshotResult,
)


class TestEvaluationHandlers(unittest.TestCase):
    """Behaviour tests for mode handlers."""

    @patch("patientflow.evaluate.handlers.plot_deltas", return_value=None)
    @patch("patientflow.evaluate.handlers.plot_epudd", return_value=None)
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

    @patch("patientflow.evaluate.handlers.plot_arrival_deltas", return_value=None)
    def test_evaluate_arrival_deltas_skips_all_empty_snapshot_windows(
        self, mock_plot
    ) -> None:
        target = EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            evaluation_mode="arrival_deltas",
            component="arrivals",
        )
        collector = ScalarsCollector()
        payloads_by_time = {
            (9, 30): ArrivalDeltaPayload(
                df=pd.DataFrame(
                    {
                        "arrival_datetime": [
                            pd.Timestamp("2026-01-01 06:00:00+00:00"),
                            pd.Timestamp("2026-01-02 06:30:00+00:00"),
                        ]
                    }
                ),
                snapshot_dates=[date(2026, 1, 1), date(2026, 1, 2)],
                prediction_window=timedelta(hours=2),
                yta_time_interval=timedelta(minutes=15),
            ),
        }

        with TemporaryDirectory() as tmpdir:
            evaluate_arrival_deltas(
                service_name="medical",
                flow_name="ed_yta_arrival_rates",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                payloads_by_time=payloads_by_time,
            )

        mock_plot.assert_not_called()
        self.assertEqual(len(collector.rows), 1)
        row = collector.rows[0]
        self.assertTrue(row["evaluated"])
        self.assertFalse(row["charts_generated"])
        self.assertEqual(row["skip_reason"], "inactive_service")

    @patch("patientflow.evaluate.handlers.plot_arrival_deltas", return_value=None)
    def test_evaluate_arrival_deltas_plots_when_any_snapshot_has_arrivals(
        self, mock_plot
    ) -> None:
        target = EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            evaluation_mode="arrival_deltas",
            component="arrivals",
        )
        collector = ScalarsCollector()
        payloads_by_time = {
            (9, 30): ArrivalDeltaPayload(
                df=pd.DataFrame(
                    {
                        "arrival_datetime": [
                            pd.Timestamp("2026-01-01 10:00:00+00:00"),
                            pd.Timestamp("2026-01-02 06:30:00+00:00"),
                        ]
                    }
                ),
                snapshot_dates=[date(2026, 1, 1), date(2026, 1, 2)],
                prediction_window=timedelta(hours=2),
                yta_time_interval=timedelta(minutes=15),
            ),
        }

        with TemporaryDirectory() as tmpdir:
            evaluate_arrival_deltas(
                service_name="medical",
                flow_name="ed_yta_arrival_rates",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                payloads_by_time=payloads_by_time,
            )

        mock_plot.assert_called_once()
        self.assertEqual(len(collector.rows), 1)
        row = collector.rows[0]
        self.assertTrue(row["evaluated"])
        self.assertIsNone(row.get("charts_generated"))

    @patch("patientflow.evaluate.handlers.plot_arrival_deltas", return_value=None)
    def test_evaluate_arrival_deltas_passes_predictor_to_plot(self, mock_plot) -> None:
        target = EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            evaluation_mode="arrival_deltas",
            component="arrivals",
        )
        collector = ScalarsCollector()
        fake_predictor = object()
        payloads_by_time = {
            (9, 30): ArrivalDeltaPayload(
                df=pd.DataFrame(
                    {
                        "arrival_datetime": [
                            pd.Timestamp("2026-01-01 10:00:00+00:00"),
                        ]
                    }
                ),
                snapshot_dates=[date(2026, 1, 1)],
                prediction_window=timedelta(hours=2),
                yta_time_interval=timedelta(minutes=15),
                predictor=fake_predictor,
                filter_key="all",
                strict_prediction_date=True,
            ),
        }

        with TemporaryDirectory() as tmpdir:
            evaluate_arrival_deltas(
                service_name="medical",
                flow_name="ed_yta_arrival_rates",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                payloads_by_time=payloads_by_time,
            )

        mock_plot.assert_called_once()
        kwargs = mock_plot.call_args.kwargs
        self.assertIs(kwargs["predictor"], fake_predictor)
        self.assertEqual(kwargs["filter_key"], "all")
        self.assertTrue(kwargs["strict_prediction_date"])

    @patch("patientflow.evaluate.handlers.plot_deltas", return_value=None)
    @patch("patientflow.evaluate.handlers.plot_epudd", return_value=None)
    def test_evaluate_distribution_emits_per_weekday_rows(
        self, _mock_epudd, _mock_deltas
    ) -> None:
        target = EvaluationTarget(
            name="ed_yta_beds",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
        )
        collector = ScalarsCollector()
        # 20 snapshots: 10 Mondays and 10 Tuesdays (alternating)
        snapshots = {}
        d = date(2026, 1, 5)  # Monday
        for i in range(20):
            snapshots[d + timedelta(days=i)] = SnapshotResult(
                predicted_pmf=np.array([0.5, 0.5]),
                observed=1 if i % 2 == 0 else 0,
            )
        snapshots_by_time = {(9, 30): snapshots}

        with TemporaryDirectory() as tmpdir:
            evaluate_distribution(
                service_name="medical",
                flow_name="ed_yta_beds",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                snapshots_by_time=snapshots_by_time,
                group_by=("weekday",),
                min_group_snapshots=2,
            )

        all_rows = [r for r in collector.rows if r["group_weekday"] == "all"]
        self.assertEqual(len(all_rows), 1)
        weekday_rows = [r for r in collector.rows if r["group_weekday"] != "all"]
        # At least some weekdays present (spans 20 consecutive days => all 7)
        self.assertTrue(len(weekday_rows) >= 3)
        for r in weekday_rows:
            self.assertIn(
                r["group_weekday"], {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
            )

    @patch("patientflow.evaluate.handlers.plot_arrival_deltas", return_value=None)
    def test_evaluate_arrival_deltas_emits_per_weekday_plots(self, mock_plot) -> None:
        target = EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            evaluation_mode="arrival_deltas",
            component="arrivals",
        )
        collector = ScalarsCollector()
        snapshot_dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(14)]
        arrivals = [
            pd.Timestamp(f"2026-01-{5 + i:02d} 10:00:00+00:00") for i in range(14)
        ]
        payloads_by_time = {
            (9, 30): ArrivalDeltaPayload(
                df=pd.DataFrame({"arrival_datetime": arrivals}),
                snapshot_dates=snapshot_dates,
                prediction_window=timedelta(hours=2),
                yta_time_interval=timedelta(minutes=15),
            ),
        }

        with TemporaryDirectory() as tmpdir:
            evaluate_arrival_deltas(
                service_name="medical",
                flow_name="ed_yta_arrival_rates",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                payloads_by_time=payloads_by_time,
                group_by=("weekday",),
                min_group_snapshots=2,
            )

        # One headline + one per weekday (7 weekdays with >=2 snapshots each)
        self.assertGreaterEqual(mock_plot.call_count, 8)
        weekday_rows = [r for r in collector.rows if r["group_weekday"] != "all"]
        self.assertEqual(len(weekday_rows), 7)

    @patch("patientflow.evaluate.handlers.plot_deltas", return_value=None)
    @patch("patientflow.evaluate.handlers.plot_epudd", return_value=None)
    def test_evaluate_distribution_skips_small_weekday_groups(
        self, _mock_epudd, _mock_deltas
    ) -> None:
        target = EvaluationTarget(
            name="ed_yta_beds",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
        )
        collector = ScalarsCollector()
        # 1 snapshot for Monday only
        snapshots = {
            date(2026, 1, 5): SnapshotResult(
                predicted_pmf=np.array([0.5, 0.5]), observed=1
            )
        }
        snapshots_by_time = {(9, 30): snapshots}

        with TemporaryDirectory() as tmpdir:
            evaluate_distribution(
                service_name="medical",
                flow_name="ed_yta_beds",
                target=target,
                prediction_times=[(9, 30)],
                collector=collector,
                output_root=Path(tmpdir),
                snapshots_by_time=snapshots_by_time,
                group_by=("weekday",),
                min_group_snapshots=10,
            )

        weekday_rows = [r for r in collector.rows if r["group_weekday"] != "all"]
        self.assertEqual(len(weekday_rows), 1)
        row = weekday_rows[0]
        self.assertEqual(row["group_weekday"], "mon")
        self.assertFalse(row["evaluated"])
        self.assertIn("below min_group_snapshots", row["reason"])


if __name__ == "__main__":
    unittest.main()
