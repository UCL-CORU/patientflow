"""Tests for observed-count helpers."""

from datetime import date, timedelta
import unittest

import pandas as pd

from patientflow.evaluate.observations import count_observed


class TestAdmittedAtSomePoint(unittest.TestCase):
    """Tests for the admitted_at_some_point mode."""

    def test_counts_admitted_patients(self) -> None:
        visits = pd.DataFrame(
            {
                "snapshot_date": [date(2026, 1, 1)] * 3,
                "prediction_time": [(10, 0)] * 3,
                "is_admitted": [True, False, True],
                "specialty": ["medical", "medical", "surgical"],
            }
        )
        result = count_observed(
            "admitted_at_some_point",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
        )
        self.assertEqual(result, 1)

    def test_counts_all_specialties_when_none(self) -> None:
        visits = pd.DataFrame(
            {
                "snapshot_date": [date(2026, 1, 1)] * 3,
                "prediction_time": [(10, 0)] * 3,
                "is_admitted": [True, True, True],
                "specialty": ["medical", "medical", "surgical"],
            }
        )
        result = count_observed(
            "admitted_at_some_point",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
        )
        self.assertEqual(result, 3)


class TestAdmittedInWindow(unittest.TestCase):
    """Tests for admitted_in_window and departed_in_window modes."""

    def setUp(self) -> None:
        self.visits = pd.DataFrame(
            {
                "arrival_datetime": [
                    pd.Timestamp("2026-01-01 09:00:00+00:00"),  # present before
                    pd.Timestamp("2026-01-01 09:30:00+00:00"),  # present before
                    pd.Timestamp("2026-01-01 10:30:00+00:00"),  # arrived after moment
                ],
                "departure_datetime": [
                    pd.Timestamp("2026-01-01 10:30:00+00:00"),  # within window
                    pd.Timestamp("2026-01-01 13:00:00+00:00"),  # after window
                    pd.Timestamp(
                        "2026-01-01 11:00:00+00:00"
                    ),  # within but arrived after
                ],
                "specialty": ["medical", "medical", "medical"],
            }
        )

    def test_counts_present_before_ended_in_window(self) -> None:
        result = count_observed(
            "admitted_in_window",
            visits=self.visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
        )
        self.assertEqual(result, 1)

    def test_departed_in_window_is_alias(self) -> None:
        result = count_observed(
            "departed_in_window",
            visits=self.visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
        )
        self.assertEqual(result, 1)


class TestArrivedInWindow(unittest.TestCase):
    """Tests for arrived_in_window mode (arrival-only check)."""

    def test_counts_arrivals_within_window(self) -> None:
        visits = pd.DataFrame(
            {
                "arrival_datetime": [
                    pd.Timestamp("2026-01-01 09:45:00+00:00"),  # before moment
                    pd.Timestamp("2026-01-01 10:15:00+00:00"),  # within window
                    pd.Timestamp("2026-01-01 12:30:00+00:00"),  # after window
                ],
                "specialty": ["medical", "medical", "medical"],
            }
        )
        result = count_observed(
            "arrived_in_window",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
            start_time_col="arrival_datetime",
        )
        self.assertEqual(result, 1)


class TestArrivedAndAdmittedInWindow(unittest.TestCase):
    """Tests for arrived_and_admitted_in_window mode."""

    def test_counts_arrived_after_and_admitted_within(self) -> None:
        visits = pd.DataFrame(
            {
                "arrival_datetime": [
                    pd.Timestamp("2026-01-01 09:45:00+00:00"),  # before moment
                    pd.Timestamp("2026-01-01 10:10:00+00:00"),  # after moment
                    pd.Timestamp("2026-01-01 10:20:00+00:00"),  # after moment
                ],
                "departure_datetime": [
                    pd.Timestamp("2026-01-01 10:30:00+00:00"),
                    pd.Timestamp("2026-01-01 10:40:00+00:00"),  # within window
                    pd.Timestamp("2026-01-01 12:10:00+00:00"),  # after window
                ],
                "specialty": ["medical", "medical", "medical"],
            }
        )
        result = count_observed(
            "arrived_and_admitted_in_window",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
        )
        self.assertEqual(result, 1)

    def test_direct_admission_gives_same_as_arrived_in_window(self) -> None:
        """When arrival == departure (direct admission), both modes agree."""
        visits = pd.DataFrame(
            {
                "arrival_datetime": [
                    pd.Timestamp("2026-01-01 10:15:00+00:00"),
                    pd.Timestamp("2026-01-01 12:30:00+00:00"),
                ],
                "departure_datetime": [
                    pd.Timestamp("2026-01-01 10:15:00+00:00"),
                    pd.Timestamp("2026-01-01 12:30:00+00:00"),
                ],
                "specialty": ["medical", "medical"],
            }
        )
        arrived = count_observed(
            "arrived_in_window",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
            start_time_col="arrival_datetime",
        )
        arrived_admitted = count_observed(
            "arrived_and_admitted_in_window",
            visits=visits,
            snapshot_date=date(2026, 1, 1),
            prediction_time=(10, 0),
            prediction_window=timedelta(hours=2),
            specialty="medical",
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
        )
        self.assertEqual(arrived, arrived_admitted)
        self.assertEqual(arrived, 1)


class TestInvalidMode(unittest.TestCase):
    """Test that unsupported modes raise ValueError."""

    def test_raises_for_non_counting_mode(self) -> None:
        visits = pd.DataFrame({"arrival_datetime": [], "departure_datetime": []})
        with self.assertRaises(ValueError):
            count_observed(
                "not_applicable",
                visits=visits,
                snapshot_date=date(2026, 1, 1),
                prediction_time=(10, 0),
                prediction_window=timedelta(hours=2),
            )


if __name__ == "__main__":
    unittest.main()
