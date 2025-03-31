import unittest
import pandas as pd
from datetime import date

from patientflow.prepare import prepare_group_snapshot_dict


class TestPrepareGroupSnapshotDict(unittest.TestCase):
    def setUp(self):
        """Set up test data that will be used in multiple tests."""
        # Create sample data with snapshot_date column
        self.sample_data = pd.DataFrame(
            {
                "snapshot_date": [
                    date(2023, 1, 1),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 4),
                    date(2023, 1, 4),
                    date(2023, 1, 4),
                    date(2023, 1, 7),
                ],
                "value": [10, 20, 30, 40, 50, 60, 70],
            }
        )

        # Expected output for the sample data without date range
        self.expected_basic = {
            date(2023, 1, 1): [0, 1],
            date(2023, 1, 2): [2],
            date(2023, 1, 4): [3, 4, 5],
            date(2023, 1, 7): [6],
        }

        # Date range parameters
        self.start_date = date(2023, 1, 1)
        self.end_date = date(
            2023, 1, 8
        )  # One day after last date to include full range

        # Expected output with date range (should include empty arrays for missing dates)
        self.expected_with_range = {
            date(2023, 1, 1): [0, 1],
            date(2023, 1, 2): [2],
            date(2023, 1, 3): [],
            date(2023, 1, 4): [3, 4, 5],
            date(2023, 1, 5): [],
            date(2023, 1, 6): [],
            date(2023, 1, 7): [6],
        }

    def test_basic_functionality(self):
        """Test the basic functionality without date range parameters."""
        result = prepare_group_snapshot_dict(self.sample_data)
        self.assertEqual(result, self.expected_basic)

    def test_with_date_range(self):
        """Test with date range parameters to ensure all dates are included."""
        result = prepare_group_snapshot_dict(
            self.sample_data, start_dt=self.start_date, end_dt=self.end_date
        )
        self.assertEqual(result, self.expected_with_range)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame that has the required column."""
        empty_df = pd.DataFrame(columns=["snapshot_date", "value"])
        result = prepare_group_snapshot_dict(empty_df)
        self.assertEqual(result, {})

    def test_missing_column(self):
        """Test the error handling when the required column is missing."""
        invalid_df = pd.DataFrame({"date": [date(2023, 1, 1)], "value": [10]})
        with self.assertRaises(ValueError) as context:
            prepare_group_snapshot_dict(invalid_df)
        self.assertTrue(
            "DataFrame must include a 'snapshot_date' column" in str(context.exception)
        )

    def test_single_date(self):
        """Test with a DataFrame containing a single date."""
        single_date_df = pd.DataFrame(
            {"snapshot_date": [date(2023, 1, 1)] * 3, "value": [10, 20, 30]}
        )
        result = prepare_group_snapshot_dict(single_date_df)
        self.assertEqual(result, {date(2023, 1, 1): [0, 1, 2]})

    def test_non_sequential_indices(self):
        """Test with a DataFrame that has non-sequential indices."""
        df_with_non_seq_indices = self.sample_data.copy()
        # Change DataFrame indices to be non-sequential
        df_with_non_seq_indices.index = [5, 10, 15, 20, 25, 30, 35]

        result = prepare_group_snapshot_dict(df_with_non_seq_indices)
        expected = {
            date(2023, 1, 1): [5, 10],
            date(2023, 1, 2): [15],
            date(2023, 1, 4): [20, 25, 30],
            date(2023, 1, 7): [35],
        }
        self.assertEqual(result, expected)

    def test_date_range_no_data(self):
        """Test with date range that doesn't overlap with any data."""
        future_start = date(2024, 1, 1)
        future_end = date(2024, 1, 5)

        result = prepare_group_snapshot_dict(
            self.sample_data, start_dt=future_start, end_dt=future_end
        )
        print(result)
        # Should have empty lists for all dates in range
        expected = {
            date(2024, 1, 1): [],
            date(2024, 1, 2): [],
            date(2024, 1, 3): [],
            date(2024, 1, 4): [],
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
