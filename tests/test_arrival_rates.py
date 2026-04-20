import unittest
import numpy as np
import pandas as pd
from collections import OrderedDict

# Import the functions to test
from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates,
    time_varying_arrival_rates_by_weekday,
    time_varying_arrival_rates_lagged,
    weekday_occurrences_in_span,
    admission_probabilities,
    weighted_arrival_rates,
    unfettered_demand_by_hour,
)
from datetime import date


class TestArrivalRates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        # Generate random arrival times over a week
        np.random.seed(42)
        n_arrivals = 1000
        random_times = [
            pd.Timestamp("2024-01-01")
            + pd.Timedelta(days=np.random.randint(0, 7))
            + pd.Timedelta(hours=np.random.randint(0, 24))
            + pd.Timedelta(minutes=np.random.randint(0, 60))
            for _ in range(n_arrivals)
        ]
        cls.test_df = pd.DataFrame(index=sorted(random_times))

        # Set up common parameters
        cls.yta_time_interval = 60
        cls.x1, cls.y1 = 4, 0.8
        cls.x2, cls.y2 = 8, 0.95

    def test_time_varying_arrival_rates_input_validation(self):
        """Test input validation for time_varying_arrival_rates function."""
        # Test invalid DataFrame
        with self.assertRaises(TypeError):
            time_varying_arrival_rates([], self.yta_time_interval)

        # Test invalid time interval
        with self.assertRaises(TypeError):
            time_varying_arrival_rates(self.test_df, "60")

        # Test negative time interval
        with self.assertRaises(ValueError):
            time_varying_arrival_rates(self.test_df, -60)

        # Test time interval that doesn't divide into 24 hours
        with self.assertRaises(ValueError):
            time_varying_arrival_rates(self.test_df, 43)

    def test_time_varying_arrival_rates_output(self):
        """Test the output format and values of time_varying_arrival_rates."""
        rates = time_varying_arrival_rates(self.test_df, self.yta_time_interval)

        # Check return type
        self.assertIsInstance(rates, OrderedDict)

        # Check that we have 24 hourly rates (when using 60-minute intervals)
        self.assertEqual(len(rates), 24)

        # Check that all rates are non-negative
        self.assertTrue(all(rate >= 0 for rate in rates.values()))

        # Check that times are properly ordered
        times = list(rates.keys())
        self.assertEqual(times, sorted(times))

    def test_time_varying_arrival_rates_lagged(self):
        """Test the lagged arrival rates calculation."""
        lag_hours = 4
        base_rates = time_varying_arrival_rates(self.test_df, self.yta_time_interval)
        lagged_rates = time_varying_arrival_rates_lagged(
            self.test_df, lagged_by=lag_hours, yta_time_interval=self.yta_time_interval
        )

        # Check that the times (keys) are the same
        self.assertEqual(list(base_rates.keys()), list(lagged_rates.keys()))

        # Check that values are shifted by lag_hours
        base_values = list(base_rates.values())
        lagged_values = list(lagged_rates.values())

        # The value at time t in lagged_rates should equal
        # the value at time (t - lag_hours) in base_rates
        for i in range(len(base_values)):
            source_idx = (i - lag_hours) % 24  # wrap around for 24-hour cycle
            self.assertEqual(lagged_values[i], base_values[source_idx])

        # Additional check: total rates should remain the same
        self.assertAlmostEqual(sum(base_rates.values()), sum(lagged_rates.values()))

    def test_admission_probabilities(self):
        """Test the admission probabilities calculation."""
        hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        prob_by_hour, prob_within_hour = admission_probabilities(
            hours, self.x1, self.y1, self.x2, self.y2
        )

        # Check shapes
        self.assertEqual(len(prob_by_hour), len(hours))
        self.assertEqual(len(prob_within_hour), len(hours) - 1)

        # Check probability ranges
        self.assertTrue(np.all(prob_by_hour >= 0))
        self.assertTrue(np.all(prob_by_hour <= 1))
        self.assertTrue(np.all(prob_within_hour >= 0))

        # Check that probabilities increase monotonically
        self.assertTrue(np.all(np.diff(prob_by_hour) >= 0))

    def test_weighted_arrival_rates(self):
        """Test the weighted arrival rates calculation."""
        # Create sample data
        weighted_rates = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        elapsed_hours = range(3)
        hour_idx = 1
        num_intervals = 3

        result = weighted_arrival_rates(
            weighted_rates, elapsed_hours, hour_idx, num_intervals
        )

        # Test that result is as expected
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_unfettered_demand_by_hour(self):
        """Test the undelayed demand calculation."""
        demand = unfettered_demand_by_hour(
            self.test_df, self.x1, self.y1, self.x2, self.y2, self.yta_time_interval
        )

        # Check return type
        self.assertIsInstance(demand, OrderedDict)

        # Check that we have 24 hourly demands
        self.assertEqual(len(demand), 24)

        # Check that all demands are non-negative
        self.assertTrue(all(d >= 0 for d in demand.values()))

        # Check that times are properly ordered
        times = list(demand.keys())
        self.assertEqual(times, sorted(times))

    def test_unfettered_demand_by_hour_input_validation(self):
        """Test input validation for unfettered_demand_by_hour function."""
        # Test invalid y coordinates
        with self.assertRaises(ValueError):
            unfettered_demand_by_hour(self.test_df, self.x1, 1.5, self.x2, self.y2)

        # Test invalid x coordinates
        with self.assertRaises(ValueError):
            unfettered_demand_by_hour(self.test_df, 8, self.y1, 4, self.y2)

        # Test invalid time interval
        with self.assertRaises(ValueError):
            unfettered_demand_by_hour(
                self.test_df, self.x1, self.y1, self.x2, self.y2, yta_time_interval=43
            )

    def test_weekday_occurrences_in_span(self):
        counts = weekday_occurrences_in_span(date(2024, 1, 1), date(2024, 1, 7))
        self.assertEqual(sum(counts), 7)
        self.assertEqual(counts[0], 1)

    def test_time_varying_arrival_rates_by_weekday(self):
        rates_by_d = time_varying_arrival_rates_by_weekday(
            self.test_df, self.yta_time_interval
        )
        self.assertEqual(len(rates_by_d), 7)
        for d in range(7):
            self.assertIsInstance(rates_by_d[d], OrderedDict)
            self.assertEqual(len(rates_by_d[d]), 24)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
        with self.assertRaises(ValueError):
            time_varying_arrival_rates(empty_df, self.yta_time_interval)

        # Test with single day of data
        single_day = pd.DataFrame(index=[pd.Timestamp("2024-01-01")])
        rates = time_varying_arrival_rates(single_day, self.yta_time_interval)
        self.assertEqual(len(rates), 24)

        # Test with maximum lag
        max_lag = 23
        lagged_rates = time_varying_arrival_rates_lagged(
            self.test_df, lagged_by=max_lag, yta_time_interval=self.yta_time_interval
        )
        self.assertEqual(len(lagged_rates), 24)


if __name__ == "__main__":
    unittest.main()
