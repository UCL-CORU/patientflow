"""Tests for the patientflow.viz module.

Tier 1: Import smoke tests — verify every viz module can be imported.
Tier 2: Unit tests for pure (non-plotting) functions in the viz module.
Tier 3: Render smoke tests — verify plotting functions produce figures without errors.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless testing
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from patientflow.viz.utils import clean_title_for_filename, format_prediction_time
from patientflow.viz.madcap import classify_age
from patientflow.viz.probability_distribution import (
    _calculate_probability_thresholds,
    plot_prob_dist,
)
from patientflow.viz.epudd import _calculate_cdf_values
from patientflow.viz.randomised_pit import _prob_to_cdf
from patientflow.viz.aspirational_curve import plot_curve
from patientflow.viz.survival_curve import plot_admission_time_survival_curve
from patientflow.viz.data_distribution import plot_data_distribution
from patientflow.viz.observed_against_expected import plot_deltas
from patientflow.viz.arrival_rates import (
    plot_arrival_rates,
    plot_cumulative_arrival_rates,
)
from patientflow.viz.trial_results import plot_trial_results
from patientflow.model_artifacts import HyperParameterTrial


# ---------------------------------------------------------------------------
# Tier 1 – Import smoke tests
# ---------------------------------------------------------------------------

_VIZ_IMPORTS = [
    ("patientflow.viz", None),
    ("patientflow.viz.utils", ["clean_title_for_filename", "format_prediction_time"]),
    (
        "patientflow.viz.arrival_rates",
        ["plot_arrival_rates", "plot_cumulative_arrival_rates"],
    ),
    ("patientflow.viz.aspirational_curve", ["plot_curve"]),
    ("patientflow.viz.calibration", ["plot_calibration"]),
    ("patientflow.viz.data_distribution", ["plot_data_distribution"]),
    ("patientflow.viz.epudd", ["plot_epudd"]),
    ("patientflow.viz.estimated_probabilities", ["plot_estimated_probabilities"]),
    ("patientflow.viz.features", ["plot_features"]),
    ("patientflow.viz.madcap", ["classify_age", "plot_madcap", "plot_madcap_by_group"]),
    ("patientflow.viz.observed_against_expected", ["plot_deltas"]),
    ("patientflow.viz.probability_distribution", ["plot_prob_dist"]),
    ("patientflow.viz.quantile_quantile", ["qq_plot"]),
    ("patientflow.viz.randomised_pit", ["plot_randomised_pit"]),
    ("patientflow.viz.survival_curve", ["plot_admission_time_survival_curve"]),
    ("patientflow.viz.trial_results", ["plot_trial_results"]),
]


class TestVizImports(unittest.TestCase):
    """Verify that every viz module can be imported without errors."""

    def test_all_viz_modules_import(self):
        import importlib

        for module_path, names in _VIZ_IMPORTS:
            with self.subTest(module=module_path):
                mod = importlib.import_module(module_path)
                if names:
                    for name in names:
                        self.assertTrue(
                            hasattr(mod, name), f"{module_path}.{name} missing"
                        )

    def test_import_shap(self):
        try:
            from patientflow.viz.shap import plot_shap  # noqa: F401
        except ImportError:
            self.skipTest("shap package not installed")


# ---------------------------------------------------------------------------
# Tier 2 – Unit tests for pure functions
# ---------------------------------------------------------------------------


class TestCleanTitleForFilename(unittest.TestCase):
    """Tests for clean_title_for_filename."""

    def test_replacements(self):
        cases = [
            ("hello world", "hello_world"),
            ("90% confidence", "90_confidence"),
            ("line1\nline2", "line1line2"),
            ("a, b, c", "a_b_c"),
            ("v1.2.3", "v123"),
            ("", ""),
            ("clean_title", "clean_title"),
        ]
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(clean_title_for_filename(title), expected)

    def test_no_special_chars_remain(self):
        result = clean_title_for_filename("Results: 90% CI,\nnew.line")
        for char in [" ", "%", "\n", ",", "."]:
            self.assertNotIn(char, result)


class TestFormatPredictionTime(unittest.TestCase):
    """Tests for format_prediction_time."""

    def test_tuple_inputs(self):
        cases = [
            ((9, 30), "09:30"),
            ((14, 0), "14:00"),
            ((0, 0), "00:00"),
            ((23, 59), "23:59"),
        ]
        for input_val, expected in cases:
            with self.subTest(input=input_val):
                self.assertEqual(format_prediction_time(input_val), expected)

    def test_string_inputs(self):
        cases = [
            ("0930", "09:30"),
            ("pred_0930", "09:30"),
            ("model_pred_1400", "14:00"),
        ]
        for input_val, expected in cases:
            with self.subTest(input=input_val):
                self.assertEqual(format_prediction_time(input_val), expected)


class TestClassifyAge(unittest.TestCase):
    """Tests for classify_age."""

    def test_numeric_ages(self):
        cases = [
            (0, "Children"),
            (5, "Children"),
            (17, "Children"),
            (18, "Adults < 65"),
            (30, "Adults < 65"),
            (64, "Adults < 65"),
            (65, "Adults 65 or over"),
            (70, "Adults 65 or over"),
            (100, "Adults 65 or over"),
        ]
        for age, expected in cases:
            with self.subTest(age=age):
                self.assertEqual(classify_age(age), expected)

    def test_string_age_groups(self):
        cases = [
            ("0-17", "Children"),
            ("18-24", "Adults < 65"),
            ("45-54", "Adults < 65"),
            ("65-74", "Adults 65 or over"),
            ("75-115", "Adults 65 or over"),
        ]
        for age_str, expected in cases:
            with self.subTest(age=age_str):
                self.assertEqual(classify_age(age_str), expected)

    def test_edge_cases(self):
        self.assertEqual(classify_age("not-a-group"), "unknown")
        self.assertEqual(classify_age(None), "unknown")
        self.assertEqual(classify_age(30.5), "Adults < 65")

    def test_custom_categories(self):
        custom = {
            "Junior": {"numeric": {"max": 12}, "groups": ["a"]},
            "Senior": {"numeric": {"min": 13}, "groups": ["b"]},
        }
        self.assertEqual(classify_age(10, age_categories=custom), "Junior")
        self.assertEqual(classify_age(15, age_categories=custom), "Senior")
        self.assertEqual(classify_age("a", age_categories=custom), "Junior")
        self.assertEqual(classify_age("z", age_categories=custom), "unknown")


class TestCalculateProbabilityThresholds(unittest.TestCase):
    """Tests for _calculate_probability_thresholds."""

    def test_basic_thresholds(self):
        pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        result = _calculate_probability_thresholds(pmf, [0.7, 0.9])
        self.assertIn(0.7, result)
        self.assertIn(0.9, result)
        self.assertIsInstance(result[0.9], (int, np.integer))
        # Higher confidence threshold -> lower or equal bed count
        self.assertLessEqual(result[0.9], result[0.7])

    def test_concentrated_distribution(self):
        pmf = [0.0, 0.0, 0.0, 1.0]
        result = _calculate_probability_thresholds(pmf, [0.5])
        self.assertEqual(result[0.5], 3)


class TestCalculateCdfValues(unittest.TestCase):
    """Tests for _calculate_cdf_values."""

    def test_simple_distribution(self):
        lower, mid, upper = _calculate_cdf_values(np.array([0.2, 0.3, 0.5]))
        np.testing.assert_array_almost_equal(upper, [0.2, 0.5, 1.0])
        np.testing.assert_array_almost_equal(lower, [0.0, 0.2, 0.5])
        np.testing.assert_array_almost_equal(mid, [0.1, 0.35, 0.75])

    def test_properties(self):
        lower, mid, upper = _calculate_cdf_values(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertAlmostEqual(lower[0], 0.0)
        self.assertAlmostEqual(upper[-1], 1.0)
        for i in range(4):
            self.assertGreaterEqual(mid[i], lower[i])
            self.assertLessEqual(mid[i], upper[i])


class TestProbToCdf(unittest.TestCase):
    """Tests for _prob_to_cdf."""

    def test_array_input(self):
        cdf = _prob_to_cdf([0.2, 0.3, 0.5])
        self.assertAlmostEqual(cdf(0), 0.2)
        self.assertAlmostEqual(cdf(1), 0.5)
        self.assertAlmostEqual(cdf(2), 1.0)

    def test_boundaries(self):
        cdf = _prob_to_cdf([0.5, 0.5])
        self.assertAlmostEqual(cdf(-1), 0.0)
        self.assertAlmostEqual(cdf(10), 1.0)

    def test_dict_input(self):
        cdf = _prob_to_cdf({2: 0.3, 0: 0.3, 1: 0.4})  # unordered keys
        self.assertAlmostEqual(cdf(0), 0.3)
        self.assertAlmostEqual(cdf(1), 0.7)
        self.assertAlmostEqual(cdf(2), 1.0)

    def test_series_and_dataframe_input(self):
        cdf_s = _prob_to_cdf(pd.Series([0.5, 0.5], index=[0, 1]))
        self.assertAlmostEqual(cdf_s(1), 1.0)

        cdf_df = _prob_to_cdf(pd.DataFrame([[0.4, 0.6]], columns=[0, 1]))
        self.assertAlmostEqual(cdf_df(0), 0.4)

    def test_monotonicity(self):
        cdf = _prob_to_cdf([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        prev = 0.0
        for x in range(6):
            self.assertGreaterEqual(cdf(x), prev)
            prev = cdf(x)


# ---------------------------------------------------------------------------
# Tier 3 – Render smoke tests
# ---------------------------------------------------------------------------


class TestPlotRendering(unittest.TestCase):
    """Verify that plotting functions produce figures without errors."""

    def tearDown(self):
        plt.close("all")

    def test_plot_prob_dist(self):
        fig = plot_prob_dist(
            [0.05, 0.1, 0.3, 0.3, 0.15, 0.1], "Test", return_figure=True
        )
        self.assertIsInstance(fig, Figure)

    def test_plot_prob_dist_with_thresholds(self):
        fig = plot_prob_dist(
            [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05],
            "Thresholds",
            probability_levels=[0.7, 0.9],
            return_figure=True,
        )
        self.assertIsInstance(fig, Figure)

    def test_plot_curve(self):
        fig = plot_curve(title="Test", x1=4, y1=0.2, x2=24, y2=0.8, return_figure=True)
        self.assertIsInstance(fig, Figure)

    def test_plot_survival_curve(self):
        np.random.seed(42)
        n = 50
        arrivals = pd.date_range("2024-01-01", periods=n, freq="h")
        departures = arrivals + pd.to_timedelta(np.random.exponential(3, n), unit="h")
        df = pd.DataFrame(
            {"arrival_datetime": arrivals, "departure_datetime": departures}
        )
        fig = plot_admission_time_survival_curve(df, return_figure=True)
        self.assertIsInstance(fig, Figure)

    def test_plot_data_distribution(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "value": np.random.normal(10, 3, 200),
                "group": np.random.choice(["A", "B"], 200),
            }
        )
        result = plot_data_distribution(
            df, "value", "group", "Group", return_figure=True
        )
        self.assertIsNotNone(result)

    def test_plot_deltas(self):
        results = {
            "pred_0930": {
                "observed": np.array([5, 8, 6, 7]),
                "expected": np.array([6, 7, 7, 6]),
            },
            "pred_1200": {
                "observed": np.array([10, 12, 11, 9]),
                "expected": np.array([11, 11, 10, 10]),
            },
        }
        fig = plot_deltas(results, return_figure=True)
        self.assertIsInstance(fig, Figure)

    def test_plot_arrival_rates(self):
        np.random.seed(42)
        times = sorted(
            pd.Timestamp("2024-01-01")
            + pd.Timedelta(days=np.random.randint(0, 7))
            + pd.Timedelta(hours=np.random.randint(0, 24))
            + pd.Timedelta(minutes=np.random.randint(0, 60))
            for _ in range(500)
        )
        df = pd.DataFrame(index=times)
        fig = plot_arrival_rates(df, "Test Rates", return_figure=True)
        self.assertIsInstance(fig, Figure)

    def test_plot_cumulative_arrival_rates(self):
        np.random.seed(42)
        times = sorted(
            pd.Timestamp("2024-01-01")
            + pd.Timedelta(days=np.random.randint(0, 7))
            + pd.Timedelta(hours=np.random.randint(0, 24))
            + pd.Timedelta(minutes=np.random.randint(0, 60))
            for _ in range(500)
        )
        df = pd.DataFrame(index=times)
        fig = plot_cumulative_arrival_rates(
            df, "Test Cumulative", hour_lines=[], return_figure=True
        )
        self.assertIsInstance(fig, Figure)

    def test_plot_trial_results(self):
        trials = [
            HyperParameterTrial(
                parameters={"lr": 0.01},
                cv_results={"valid_auc": 0.85, "valid_logloss": 0.4},
            ),
            HyperParameterTrial(
                parameters={"lr": 0.1},
                cv_results={"valid_auc": 0.88, "valid_logloss": 0.35},
            ),
            HyperParameterTrial(
                parameters={"lr": 0.001},
                cv_results={"valid_auc": 0.82, "valid_logloss": 0.45},
            ),
        ]
        fig = plot_trial_results(trials, return_figure=True)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 2)


if __name__ == "__main__":
    unittest.main()
