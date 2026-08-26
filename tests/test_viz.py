"""Tests for the patientflow.viz module.

Tier 1: Import smoke tests — verify every viz module can be imported.
Tier 2: Unit tests for pure (non-plotting) functions in the viz module.
Tier 3: Render smoke tests — verify plotting functions produce figures without errors.
"""

import unittest
import warnings
from unittest.mock import patch
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless testing
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from patientflow.viz.utils import (
    clean_title_for_filename,
    format_prediction_time,
    pyplot_show_if,
)
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
from patientflow.viz.observed_against_expected import (
    plot_deltas,
    plot_arrival_deltas,
    _final_arrival_deltas_for_clock,
    _predictor_rates_for_window,
)
from patientflow.viz.arrival_rates import (
    plot_arrival_rates,
    plot_cumulative_arrival_rates,
)
from patientflow.viz.trial_results import plot_trial_results
from patientflow.model_artifacts import HyperParameterTrial
from patientflow.predictors.incoming_admission_predictors import (
    DirectAdmissionPredictor,
)


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
    ("patientflow.viz.shap", ["plot_shap", "SHAP_AVAILABLE"]),
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

    def test_import_shap_submodule(self):
        import importlib

        mod = importlib.import_module("patientflow.viz.shap")
        self.assertTrue(hasattr(mod, "plot_shap"))
        self.assertTrue(hasattr(mod, "SHAP_AVAILABLE"))


class TestShapOptional(unittest.TestCase):
    def test_plot_shap_import_error_when_shap_unavailable(self):
        import patientflow.viz.shap as viz_shap

        with patch.object(viz_shap, "SHAP_AVAILABLE", False):
            with self.assertRaises(ImportError) as ctx:
                viz_shap.plot_shap([], pd.DataFrame())
        self.assertIn("pip install shap", str(ctx.exception).lower())

    @unittest.skipUnless(
        __import__("patientflow.viz.shap", fromlist=["SHAP_AVAILABLE"]).SHAP_AVAILABLE,
        "shap not installed",
    )
    def test_plot_shap_show_false_does_not_call_pyplot_show(self):
        import numpy as np

        import patientflow.viz.shap as viz_shap
        from patientflow.train.classifiers import train_classifier

        n = 200
        train_visits = pd.DataFrame(
            {
                "visit_number": range(n),
                "age": np.random.randint(0, 100, n),
                "sex": pd.Series(np.random.choice(["M", "F"], n), dtype="object"),
                "arrival_method": pd.Series(
                    np.random.choice(["ambulance", "walk-in", "referral"], n),
                    dtype="object",
                ),
                "is_admitted": np.random.choice([0, 1], n, p=[0.7, 0.3]),
                "snapshot_time": pd.date_range(start="2023-01-01", periods=n, freq="h"),
                "prediction_time": [(4, 0)] * n,
            }
        )
        grid = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [20]}
        ordinal = {"arrival_method": ["walk-in", "referral", "ambulance"]}
        model = train_classifier(
            train_visits=train_visits,
            valid_visits=train_visits,
            prediction_time=(4, 0),
            exclude_from_training_data=[
                "snapshot_time",
                "visit_number",
                "prediction_time",
            ],
            grid=grid,
            ordinal_mappings=ordinal,
            visit_col="visit_number",
            evaluate_on_test=False,
            calibrate_probabilities=False,
        )
        with patch.object(viz_shap.plt, "show") as mock_show:
            viz_shap.plot_shap([model], train_visits, show=False, return_figure=False)
        mock_show.assert_not_called()

    def test_summary_plot_kwargs_omits_rng_when_unsupported(self):
        import patientflow.viz.shap as viz_shap

        def summary_plot_without_rng(*args, **kwargs):
            pass

        with patch.object(viz_shap, "shap") as mock_shap:
            mock_shap.summary_plot = summary_plot_without_rng
            kwargs = viz_shap._summary_plot_kwargs(feature_names=["a"], show=False)
        self.assertEqual(kwargs, {"feature_names": ["a"], "show": False})

    def test_summary_plot_kwargs_includes_rng_when_supported(self):
        import patientflow.viz.shap as viz_shap

        def summary_plot_with_rng(*args, rng=None, **kwargs):
            pass

        with patch.object(viz_shap, "shap") as mock_shap:
            mock_shap.summary_plot = summary_plot_with_rng
            kwargs = viz_shap._summary_plot_kwargs(feature_names=["a"], show=False)
        self.assertEqual(kwargs["feature_names"], ["a"])
        self.assertFalse(kwargs["show"])
        self.assertIn("rng", kwargs)
        self.assertIsNotNone(kwargs["rng"])


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


class TestPyplotShowIf(unittest.TestCase):
    """Tests for pyplot_show_if."""

    def test_false_does_not_call_pyplot_show(self):
        with patch.object(plt, "show") as mock_show:
            pyplot_show_if(False)
        mock_show.assert_not_called()

    def test_true_calls_pyplot_show(self):
        with patch.object(plt, "show") as mock_show:
            pyplot_show_if(True)
        mock_show.assert_called_once()


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


class TestPlotArrivalDeltas(unittest.TestCase):
    """Tests for plot_arrival_deltas with and without a fitted arrival-rate model."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        rows = []
        for w in range(8):
            base = datetime(2024, 1, 1) + timedelta(weeks=w)
            for d in range(7):
                day = base + timedelta(days=d)
                # Mondays receive many arrivals; other days receive few.
                n = 40 if d == 0 else 4
                for _ in range(n):
                    rows.append(
                        pd.Timestamp(
                            day
                            + timedelta(
                                hours=int(np.random.randint(0, 24)),
                                minutes=int(np.random.randint(0, 60)),
                            )
                        )
                    )
        cls.df = pd.DataFrame({"arrival_datetime": sorted(rows)})
        cls.prediction_window = timedelta(hours=4)
        cls.yta_time_interval = timedelta(hours=1)

        # Snapshot dates: a Monday and a Tuesday (matching the synthetic gradient).
        cls.snapshot_dates = [date(2024, 1, 1), date(2024, 1, 2)]

    def tearDown(self):
        plt.close("all")

    def _make_predictor(self, *, stratify_by_weekday=True):
        df_for_fit = self.df.set_index("arrival_datetime")
        predictor = DirectAdmissionPredictor(filters=None)
        predictor.fit(
            df_for_fit,
            yta_time_interval=self.yta_time_interval,
            num_days=56,
            stratify_by_weekday=stratify_by_weekday,
        )
        return predictor

    def test_predictor_rates_for_window_uses_weekday_profile(self):
        """Helper returns rates from arrival_rates_by_weekday when available."""
        predictor = self._make_predictor()
        rates_mon = _predictor_rates_for_window(
            predictor,
            "unfiltered",
            (8, 0),
            self.prediction_window,
            date(2024, 1, 1),  # Monday
        )
        rates_tue = _predictor_rates_for_window(
            predictor,
            "unfiltered",
            (8, 0),
            self.prediction_window,
            date(2024, 1, 2),  # Tuesday
        )
        self.assertGreater(sum(rates_mon.values()), sum(rates_tue.values()))

    def test_plot_with_arrival_rate_model_uses_weekday_baseline(self):
        """Baseline provenance is off by default; opt in with show_baseline_label."""
        predictor = self._make_predictor()
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            arrival_rate_model=predictor,
            return_figure=True,
        )
        self.assertIsInstance(fig, Figure)
        title_text = fig.axes[0].get_title()
        self.assertEqual(title_text, "Arrival delta plot for 8:00")
        self.assertNotIn(
            "Expected baseline",
            " ".join(t.get_text() for t in fig.texts),
        )

        fig_labelled = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            arrival_rate_model=predictor,
            show_baseline_label=True,
            return_figure=True,
        )
        self.assertIn(
            "weekday-specific rates (from fitted model)",
            " ".join(t.get_text() for t in fig_labelled.texts),
        )

    def test_plot_predictor_alias_matches_arrival_rate_model(self):
        """1.6.2 predictor= keyword still selects the fitted baseline."""
        predictor = self._make_predictor()
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            predictor=predictor,
            show_baseline_label=True,
            return_figure=True,
        )
        self.assertIsInstance(fig, Figure)
        self.assertIn(
            "weekday-specific rates (from fitted model)",
            " ".join(t.get_text() for t in fig.texts),
        )

    def test_plot_raises_when_predictor_and_arrival_rate_model_both_passed(self):
        predictor = self._make_predictor()
        with self.assertRaises(ValueError) as cm:
            plot_arrival_deltas(
                self.df,
                prediction_times=(8, 0),
                snapshot_dates=self.snapshot_dates,
                prediction_window=self.prediction_window,
                yta_time_interval=self.yta_time_interval,
                predictor=predictor,
                arrival_rate_model=predictor,
                return_figure=True,
            )
        self.assertIn("not both", str(cm.exception))

    def test_plot_without_arrival_rate_model_uses_pooled_baseline(self):
        """Default path falls back to pooled rates derived from the dataframe."""
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            show_baseline_label=True,
            return_figure=True,
        )
        self.assertIsInstance(fig, Figure)
        title_text = fig.axes[0].get_title()
        self.assertNotIn("Expected baseline", title_text)
        self.assertEqual(title_text, "Arrival delta plot for 8:00")
        self.assertIn(
            "pooled rates (from dataframe)",
            " ".join(t.get_text() for t in fig.texts),
        )

    def test_plot_raises_on_yta_interval_mismatch(self):
        """yta_time_interval must match arrival_rate_model.yta_time_interval."""
        predictor = self._make_predictor()
        with self.assertRaises(ValueError) as cm:
            plot_arrival_deltas(
                self.df,
                prediction_times=(8, 0),
                snapshot_dates=self.snapshot_dates,
                prediction_window=self.prediction_window,
                yta_time_interval=timedelta(minutes=30),
                arrival_rate_model=predictor,
                return_figure=True,
            )
        self.assertIn("yta_time_interval mismatch", str(cm.exception))

    def test_plot_with_pooled_model_uses_pooled_model_baseline(self):
        """Model without weekday profiles → 'pooled rates (from fitted model)'."""
        predictor = self._make_predictor(stratify_by_weekday=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig = plot_arrival_deltas(
                self.df,
                prediction_times=(8, 0),
                snapshot_dates=self.snapshot_dates,
                prediction_window=self.prediction_window,
                yta_time_interval=self.yta_time_interval,
                arrival_rate_model=predictor,
                show_baseline_label=True,
                return_figure=True,
            )
        self.assertIsInstance(fig, Figure)
        self.assertNotIn("Expected baseline", fig.axes[0].get_title())
        self.assertIn(
            "pooled rates (from fitted model)",
            " ".join(t.get_text() for t in fig.texts),
        )

    def test_plot_strict_raises_when_model_lacks_weekday(self):
        """strict_prediction_date=True surfaces missing weekday profiles."""
        predictor = self._make_predictor(stratify_by_weekday=False)
        with self.assertRaises(ValueError):
            plot_arrival_deltas(
                self.df,
                prediction_times=(8, 0),
                snapshot_dates=self.snapshot_dates,
                prediction_window=self.prediction_window,
                yta_time_interval=self.yta_time_interval,
                arrival_rate_model=predictor,
                strict_prediction_date=True,
                return_figure=True,
            )

    def test_plot_with_suptitle_renders_supertitle(self):
        """suptitle renders as the figure-level title."""
        predictor = self._make_predictor()
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            arrival_rate_model=predictor,
            suptitle="Medical service",
            return_figure=True,
        )
        self.assertEqual(fig._suptitle.get_text(), "Medical service")

    def test_plot_with_alternate_arrival_datetime_col(self):
        """arrival_datetime_col selects a non-default timestamp column."""
        df = self.df.rename(columns={"arrival_datetime": "arrived_at"})
        fig = plot_arrival_deltas(
            df,
            prediction_times=(8, 0),
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            arrival_datetime_col="arrived_at",
            return_figure=True,
        )
        self.assertIsInstance(fig, Figure)

    def test_quiet_day_included_with_positive_expected(self):
        """Zero-arrival days still contribute when expected mass is non-zero."""
        predictor = self._make_predictor()
        # 2024-01-03 is a Wednesday with arrivals in the synthetic set, but none
        # in the 08:00–12:00 prediction window if we use an empty day outside the
        # construction range. Use a date with no rows at all.
        quiet_date = date(2024, 3, 1)
        busy_date = date(2024, 1, 1)  # Monday with many arrivals
        snapshot_dates = [busy_date, quiet_date]

        deltas = _final_arrival_deltas_for_clock(
            self.df,
            (8, 0),
            snapshot_dates,
            self.prediction_window,
            self.yta_time_interval,
            arrival_rate_model=predictor,
        )
        self.assertEqual(len(deltas), 2)
        rates_quiet = _predictor_rates_for_window(
            predictor,
            "unfiltered",
            (8, 0),
            self.prediction_window,
            quiet_date,
        )
        expected_quiet = sum(rates_quiet.values())
        self.assertGreater(expected_quiet, 0)
        self.assertAlmostEqual(deltas[1], 0.0 - expected_quiet)

        fig = plot_arrival_deltas(
            self.df,
            prediction_times=(8, 0),
            snapshot_dates=snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            arrival_rate_model=predictor,
            return_figure=True,
        )
        # Histogram bar heights should sum to the number of snapshot dates.
        heights = [patch.get_height() for patch in fig.axes[0].patches]
        self.assertEqual(sum(heights), len(snapshot_dates))

    def test_plot_panels_one_axis_per_clock(self):
        """Multiple clocks produce one histogram panel each."""
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=[(8, 0), (12, 0), (16, 0)],
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            return_figure=True,
        )
        self.assertEqual(len(fig.axes), 3)

    def test_plot_panels_share_x_limits(self):
        """Panels use a shared x-axis range like plot_deltas."""
        fig = plot_arrival_deltas(
            self.df,
            prediction_times=[(8, 0), (12, 0), (16, 0)],
            snapshot_dates=self.snapshot_dates,
            prediction_window=self.prediction_window,
            yta_time_interval=self.yta_time_interval,
            return_figure=True,
        )
        xlims = [ax.get_xlim() for ax in fig.axes]
        self.assertTrue(all(xlim == xlims[0] for xlim in xlims))
        # Symmetric about zero, matching plot_deltas.
        left, right = xlims[0]
        self.assertAlmostEqual(left + right, 0.0, places=6)
        self.assertEqual(fig.axes[0].get_ylabel(), "Frequency")


if __name__ == "__main__":
    unittest.main()
