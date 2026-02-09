"""Tests for the patientflow.viz module.

Tier 1: Import smoke tests — verify every viz module can be imported.
Tier 2: Unit tests for pure (non-plotting) functions in the viz module.
"""

import unittest
import numpy as np
import pandas as pd

from patientflow.viz.utils import clean_title_for_filename, format_prediction_time
from patientflow.viz.madcap import classify_age, DEFAULT_AGE_CATEGORIES
from patientflow.viz.probability_distribution import _calculate_probability_thresholds
from patientflow.viz.epudd import _calculate_cdf_values
from patientflow.viz.randomised_pit import _prob_to_cdf


# ---------------------------------------------------------------------------
# Tier 1 – Import smoke tests
# ---------------------------------------------------------------------------


class TestVizImports(unittest.TestCase):
    """Verify that every viz module can be imported without errors."""

    def test_import_viz_package(self):
        import patientflow.viz  # noqa: F401

    def test_import_utils(self):
        from patientflow.viz.utils import (  # noqa: F401
            clean_title_for_filename,
            format_prediction_time,
        )

    def test_import_arrival_rates(self):
        from patientflow.viz.arrival_rates import (  # noqa: F401
            plot_arrival_rates,
            plot_cumulative_arrival_rates,
        )

    def test_import_aspirational_curve(self):
        from patientflow.viz.aspirational_curve import plot_curve  # noqa: F401

    def test_import_calibration(self):
        from patientflow.viz.calibration import plot_calibration  # noqa: F401

    def test_import_data_distribution(self):
        from patientflow.viz.data_distribution import (  # noqa: F401
            plot_data_distribution,
        )

    def test_import_epudd(self):
        from patientflow.viz.epudd import plot_epudd  # noqa: F401

    def test_import_estimated_probabilities(self):
        from patientflow.viz.estimated_probabilities import (  # noqa: F401
            plot_estimated_probabilities,
        )

    def test_import_features(self):
        from patientflow.viz.features import plot_features  # noqa: F401

    def test_import_madcap(self):
        from patientflow.viz.madcap import (  # noqa: F401
            classify_age,
            plot_madcap,
            plot_madcap_by_group,
        )

    def test_import_observed_against_expected(self):
        from patientflow.viz.observed_against_expected import (  # noqa: F401
            plot_deltas,
            plot_arrival_delta_single_instance,
            plot_arrival_deltas,
        )

    def test_import_probability_distribution(self):
        from patientflow.viz.probability_distribution import (  # noqa: F401
            plot_prob_dist,
        )

    def test_import_quantile_quantile(self):
        from patientflow.viz.quantile_quantile import qq_plot  # noqa: F401

    def test_import_randomised_pit(self):
        from patientflow.viz.randomised_pit import (  # noqa: F401
            plot_randomised_pit,
        )

    def test_import_shap(self):
        try:
            from patientflow.viz.shap import plot_shap  # noqa: F401
        except ImportError:
            self.skipTest("shap package not installed")

    def test_import_survival_curve(self):
        from patientflow.viz.survival_curve import (  # noqa: F401
            plot_admission_time_survival_curve,
        )

    def test_import_trial_results(self):
        from patientflow.viz.trial_results import plot_trial_results  # noqa: F401


# ---------------------------------------------------------------------------
# Tier 2 – Unit tests for pure functions
# ---------------------------------------------------------------------------


class TestCleanTitleForFilename(unittest.TestCase):
    """Tests for patientflow.viz.utils.clean_title_for_filename."""

    def test_spaces_replaced_with_underscores(self):
        self.assertEqual(clean_title_for_filename("hello world"), "hello_world")

    def test_percent_removed(self):
        self.assertEqual(clean_title_for_filename("90% confidence"), "90_confidence")

    def test_newlines_removed(self):
        self.assertEqual(clean_title_for_filename("line1\nline2"), "line1line2")

    def test_commas_removed(self):
        # "a, b, c" -> spaces become _, commas removed -> "a_b_c"
        self.assertEqual(clean_title_for_filename("a, b, c"), "a_b_c")

    def test_periods_removed(self):
        self.assertEqual(clean_title_for_filename("v1.2.3"), "v123")

    def test_combined_special_characters(self):
        result = clean_title_for_filename("Results: 90% CI,\nnew.line")
        # spaces -> _, % removed, comma removed, \n removed, . removed
        self.assertNotIn(" ", result)
        self.assertNotIn("%", result)
        self.assertNotIn("\n", result)
        self.assertNotIn(",", result)
        self.assertNotIn(".", result)

    def test_empty_string(self):
        self.assertEqual(clean_title_for_filename(""), "")

    def test_no_special_characters(self):
        self.assertEqual(clean_title_for_filename("clean_title"), "clean_title")


class TestFormatPredictionTime(unittest.TestCase):
    """Tests for patientflow.viz.utils.format_prediction_time."""

    def test_tuple_single_digit_hour(self):
        self.assertEqual(format_prediction_time((9, 30)), "09:30")

    def test_tuple_double_digit_hour(self):
        self.assertEqual(format_prediction_time((14, 0)), "14:00")

    def test_tuple_midnight(self):
        self.assertEqual(format_prediction_time((0, 0)), "00:00")

    def test_tuple_end_of_day(self):
        self.assertEqual(format_prediction_time((23, 59)), "23:59")

    def test_string_simple(self):
        self.assertEqual(format_prediction_time("0930"), "09:30")

    def test_string_with_prefix(self):
        self.assertEqual(format_prediction_time("pred_0930"), "09:30")

    def test_string_with_multiple_underscores(self):
        self.assertEqual(format_prediction_time("model_pred_1400"), "14:00")


class TestClassifyAge(unittest.TestCase):
    """Tests for patientflow.viz.madcap.classify_age."""

    # --- Numeric inputs with default categories ---

    def test_child_numeric(self):
        self.assertEqual(classify_age(5), "Children")

    def test_child_numeric_boundary(self):
        self.assertEqual(classify_age(17), "Children")

    def test_adult_under_65_numeric(self):
        self.assertEqual(classify_age(30), "Adults < 65")

    def test_adult_under_65_lower_boundary(self):
        self.assertEqual(classify_age(18), "Adults < 65")

    def test_adult_under_65_upper_boundary(self):
        self.assertEqual(classify_age(64), "Adults < 65")

    def test_adult_65_or_over_numeric(self):
        self.assertEqual(classify_age(70), "Adults 65 or over")

    def test_adult_65_or_over_boundary(self):
        self.assertEqual(classify_age(65), "Adults 65 or over")

    def test_very_old_numeric(self):
        self.assertEqual(classify_age(100), "Adults 65 or over")

    def test_infant_numeric(self):
        self.assertEqual(classify_age(0), "Children")

    # --- String inputs with default categories ---

    def test_child_string(self):
        self.assertEqual(classify_age("0-17"), "Children")

    def test_young_adult_string(self):
        self.assertEqual(classify_age("18-24"), "Adults < 65")

    def test_middle_adult_string(self):
        self.assertEqual(classify_age("45-54"), "Adults < 65")

    def test_older_adult_string(self):
        self.assertEqual(classify_age("65-74"), "Adults 65 or over")

    def test_elderly_string(self):
        self.assertEqual(classify_age("75-115"), "Adults 65 or over")

    # --- Edge cases ---

    def test_unknown_string(self):
        self.assertEqual(classify_age("not-a-group"), "unknown")

    def test_none_returns_unknown(self):
        self.assertEqual(classify_age(None), "unknown")

    def test_float_age(self):
        self.assertEqual(classify_age(30.5), "Adults < 65")

    # --- Custom categories ---

    def test_custom_categories_numeric(self):
        custom = {
            "Junior": {"numeric": {"max": 12}, "groups": []},
            "Senior": {"numeric": {"min": 13}, "groups": []},
        }
        self.assertEqual(classify_age(10, age_categories=custom), "Junior")
        self.assertEqual(classify_age(15, age_categories=custom), "Senior")

    def test_custom_categories_string(self):
        custom = {
            "GroupA": {"numeric": {}, "groups": ["a", "b"]},
            "GroupB": {"numeric": {}, "groups": ["c", "d"]},
        }
        self.assertEqual(classify_age("a", age_categories=custom), "GroupA")
        self.assertEqual(classify_age("d", age_categories=custom), "GroupB")
        self.assertEqual(classify_age("z", age_categories=custom), "unknown")


class TestCalculateProbabilityThresholds(unittest.TestCase):
    """Tests for patientflow.viz.probability_distribution._calculate_probability_thresholds."""

    def test_basic_thresholds(self):
        # PMF: P(X=0)=0.05, P(X=1)=0.1, P(X=2)=0.2, P(X=3)=0.3,
        #       P(X=4)=0.2, P(X=5)=0.1, P(X=6)=0.05
        pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        result = _calculate_probability_thresholds(pmf, [0.7, 0.9])

        # Result should be a dict with the two probability levels as keys
        self.assertIsInstance(result, dict)
        self.assertIn(0.7, result)
        self.assertIn(0.9, result)

    def test_returns_integer_thresholds(self):
        pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        result = _calculate_probability_thresholds(pmf, [0.9])
        self.assertIsInstance(result[0.9], (int, np.integer))

    def test_higher_threshold_gives_lower_or_equal_index(self):
        # A higher probability threshold (e.g. 0.9) means we need fewer
        # resources guaranteed, so the index should be <= that for a lower
        # threshold (0.7). The find_probability_threshold_index finds the
        # first index where cumulative prob >= (1 - threshold).
        pmf = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        result = _calculate_probability_thresholds(pmf, [0.7, 0.9])
        self.assertLessEqual(result[0.9], result[0.7])

    def test_single_probability_level(self):
        pmf = [0.0, 0.0, 0.0, 1.0]
        result = _calculate_probability_thresholds(pmf, [0.5])
        self.assertIn(0.5, result)
        # All probability is at index 3, so threshold should be 3
        self.assertEqual(result[0.5], 3)

    def test_uniform_distribution(self):
        pmf = [0.25, 0.25, 0.25, 0.25]
        result = _calculate_probability_thresholds(pmf, [0.5])
        self.assertIsInstance(result[0.5], (int, np.integer))


class TestCalculateCdfValues(unittest.TestCase):
    """Tests for patientflow.viz.epudd._calculate_cdf_values."""

    def test_simple_distribution(self):
        agg_predicted = np.array([0.2, 0.3, 0.5])
        lower, mid, upper = _calculate_cdf_values(agg_predicted)

        # Upper should be cumulative sum
        np.testing.assert_array_almost_equal(upper, [0.2, 0.5, 1.0])

        # Lower should be shifted: [0, 0.2, 0.5]
        np.testing.assert_array_almost_equal(lower, [0.0, 0.2, 0.5])

        # Mid should be average of lower and upper
        np.testing.assert_array_almost_equal(mid, [0.1, 0.35, 0.75])

    def test_output_shapes(self):
        agg_predicted = np.array([0.1, 0.2, 0.3, 0.4])
        lower, mid, upper = _calculate_cdf_values(agg_predicted)

        self.assertEqual(len(lower), 4)
        self.assertEqual(len(mid), 4)
        self.assertEqual(len(upper), 4)

    def test_upper_ends_at_one(self):
        agg_predicted = np.array([0.1, 0.2, 0.3, 0.4])
        _, _, upper = _calculate_cdf_values(agg_predicted)
        self.assertAlmostEqual(upper[-1], 1.0)

    def test_lower_starts_at_zero(self):
        agg_predicted = np.array([0.5, 0.5])
        lower, _, _ = _calculate_cdf_values(agg_predicted)
        self.assertAlmostEqual(lower[0], 0.0)

    def test_mid_between_lower_and_upper(self):
        agg_predicted = np.array([0.1, 0.2, 0.3, 0.4])
        lower, mid, upper = _calculate_cdf_values(agg_predicted)

        for i in range(len(agg_predicted)):
            self.assertGreaterEqual(mid[i], lower[i])
            self.assertLessEqual(mid[i], upper[i])

    def test_single_element(self):
        agg_predicted = np.array([1.0])
        lower, mid, upper = _calculate_cdf_values(agg_predicted)

        np.testing.assert_array_almost_equal(lower, [0.0])
        np.testing.assert_array_almost_equal(mid, [0.5])
        np.testing.assert_array_almost_equal(upper, [1.0])


class TestProbToCdf(unittest.TestCase):
    """Tests for patientflow.viz.randomised_pit._prob_to_cdf."""

    def test_array_input(self):
        probs = [0.2, 0.3, 0.5]
        cdf = _prob_to_cdf(probs)

        # CDF at each value
        self.assertAlmostEqual(cdf(0), 0.2)
        self.assertAlmostEqual(cdf(1), 0.5)
        self.assertAlmostEqual(cdf(2), 1.0)

    def test_below_range_returns_zero(self):
        probs = [0.5, 0.5]
        cdf = _prob_to_cdf(probs)
        self.assertAlmostEqual(cdf(-1), 0.0)

    def test_above_range_returns_one(self):
        probs = [0.5, 0.5]
        cdf = _prob_to_cdf(probs)
        self.assertAlmostEqual(cdf(10), 1.0)

    def test_dict_input(self):
        probs = {0: 0.3, 1: 0.4, 2: 0.3}
        cdf = _prob_to_cdf(probs)

        self.assertAlmostEqual(cdf(0), 0.3)
        self.assertAlmostEqual(cdf(1), 0.7)
        self.assertAlmostEqual(cdf(2), 1.0)

    def test_dict_input_unordered_keys(self):
        # Keys should be sorted internally
        probs = {2: 0.3, 0: 0.3, 1: 0.4}
        cdf = _prob_to_cdf(probs)

        self.assertAlmostEqual(cdf(0), 0.3)
        self.assertAlmostEqual(cdf(1), 0.7)
        self.assertAlmostEqual(cdf(2), 1.0)

    def test_series_input(self):
        probs = pd.Series([0.5, 0.5], index=[0, 1])
        cdf = _prob_to_cdf(probs)

        self.assertAlmostEqual(cdf(0), 0.5)
        self.assertAlmostEqual(cdf(1), 1.0)

    def test_dataframe_input(self):
        probs = pd.DataFrame([[0.4, 0.6]], columns=[0, 1])
        cdf = _prob_to_cdf(probs)

        self.assertAlmostEqual(cdf(0), 0.4)
        self.assertAlmostEqual(cdf(1), 1.0)

    def test_returns_callable(self):
        probs = [1.0]
        cdf = _prob_to_cdf(probs)
        self.assertTrue(callable(cdf))

    def test_cdf_is_non_decreasing(self):
        probs = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
        cdf = _prob_to_cdf(probs)

        prev = 0.0
        for x in range(len(probs)):
            current = cdf(x)
            self.assertGreaterEqual(current, prev)
            prev = current


if __name__ == "__main__":
    unittest.main()
