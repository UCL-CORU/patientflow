"""Tests for the patientflow.evaluate package-level backward-compatible API.

These tests verify that the public surface available via
``from patientflow.evaluate import ...`` works correctly, covering
both the legacy scalar helpers and the typed evaluation entry points.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from patientflow.evaluate import (
    EvaluationTarget,
    calc_mae_mpe,
    calculate_results,
    get_default_evaluation_targets,
)
from patientflow.evaluate.handlers import _metrics_from_training_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trained_model(prediction_time=(9, 30), metrics=None):
    selected = metrics or {
        "log_loss": 0.35,
        "auroc": 0.88,
        "auprc": 0.72,
        "n_samples": 200,
        "n_positive_cases": 80,
    }
    training_results = SimpleNamespace(
        prediction_time=prediction_time,
        selected_eval_metrics=selected,
    )
    return SimpleNamespace(
        training_results=training_results,
        selected_eval_metrics=selected,
    )


# ---------------------------------------------------------------------------
# Tier 1 – Legacy scalar helpers
# ---------------------------------------------------------------------------


class TestCalculateResults(unittest.TestCase):

    def test_basic_computation(self):
        result = calculate_results([10, 20, 30], [12.0, 18.0, 33.0])
        self.assertAlmostEqual(result["mae"], np.mean([2, 2, 3]))
        self.assertGreater(result["mpe"], 0)

    def test_perfect_prediction(self):
        result = calculate_results([5, 10, 15], [5.0, 10.0, 15.0])
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["mpe"], 0.0)

    def test_empty_inputs(self):
        result = calculate_results([], [])
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["mpe"], 0.0)

    def test_observed_zeros_excluded_from_mpe(self):
        result = calculate_results([5, 0], [0.0, 0.0])
        self.assertAlmostEqual(result["mpe"], 0.0)
        self.assertAlmostEqual(result["mae"], 2.5)


class TestCalcMaeMpe(unittest.TestCase):

    def test_single_time(self):
        prob_dist_dict_all = {
            "admissions_0930": {
                "2026-01-01": {
                    "agg_predicted": pd.DataFrame(
                        {"agg_proba": [0.0, 1.0]}, index=[0, 1]
                    ),
                    "agg_observed": 1,
                }
            }
        }
        result = calc_mae_mpe(prob_dist_dict_all)
        self.assertIn("admissions_0930", result)
        self.assertIn("mae", result["admissions_0930"])


# ---------------------------------------------------------------------------
# Tier 2 – Classifier metrics extraction
# ---------------------------------------------------------------------------


class TestMetricsFromTrainingArtifacts(unittest.TestCase):

    def test_selected_eval_metrics(self):
        model = _make_trained_model()
        result = _metrics_from_training_artifacts(model)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["auroc"], 0.88)
        self.assertTrue(result["reliable"])

    def test_no_training_results(self):
        model = SimpleNamespace()
        self.assertIsNone(_metrics_from_training_artifacts(model))

    def test_missing_core_metrics_returns_none(self):
        model = _make_trained_model(metrics={"n_samples": 100})
        self.assertIsNone(_metrics_from_training_artifacts(model))

    def test_fallback_to_test_results(self):
        training_results = SimpleNamespace(
            prediction_time=(9, 30),
            selected_eval_metrics={},
            test_results={
                "test_logloss": 0.4,
                "test_auc": 0.85,
                "test_auprc": 0.7,
            },
        )
        model = SimpleNamespace(
            training_results=training_results,
            selected_eval_metrics={},
        )
        result = _metrics_from_training_artifacts(model)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["auroc"], 0.85)


# ---------------------------------------------------------------------------
# Tier 3 – Target registry
# ---------------------------------------------------------------------------


class TestDefaultTargetRegistry(unittest.TestCase):

    def test_returns_typed_targets(self):
        targets = get_default_evaluation_targets()
        self.assertIsInstance(targets, dict)
        self.assertGreater(len(targets), 0)
        for name, target in targets.items():
            self.assertIsInstance(target, EvaluationTarget)
            self.assertEqual(target.name, name)

    def test_aspirational_property(self):
        targets = get_default_evaluation_targets()
        aspirational_targets = {
            k: v for k, v in targets.items() if v.aspirational
        }
        self.assertGreater(len(aspirational_targets), 0)
        for target in aspirational_targets.values():
            self.assertEqual(target.evaluation_mode, "aspirational_skip")


if __name__ == "__main__":
    unittest.main()
