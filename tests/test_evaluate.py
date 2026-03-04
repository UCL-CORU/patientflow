"""Tests for the patientflow.evaluate module.

Tier 1: Unit tests for calculate_results and data-structure helpers.
Tier 2: Orchestration tests verifying correct routing and scalar output.
"""

import json
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from patientflow.evaluate import (
    EvaluationTarget,
    _metrics_from_training_artifacts,
    _upsert_scalar_metadata,
    calculate_results,
    evaluate_classifier,
    evaluate_flow,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluation_target(**overrides):
    defaults = dict(
        name="test_target",
        flow_type="pmf",
        aspirational=False,
        components=["arrivals"],
        flow_selection=None,
        evaluation_mode="distribution",
    )
    defaults.update(overrides)
    return EvaluationTarget(**defaults)


def _make_trained_model(prediction_time=(9, 30), metrics=None):
    """Build a minimal mock trained model with the given metrics."""
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
# Tier 1 – Core logic and data-structure helpers
# ---------------------------------------------------------------------------


class TestCalculateResults(unittest.TestCase):
    """Tests for calculate_results."""

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


class TestUpsertScalarMetadata(unittest.TestCase):
    """Tests for _upsert_scalar_metadata."""

    def test_basic_insertion(self):
        scalars = {}
        target = _make_evaluation_target()
        _upsert_scalar_metadata(
            scalars=scalars,
            flow_name="my_flow",
            evaluation_target=target,
            prediction_time_key="0930",
            evaluated=True,
            component_name="arrivals",
        )
        node = scalars["my_flow"]
        self.assertEqual(node["flow_type"], "pmf")
        time_node = node["components"]["arrivals"]["prediction_times"]["0930"]
        self.assertTrue(time_node["evaluated"])

    def test_with_service_name(self):
        scalars = {}
        target = _make_evaluation_target()
        _upsert_scalar_metadata(
            scalars=scalars,
            flow_name="my_flow",
            evaluation_target=target,
            prediction_time_key="1200",
            evaluated=False,
            reason="test reason",
            service_name="medical",
            component_name="arrivals",
        )
        time_node = scalars["my_flow"]["services"]["medical"]["components"]["arrivals"][
            "prediction_times"
        ]["1200"]
        self.assertFalse(time_node["evaluated"])
        self.assertEqual(time_node["reason"], "test reason")

    def test_metrics_merged(self):
        scalars = {}
        target = _make_evaluation_target()
        _upsert_scalar_metadata(
            scalars=scalars,
            flow_name="my_flow",
            evaluation_target=target,
            prediction_time_key="0930",
            evaluated=True,
            metrics={"mae": 1.5, "mpe": 12.0},
            component_name="arrivals",
        )
        time_node = scalars["my_flow"]["components"]["arrivals"]["prediction_times"][
            "0930"
        ]
        self.assertAlmostEqual(time_node["mae"], 1.5)


class TestMetricsFromTrainingArtifacts(unittest.TestCase):
    """Tests for _metrics_from_training_artifacts."""

    def test_selected_eval_metrics(self):
        model = _make_trained_model()
        result = _metrics_from_training_artifacts(model)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["auroc"], 0.88)
        self.assertTrue(result["reliability"]["is_reliable"])

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
# Tier 2 – Orchestration routing tests
# ---------------------------------------------------------------------------


class TestRunEvaluation(unittest.TestCase):
    """Smoke tests for run_evaluation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_requires_prediction_times(self):
        with self.assertRaises(ValueError):
            run_evaluation(output_root=self.tmpdir, prediction_times=None)
        with self.assertRaises(ValueError):
            run_evaluation(output_root=self.tmpdir, prediction_times=[])

    def test_minimal_run_creates_scalars(self):
        result = run_evaluation(
            output_root=self.tmpdir,
            prediction_times=[(9, 30)],
            run_label="test_run",
            services=["test_service"],
        )
        scalars_path = Path(result["scalars_path"])
        self.assertTrue(scalars_path.exists())
        scalars = json.loads(scalars_path.read_text())
        self.assertIn("_meta", scalars)
        self.assertEqual(scalars["_meta"]["schema_version"], "phase2")

    def test_output_directory_structure(self):
        result = run_evaluation(
            output_root=self.tmpdir,
            prediction_times=[(9, 30)],
            run_label="test_run",
            services=[],
        )
        root = Path(result["output_root"])
        self.assertTrue((root / "classifiers").is_dir())
        self.assertTrue((root / "services").is_dir())
        self.assertTrue((root / "scalars.json").is_file())

    def test_return_structure(self):
        result = run_evaluation(
            output_root=self.tmpdir,
            prediction_times=[(9, 30), (12, 0)],
            run_label="test_run",
            services=[],
        )
        self.assertIn("output_root", result)
        self.assertIn("run_label", result)
        self.assertIn("scalars_path", result)
        self.assertIn("n_flows", result)
        self.assertIn("prediction_times", result)
        self.assertEqual(result["prediction_times"], ["0930", "1200"])

    def test_custom_run_label(self):
        result = run_evaluation(
            output_root=self.tmpdir,
            prediction_times=[(9, 30)],
            run_label="my_custom_label",
            services=[],
        )
        self.assertEqual(result["run_label"], "my_custom_label")
        self.assertIn("my_custom_label", result["output_root"])


class TestEvaluateClassifier(unittest.TestCase):
    """Tests for evaluate_classifier."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_root = Path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_with_trained_model(self):
        scalars = {}
        target = _make_evaluation_target(
            name="test_clf", flow_type="classifier", evaluation_mode="classifier"
        )
        model = _make_trained_model(prediction_time=(9, 30))
        evaluate_classifier(
            classifier_name="test_clf",
            prediction_times=[(9, 30)],
            scalars=scalars,
            evaluation_target=target,
            output_root=self.output_root,
            classifier_input={"trained_models": [model]},
        )
        time_node = scalars["test_clf"]["components"]["classifier"][
            "prediction_times"
        ]["0930"]
        self.assertTrue(time_node["evaluated"])
        self.assertAlmostEqual(time_node["auroc"], 0.88)


class TestEvaluateFlow(unittest.TestCase):
    """Tests for evaluate_flow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_root = Path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_aspirational_skips_evaluation(self):
        scalars = {}
        target = _make_evaluation_target(aspirational=True)
        evaluate_flow(
            service_name="medical",
            flow_name="test_flow",
            prediction_times=[(9, 30)],
            scalars=scalars,
            evaluation_target=target,
            output_root=self.output_root,
        )
        time_node = scalars["test_flow"]["services"]["medical"]["components"][
            "arrivals"
        ]["prediction_times"]["0930"]
        self.assertFalse(time_node["evaluated"])
        self.assertIn("Aspirational", time_node["reason"])

    def test_arrival_deltas_missing_keys(self):
        scalars = {}
        target = _make_evaluation_target(evaluation_mode="arrival_deltas")
        evaluate_flow(
            service_name="medical",
            flow_name="test_flow",
            prediction_times=[(9, 30)],
            scalars=scalars,
            evaluation_target=target,
            output_root=self.output_root,
            flow_input={(9, 30): {"incomplete": True}},
        )
        time_node = scalars["test_flow"]["services"]["medical"]["components"][
            "arrivals"
        ]["prediction_times"]["0930"]
        self.assertFalse(time_node["evaluated"])
        self.assertIn("Arrival delta", time_node["reason"])

    def test_survival_curve_missing_keys(self):
        scalars = {}
        target = _make_evaluation_target(evaluation_mode="survival_curve")
        evaluate_flow(
            service_name="medical",
            flow_name="test_flow",
            prediction_times=[(9, 30)],
            scalars=scalars,
            evaluation_target=target,
            output_root=self.output_root,
            flow_input={(9, 30): {"only_train": True}},
        )
        time_node = scalars["test_flow"]["services"]["medical"]["components"][
            "arrivals"
        ]["prediction_times"]["0930"]
        self.assertFalse(time_node["evaluated"])
        self.assertIn("Survival curve", time_node["reason"])


if __name__ == "__main__":
    unittest.main()
