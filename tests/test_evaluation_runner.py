"""Tests for the typed evaluation runner."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from patientflow.evaluate.runner import run_evaluation
from patientflow.evaluate.types import EvaluationInputs, EvaluationTarget


class TestEvaluationRunner(unittest.TestCase):
    """Integration-style tests for typed run_evaluation."""

    def test_requires_prediction_times(self) -> None:
        inputs = EvaluationInputs(
            prediction_times=[],
            evaluation_targets={},
            classifier_inputs={},
            flow_inputs_by_service={},
        )
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_evaluation(output_root=tmpdir, inputs=inputs)

    def test_writes_flat_scalars_payload(self) -> None:
        inputs = EvaluationInputs(
            prediction_times=[(9, 30)],
            evaluation_targets={
                "ed_current_admission_classifier": EvaluationTarget(
                    name="ed_current_admission_classifier",
                    flow_type="classifier",
                    evaluation_mode="classifier",
                    component="classifier",
                ),
                "combined_net_emergency": EvaluationTarget(
                    name="combined_net_emergency",
                    flow_type="pmf",
                    evaluation_mode="aspirational_skip",
                    component="net_flow",
                ),
            },
            classifier_inputs={},
            flow_inputs_by_service={"medical": {}},
        )
        with TemporaryDirectory() as tmpdir:
            result = run_evaluation(
                output_root=tmpdir,
                inputs=inputs,
                run_label="typed_run",
                services=["medical"],
            )
            scalars_path = Path(result["scalars_path"])
            payload = json.loads(scalars_path.read_text())
            self.assertIn("_meta", payload)
            self.assertEqual(payload["_meta"]["schema_version"], 4)
            self.assertIn("results", payload)
            self.assertEqual(len(payload["results"]), 2)
            rows_by_flow = {row["flow"]: row for row in payload["results"]}
            self.assertFalse(rows_by_flow["ed_current_admission_classifier"]["evaluated"])
            self.assertFalse(rows_by_flow["combined_net_emergency"]["evaluated"])
            self.assertTrue(rows_by_flow["combined_net_emergency"]["aspirational"])

    def test_rejects_unsupported_group_by(self) -> None:
        inputs = EvaluationInputs(
            prediction_times=[(9, 30)],
            evaluation_targets={},
            classifier_inputs={},
            flow_inputs_by_service={},
        )
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_evaluation(
                    output_root=tmpdir,
                    inputs=inputs,
                    group_by=("service_size",),
                )

    def test_records_meta_group_by_when_active(self) -> None:
        inputs = EvaluationInputs(
            prediction_times=[(9, 30)],
            evaluation_targets={
                "combined_net_emergency": EvaluationTarget(
                    name="combined_net_emergency",
                    flow_type="pmf",
                    evaluation_mode="aspirational_skip",
                    component="net_flow",
                ),
            },
            classifier_inputs={},
            flow_inputs_by_service={"medical": {}},
        )
        with TemporaryDirectory() as tmpdir:
            result = run_evaluation(
                output_root=tmpdir,
                inputs=inputs,
                run_label="grouped_run",
                services=["medical"],
                group_by=("weekday",),
            )
            payload = json.loads(Path(result["scalars_path"]).read_text())
            self.assertEqual(payload["_meta"]["group_by"], ["weekday"])

    def test_unsupported_mode_records_skipped(self) -> None:
        inputs = EvaluationInputs(
            prediction_times=[(12, 0)],
            evaluation_targets={
                "future_transfer_metric": EvaluationTarget(
                    name="future_transfer_metric",
                    flow_type="special",
                    evaluation_mode="transfer_matrix",
                    component="arrivals",
                )
            },
            classifier_inputs={},
            flow_inputs_by_service={"medical": {}},
        )
        with TemporaryDirectory() as tmpdir:
            result = run_evaluation(
                output_root=tmpdir,
                inputs=inputs,
                run_label="typed_run",
                services=["medical"],
            )
            payload = json.loads(Path(result["scalars_path"]).read_text())
            self.assertEqual(len(payload["results"]), 1)
            row = payload["results"][0]
            self.assertFalse(row["evaluated"])
            self.assertIn("Unsupported evaluation mode", row["reason"])


if __name__ == "__main__":
    unittest.main()
