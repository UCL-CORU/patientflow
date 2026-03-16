"""Tests for typed evaluation target conversion helpers."""

import unittest
from types import SimpleNamespace

from patientflow.evaluate.targets import (
    convert_legacy_target,
    convert_legacy_targets,
    get_default_evaluation_targets,
)
from patientflow.predict.types import FlowSelection


class TestEvaluationTargets(unittest.TestCase):
    """Tests for legacy -> typed evaluation target conversion."""

    def test_convert_legacy_target_single_component(self):
        legacy = SimpleNamespace(
            name="ed_current_beds",
            flow_type="pmf",
            aspirational=False,
            components=["arrivals"],
            flow_selection=None,
            evaluation_mode="distribution",
        )
        converted = convert_legacy_target("ed_current_beds", legacy)
        self.assertEqual(converted.component, "arrivals")
        self.assertFalse(converted.aspirational)
        self.assertEqual(converted.evaluation_mode, "distribution")

    def test_convert_legacy_target_infers_component(self):
        legacy = SimpleNamespace(
            name="ed_current_beds",
            flow_type="pmf",
            aspirational=False,
            components=None,
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=False,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
            ),
            evaluation_mode="distribution",
        )
        converted = convert_legacy_target("ed_current_beds", legacy)
        self.assertEqual(converted.component, "arrivals")

    def test_convert_legacy_target_multiple_components_raises(self):
        legacy = SimpleNamespace(
            name="mixed_flow",
            flow_type="pmf",
            aspirational=False,
            components=["arrivals", "departures"],
            flow_selection=None,
            evaluation_mode="distribution",
        )
        with self.assertRaises(ValueError):
            convert_legacy_target("mixed_flow", legacy)

    def test_convert_legacy_targets_registry(self):
        legacy_targets = {
            "discharge_elective": SimpleNamespace(
                name="discharge_elective",
                flow_type="pmf",
                aspirational=False,
                components=["departures"],
                flow_selection=None,
                evaluation_mode="distribution",
            )
        }
        converted = convert_legacy_targets(legacy_targets)
        self.assertIn("discharge_elective", converted)
        self.assertEqual(converted["discharge_elective"].component, "departures")

    def test_default_registry_conversion(self):
        defaults = get_default_evaluation_targets()
        self.assertIn("ed_current_admission_classifier", defaults)
        self.assertIn("combined_net_elective", defaults)
        self.assertEqual(
            defaults["ed_current_admission_classifier"].component, "arrivals"
        )
        self.assertTrue(defaults["combined_net_emergency"].aspirational)


if __name__ == "__main__":
    unittest.main()
