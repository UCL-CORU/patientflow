"""Tests for subgroup-aware transfer routing in patientflow.predict.transfers."""

import unittest
import warnings

import numpy as np
import pandas as pd

from patientflow.predict.transfers import (
    build_per_patient_probabilities,
    compute_transfer_arrivals,
    transfer_weight_to_target,
)
from patientflow.predictors.subgroup_definitions import (
    assign_patient_subgroups,
    create_subgroup_functions,
    resolve_patient_subgroup,
)
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator


def _expected_value(pmf: np.ndarray) -> float:
    return float(np.sum(np.arange(len(pmf)) * pmf))


def _build_sex_split_model(services):
    """Fit a model where females route to gynae and males to surgery.

    The pooled ``["services"]`` row mixes both, so it assigns gynae mass to all
    cardiology patients, whereas subgroup routing sends gynae mass only to
    females.
    """
    females = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 10,
            "next_subspecialty": ["gynae"] * 10,
            "admission_type": ["emergency"] * 10,
            "age_on_arrival": [30] * 10,
            "sex": ["F"] * 10,
        }
    )
    males = pd.DataFrame(
        {
            "current_subspecialty": ["cardiology"] * 10,
            "next_subspecialty": ["surgery"] * 10,
            "admission_type": ["emergency"] * 10,
            "age_on_arrival": [30] * 10,
            "sex": ["M"] * 10,
        }
    )
    X = pd.concat([females, males], ignore_index=True)
    model = TransferProbabilityEstimator(cohort_col="admission_type")
    model.fit(X, set(services))
    return model


class TestResolvePatientSubgroup(unittest.TestCase):
    def setUp(self):
        self.funcs = create_subgroup_functions()

    def test_resolves_paediatric(self):
        row = pd.Series({"age_on_arrival": 5, "sex": None})
        self.assertEqual(resolve_patient_subgroup(row, self.funcs), "paediatric")

    def test_resolves_adult_male_young(self):
        row = pd.Series({"age_on_arrival": 30, "sex": "M"})
        self.assertEqual(resolve_patient_subgroup(row, self.funcs), "adult_male_young")

    def test_unmatched_adult_missing_sex(self):
        row = pd.Series({"age_on_arrival": 40, "sex": None})
        self.assertIsNone(resolve_patient_subgroup(row, self.funcs))

    def test_unmatched_adult_invalid_sex(self):
        row = pd.Series({"age_on_arrival": 40, "sex": "X"})
        self.assertIsNone(resolve_patient_subgroup(row, self.funcs))

    def test_overlapping_masks_raise(self):
        overlapping = {"a": lambda r: True, "b": lambda r: True}
        row = pd.Series({"age_on_arrival": 40, "sex": "M"})
        with self.assertRaises(ValueError):
            resolve_patient_subgroup(row, overlapping)

    def test_assign_patient_subgroups_frame(self):
        df = pd.DataFrame(
            {
                "age_on_arrival": [5, 30, 70, 40],
                "sex": ["M", "F", "M", None],
            }
        )
        result = assign_patient_subgroups(df, self.funcs)
        self.assertEqual(result.iloc[0], "paediatric")
        self.assertEqual(result.iloc[1], "adult_female_young")
        self.assertEqual(result.iloc[2], "adult_male_senior")
        self.assertIsNone(result.iloc[3])

    def test_assign_patient_subgroups_overlap_raises(self):
        df = pd.DataFrame({"age_on_arrival": [40], "sex": ["M"]})
        with self.assertRaises(ValueError):
            assign_patient_subgroups(df, {"a": lambda r: True, "b": lambda r: True})


class TestTransferWeightToTarget(unittest.TestCase):
    def setUp(self):
        self.services = ["cardiology", "surgery", "gynae"]
        self.model = _build_sex_split_model(self.services)

    def test_female_routes_to_gynae(self):
        row = pd.Series({"age_on_arrival": 30, "sex": "F"})
        weight = transfer_weight_to_target(
            row, "cardiology", "gynae", "emergency", self.model
        )
        self.assertAlmostEqual(weight, 1.0)

    def test_male_does_not_route_to_gynae(self):
        row = pd.Series({"age_on_arrival": 30, "sex": "M"})
        weight = transfer_weight_to_target(
            row, "cardiology", "gynae", "emergency", self.model
        )
        self.assertEqual(weight, 0.0)

    def test_unmatched_adult_zero_weight(self):
        row = pd.Series({"age_on_arrival": 40, "sex": None})
        weight = transfer_weight_to_target(
            row, "cardiology", "gynae", "emergency", self.model
        )
        self.assertEqual(weight, 0.0)


class TestBuildPerPatientProbabilities(unittest.TestCase):
    """Per-event routing matrix for transition-matrix evaluation."""

    def setUp(self):
        self.services = ["cardiology", "surgery", "gynae", "medicine"]
        females = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"] * 20,
                "next_subspecialty": ["gynae"] * 20,
                "admission_type": ["emergency"] * 20,
                "age_on_arrival": [30] * 20,
                "sex": ["F"] * 20,
            }
        )
        males = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"] * 20,
                "next_subspecialty": ["surgery"] * 20,
                "admission_type": ["emergency"] * 20,
                "age_on_arrival": [30] * 20,
                "sex": ["M"] * 20,
            }
        )
        X = pd.concat([females, males], ignore_index=True)
        self.model = TransferProbabilityEstimator(cohort_col="admission_type")
        self.model.fit(X, set(self.services))
        self.destinations = list(self.model.get_transition_matrix("emergency").columns)

    def test_subgroup_routing_and_exclusion(self):
        """Male and female rows get different p_i; unmatched adults are all-discharge."""
        events = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology", "cardiology", "cardiology"],
                "next_subspecialty": ["gynae", "surgery", None],
                "admission_type": ["emergency"] * 3,
                "age_on_arrival": [30, 30, 40],
                "sex": ["F", "M", None],
            }
        )
        result = build_per_patient_probabilities(
            events,
            "cardiology",
            "emergency",
            self.destinations,
            self.model,
        )

        self.assertEqual(result.routing_matrix.shape, (3, len(self.destinations)))
        np.testing.assert_allclose(result.routing_matrix.sum(axis=1), 1.0)

        gynae_idx = self.destinations.index("gynae")
        surgery_idx = self.destinations.index("surgery")
        discharge_idx = self.destinations.index("Discharge")

        # Female -> gynae; male -> surgery; missing sex -> Discharge (no pooled fallback).
        self.assertAlmostEqual(result.routing_matrix[0, gynae_idx], 1.0)
        self.assertAlmostEqual(result.routing_matrix[1, surgery_idx], 1.0)
        self.assertAlmostEqual(result.routing_matrix[2, discharge_idx], 1.0)
        self.assertEqual(result.n_excluded_unmatched, 1)
        self.assertEqual(result.n_subgroups_used, 2)


class TestComputeTransferArrivalsSubgroup(unittest.TestCase):
    def setUp(self):
        self.services = ["cardiology", "surgery", "gynae"]
        self.model = _build_sex_split_model(self.services)

    def _snapshots(self, rows):
        df = pd.DataFrame(rows)
        return df

    def test_male_only_sends_no_gynae_mass(self):
        """Subgroup routing keeps male inpatients away from gynae.

        The pooled ``["services"]`` row would assign gynae mass to the male
        (compound prob > 0), demonstrating the subgroup path differs from the
        former scalar-thin path.
        """
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "M",
                }
            ]
        )
        prob_emergency = pd.DataFrame({"pred_proba": [1.0]}, index=snapshots.index)

        result = compute_transfer_arrivals(
            snapshots,
            self.model,
            self.services,
            prob_departure_after_emergency=prob_emergency,
        )

        # Subgroup routing: male sends zero mass to gynae.
        np.testing.assert_allclose(result["emergency"]["gynae"], np.array([1.0]))
        # But the pooled (former) path would have assigned positive mass.
        pooled_compound = self.model.get_transfer_prob(
            "cardiology", "emergency"
        ) * self.model.get_destination_distribution("cardiology", "emergency").get(
            "gynae", 0.0
        )
        self.assertGreater(pooled_compound, 0.0)
        # Male routes to surgery instead.
        self.assertAlmostEqual(result["emergency"]["surgery"][1], 1.0, places=5)

    def test_female_only_sends_gynae_mass(self):
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "F",
                }
            ]
        )
        prob_emergency = pd.DataFrame({"pred_proba": [1.0]}, index=snapshots.index)

        result = compute_transfer_arrivals(
            snapshots,
            self.model,
            self.services,
            prob_departure_after_emergency=prob_emergency,
        )

        self.assertAlmostEqual(result["emergency"]["gynae"][1], 1.0, places=5)
        np.testing.assert_allclose(result["emergency"]["surgery"], np.array([1.0]))

    def test_mixed_cohort_routes_per_subgroup(self):
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "M",
                },
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "F",
                },
            ]
        )
        prob_emergency = pd.DataFrame({"pred_proba": [1.0, 1.0]}, index=snapshots.index)

        result = compute_transfer_arrivals(
            snapshots,
            self.model,
            self.services,
            prob_departure_after_emergency=prob_emergency,
        )

        # Exactly one arrival to each of gynae (female) and surgery (male).
        self.assertAlmostEqual(_expected_value(result["emergency"]["gynae"]), 1.0)
        self.assertAlmostEqual(_expected_value(result["emergency"]["surgery"]), 1.0)

    def test_unmatched_adult_excluded_and_warns(self):
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 40,
                    "sex": None,
                }
            ]
        )
        prob_emergency = pd.DataFrame({"pred_proba": [1.0]}, index=snapshots.index)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = compute_transfer_arrivals(
                snapshots,
                self.model,
                self.services,
                prob_departure_after_emergency=prob_emergency,
            )

        # No transfer arrivals anywhere for the excluded row.
        for service in self.services:
            np.testing.assert_allclose(result["emergency"][service], np.array([1.0]))
        # Exactly one summary warning about excluded rows.
        exclusion_warnings = [
            w for w in recorded if "excluded from transfer routing" in str(w.message)
        ]
        self.assertEqual(len(exclusion_warnings), 1)

    def test_unfitted_model_raises(self):
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "M",
                }
            ]
        )
        unfitted = TransferProbabilityEstimator(cohort_col="admission_type")
        with self.assertRaises(ValueError):
            compute_transfer_arrivals(snapshots, unfitted, self.services)

    def test_overlapping_subgroup_masks_raise(self):
        X = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"] * 4,
                "next_subspecialty": ["surgery"] * 4,
                "admission_type": ["emergency"] * 4,
                "age_on_arrival": [30] * 4,
                "sex": ["M", "F", "M", "F"],
            }
        )
        model = TransferProbabilityEstimator(
            cohort_col="admission_type",
            subgroup_functions={"a": lambda r: True, "b": lambda r: True},
        )
        model.fit(X, {"cardiology", "surgery"})

        snapshots = pd.DataFrame(
            {
                "current_subspecialty": ["cardiology"],
                "admission_type": ["emergency"],
                "age_on_arrival": [30],
                "sex": ["M"],
            }
        )
        prob_emergency = pd.DataFrame({"pred_proba": [1.0]}, index=snapshots.index)
        with self.assertRaises(ValueError):
            compute_transfer_arrivals(
                snapshots,
                model,
                ["cardiology", "surgery"],
                prob_departure_after_emergency=prob_emergency,
            )

    def test_empty_snapshots_returns_zero_arrivals(self):
        result = compute_transfer_arrivals(None, self.model, self.services)
        for admission_type in ("elective", "emergency"):
            for service in self.services:
                np.testing.assert_allclose(
                    result[admission_type][service], np.array([1.0])
                )

    def test_departure_probability_scales_arrivals(self):
        """A patient excluded from routing does not affect departure inputs.

        compute_transfer_arrivals consumes p_depart_i only as the Bernoulli base
        for routed arrivals; an excluded (unmatched) row contributes zero
        transfer mass regardless of its departure probability.
        """
        snapshots = self._snapshots(
            [
                {
                    "current_subspecialty": "cardiology",
                    "admission_type": "emergency",
                    "age_on_arrival": 30,
                    "sex": "F",
                }
            ]
        )
        prob_half = pd.DataFrame({"pred_proba": [0.5]}, index=snapshots.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_transfer_arrivals(
                snapshots,
                self.model,
                self.services,
                prob_departure_after_emergency=prob_half,
            )

        # Female routes to gynae with effective prob 0.5 -> EV 0.5.
        self.assertAlmostEqual(_expected_value(result["emergency"]["gynae"]), 0.5)

    def test_non_default_source_col(self):
        """source_col selects the service-unit column used for filtering sources."""
        snapshots = pd.DataFrame(
            {
                "reporting_unit": ["cardiology", "cardiology"],
                "admission_type": ["emergency", "emergency"],
                "age_on_arrival": [30, 30],
                "sex": ["F", "M"],
            }
        )
        prob = pd.DataFrame({"pred_proba": [1.0, 1.0]}, index=snapshots.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_transfer_arrivals(
                snapshots,
                self.model,
                self.services,
                prob_departure_after_emergency=prob,
                source_col="reporting_unit",
            )
        # Female -> gynae, male -> surgery; both with p_depart=1.
        self.assertAlmostEqual(_expected_value(result["emergency"]["gynae"]), 1.0)
        self.assertAlmostEqual(_expected_value(result["emergency"]["surgery"]), 1.0)


if __name__ == "__main__":
    unittest.main()
