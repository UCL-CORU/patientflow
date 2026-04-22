"""Tests for EvaluationInputsBuilder."""

from datetime import date, timedelta
import unittest

import numpy as np
import pandas as pd

from patientflow.evaluate.builder import EvaluationInputsBuilder
from patientflow.evaluate.types import ArrivalDeltaPayload, SurvivalCurvePayload
from patientflow.load import get_model_key


class TestEvaluationInputsBuilder(unittest.TestCase):
    """Tests for builder-based input assembly."""

    def test_add_classifier(self) -> None:
        visits_df = pd.DataFrame(
            {"visit_number": [1, 2], "snapshot_date": ["2026-01-01", "2026-01-01"]}
        )
        builder = EvaluationInputsBuilder(
            prediction_times=[(9, 30)], evaluation_targets={}
        )
        builder.add_classifier(
            "ed_current_admission_classifier",
            trained_models=[object()],
            visits_df=visits_df,
            label_col="is_admitted",
        )
        built = builder.build()
        self.assertIn("ed_current_admission_classifier", built.classifier_inputs)
        payload = built.classifier_inputs["ed_current_admission_classifier"]
        self.assertEqual(len(payload.trained_models), 1)
        self.assertTrue(payload.visits_df.equals(visits_df))

    def test_add_distributions_from_service_dict(self) -> None:
        prediction_times = [(9, 30)]
        builder = EvaluationInputsBuilder(
            prediction_times=prediction_times,
            evaluation_targets={},
        )
        model_key = get_model_key("admissions", (9, 30))
        prob_dist_by_service = {
            model_key: {
                "medical": {
                    date(2026, 1, 1): {
                        "agg_predicted": pd.DataFrame(
                            {"agg_proba": [0.2, 0.8]},
                            index=[0, 1],
                        ),
                        "agg_observed": 1,
                    }
                }
            }
        }
        builder.add_distributions_from_service_dict(
            "ed_current_beds",
            prob_dist_by_service=prob_dist_by_service,
            model_name="admissions",
        )
        built = builder.build()
        snapshot_result = built.flow_inputs_by_service["medical"]["ed_current_beds"][
            (9, 30)
        ][date(2026, 1, 1)]
        np.testing.assert_allclose(snapshot_result.predicted_pmf, np.array([0.2, 0.8]))
        self.assertEqual(snapshot_result.observed, 1)

    def test_add_arrival_deltas_and_survival_curve(self) -> None:
        builder = EvaluationInputsBuilder(
            prediction_times=[(9, 30), (12, 0)],
            evaluation_targets={},
        )
        arrivals_df = pd.DataFrame({"arrival_datetime": ["2026-01-01T09:00:00Z"]})
        train_df = pd.DataFrame({"arrival_datetime": ["2026-01-01T08:00:00Z"]})
        test_df = pd.DataFrame({"arrival_datetime": ["2026-01-02T08:00:00Z"]})

        builder.add_arrival_deltas(
            "ed_yta_arrival_rates",
            arrivals_by_service={"medical": arrivals_df},
            snapshot_dates=[date(2026, 1, 1)],
            prediction_window=timedelta(hours=4),
        )
        builder.add_survival_curve(
            "ed_current_window_prob",
            service_name="medical",
            train_df=train_df,
            test_df=test_df,
        )
        built = builder.build()

        delta_payload = built.flow_inputs_by_service["medical"]["ed_yta_arrival_rates"][
            (9, 30)
        ]
        self.assertIsInstance(delta_payload, ArrivalDeltaPayload)
        self.assertIsNone(delta_payload.predictor)
        self.assertIsNone(delta_payload.filter_key)
        self.assertFalse(delta_payload.strict_prediction_date)

        survival_payload = built.flow_inputs_by_service["medical"][
            "ed_current_window_prob"
        ][(12, 0)]
        self.assertIsInstance(survival_payload, SurvivalCurvePayload)

    def test_add_arrival_deltas_with_predictors(self) -> None:
        builder = EvaluationInputsBuilder(
            prediction_times=[(9, 30)],
            evaluation_targets={},
        )
        arrivals_df = pd.DataFrame({"arrival_datetime": ["2026-01-01T09:00:00Z"]})
        stub_predictor = object()
        builder.add_arrival_deltas(
            "ed_yta_arrival_rates",
            arrivals_by_service={"medical": arrivals_df},
            snapshot_dates=[date(2026, 1, 1)],
            prediction_window=timedelta(hours=4),
            predictors_by_service={"medical": stub_predictor},
            filter_keys_by_service={"medical": "all"},
            strict_prediction_date=True,
        )
        built = builder.build()
        payload = built.flow_inputs_by_service["medical"]["ed_yta_arrival_rates"][
            (9, 30)
        ]
        self.assertIs(payload.predictor, stub_predictor)
        self.assertEqual(payload.filter_key, "all")
        self.assertTrue(payload.strict_prediction_date)

    def test_add_distribution_observations(self) -> None:
        builder = EvaluationInputsBuilder(
            prediction_times=[(9, 30)],
            evaluation_targets={},
        )
        observations = pd.DataFrame(
            {
                "arrival_datetime": [pd.Timestamp("2026-01-01 10:00:00+00:00")],
                "departure_datetime": [pd.Timestamp("2026-01-01 11:00:00+00:00")],
                "specialty": ["medical"],
            }
        )

        builder.add_distribution_observations(
            "ed_yta_beds",
            observations_by_service={"medical": observations},
            prediction_window=timedelta(hours=4),
            start_time_col="arrival_datetime",
            end_time_col="departure_datetime",
        )
        built = builder.build()
        observation_input = built.observation_inputs_by_service["medical"][
            "ed_yta_beds"
        ][(9, 30)]
        self.assertEqual(observation_input.start_time_col, "arrival_datetime")
        self.assertEqual(observation_input.end_time_col, "departure_datetime")
        self.assertEqual(observation_input.prediction_window, timedelta(hours=4))


if __name__ == "__main__":
    unittest.main()
