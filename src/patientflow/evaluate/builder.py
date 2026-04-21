"""Builder utilities for assembling ``EvaluationInputs``.

The builder in this module converts common upstream outputs into the nested
typed input structure expected by the evaluation runner, avoiding manual
dictionary reshaping in notebooks and scripts.

Supported assembly paths include:

- classifier payload registration
- distribution payload ingestion from service-level dictionaries
- arrival-delta payload registration
- survival-curve payload registration
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from patientflow.evaluate.adapters import (
    from_legacy_prediction_dict,
    from_legacy_prob_dist_dict,
)
from patientflow.evaluate.targets import get_default_evaluation_targets
from patientflow.evaluate.types import (
    ArrivalDeltaPayload,
    ClassifierInput,
    EvaluationInputs,
    ObservationInput,
    EvaluationTarget,
    SurvivalCurvePayload,
)
from patientflow.load import get_model_key


class EvaluationInputsBuilder:
    """Helper for assembling typed evaluation inputs from pipeline outputs.

    Parameters
    ----------
    prediction_times
        Prediction times that should be represented in the built inputs.
    evaluation_targets
        Optional explicit target registry. If omitted, defaults are used.
    """

    def __init__(
        self,
        prediction_times: List[Tuple[int, int]],
        evaluation_targets: Optional[Dict[str, EvaluationTarget]] = None,
    ):
        self.prediction_times = prediction_times
        self.evaluation_targets = evaluation_targets or get_default_evaluation_targets()
        self.classifier_inputs: Dict[str, ClassifierInput] = {}
        self.flow_inputs_by_service: Dict[str, Dict[str, Dict[Tuple[int, int], Any]]] = {}
        self.observation_inputs_by_service: Dict[
            str, Dict[str, Dict[Tuple[int, int], ObservationInput]]
        ] = {}

    def add_classifier(
        self,
        flow_name: str,
        *,
        trained_models: Any,
        visits_df: Optional[pd.DataFrame] = None,
        label_col: str = "is_admitted",
    ) -> "EvaluationInputsBuilder":
        """Register classifier payload for one target.

        Parameters
        ----------
        flow_name
            Classifier target name.
        trained_models
            Iterable or mapping of trained classifier artifacts.
        visits_df
            Evaluation dataframe used by classifier diagnostics.
        label_col
            Binary outcome label column in ``visits_df``.

        Returns
        -------
        EvaluationInputsBuilder
            Builder instance for fluent chaining.
        """
        models = list(trained_models.values()) if isinstance(trained_models, dict) else list(trained_models)
        self.classifier_inputs[flow_name] = ClassifierInput(
            trained_models=models,
            visits_df=visits_df,
            label_col=label_col,
        )
        return self

    def add_distributions_from_service_dict(
        self,
        flow_name: str,
        *,
        prob_dist_by_service: Mapping[str, Mapping[str, Mapping[date, Mapping[str, Any]]]],
        model_name: str,
    ) -> "EvaluationInputsBuilder":
        """Ingest service distribution outputs for one flow target.

        Parameters
        ----------
        flow_name
            Distribution flow target name.
        prob_dist_by_service
            Mapping returned by ``get_prob_dist_by_service`` keyed by model key.
            Payload entries with ``agg_observed`` are stored as
            :class:`SnapshotResult`; prediction-only entries are stored as
            :class:`PredictedSnapshotResult`.
        model_name
            Base model name used to derive model keys for prediction times.

        Returns
        -------
        EvaluationInputsBuilder
            Builder instance for fluent chaining.
        """
        for prediction_time in self.prediction_times:
            model_key = get_model_key(model_name, prediction_time)
            by_service = prob_dist_by_service.get(model_key, {})
            for service_name, legacy_prob_dist_dict in by_service.items():
                first_payload = next(iter(legacy_prob_dist_dict.values()), {})
                has_observed = "agg_observed" in first_payload
                if has_observed:
                    typed_prob_dist_dict = from_legacy_prob_dist_dict(
                        legacy_prob_dist_dict
                    )
                else:
                    typed_prob_dist_dict = from_legacy_prediction_dict(
                        legacy_prob_dist_dict
                    )
                self.flow_inputs_by_service.setdefault(service_name, {}).setdefault(
                    flow_name, {}
                )[prediction_time] = typed_prob_dist_dict
        return self

    def add_distribution_observations(
        self,
        flow_name: str,
        *,
        observations_by_service: Mapping[str, pd.DataFrame],
        prediction_window: timedelta,
        prediction_times: Optional[Iterable[Tuple[int, int]]] = None,
        label_col: str = "is_admitted",
        service_col: str = "specialty",
        start_time_col: str = "arrival_datetime",
        end_time_col: str = "departure_datetime",
        apply_service_filter: bool = True,
    ) -> "EvaluationInputsBuilder":
        """Register observation datasets for a distribution target.

        These inputs are consumed by ``run_evaluation`` to compute observed
        counts using each target's ``observation_mode`` at evaluation time.
        """
        selected_times = list(prediction_times or self.prediction_times)
        for service_name, visits in observations_by_service.items():
            for prediction_time in selected_times:
                self.observation_inputs_by_service.setdefault(service_name, {}).setdefault(
                    flow_name, {}
                )[prediction_time] = ObservationInput(
                    visits=visits,
                    prediction_window=prediction_window,
                    label_col=label_col,
                    service_col=service_col,
                    start_time_col=start_time_col,
                    end_time_col=end_time_col,
                    apply_service_filter=apply_service_filter,
                )
        return self

    def add_arrival_deltas(
        self,
        flow_name: str,
        *,
        arrivals_by_service: Mapping[str, pd.DataFrame],
        snapshot_dates: List[date],
        prediction_window: timedelta,
        yta_time_interval: timedelta = timedelta(minutes=15),
        prediction_times: Optional[Iterable[Tuple[int, int]]] = None,
        predictors_by_service: Optional[Mapping[str, Any]] = None,
        filter_keys_by_service: Optional[Mapping[str, str]] = None,
        strict_prediction_date: bool = False,
    ) -> "EvaluationInputsBuilder":
        """Register arrival-delta payloads for multiple services and times.

        Parameters
        ----------
        flow_name
            Arrival-delta target name.
        arrivals_by_service
            Mapping from service name to arrivals dataframe.
        snapshot_dates
            Snapshot dates to evaluate.
        prediction_window
            Forecast window associated with each prediction time.
        yta_time_interval
            Yet-to-arrive interval used by delta calculations.
        prediction_times
            Optional subset of prediction times. Defaults to builder times.
        predictors_by_service
            Optional mapping from service name to a fitted
            ``IncomingAdmissionPredictor`` (or compatible object). When
            provided for a service, the diagnostic uses the predictor's
            stored arrival rates as the expected baseline — matching the
            rate profile the deployed model uses. Services without a
            predictor fall back to the pooled rates derived from the
            arrivals dataframe.
        filter_keys_by_service
            Optional mapping from service name to the predictor filter
            key to read rates from. Only required when the fitted
            predictor has more than one weight key.
        strict_prediction_date
            When a weekday-aware predictor lacks per-weekday profiles,
            raise instead of silently falling back to pooled rates.

        Returns
        -------
        EvaluationInputsBuilder
            Builder instance for fluent chaining.
        """
        selected_times = list(prediction_times or self.prediction_times)
        predictors_by_service = predictors_by_service or {}
        filter_keys_by_service = filter_keys_by_service or {}
        for service_name, arrivals_df in arrivals_by_service.items():
            predictor = predictors_by_service.get(service_name)
            filter_key = filter_keys_by_service.get(service_name)
            for prediction_time in selected_times:
                payload = ArrivalDeltaPayload(
                    df=arrivals_df,
                    snapshot_dates=snapshot_dates,
                    prediction_window=prediction_window,
                    yta_time_interval=yta_time_interval,
                    predictor=predictor,
                    filter_key=filter_key,
                    strict_prediction_date=strict_prediction_date,
                )
                self.flow_inputs_by_service.setdefault(service_name, {}).setdefault(
                    flow_name, {}
                )[prediction_time] = payload
        return self

    def add_survival_curve(
        self,
        flow_name: str,
        *,
        service_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        start_time_col: str = "arrival_datetime",
        end_time_col: str = "departure_datetime",
        prediction_times: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> "EvaluationInputsBuilder":
        """Register survival-curve payload for one service.

        Parameters
        ----------
        flow_name
            Survival-curve target name.
        service_name
            Service to evaluate.
        train_df
            Training-period dataframe.
        test_df
            Test-period dataframe.
        start_time_col
            Start-time column name for duration calculation.
        end_time_col
            End-time column name for duration calculation.
        prediction_times
            Optional subset of prediction times. Defaults to builder times.

        Returns
        -------
        EvaluationInputsBuilder
            Builder instance for fluent chaining.
        """
        selected_times = list(prediction_times or self.prediction_times)
        for prediction_time in selected_times:
            self.flow_inputs_by_service.setdefault(service_name, {}).setdefault(
                flow_name, {}
            )[prediction_time] = SurvivalCurvePayload(
                train_df=train_df,
                test_df=test_df,
                start_time_col=start_time_col,
                end_time_col=end_time_col,
            )
        return self

    def build(self) -> EvaluationInputs:
        """Build final evaluation inputs object.

        Returns
        -------
        EvaluationInputs
            Fully assembled typed input container for ``run_evaluation``.
        """
        return EvaluationInputs(
            prediction_times=list(self.prediction_times),
            evaluation_targets=dict(self.evaluation_targets),
            classifier_inputs=dict(self.classifier_inputs),
            flow_inputs_by_service=dict(self.flow_inputs_by_service),
            observation_inputs_by_service=dict(self.observation_inputs_by_service),
        )
