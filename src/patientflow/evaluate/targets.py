"""Default `EvaluationTarget` list for a full evaluation matrix.

These targets use `flow_name="main"`. Swap or extend the list with
`EvaluationInputsBuilder.with_evaluation_targets` when your builder uses
different flow keys.

See Also
--------
patientflow.evaluate.inputs.EvaluationInputsBuilder.with_evaluation_targets
patientflow.evaluate.runner.run_evaluation
"""

from __future__ import annotations

from typing import List

from patientflow.evaluate.inputs import EvaluationTarget


def get_default_evaluation_targets() -> List[EvaluationTarget]:
    """Return the default registry of `EvaluationTarget` instances.

    Includes at least one target per `evaluation_mode` implemented by the
    runner, plus separate distribution targets for departures (elective,
    emergency, and all-inpatient) where the default matrix requires them.
    All targets use `flow_name="main"`; callers should replace with
    `EvaluationInputsBuilder.with_evaluation_targets` when their
    builder keys differ.

    Returns
    -------
    list of EvaluationTarget
        Frozen targets ready to pass to the builder or to filter.

    See Also
    --------
    patientflow.evaluate.inputs.EvaluationInputsBuilder.with_evaluation_targets
    patientflow.evaluate.observations.OBSERVATION_MODES
    """
    return [
        EvaluationTarget(
            flow_name="main",
            name="classifier_model_diagnostics_admissions",
            flow_type="admissions",
            evaluation_mode="classifier_model_diagnostics",
            component="model_diagnostics",
            observation_mode="admitted_at_some_point",
        ),
        EvaluationTarget(
            flow_name="main",
            name="classifier_probability_quality_admissions",
            flow_type="admissions",
            evaluation_mode="classifier_probability_quality",
            component="madcap_calibration",
            observation_mode="admitted_at_some_point",
        ),
        EvaluationTarget(
            flow_name="main",
            name="distribution_epudd_admissions",
            flow_type="admissions",
            evaluation_mode="distribution",
            component="epudd",
            observation_mode="admitted_at_some_point",
        ),
        EvaluationTarget(
            flow_name="main",
            name="distribution_epudd_departures_elective",
            flow_type="departures",
            evaluation_mode="distribution",
            component="epudd_departures_elective",
            observation_mode="departed_in_window",
        ),
        EvaluationTarget(
            flow_name="main",
            name="distribution_epudd_departures_emergency",
            flow_type="departures",
            evaluation_mode="distribution",
            component="epudd_departures_emergency",
            observation_mode="departed_in_window",
        ),
        EvaluationTarget(
            flow_name="main",
            name="distribution_epudd_departures_all_inpatient",
            flow_type="departures",
            evaluation_mode="distribution",
            component="epudd_departures_all_inpatient",
            observation_mode="departed_in_window",
        ),
        EvaluationTarget(
            flow_name="main",
            name="arrival_deltas_admissions",
            flow_type="admissions",
            evaluation_mode="arrival_deltas",
            component="cumulative_arrival_delta",
            observation_mode="arrived_in_window",
        ),
        EvaluationTarget(
            flow_name="main",
            name="survival_admission_time",
            flow_type="admissions",
            evaluation_mode="survival_curve",
            component="admission_time_survival",
            observation_mode="admitted_at_some_point",
        ),
    ]
