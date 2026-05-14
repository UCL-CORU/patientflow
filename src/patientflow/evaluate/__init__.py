"""Evaluation package: typed inputs, runner, and per-mode handlers.

Typical use is to build `EvaluationInputs` with `EvaluationInputsBuilder`,
then call `run_evaluation` to write a timestamped run directory (charts plus
`scalars.json`). This package does not re-export symbols; import from
submodules.

For example::

    from patientflow.evaluate.runner import run_evaluation
    from patientflow.evaluate.legacy_api import calc_mae_mpe

See Also
--------
patientflow.evaluate.inputs : Builder and input datatypes.
patientflow.evaluate.runner : Orchestration and artefact layout.
patientflow.evaluate.observations : Observation counting for distribution targets.
"""
