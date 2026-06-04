"""Evaluation package: typed inputs, runner, and per-mode handlers.

Typical use is to build `EvaluationInputs` with `EvaluationInputsBuilder`,
then call `run_evaluation` to write a timestamped run directory (charts plus
`scalars.json`).

Legacy scalar helpers from the former ``evaluate`` module are re-exported at
package scope for 1.6.2-style imports::

    from patientflow.evaluate import calc_mae_mpe, calculate_results

New evaluation API symbols live in submodules, for example::

    from patientflow.evaluate.runner import run_evaluation

See Also
--------
patientflow.evaluate.inputs : Builder and input datatypes.
patientflow.evaluate.runner : Orchestration and artefact layout.
patientflow.evaluate.legacy_api : Legacy MAE/MPE helpers.
patientflow.evaluate.observations : Observation counting for distribution targets.
"""

from patientflow.evaluate.legacy_api import calc_mae_mpe, calculate_results

__all__ = ["calc_mae_mpe", "calculate_results"]
