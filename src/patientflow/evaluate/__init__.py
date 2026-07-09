"""Evaluation package: typed inputs, runner, and per-mode handlers.

Typical use is to build `EvaluationInputs` with `EvaluationInputsBuilder`,
then call `run_evaluation` to write a timestamped run directory (charts plus
`scalars.json`). Evaluation modes include classifier diagnostics, distribution
comparison, arrival deltas, survival curves, and transition-matrix row
calibration.

Legacy scalar helpers from the former `evaluate` module are re-exported at
package scope for backward-compatible imports, for example
`from patientflow.evaluate import calc_mae_mpe, calculate_results`.

New evaluation API symbols live in submodules, for example
`from patientflow.evaluate.runner import run_evaluation`. See
`patientflow.evaluate.inputs`, `patientflow.evaluate.runner`,
`patientflow.evaluate.legacy_api`, and `patientflow.evaluate.observations`.
"""

from patientflow.evaluate.legacy_api import calc_mae_mpe, calculate_results

__all__ = ["calc_mae_mpe", "calculate_results"]
