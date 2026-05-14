"""Materialise evaluation runs: directory layout, config snapshot, and dispatch.

The runner writes timestamped artefacts under a base `output_root` and calls
per-mode functions in `patientflow.evaluate.handlers` (there is no handler
registry).

See Also
--------
patientflow.evaluate.handlers
patientflow.evaluate.inputs.EvaluationInputs
patientflow.evaluate.inputs.EvaluationInputsBuilder
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from patientflow.evaluate.handlers import (
    evaluate_arrival_deltas,
    evaluate_classifier_model_diagnostics,
    evaluate_classifier_probability_quality,
    evaluate_distribution,
    evaluate_survival_curve,
)
from patientflow.evaluate.inputs import EvaluationInputs
from patientflow.evaluate.scalars import ScalarsCollector

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover
    pass


def run_evaluation(
    output_root: Path,
    inputs: EvaluationInputs,
    *,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute every `EvaluationTarget` in `inputs` and write artefacts.

    Creates a timestamped subdirectory under `output_root` containing:

    - `config.yaml` — `flow_selection`, `prediction_times`, and run metadata.
    - `scalars.json` — `evaluation_rows` plus optional `_service_summary`
      fragments merged by handlers (distribution and arrival modes attach
      per-slice service coverage).

    Notes
    -----
    Row de-duplication uses `patientflow.evaluate.scalars.scalar_merge_key`.
    Classifier diagnostics, probability-quality, and distribution rows use
    distinct `evaluation_mode` / `model_name` combinations so keys do not clash.
    Survival rows use `prediction_time: null` and `service: _all_`.

    Dispatch uses `match` / `case` on `target.evaluation_mode` (no handler
    registry).

    Parameters
    ----------
    output_root : pathlib.Path
        Base directory for evaluation runs.
    inputs : EvaluationInputs
        Immutable inputs from `EvaluationInputsBuilder.build()`.
    run_name : str, optional
        Subdirectory name; default is `YYYYMMDD_HHMMSS` from the current time.

    Returns
    -------
    dict
        Keys `run_dir`, `scalars_path` (each a `pathlib.Path`), and
        `n_targets` (`int`).
    """
    output_root = Path(output_root)
    stamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    scalars_path = run_dir / "scalars.json"
    collector = ScalarsCollector()
    if scalars_path.is_file():
        collector.load_prior_from_path(scalars_path)

    config = {
        "output_root": str(output_root),
        "run_name": stamp,
        "flow_selection": asdict(inputs.flow_selection),
        "prediction_times": [list(t) for t in inputs.prediction_times],
        "n_targets": len(inputs.evaluation_targets),
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    classifiers_dir = run_dir / "classifiers"
    services_dir = run_dir / "services"
    distributions_dir = run_dir / "distributions"
    arrivals_dir = run_dir / "arrivals"
    survival_dir = run_dir / "survival"

    for target in inputs.evaluation_targets:
        match target.evaluation_mode:
            case "classifier_model_diagnostics":
                if inputs.classifier_by_flow.get(target.flow_name):
                    if not classifiers_dir.exists():
                        classifiers_dir.mkdir(parents=True, exist_ok=True)
                evaluate_classifier_model_diagnostics(
                    inputs,
                    target,
                    classifiers_dir=classifiers_dir / target.flow_name,
                    collector=collector,
                )
            case "classifier_probability_quality":
                if inputs.classifier_by_flow.get(target.flow_name):
                    if not services_dir.exists():
                        services_dir.mkdir(parents=True, exist_ok=True)
                evaluate_classifier_probability_quality(
                    inputs,
                    target,
                    services_dir=services_dir / target.flow_name,
                    collector=collector,
                )
            case "distribution":
                evaluate_distribution(
                    inputs,
                    target,
                    distributions_dir=distributions_dir,
                    collector=collector,
                )
            case "arrival_deltas":
                evaluate_arrival_deltas(
                    inputs,
                    target,
                    arrivals_dir=arrivals_dir,
                    collector=collector,
                )
            case "survival_curve":
                survival_dir.mkdir(parents=True, exist_ok=True)
                evaluate_survival_curve(
                    inputs,
                    target,
                    survival_dir=survival_dir,
                    collector=collector,
                )
            case _:
                raise ValueError(f"Unknown evaluation_mode: {target.evaluation_mode!r}")

    collector.write_json(scalars_path)
    return {
        "run_dir": run_dir,
        "scalars_path": scalars_path,
        "n_targets": len(inputs.evaluation_targets),
    }
