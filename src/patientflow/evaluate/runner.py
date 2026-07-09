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

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from patientflow.evaluate.handlers import (
    evaluate_arrival_deltas,
    evaluate_classifier_model_diagnostics,
    evaluate_classifier_probability_quality,
    evaluate_distribution,
    evaluate_survival_curve,
    evaluate_transition_matrix,
)
from patientflow.evaluate.inputs import EvaluationInputs
from patientflow.evaluate.scalars import ScalarsCollector

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover
    pass

from importlib.metadata import version as package_version

EVALUATION_RUN_MANIFEST = "evaluation_run.yaml"


def _patientflow_version() -> str:
    try:
        return package_version("patientflow")
    except Exception:
        return "unknown"


def evaluation_targets_for_manifest(
    inputs: EvaluationInputs,
) -> list[Dict[str, Any]]:
    """Serialise ``EvaluationTarget`` rows for ``evaluation_run.yaml``."""
    return [
        {
            "flow_name": t.flow_name,
            "flow_type": t.flow_type,
            "evaluation_mode": t.evaluation_mode,
            "component": t.component,
            "observation_mode": t.observation_mode,
        }
        for t in inputs.evaluation_targets
    ]


def prediction_dict_for_manifest(
    prediction_dict: Mapping[Tuple[int, int], timedelta],
) -> Dict[str, float]:
    """Serialise a prediction schedule for ``evaluation_run.yaml``.

    Keys use uclhflow-style stringified ``[hour, minute]`` tuples; values are
    horizon length in hours.
    """
    return {
        json.dumps(list(pt)): window.total_seconds() / 3600.0
        for pt, window in sorted(prediction_dict.items())
    }


def write_evaluation_run_manifest(
    run_dir: Path,
    *,
    output_root: Path,
    run_name: str,
    inputs: EvaluationInputs,
    training_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``evaluation_run.yaml`` describing this evaluation run.

    The ``evaluation`` block records run settings and the ``prediction_dict``
    schedule used. An optional ``training_metadata`` block may be supplied by the
    caller (for example uclhflow splits or aspirational curve parameters);
    patientflow does not load or copy any repository ``config.yaml`` by default.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluation run directory.
    output_root : pathlib.Path
        Base directory for evaluation runs.
    run_name : str
        Run subdirectory name (timestamp or custom).
    inputs : EvaluationInputs
        Built evaluation inputs.
    training_metadata : mapping, optional
        Caller-supplied YAML-serialisable training context (not auto-discovered).

    Returns
    -------
    pathlib.Path
        Path to the written manifest file.
    """
    manifest: Dict[str, Any] = {
        "evaluation": {
            "output_root": str(output_root),
            "run_name": run_name,
            "eval_split": inputs.eval_split,
            "flow_selection": asdict(inputs.flow_selection),
            "n_targets": len(inputs.evaluation_targets),
            "evaluation_targets": evaluation_targets_for_manifest(inputs),
            "prediction_dict": prediction_dict_for_manifest(inputs.prediction_dict),
            "patientflow_version": _patientflow_version(),
        },
    }
    if training_metadata is not None:
        manifest["training_metadata"] = dict(training_metadata)

    manifest_path = run_dir / EVALUATION_RUN_MANIFEST
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path


def run_evaluation(
    output_root: Path,
    inputs: EvaluationInputs,
    *,
    run_name: Optional[str] = None,
    training_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute every `EvaluationTarget` in `inputs` and write artefacts.

    Creates a timestamped subdirectory under `output_root` containing:

    - `evaluation_run.yaml` — run settings under ``evaluation:`` (including
      ``evaluation_targets`` with ``observation_mode``, ``prediction_dict``,
      and ``eval_split``), plus optional caller ``training_metadata``.
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
    training_metadata : mapping, optional
        Optional training context written under ``training_metadata`` in the
        manifest (caller-defined; not loaded from patientflow ``config.yaml``).

    Returns
    -------
    dict
        Keys `run_dir`, `scalars_path`, `manifest_path` (each a `pathlib.Path`),
        and `n_targets` (`int`).
    """
    output_root = Path(output_root)
    stamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    scalars_path = run_dir / "scalars.json"
    collector = ScalarsCollector()
    if scalars_path.is_file():
        collector.load_prior_from_path(scalars_path)

    manifest_path = write_evaluation_run_manifest(
        run_dir,
        output_root=output_root,
        run_name=stamp,
        inputs=inputs,
        training_metadata=training_metadata,
    )

    classifiers_dir = run_dir / "classifiers"
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
                    if not classifiers_dir.exists():
                        classifiers_dir.mkdir(parents=True, exist_ok=True)
                evaluate_classifier_probability_quality(
                    inputs,
                    target,
                    classifiers_dir=classifiers_dir / target.flow_name,
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
            case "transition_matrix":
                evaluate_transition_matrix(
                    inputs,
                    target,
                    transitions_dir=run_dir / "transitions",
                    collector=collector,
                )
            case _:
                raise ValueError(f"Unknown evaluation_mode: {target.evaluation_mode!r}")

    collector.write_json(scalars_path)
    return {
        "run_dir": run_dir,
        "scalars_path": scalars_path,
        "manifest_path": manifest_path,
        "n_targets": len(inputs.evaluation_targets),
    }
