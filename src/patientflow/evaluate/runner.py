"""Top-level orchestration for typed evaluation runs.

This module defines the entry point that coordinates:

- target iteration and per-mode dispatch
- service-level evaluation where applicable
- output folder creation and config copying
- flat scalar payload generation
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from patientflow.evaluate.constants import RELIABILITY_THRESHOLDS
from patientflow.evaluate.handlers import (
    evaluate_arrival_deltas,
    evaluate_aspirational_skip,
    evaluate_classifier,
    evaluate_distribution,
    evaluate_survival_curve,
)
from patientflow.evaluate.scalars import ScalarsCollector, default_scalars_meta
from patientflow.evaluate.types import EvaluationInputs


def run_evaluation(
    output_root: Union[str, Path],
    inputs: EvaluationInputs,
    *,
    config_path: Optional[Union[str, Path]] = None,
    run_label: Optional[str] = None,
    services: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run typed evaluation targets and write plots/scalars to disk.

    Parameters
    ----------
    output_root
        Root directory under which a run directory is created.
    inputs
        Typed evaluation inputs including targets, prediction times, and payloads.
    config_path
        Optional config file copied to ``config.yaml`` in the run directory.
    run_label
        Optional explicit run label. If omitted, a UTC timestamp is used.
    services
        Optional service subset for non-classifier evaluations. If omitted,
        services are inferred from ``inputs.flow_inputs_by_service``.

    Returns
    -------
    Dict[str, Any]
        Run summary including output paths, counts, and prediction times.

    Raises
    ------
    ValueError
        If no prediction times are provided.
    """
    if not inputs.prediction_times:
        raise ValueError("prediction_times must be provided explicitly.")

    root = Path(output_root)
    effective_run_label = run_label or datetime.now(timezone.utc).strftime(
        "run_%Y%m%d_%H%M%S"
    )
    root = root / effective_run_label
    root.mkdir(parents=True, exist_ok=True)
    (root / "classifiers").mkdir(exist_ok=True)
    (root / "services").mkdir(exist_ok=True)

    if config_path is not None:
        source = Path(config_path)
        destination = root / "config.yaml"
        if source.exists():
            shutil.copy2(source, destination)

    collector = ScalarsCollector()
    services_to_process = services or sorted(inputs.flow_inputs_by_service.keys())

    for flow_name, target in inputs.evaluation_targets.items():
        mode = target.evaluation_mode

        if mode == "classifier":
            evaluate_classifier(
                flow_name=flow_name,
                target=target,
                prediction_times=inputs.prediction_times,
                collector=collector,
                output_root=root,
                classifier_input=inputs.classifier_inputs.get(flow_name),
            )
            continue

        for service_name in services_to_process:
            payload_by_time = (
                inputs.flow_inputs_by_service.get(service_name, {}).get(flow_name, {})
            )

            if mode == "distribution":
                evaluate_distribution(
                    service_name=service_name,
                    flow_name=flow_name,
                    target=target,
                    prediction_times=inputs.prediction_times,
                    collector=collector,
                    output_root=root,
                    snapshots_by_time=payload_by_time,
                )
            elif mode == "arrival_deltas":
                evaluate_arrival_deltas(
                    service_name=service_name,
                    flow_name=flow_name,
                    target=target,
                    prediction_times=inputs.prediction_times,
                    collector=collector,
                    output_root=root,
                    payloads_by_time=payload_by_time,
                )
            elif mode == "survival_curve":
                evaluate_survival_curve(
                    service_name=service_name,
                    flow_name=flow_name,
                    target=target,
                    prediction_times=inputs.prediction_times,
                    collector=collector,
                    output_root=root,
                    payloads_by_time=payload_by_time,
                )
            elif mode == "aspirational_skip":
                evaluate_aspirational_skip(
                    service_name=service_name,
                    flow_name=flow_name,
                    target=target,
                    prediction_times=inputs.prediction_times,
                    collector=collector,
                )
            else:
                for prediction_time in inputs.prediction_times:
                    collector.record(
                        flow=flow_name,
                        service=service_name,
                        component=target.component,
                        prediction_time=f"{prediction_time[0]:02d}{prediction_time[1]:02d}",
                        flow_type=target.flow_type,
                        aspirational=target.aspirational,
                        evaluated=False,
                        reason=f"Unsupported evaluation mode: {mode}",
                    )

    scalars_path = root / "scalars.json"
    meta = default_scalars_meta(RELIABILITY_THRESHOLDS, schema_version=3)
    scalars_path.write_text(
        json.dumps(collector.to_payload(meta=meta), indent=2, sort_keys=True)
    )

    return {
        "output_root": str(root),
        "run_label": effective_run_label,
        "scalars_path": str(scalars_path),
        "n_flows": len(inputs.evaluation_targets),
        "n_services": len(services_to_process),
        "prediction_times": [f"{pt[0]:02d}{pt[1]:02d}" for pt in inputs.prediction_times],
    }
