"""Target registry helpers for typed evaluation.

This module contains utilities that:

- resolve a single component per target
- convert target-like registry entries into typed ``EvaluationTarget`` objects
- provide the default typed target registry used by builders/runners
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional

from patientflow.evaluation.types import EvaluationTarget
from patientflow.predict.types import FlowSelection


def _infer_components_from_flow_selection(flow_selection: FlowSelection) -> List[str]:
    components: List[str] = []
    has_inflow = any(
        [
            flow_selection.include_ed_current,
            flow_selection.include_ed_yta,
            flow_selection.include_non_ed_yta,
            flow_selection.include_elective_yta,
            flow_selection.include_transfers_in,
        ]
    )
    if has_inflow:
        components.append("arrivals")
    if flow_selection.include_departures:
        components.append("departures")
    if has_inflow and flow_selection.include_departures:
        components.append("net_flow")
    return components


def _resolve_component(
    *,
    components: Optional[Iterable[str]],
    flow_selection: Optional[FlowSelection],
    target_name: str,
) -> str:
    resolved: List[str] = []
    if components:
        resolved = list(components)
    elif flow_selection is not None:
        resolved = _infer_components_from_flow_selection(flow_selection)

    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) == 0:
        # Preserve a stable fallback marker for registry entries with no component.
        return "unspecified"
    raise ValueError(
        f"Target '{target_name}' resolves to multiple components {resolved}. "
        "Split this target into one target per component."
    )


def convert_legacy_target(name: str, legacy_target: Any) -> EvaluationTarget:
    """Convert a target-like object into a typed evaluation target.

    Parameters
    ----------
    name
        Registry key for the target.
    legacy_target
        Object with target attributes (for example ``name``, ``flow_type``,
        ``evaluation_mode``, and optional ``components`` / ``flow_selection``).

    Returns
    -------
    EvaluationTarget
        Typed target with a single resolved component.

    Raises
    ------
    ValueError
        If the source target resolves to multiple components.
    """
    component = _resolve_component(
        components=getattr(legacy_target, "components", None),
        flow_selection=getattr(legacy_target, "flow_selection", None),
        target_name=name,
    )
    return EvaluationTarget(
        name=getattr(legacy_target, "name", name),
        flow_type=getattr(legacy_target, "flow_type"),
        evaluation_mode=getattr(legacy_target, "evaluation_mode"),
        component=component,
        flow_selection=getattr(legacy_target, "flow_selection", None),
    )


def convert_legacy_targets(
    legacy_targets: Mapping[str, Any],
) -> Dict[str, EvaluationTarget]:
    """Convert a target registry into typed targets.

    Parameters
    ----------
    legacy_targets
        Mapping of target names to target-like objects.

    Returns
    -------
    Dict[str, EvaluationTarget]
        Typed target registry with the same keys.
    """
    return {
        target_name: convert_legacy_target(target_name, target)
        for target_name, target in legacy_targets.items()
    }


def get_default_evaluation_targets() -> Dict[str, EvaluationTarget]:
    """Return typed default target registry.

    Returns
    -------
    Dict[str, EvaluationTarget]
        Typed default evaluation target registry.

    Notes
    -----
    The current implementation derives defaults from
    ``patientflow.evaluate.get_default_evaluation_targets()`` so the two
    registries remain aligned.
    """
    from patientflow.evaluate import get_default_evaluation_targets as _legacy_defaults

    return convert_legacy_targets(_legacy_defaults())
