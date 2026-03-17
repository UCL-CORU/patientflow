"""Target registry helpers for typed evaluation.

This module contains utilities that:

- resolve a single component per target
- convert target-like registry entries into typed ``EvaluationTarget`` objects
- provide the default typed target registry used by builders/runners

Row references in :func:`get_default_evaluation_targets` (e.g. "Row 3",
"Row 5b") refer to the flow evaluation matrix in
``docs/evaluation_plan.md``.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional

from patientflow.evaluate.types import EvaluationTarget
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
    kwargs: dict = dict(
        name=getattr(legacy_target, "name", name),
        flow_type=getattr(legacy_target, "flow_type"),
        evaluation_mode=getattr(legacy_target, "evaluation_mode"),
        component=component,
        flow_selection=getattr(legacy_target, "flow_selection", None),
    )
    obs_mode = getattr(legacy_target, "observation_mode", None)
    if obs_mode is not None:
        kwargs["observation_mode"] = obs_mode
    return EvaluationTarget(**kwargs)


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
    """Return the default typed target registry.

    Returns
    -------
    Dict[str, EvaluationTarget]
        Typed default evaluation target registry covering all currently
        supported flows and diagnostic modes.
    """
    return {
        # Row 1: P(admission after ED) — patient-level binary classifier
        "ed_current_admission_classifier": EvaluationTarget(
            name="ed_current_admission_classifier",
            flow_type="classifier",
            evaluation_mode="classifier",
            component="arrivals",
            observation_mode="classifier_binary",
        ),
        # Row 10: P(discharge within window) — patient-level binary classifier
        "discharge_classifier": EvaluationTarget(
            name="discharge_classifier",
            flow_type="classifier",
            evaluation_mode="classifier",
            component="departures",
            observation_mode="classifier_binary",
        ),
        # Row 3: admitted at some point (no window constraint)
        "ed_current_beds": EvaluationTarget(
            name="ed_current_beds",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_at_some_point",
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=False,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
            ),
        ),
        # Row 4b: survival curve comparison (train vs test)
        "ed_current_window_prob": EvaluationTarget(
            name="ed_current_window_prob",
            flow_type="special",
            evaluation_mode="survival_curve",
            component="arrivals",
            observation_mode="survival_comparison",
        ),
        # Row 4a: aspirational — no observation
        "ed_current_window_prob_aspirational": EvaluationTarget(
            name="ed_current_window_prob_aspirational",
            flow_type="special",
            evaluation_mode="aspirational_skip",
            component="arrivals",
            observation_mode="not_applicable",
        ),
        # Row 5b: beds needed in window (survival-curve-based)
        "ed_current_window_beds": EvaluationTarget(
            name="ed_current_window_beds",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_in_window",
        ),
        # Row 5a: aspirational — no observation
        "ed_current_window_beds_aspirational": EvaluationTarget(
            name="ed_current_window_beds_aspirational",
            flow_type="pmf",
            evaluation_mode="aspirational_skip",
            component="arrivals",
            observation_mode="not_applicable",
        ),
        # Row 6: arrival rate comparison
        "ed_yta_arrival_rates": EvaluationTarget(
            name="ed_yta_arrival_rates",
            flow_type="special",
            evaluation_mode="arrival_deltas",
            component="arrivals",
            observation_mode="arrival_rates",
        ),
        # Row 7b: yet-to-arrive beds needed in window
        "ed_yta_beds": EvaluationTarget(
            name="ed_yta_beds",
            flow_type="poisson",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_in_window",
            flow_selection=FlowSelection.custom(
                include_ed_current=False,
                include_ed_yta=True,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
            ),
        ),
        # Row 7a: aspirational — no observation
        "ed_yta_beds_aspirational": EvaluationTarget(
            name="ed_yta_beds_aspirational",
            flow_type="poisson",
            evaluation_mode="aspirational_skip",
            component="arrivals",
            observation_mode="not_applicable",
        ),
        # Row 8: emergency admissions not via ED, in window
        "non_ed_yta_beds": EvaluationTarget(
            name="non_ed_yta_beds",
            flow_type="poisson",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_in_window",
        ),
        # Row 9: elective admissions in window
        "elective_yta_beds": EvaluationTarget(
            name="elective_yta_beds",
            flow_type="poisson",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_in_window",
        ),
        # Row 11: emergency departures in window
        "discharge_emergency": EvaluationTarget(
            name="discharge_emergency",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="departures",
            observation_mode="count_in_window",
        ),
        # Row 12: elective departures in window
        "discharge_elective": EvaluationTarget(
            name="discharge_elective",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="departures",
            observation_mode="count_in_window",
        ),
        # Row 16: aspirational — no observation
        "combined_emergency_arrivals": EvaluationTarget(
            name="combined_emergency_arrivals",
            flow_type="pmf",
            evaluation_mode="aspirational_skip",
            component="arrivals",
            observation_mode="not_applicable",
            flow_selection=FlowSelection.emergency_only(),
        ),
        # Row 17: elective arrivals in window
        "combined_elective_arrivals": EvaluationTarget(
            name="combined_elective_arrivals",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="arrivals",
            observation_mode="count_in_window",
            flow_selection=FlowSelection.elective_only(),
        ),
        # Row 18: aspirational — no observation
        "combined_net_emergency": EvaluationTarget(
            name="combined_net_emergency",
            flow_type="pmf",
            evaluation_mode="aspirational_skip",
            component="net_flow",
            observation_mode="not_applicable",
            flow_selection=FlowSelection.emergency_only(),
        ),
        # Row 19: elective net flow in window
        "combined_net_elective": EvaluationTarget(
            name="combined_net_elective",
            flow_type="pmf",
            evaluation_mode="distribution",
            component="net_flow",
            observation_mode="count_in_window",
            flow_selection=FlowSelection.elective_only(),
        ),
    }
