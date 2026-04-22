"""Guards and predicates for :class:`FlowSelection` used when building demand inputs.

Used by ``build_service_data`` validation so callers only supply inputs and
parameters relevant to the selected flows."""

from __future__ import annotations

from typing import Any, Optional

from patientflow.predict.types import FlowSelection


def cohort_allows_emergency(fs: FlowSelection) -> bool:
    return fs.cohort in ("all", "emergency")


def cohort_allows_elective(fs: FlowSelection) -> bool:
    return fs.cohort in ("all", "elective")


def requires_ed_snapshots(fs: FlowSelection) -> bool:
    return fs.include_ed_current


def requires_inpatient_snapshots(
    fs: FlowSelection, transfer_model: Optional[Any]
) -> bool:
    if fs.include_departures:
        return True
    return bool(fs.include_transfers_in and transfer_model is not None)


def validate_ed_classifier(fs: FlowSelection) -> bool:
    return fs.include_ed_current and cohort_allows_emergency(fs)


def validate_spec_model_for_ed(fs: FlowSelection) -> bool:
    return validate_ed_classifier(fs)


def validate_inpatient_classifier(
    fs: FlowSelection, transfer_model: Optional[Any]
) -> bool:
    return requires_inpatient_snapshots(fs, transfer_model)


def validate_yta_ed(fs: FlowSelection) -> bool:
    return fs.include_ed_yta and cohort_allows_emergency(fs)


def validate_non_ed_yta(fs: FlowSelection) -> bool:
    return fs.include_non_ed_yta and cohort_allows_emergency(fs)


def validate_elective_yta(fs: FlowSelection) -> bool:
    return fs.include_elective_yta and cohort_allows_elective(fs)


def validate_transfer_model(fs: FlowSelection, transfer_model: Optional[Any]) -> bool:
    return fs.include_transfers_in and transfer_model is not None


def requires_admission_curve_params(
    fs: FlowSelection,
    yet_to_arrive_model: Optional[Any],
    use_admission_in_window_prob: bool,
) -> bool:
    if yet_to_arrive_model is None:
        return False
    from patientflow.predictors.incoming_admission_predictors import (
        ParametricIncomingAdmissionPredictor,
    )

    if not isinstance(yet_to_arrive_model, ParametricIncomingAdmissionPredictor):
        return False
    if validate_yta_ed(fs):
        return True
    if (
        fs.include_ed_current
        and cohort_allows_emergency(fs)
        and use_admission_in_window_prob
    ):
        return True
    return False


def assert_component_matches_flow_selection(
    component: str, flow_selection: FlowSelection
) -> None:
    """Raise ValueError if *component* cannot be produced from *flow_selection*."""
    has_inflow = any(
        [
            flow_selection.include_ed_current,
            flow_selection.include_ed_yta,
            flow_selection.include_non_ed_yta,
            flow_selection.include_elective_yta,
            flow_selection.include_transfers_in,
        ]
    )
    has_outflow = flow_selection.include_departures

    if component == "arrivals" and not has_inflow:
        raise ValueError(
            "component='arrivals' requires at least one inflow family enabled "
            "in flow_selection"
        )
    if component == "departures" and not has_outflow:
        raise ValueError(
            "component='departures' requires flow_selection.include_departures=True"
        )
    if component == "net_flow" and not (has_inflow and has_outflow):
        raise ValueError(
            "component='net_flow' requires at least one inflow family and "
            "include_departures=True in flow_selection"
        )
