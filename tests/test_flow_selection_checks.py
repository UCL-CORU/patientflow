"""Tests for patientflow.predict.flow_selection_checks."""

import pytest

from patientflow.predict.flow_selection_checks import (
    assert_component_matches_flow_selection,
    requires_admission_curve_params,
    requires_ed_snapshots,
    requires_inpatient_snapshots,
)
from patientflow.predict.types import FlowSelection
from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
)


@pytest.mark.parametrize(
    "fs,expected",
    [
        (FlowSelection.default(), True),
        (FlowSelection.custom(include_ed_current=False), False),
        (FlowSelection.outgoing_only(), False),
    ],
)
def test_requires_ed_snapshots(fs, expected):
    assert requires_ed_snapshots(fs) is expected


@pytest.mark.parametrize(
    "fs,transfer_model,expected",
    [
        (FlowSelection.default(), object(), True),
        (
            FlowSelection.custom(include_departures=False, include_transfers_in=True),
            object(),
            True,
        ),
        (FlowSelection.incoming_only(), None, False),
        (
            FlowSelection.custom(include_departures=False, include_transfers_in=True),
            None,
            False,
        ),
    ],
)
def test_requires_inpatient_snapshots(fs, transfer_model, expected):
    assert requires_inpatient_snapshots(fs, transfer_model) is expected


def test_requires_admission_curve_params_without_model():
    fs = FlowSelection.custom(include_ed_yta=True)
    assert not requires_admission_curve_params(
        fs,
        None,
        use_admission_in_window_prob=True,
        has_ed_snapshots=True,
    )


def test_requires_admission_curve_params_parametric_ed_yta():
    p = ParametricIncomingAdmissionPredictor.__new__(
        ParametricIncomingAdmissionPredictor
    )
    fs = FlowSelection.custom(include_ed_yta=True)
    assert requires_admission_curve_params(
        fs,
        p,
        use_admission_in_window_prob=False,
        has_ed_snapshots=False,
    )


def test_assert_component_mismatches():
    with pytest.raises(ValueError, match="no arrival flows"):
        assert_component_matches_flow_selection(
            "arrivals",
            FlowSelection.outgoing_only(),
        )
    with pytest.raises(ValueError, match="include_departures=False"):
        assert_component_matches_flow_selection(
            "departures",
            FlowSelection.incoming_only(),
        )


def test_assert_component_invalid_name():
    with pytest.raises(ValueError, match="component must be"):
        assert_component_matches_flow_selection("totals", FlowSelection.default())
