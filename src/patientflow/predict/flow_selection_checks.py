"""Predicates and assertions for `FlowSelection` (`patientflow.predict.types`).

These helpers support `build_service_data` in `patientflow.predict.service`:
only models and snapshot inputs required by the active flow configuration are
enforced.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from patientflow.model_artifacts import TrainedClassifier
from patientflow.predict.types import FlowSelection
from patientflow.predictors.incoming_admission_predictors import (
    DirectAdmissionPredictor,
    EmpiricalIncomingAdmissionPredictor,
    ParametricIncomingAdmissionPredictor,
)
from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)
from patientflow.predictors.subgroup_predictor import MultiSubgroupPredictor
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator
from patientflow.predictors.value_to_outcome_predictor import ValueToOutcomePredictor


def requires_ed_snapshots(fs: FlowSelection) -> bool:
    """Return whether ED snapshot rows are required for the selection.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.

    Returns
    -------
    bool
        `True` when `include_ed_current` is enabled (current-ED cohort).
    """
    return fs.include_ed_current


def requires_inpatient_snapshots(
    fs: FlowSelection, transfer_model: Optional[Any]
) -> bool:
    """Return whether inpatient snapshot rows are required.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    transfer_model : object or None
        Transfer model instance when present; used to decide whether transfer
        PMFs require inpatient departure inputs.

    Returns
    -------
    bool
        `True` when departures are included, or when transfers are included
        and a transfer model is supplied.
    """
    if fs.include_departures:
        return True
    if fs.include_transfers_in and transfer_model is not None:
        return True
    return False


def requires_admission_curve_params(
    fs: FlowSelection,
    yet_to_arrive_model: Optional[Any],
    *,
    use_admission_in_window_prob: bool,
    has_ed_snapshots: bool,
) -> bool:
    """Return whether aspirational curve parameters `x1`--`y2` are required.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    yet_to_arrive_model : object or None
        ED yet-to-arrive model (parametric or empirical), or `None`.
    use_admission_in_window_prob : bool
        Whether current-ED rows are weighted by in-window admission probability.
    has_ed_snapshots : bool
        Whether an ED snapshot dataframe is present.

    Returns
    -------
    bool
        `True` when a parametric ED YTA model is used, or when parametric
        in-window weighting applies to current ED patients.
    """
    if yet_to_arrive_model is None:
        return False
    if not isinstance(yet_to_arrive_model, ParametricIncomingAdmissionPredictor):
        return False
    if fs.include_ed_yta:
        return True
    if use_admission_in_window_prob and has_ed_snapshots and fs.include_ed_current:
        return True
    return False


def validate_ed_classifier(fs: FlowSelection, ed_classifier: Optional[Any]) -> None:
    """Require an ED classifier when current-ED flow is selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    ed_classifier : TrainedClassifier or None
        ED admission classifier.

    Raises
    ------
    ValueError
        If `include_ed_current` is `True` and `ed_classifier` is `None`.
    """
    if fs.include_ed_current and ed_classifier is None:
        raise ValueError(
            "flow_selection includes current ED admissions (include_ed_current=True) "
            "but ed_classifier was not supplied"
        )


def validate_spec_model_for_ed(fs: FlowSelection, spec_model: Optional[Any]) -> None:
    """Require a specialty model when current-ED flow is selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    spec_model : object or None
        Specialty routing model for ED rows.

    Raises
    ------
    ValueError
        If `include_ed_current` is `True` and `spec_model` is `None`.
    """
    if fs.include_ed_current and spec_model is None:
        raise ValueError(
            "flow_selection includes current ED admissions (include_ed_current=True) "
            "but spec_model was not supplied"
        )


def validate_inpatient_classifier(
    fs: FlowSelection, inpatient_classifier: Optional[Any]
) -> None:
    """Require an inpatient classifier when departures or transfers need it.

    Transfer arrivals are derived from per-service departure PMFs, which need
    the inpatient departure classifier even when `include_departures` is
    `False`.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    inpatient_classifier : TrainedClassifier or None
        Inpatient departure classifier.

    Raises
    ------
    ValueError
        If departures or (transfers with a transfer model) are expected but
        `inpatient_classifier` is `None`.
    """
    needs = fs.include_departures or fs.include_transfers_in
    if needs and inpatient_classifier is None:
        raise ValueError(
            "flow_selection includes departures or transfers but "
            "inpatient_classifier was not supplied"
        )


def validate_yta_ed(
    fs: FlowSelection,
    yet_to_arrive_model: Optional[Any],
) -> None:
    """Require an ED yet-to-arrive model when that flow is selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    yet_to_arrive_model : object or None
        ED YTA model (parametric or empirical).

    Raises
    ------
    ValueError
        If `include_ed_yta` is `True` and `yet_to_arrive_model` is `None`.
    """
    if fs.include_ed_yta and yet_to_arrive_model is None:
        raise ValueError(
            "flow_selection includes ED yet-to-arrive (include_ed_yta=True) "
            "but the ED yet-to-arrive model (ed_yta_model) was not supplied"
        )


def validate_non_ed_yta(fs: FlowSelection, non_ed_yta_model: Optional[Any]) -> None:
    """Require a non-ED YTA model when that flow is selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    non_ed_yta_model : DirectAdmissionPredictor or None
        Non-ED emergency yet-to-arrive model.

    Raises
    ------
    ValueError
        If `include_non_ed_yta` is `True` and `non_ed_yta_model` is `None`.
    """
    if fs.include_non_ed_yta and non_ed_yta_model is None:
        raise ValueError(
            "flow_selection includes non-ED yet-to-arrive (include_non_ed_yta=True) "
            "but non_ed_yta_model was not supplied"
        )


def validate_elective_yta(fs: FlowSelection, elective_yta_model: Optional[Any]) -> None:
    """Require an elective YTA model when that flow is selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    elective_yta_model : DirectAdmissionPredictor or None
        Elective yet-to-arrive model.

    Raises
    ------
    ValueError
        If `include_elective_yta` is `True` and `elective_yta_model` is `None`.
    """
    if fs.include_elective_yta and elective_yta_model is None:
        raise ValueError(
            "flow_selection includes elective yet-to-arrive (include_elective_yta=True) "
            "but elective_yta_model was not supplied"
        )


def validate_transfer_model(fs: FlowSelection, transfer_model: Optional[Any]) -> None:
    """Require a transfer model when transfer arrivals are selected.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    transfer_model : TransferProbabilityEstimator or None
        Internal transfer model.

    Raises
    ------
    ValueError
        If `include_transfers_in` is `True` and `transfer_model` is `None`.
    """
    if fs.include_transfers_in and transfer_model is None:
        raise ValueError(
            "flow_selection includes transfers in (include_transfers_in=True) "
            "but transfer_model was not supplied"
        )


def validate_ed_snapshots_present(
    fs: FlowSelection, ed_snapshots: Optional[pd.DataFrame]
) -> None:
    """Require ED snapshots when the selection needs current-ED data.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    ed_snapshots : pandas.DataFrame or None
        ED patient snapshot at the prediction moment.

    Raises
    ------
    ValueError
        If `requires_ed_snapshots(fs)` is true and `ed_snapshots` is `None`.
    """
    if requires_ed_snapshots(fs) and ed_snapshots is None:
        raise ValueError(
            "flow_selection requires ED snapshots (include_ed_current=True) "
            "but ed_snapshots was not supplied"
        )


def validate_inpatient_snapshots_present(
    fs: FlowSelection,
    transfer_model: Optional[Any],
    inpatient_snapshots: Optional[pd.DataFrame],
) -> None:
    """Require inpatient snapshots when departures or transfers need them.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    transfer_model : object or None
        Transfer model instance when present.
    inpatient_snapshots : pandas.DataFrame or None
        Inpatient snapshot at the prediction moment.

    Raises
    ------
    ValueError
        If `requires_inpatient_snapshots(fs, transfer_model)` is true and
        `inpatient_snapshots` is `None`.
    """
    if requires_inpatient_snapshots(fs, transfer_model) and inpatient_snapshots is None:
        raise ValueError(
            "flow_selection requires inpatient snapshots for departures or transfers "
            "but inpatient_snapshots was not supplied"
        )


def validate_admission_curve_params(
    fs: FlowSelection,
    yet_to_arrive_model: Optional[Any],
    *,
    use_admission_in_window_prob: bool,
    has_ed_snapshots: bool,
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
) -> None:
    """Require curve parameters when the parametric path is active.

    Parameters
    ----------
    fs : FlowSelection
        Active flow configuration.
    yet_to_arrive_model : object or None
        ED yet-to-arrive model.
    use_admission_in_window_prob : bool
        Whether in-window admission weighting is enabled for current ED.
    has_ed_snapshots : bool
        Whether ED snapshots are present.
    x1, y1, x2, y2 : float or None
        Aspirational survival-curve parameters.

    Raises
    ------
    ValueError
        If `requires_admission_curve_params(...)` is true and any of
        `x1`, `y1`, `x2`, `y2` is `None`.
    """
    if not requires_admission_curve_params(
        fs,
        yet_to_arrive_model,
        use_admission_in_window_prob=use_admission_in_window_prob,
        has_ed_snapshots=has_ed_snapshots,
    ):
        return
    missing = [
        name
        for name, val in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2))
        if val is None
    ]
    if missing:
        raise ValueError(
            "Parametric ED yet-to-arrive / admission-in-window curve parameters "
            f"are required for this flow_selection but these were not supplied: {', '.join(missing)}"
        )


def assert_component_matches_flow_selection(component: str, fs: FlowSelection) -> None:
    """Check that an evaluation component is compatible with `fs`.

    Parameters
    ----------
    component : {'arrivals', 'departures', 'net_flow'}
        Bundle field used for observed-vs-predicted comparison.
    fs : FlowSelection
        Active flow configuration.

    Raises
    ------
    ValueError
        If `component` is not recognised, or if it names flows that
        `fs` does not include.
    """
    if component not in {"arrivals", "departures", "net_flow"}:
        raise ValueError(
            f"component must be 'arrivals', 'departures', or 'net_flow', got {component!r}"
        )
    has_arrivals = (
        fs.include_ed_current
        or fs.include_ed_yta
        or fs.include_non_ed_yta
        or fs.include_elective_yta
        or fs.include_transfers_in
    )
    if component == "arrivals" and not has_arrivals:
        raise ValueError(
            "component is 'arrivals' but flow_selection includes no arrival flows"
        )
    if component == "departures" and not fs.include_departures:
        raise ValueError(
            "component is 'departures' but flow_selection has include_departures=False"
        )
    if component == "net_flow" and not (has_arrivals or fs.include_departures):
        raise ValueError(
            "component is 'net_flow' but flow_selection includes no arrival or departure flows"
        )


def assert_model_types_for_flow(
    _flow_selection: FlowSelection,
    *,
    ed_classifier: Optional[TrainedClassifier],
    inpatient_classifier: Optional[TrainedClassifier],
    spec_model: Optional[
        Union[
            SequenceToOutcomePredictor,
            ValueToOutcomePredictor,
            MultiSubgroupPredictor,
        ]
    ],
    yet_to_arrive_model: Optional[
        Union[
            ParametricIncomingAdmissionPredictor,
            EmpiricalIncomingAdmissionPredictor,
        ]
    ],
    non_ed_yta_model: Optional[DirectAdmissionPredictor],
    elective_yta_model: Optional[DirectAdmissionPredictor],
    transfer_model: Optional[TransferProbabilityEstimator],
) -> None:
    """Validate concrete types for each supplied model slot.

    Parameters
    ----------
    _flow_selection : FlowSelection
        Reserved for future flow-specific rules; callers should pass the same
        `FlowSelection` used for presence validation.
    ed_classifier : TrainedClassifier or None
        ED classifier slot.
    inpatient_classifier : TrainedClassifier or None
        Inpatient classifier slot.
    spec_model : object or None
        Specialty model slot.
    yet_to_arrive_model : object or None
        ED YTA model slot.
    non_ed_yta_model : DirectAdmissionPredictor or None
        Non-ED YTA slot.
    elective_yta_model : DirectAdmissionPredictor or None
        Elective YTA slot.
    transfer_model : TransferProbabilityEstimator or None
        Transfer model slot.

    Raises
    ------
    TypeError
        If any non-`None` model is not an instance of the expected type.
    """
    if ed_classifier is not None and not isinstance(ed_classifier, TrainedClassifier):
        raise TypeError("ed_classifier must be of type TrainedClassifier")
    if inpatient_classifier is not None and not isinstance(
        inpatient_classifier, TrainedClassifier
    ):
        raise TypeError("inpatient_classifier must be of type TrainedClassifier")
    if spec_model is not None and not isinstance(
        spec_model,
        (SequenceToOutcomePredictor, ValueToOutcomePredictor, MultiSubgroupPredictor),
    ):
        raise TypeError(
            "spec_model must be SequenceToOutcomePredictor, "
            "ValueToOutcomePredictor, or MultiSubgroupPredictor"
        )
    if yet_to_arrive_model is not None:
        if not isinstance(
            yet_to_arrive_model,
            (ParametricIncomingAdmissionPredictor, EmpiricalIncomingAdmissionPredictor),
        ):
            raise TypeError(
                "ed_yta_model must be ParametricIncomingAdmissionPredictor "
                "or EmpiricalIncomingAdmissionPredictor"
            )
    if non_ed_yta_model is not None and not isinstance(
        non_ed_yta_model, DirectAdmissionPredictor
    ):
        raise TypeError("non_ed_yta_model must be of type DirectAdmissionPredictor")
    if elective_yta_model is not None and not isinstance(
        elective_yta_model, DirectAdmissionPredictor
    ):
        raise TypeError("elective_yta_model must be of type DirectAdmissionPredictor")
    if transfer_model is not None and not isinstance(
        transfer_model, TransferProbabilityEstimator
    ):
        raise TypeError("transfer_model must be of type TransferProbabilityEstimator")
