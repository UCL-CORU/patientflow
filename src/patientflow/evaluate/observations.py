"""Observation counting strategies for demand evaluation.

Each public function implements one counting rule so observed totals align with
the predictor or flow they accompany. The `count_observed` dispatcher selects
a strategy by string name; see `OBSERVATION_MODES` for allowed values.

Notes
-----
Prediction-observation pairing: one distribution comparison must match exactly
one row below. The PMF and `agg_observed` must describe the same quantity *X*
for that snapshot and horizon. Row headings match `observation_mode` where that
mode uniquely identifies the contract; when the same mode applies in more than
one scenario, disambiguate with frame kwarg and prediction source in the row
body.

admitted_at_some_point
    ED current (`get_prob_dist_by_service`, component=`arrivals`). *X*: ED
    snapshot rows at (*snapshot_date*, *prediction_time*) with `is_admitted`
    true. Frame: `ed_visits`. *use_admission_in_window_prob*: False.

admitted_in_window
    ED current (`get_prob_dist_by_service`, component=`arrivals`). *X*: among
    patients in the ED snapshot cohort at the moment, those with `is_admitted`
    true whose `departure_datetime` (leave-ED / ward admission) falls in
    (moment, moment + *prediction_window*]. Frame: `ed_visits`.
    *use_admission_in_window_prob*: True.

arrived_in_window
  Pre-filtered `inpatient_arrivals` cohort (caller supplies the appropriate
  rows). *X*: `arrival_datetime` in (moment, moment + *prediction_window*]
  only; ward departure time is out of scope.

arrived_and_admitted_in_window
  Pre-filtered `inpatient_arrivals` cohort (caller chooses direct-admission vs
  ED YTA pathway rows). *X*: rows with `arrival_datetime` and
  `departure_datetime` (ward admission / leave-ED) both in (moment, moment +
  *prediction_window*]. Frame: `inpatient_arrivals`.

departed_in_window
    Inpatient departures (`get_prob_dist_by_service`, component=`departures`).
    *X*: snapshot cohort rows with `left_subspecialty_in_window` true (default
    outcome column). Frame: `inpatient_visits`.

"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

OBSERVATION_MODES: tuple[str, ...] = (
    "admitted_at_some_point",
    "admitted_in_window",
    "departed_in_window",
    "arrived_in_window",
    "arrived_and_admitted_in_window",
)  #: Allowed observation_mode strings accepted by count_observed.

DEFAULT_ADMISSION_LABEL_COL = "is_admitted"
DEFAULT_DEPARTURE_OUTCOME_COLUMN = "left_subspecialty_in_window"

# Context keys on ``EvaluationInputs.observation_contexts[flow][service]``.
OBSERVATION_CONTEXT_FRAME_KEYS: dict[str, str] = {
    "admitted_at_some_point": "ed_visits",
    "admitted_in_window": "ed_visits",
    "arrived_in_window": "inpatient_arrivals",
    "arrived_and_admitted_in_window": "inpatient_arrivals",
    "departed_in_window": "inpatient_visits",
}

# ``EvaluationTarget.component`` for departures distribution → ``admission_type`` filter.
# Keys must match ``target.component`` exactly; an unknown component applies no route filter.
DEPARTURES_DISTRIBUTION_ADMISSION_TYPE: dict[str, str] = {
    "departures_elective": "elective",
    "departures_emergency": "emergency",
}

_ARRIVAL_OBSERVATION_MODES = frozenset(
    {
        "admitted_at_some_point",
        "admitted_in_window",
        "arrived_in_window",
        "arrived_and_admitted_in_window",
    }
)
_ED_CURRENT_ARRIVAL_OBSERVATION_MODES = frozenset(
    {"admitted_at_some_point", "admitted_in_window"}
)
_DEPARTURE_OBSERVATION_MODES = frozenset({"departed_in_window"})


def count_observed_label_kwargs(
    observation_mode: str,
    benchmark_cohorts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Return ``admission_label_col`` / ``outcome_column`` for ``count_observed``.

    Uses ``benchmark_cohorts`` from ``EvaluationInputs.distribution_benchmark_cohorts``
    when registered via ``add_distribution_benchmark_cohort``; otherwise defaults.
    """
    kwargs: dict[str, str] = {}
    if observation_mode in ("admitted_at_some_point", "admitted_in_window"):
        spec = benchmark_cohorts.get("admissions") or {}
        kwargs["admission_label_col"] = str(
            spec.get("label_col", DEFAULT_ADMISSION_LABEL_COL)
        )
    elif observation_mode == "departed_in_window":
        spec = benchmark_cohorts.get("departures") or {}
        kwargs["outcome_column"] = str(
            spec.get("label_col", DEFAULT_DEPARTURE_OUTCOME_COLUMN)
        )
    return kwargs


def _prediction_moment(
    snapshot_date: date, prediction_time: Tuple[int, int]
) -> datetime:
    """Combine snapshot date and prediction clock time into a naive datetime."""
    return datetime.combine(snapshot_date, time(prediction_time[0], prediction_time[1]))


def _align_tz(moment: datetime, series: pd.Series) -> datetime:
    """Match timezone awareness of *moment* to a datetime-like *series*."""
    if hasattr(series, "dt") and series.dt.tz is not None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def count_observed_admitted_at_some_point(
    ed_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    specialty_col: str = "specialty",
    admission_label_col: str = DEFAULT_ADMISSION_LABEL_COL,
) -> int:
    """Count ED snapshot rows with a true admission label for the prediction moment.

    The prediction window is accepted for API compatibility with other
    strategies but does not affect the count: the cohort is the snapshot at
    *prediction_time* on *snapshot_date*.

    Parameters
    ----------
    ed_visits : pandas.DataFrame
        ED visits with columns *snapshot_date*, *prediction_time*,
        *admission_label_col*, and *specialty_col* when *specialty* is not None.
    admission_label_col : str, optional
        Boolean admission label column. Default is ``DEFAULT_ADMISSION_LABEL_COL``.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Unused; retained for a uniform call signature across strategies.
    specialty : str, optional
        If given, restrict to this specialty value. Default is None (no filter).
    specialty_col : str, default='specialty'
        Column used when *specialty* is set.

    Returns
    -------
    int
        Number of matching admitted rows.
    """
    del prediction_window  # retained for signature compatibility
    if admission_label_col not in ed_visits.columns:
        raise ValueError(
            f"admitted_at_some_point requires column {admission_label_col!r} on ed_visits"
        )
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (ed_visits[admission_label_col].astype(bool))
    )
    if specialty is not None:
        mask = mask & (ed_visits[specialty_col] == specialty)
    return int(mask.sum())


def count_observed_admitted_in_window(
    ed_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    specialty_col: str = "specialty",
    admission_label_col: str = DEFAULT_ADMISSION_LABEL_COL,
    departure_datetime_col: str = "departure_datetime",
) -> int:
    """Count snapshot-cohort ED rows admitted with leave-ED time in the window.

    The cohort is the same ED snapshot as `count_observed_admitted_at_some_point`:
    rows at (*snapshot_date*, *prediction_time*) — patients present in ED at the
    prediction moment. Among those rows, count patients with `is_admitted` true
    whose *departure_datetime_col* (typically leave-ED / ward admission time)
    falls in (moment, moment + *prediction_window*].

    This pairs with `use_admission_in_window_prob=True` on the prediction path,
    which weights in-ED patients by probability of admission before the window
    ends.

    Parameters
    ----------
    ed_visits : pandas.DataFrame
        Must include *departure_datetime_col*, *snapshot_date*, *prediction_time*,
        *admission_label_col*, and *specialty_col* if *specialty* is set.
    admission_label_col : str, optional
        Boolean admission label column. Default is ``DEFAULT_ADMISSION_LABEL_COL``.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment for the inclusion window.
    specialty : str, optional
        If given, restrict to this specialty. Default is None (no filter).
    specialty_col : str, default='specialty'
        Column used when *specialty* is set.
    departure_datetime_col : str, optional
        Column for leave-ED (ward admission) time. Default is
        `departure_datetime`.

    Returns
    -------
    int
        Number of admitted snapshot-cohort rows whose leave-ED time is in the
        window.

    Raises
    ------
    ValueError
        If *departure_datetime_col* is missing from *ed_visits*.
    """
    if departure_datetime_col not in ed_visits.columns:
        raise ValueError(
            f"admitted_in_window requires column {departure_datetime_col!r} on ed_visits"
        )
    if admission_label_col not in ed_visits.columns:
        raise ValueError(
            f"admitted_in_window requires column {admission_label_col!r} on ed_visits"
        )
    moment = _prediction_moment(snapshot_date, prediction_time)
    col = ed_visits[departure_datetime_col]
    moment = _align_tz(moment, col)
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (ed_visits[admission_label_col].astype(bool))
        & (col > moment)
        & (col <= moment + prediction_window)
    )
    if specialty is not None:
        mask = mask & (ed_visits[specialty_col] == specialty)
    return int(mask.sum())


def observation_context_frame_key(observation_mode: str) -> str:
    """Return the ``observation_contexts`` key for *observation_mode*."""
    try:
        return OBSERVATION_CONTEXT_FRAME_KEYS[observation_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown observation_mode {observation_mode!r}; "
            f"expected one of {OBSERVATION_MODES}"
        ) from exc


def count_observed_applies_specialty_filter(observation_mode: str) -> bool:
    """Whether ``count_observed`` should filter by the service name on a specialty column.

    ED current modes register the same ``ed_visits`` snapshot frame for every
    service; the service key selects rows via *specialty_col* (default
    ``specialty``). Arrival and inpatient snapshot modes use per-service frames
    from ``add_distribution_observations`` (for example YTA ``is_child`` for
    paediatric); those cohorts are already scoped.
    """
    return observation_mode in _ED_CURRENT_ARRIVAL_OBSERVATION_MODES


def admission_type_filter_for_distribution_component(
    component: str,
) -> Optional[str]:
    """Return ``admission_type`` filter for departures distribution components.

    Returns ``None`` for all-inpatient departures or non-departures components.
    """
    return DEPARTURES_DISTRIBUTION_ADMISSION_TYPE.get(component)


def count_observed_departed_in_window(
    inpatient_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    specialty_col: str = "current_subspecialty",
    outcome_column: str = DEFAULT_DEPARTURE_OUTCOME_COLUMN,
    admission_type: Optional[str] = None,
) -> int:
    """Count inpatients with a true departure label on the snapshot cohort.

    Uses boolean *outcome_column* on inpatient snapshot rows at
    (*snapshot_date*, *prediction_time*). The label encodes whether the patient
    left the current service unit within the prediction window.

    Parameters
    ----------
    inpatient_visits : pandas.DataFrame
        Inpatient snapshots with *snapshot_date*, *prediction_time*, and
        *outcome_column*. Optional *specialty_col* when *specialty* is set.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Unused; retained for a uniform call signature across strategies.
    specialty : str, optional
        If given, filter by *specialty_col* when that column is present.
        Default is None (no filter).
    specialty_col : str, default='current_subspecialty'
        Service-unit column used when *specialty* is set. Frames that store
        the unit under another name (e.g. ``specialty``) should pass that name.
    outcome_column : str, optional
        Boolean column naming departure within the window on the snapshot
        cohort. Default is `left_subspecialty_in_window`.
    admission_type : str, optional
        If given (e.g. ``"elective"`` or ``"emergency"``), keep rows with that
        value in an ``admission_type`` column. Default is None (no route filter).

    Returns
    -------
    int
        Number of rows in the snapshot cohort with a true *outcome_column*.

    Raises
    ------
    ValueError
        If *outcome_column* is missing from *inpatient_visits*.
    """
    del prediction_window  # label is defined on the snapshot cohort
    if outcome_column not in inpatient_visits.columns:
        raise ValueError(
            f"departed_in_window requires column {outcome_column!r} on inpatient_visits"
        )
    mask = (
        (inpatient_visits["snapshot_date"] == snapshot_date)
        & (inpatient_visits["prediction_time"] == prediction_time)
        & (inpatient_visits[outcome_column].astype(bool))
    )
    if specialty is not None and specialty_col in inpatient_visits.columns:
        mask = mask & (inpatient_visits[specialty_col] == specialty)
    if admission_type is not None:
        if "admission_type" not in inpatient_visits.columns:
            raise ValueError(
                "departed_in_window with admission_type requires column "
                "'admission_type' on inpatient_visits"
            )
        mask = mask & (inpatient_visits["admission_type"] == admission_type)
    return int(mask.sum())


def count_observed_arrived_in_window(
    inpatient_arrivals: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    specialty_col: str = "specialty",
    arrival_datetime_col: str = "arrival_datetime",
) -> int:
    """Count arrivals with arrival time in `(moment, moment + window]`.

    Caller supplies a pre-filtered *inpatient_arrivals* cohort; counts
    *arrival_datetime_col* in (moment, moment + window] only.

    Parameters
    ----------
    inpatient_arrivals : pandas.DataFrame
        Pre-filtered pathway rows; must include *arrival_datetime_col*.
    snapshot_date : datetime.date
        Snapshot calendar date; defines the prediction moment.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment.
    specialty : str, optional
        If given and *specialty_col* exists, filter to that value.
        Default is None (no filter).
    specialty_col : str, default='specialty'
        Service-unit column used when *specialty* is set.
    arrival_datetime_col : str, optional
        Column name for arrival timestamp. Default is `arrival_datetime`.

    Returns
    -------
    int
        Number of rows in the arrival time window.

    Raises
    ------
    ValueError
        If *arrival_datetime_col* is missing from *inpatient_arrivals*.
    """
    if arrival_datetime_col not in inpatient_arrivals.columns:
        raise ValueError(
            f"arrived_in_window requires column {arrival_datetime_col!r} "
            "on inpatient_arrivals"
        )
    moment = _prediction_moment(snapshot_date, prediction_time)
    col = inpatient_arrivals[arrival_datetime_col]
    moment = _align_tz(moment, col)
    mask = (col > moment) & (col <= moment + prediction_window)
    if specialty is not None and specialty_col in inpatient_arrivals.columns:
        mask = mask & (inpatient_arrivals[specialty_col] == specialty)
    return int(mask.sum())


def count_observed_arrived_and_admitted_in_window(
    inpatient_arrivals: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    specialty_col: str = "specialty",
    arrival_datetime_col: str = "arrival_datetime",
    departure_datetime_col: str = "departure_datetime",
) -> int:
    """Count pathway rows with arrival and departure time in the window.

    Caller supplies a pre-filtered *inpatient_arrivals* cohort (e.g. direct
    admission or ED yet-to-arrive). Counts rows whose *arrival_datetime_col*
    and *departure_datetime_col* (ward admission / leave-ED) both fall in
    (moment, moment + *prediction_window*].

    Parameters
    ----------
    inpatient_arrivals : pandas.DataFrame
        Pre-filtered pathway rows. Must include *arrival_datetime_col* and
        *departure_datetime_col*.
    snapshot_date : datetime.date
        Snapshot calendar date (defines the prediction moment).
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment for both windows.
    specialty : str, optional
        If given and *specialty_col* exists, filter to that value.
        Default is None (no filter).
    specialty_col : str, default='specialty'
        Service-unit column used when *specialty* is set.
    arrival_datetime_col : str, optional
        Column name for arrival time. Default is `arrival_datetime`.
    departure_datetime_col : str, optional
        Column name for ward admission / leave-ED time. Default is
        `departure_datetime`.

    Returns
    -------
    int
        Number of rows with both times in (moment, moment + *prediction_window*].

    Raises
    ------
    ValueError
        If a required datetime column is missing from *inpatient_arrivals*.
    """
    for coln in (arrival_datetime_col, departure_datetime_col):
        if coln not in inpatient_arrivals.columns:
            raise ValueError(
                f"arrived_and_admitted_in_window requires column {coln!r} "
                "on inpatient_arrivals"
            )
    moment = _prediction_moment(snapshot_date, prediction_time)
    arr = inpatient_arrivals[arrival_datetime_col]
    dep = inpatient_arrivals[departure_datetime_col]
    moment_arr = _align_tz(moment, arr)
    moment_dep = _align_tz(moment, dep)
    window_end_arr = moment_arr + prediction_window
    window_end_dep = moment_dep + prediction_window
    mask = (
        (arr > moment_arr)
        & (arr <= window_end_arr)
        & (dep > moment_dep)
        & (dep <= window_end_dep)
    )
    if specialty is not None and specialty_col in inpatient_arrivals.columns:
        mask = mask & (inpatient_arrivals[specialty_col] == specialty)
    return int(mask.sum())


def validate_observation_mode_for_component(
    component: str,
    observation_mode: str,
) -> None:
    """Raise if *observation_mode* is not allowed for aggregate *component*.

    Parameters
    ----------
    component : str
        One of `"arrivals"`, `"departures"`, or `"net_flow"`.
    observation_mode : str
        One of the strings in `OBSERVATION_MODES`.

    Raises
    ------
    ValueError
        If the pairing is invalid or *component* is `"net_flow"`.
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unknown observation_mode {observation_mode!r}; "
            f"expected one of {OBSERVATION_MODES}"
        )
    if component == "net_flow":
        raise ValueError(
            "net_flow distribution evaluation is out of scope: pass "
            "component='arrivals' or 'departures' with an explicit observation_mode"
        )
    if component == "arrivals" and observation_mode not in _ARRIVAL_OBSERVATION_MODES:
        raise ValueError(
            f"component='arrivals' requires observation_mode in "
            f"{sorted(_ARRIVAL_OBSERVATION_MODES)}, got {observation_mode!r}"
        )
    if (
        component == "departures"
        and observation_mode not in _DEPARTURE_OBSERVATION_MODES
    ):
        raise ValueError(
            f"component='departures' requires observation_mode "
            f"'departed_in_window', got {observation_mode!r}"
        )


def count_observed(
    observation_mode: str,
    *,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    ed_visits: Optional[pd.DataFrame] = None,
    inpatient_visits: Optional[pd.DataFrame] = None,
    inpatient_arrivals: Optional[pd.DataFrame] = None,
    specialty: Optional[str] = None,
    specialty_col: Optional[str] = None,
    admission_label_col: str = DEFAULT_ADMISSION_LABEL_COL,
    outcome_column: str = DEFAULT_DEPARTURE_OUTCOME_COLUMN,
    admission_type: Optional[str] = None,
    arrival_datetime_col: str = "arrival_datetime",
    departure_datetime_col: str = "departure_datetime",
) -> int:
    """Dispatch to a counting strategy by name.

    Parameters
    ----------
    observation_mode : str
        One of the strings in `OBSERVATION_MODES`.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Prediction horizon passed through to the underlying counter.
    ed_visits : pandas.DataFrame, optional
        ED snapshot frame; required for `admitted_*` modes.
    inpatient_visits : pandas.DataFrame, optional
        Inpatient snapshot frame; required for `departed_in_window`.
    inpatient_arrivals : pandas.DataFrame, optional
        Pre-filtered pathway rows; required for `arrived_in_window` and
        `arrived_and_admitted_in_window`.
    specialty : str, optional
        Passed through to the underlying counter when supported.
        Default is None (no filter).
    specialty_col : str, optional
        Service-unit column name passed to the underlying counter. When omitted,
        each counter uses its own default (``specialty`` for ED/arrivals modes;
        ``current_subspecialty`` for `departed_in_window`).
    departure_datetime_col : str, optional
        Passed to `count_observed_admitted_in_window` and
        `count_observed_arrived_and_admitted_in_window`.
        Default is `departure_datetime`.
    admission_label_col : str, optional
        Passed to `count_observed_admitted_in_window`.
        Default is ``DEFAULT_ADMISSION_LABEL_COL``.
    outcome_column : str, optional
        Passed to `count_observed_departed_in_window`.
        Default is ``DEFAULT_DEPARTURE_OUTCOME_COLUMN``.
    admission_type : str, optional
        Passed to `count_observed_departed_in_window` for route-specific
        departures targets (e.g. ``"elective"``). Default is None.
    arrival_datetime_col : str, optional
        Passed to `count_observed_arrived_in_window` and
        `count_observed_arrived_and_admitted_in_window`.
        Default is `arrival_datetime`.

    Returns
    -------
    int
        Observed count. Returns 0 when a required frame is missing (`None`).

    Raises
    ------
    ValueError
        If *observation_mode* is not recognised.

    See Also
    --------
    count_observed_admitted_at_some_point
    count_observed_admitted_in_window
    count_observed_departed_in_window
    count_observed_arrived_in_window
    count_observed_arrived_and_admitted_in_window
    validate_observation_mode_for_component

    Notes
    -----
    See the module docstring pairing table for which frame (*ed_visits*,
    *inpatient_arrivals*, *inpatient_visits*) pairs with each *observation_mode*.
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unknown observation_mode {observation_mode!r}; "
            f"expected one of {OBSERVATION_MODES}"
        )
    specialty_kwargs: Dict[str, str] = (
        {"specialty_col": specialty_col} if specialty_col is not None else {}
    )
    if observation_mode == "admitted_at_some_point":
        if ed_visits is None:
            return 0
        return count_observed_admitted_at_some_point(
            ed_visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            admission_label_col=admission_label_col,
            **specialty_kwargs,
        )
    if observation_mode == "admitted_in_window":
        if ed_visits is None:
            return 0
        return count_observed_admitted_in_window(
            ed_visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            admission_label_col=admission_label_col,
            departure_datetime_col=departure_datetime_col,
            **specialty_kwargs,
        )
    if observation_mode == "departed_in_window":
        if inpatient_visits is None:
            return 0
        return count_observed_departed_in_window(
            inpatient_visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            outcome_column=outcome_column,
            admission_type=admission_type,
            **specialty_kwargs,
        )
    if observation_mode == "arrived_in_window":
        if inpatient_arrivals is None:
            return 0
        return count_observed_arrived_in_window(
            inpatient_arrivals,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            arrival_datetime_col=arrival_datetime_col,
            **specialty_kwargs,
        )
    if observation_mode == "arrived_and_admitted_in_window":
        if inpatient_arrivals is None:
            return 0
        return count_observed_arrived_and_admitted_in_window(
            inpatient_arrivals,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            arrival_datetime_col=arrival_datetime_col,
            departure_datetime_col=departure_datetime_col,
            **specialty_kwargs,
        )
    raise RuntimeError("unreachable")  # pragma: no cover
