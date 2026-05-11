"""Observation counting strategies for demand evaluation.

Each public function implements one counting rule so observed totals align with
the predictor or flow they accompany. The `count_observed` dispatcher selects
a strategy by string name; see the `OBSERVATION_MODES` tuple for allowed values.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

OBSERVATION_MODES: tuple[str, ...] = (
    "admitted_at_some_point",
    "admitted_in_window",
    "departed_in_window",
    "arrived_in_window",
    "arrived_and_admitted_in_window",
)  #: Allowed observation_mode strings accepted by count_observed.


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
) -> int:
    """Count ED snapshot rows with `is_admitted` for the prediction moment.

    The prediction window is accepted for API compatibility with other
    strategies but does not affect the count: the cohort is the snapshot at
    `prediction_time` on `snapshot_date`.

    Parameters
    ----------
    ed_visits : pandas.DataFrame
        ED visits with columns *snapshot_date*, *prediction_time*,
        *is_admitted*, and *specialty* when *specialty* is not None.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Unused; retained for a uniform call signature across strategies.
    specialty : str, optional
        If given, restrict to this specialty value.

    Returns
    -------
    int
        Number of matching admitted rows.
    """
    del prediction_window  # retained for signature compatibility
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (ed_visits["is_admitted"].astype(bool))
    )
    if specialty is not None:
        mask = mask & (ed_visits["specialty"] == specialty)
    return int(mask.sum())


def count_observed_admitted_in_window(
    ed_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    admission_datetime_col: str = "admission_datetime",
) -> int:
    """Count admissions with admission time in `(moment, moment + window]`.

    Parameters
    ----------
    ed_visits : pandas.DataFrame
        Must include *admission_datetime_col* (datetime-like), *snapshot_date*,
        *prediction_time*, *is_admitted*, and *specialty* if *specialty*
        is set.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment for the inclusion window.
    specialty : str, optional
        If given, restrict to this specialty.
    admission_datetime_col : str, default "admission_datetime"
        Column name for admission timestamp.

    Returns
    -------
    int
        Number of admitted rows whose admission time falls in the window.

    Raises
    ------
    ValueError
        If *admission_datetime_col* is missing from *ed_visits*.
    """
    if admission_datetime_col not in ed_visits.columns:
        raise ValueError(
            f"admitted_in_window requires column {admission_datetime_col!r} on ed_visits"
        )
    moment = _prediction_moment(snapshot_date, prediction_time)
    col = ed_visits[admission_datetime_col]
    moment = _align_tz(moment, col)
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (ed_visits["is_admitted"].astype(bool))
        & (col > moment)
        & (col <= moment + prediction_window)
    )
    if specialty is not None:
        mask = mask & (ed_visits["specialty"] == specialty)
    return int(mask.sum())


def count_observed_departed_in_window(
    inpatient_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    departure_datetime_col: str = "departure_datetime",
) -> int:
    """Count inpatient departures in `(moment, moment + window]`.

    Parameters
    ----------
    inpatient_visits : pandas.DataFrame
        Must include *departure_datetime_col*, *snapshot_date*, and
        *prediction_time*. Optional *current_subspecialty* or *specialty*
        is used when *specialty* is set.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment.
    specialty : str, optional
        If given, filter by *current_subspecialty* or *specialty* when present.
    departure_datetime_col : str, default "departure_datetime"
        Column name for departure timestamp.

    Returns
    -------
    int
        Number of rows with departure time in the window.

    Raises
    ------
    ValueError
        If *departure_datetime_col* is missing from *inpatient_visits*.
    """
    if departure_datetime_col not in inpatient_visits.columns:
        raise ValueError(
            f"departed_in_window requires column {departure_datetime_col!r} "
            "on inpatient_visits"
        )
    moment = _prediction_moment(snapshot_date, prediction_time)
    col = inpatient_visits[departure_datetime_col]
    moment = _align_tz(moment, col)
    mask = (
        (inpatient_visits["snapshot_date"] == snapshot_date)
        & (inpatient_visits["prediction_time"] == prediction_time)
        & (col > moment)
        & (col <= moment + prediction_window)
    )
    if specialty is not None and "current_subspecialty" in inpatient_visits.columns:
        mask = mask & (inpatient_visits["current_subspecialty"] == specialty)
    elif specialty is not None and "specialty" in inpatient_visits.columns:
        mask = mask & (inpatient_visits["specialty"] == specialty)
    return int(mask.sum())


def count_observed_arrived_in_window(
    visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    arrival_datetime_col: str = "arrival_datetime",
) -> int:
    """Count arrivals with arrival time in `(moment, moment + window]`.

    Suitable for direct-admission (non-ED) yet-to-arrive style comparisons.

    Parameters
    ----------
    visits : pandas.DataFrame
        Must include *arrival_datetime_col*. If *snapshot_date* and/or
        *prediction_time* columns exist, they are filtered to match.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound added to the prediction moment.
    specialty : str, optional
        If given and a *specialty* column exists, filter to that value.
    arrival_datetime_col : str, default "arrival_datetime"
        Column name for arrival timestamp.

    Returns
    -------
    int
        Number of rows in the arrival time window.

    Raises
    ------
    ValueError
        If *arrival_datetime_col* is missing from *visits*.
    """
    if arrival_datetime_col not in visits.columns:
        raise ValueError(
            f"arrived_in_window requires column {arrival_datetime_col!r} on visits"
        )
    moment = _prediction_moment(snapshot_date, prediction_time)
    col = visits[arrival_datetime_col]
    moment = _align_tz(moment, col)
    mask = (col > moment) & (col <= moment + prediction_window)
    if "snapshot_date" in visits.columns:
        mask = mask & (visits["snapshot_date"] == snapshot_date)
    if "prediction_time" in visits.columns:
        mask = mask & (visits["prediction_time"] == prediction_time)
    if specialty is not None and "specialty" in visits.columns:
        mask = mask & (visits["specialty"] == specialty)
    return int(mask.sum())


def count_observed_arrived_and_admitted_in_window(
    ed_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    specialty: Optional[str] = None,
    arrival_datetime_col: str = "arrival_datetime",
    admission_datetime_col: str = "admission_datetime",
) -> int:
    """Count ED rows with arrival after the moment and admission within the window.

    Parameters
    ----------
    ed_visits : pandas.DataFrame
        Must include *arrival_datetime_col*, *admission_datetime_col*,
        *snapshot_date*, *prediction_time*, and *is_admitted*.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Upper bound on admission time after the prediction moment.
    specialty : str, optional
        If given, restrict to this *specialty*.
    arrival_datetime_col : str, default "arrival_datetime"
        Column name for ED arrival time.
    admission_datetime_col : str, default "admission_datetime"
        Column name for admission time.

    Returns
    -------
    int
        Number of admitted rows matching the ED YTA-style window rule.

    Raises
    ------
    ValueError
        If a required datetime column is missing from *ed_visits*.
    """
    for coln in (arrival_datetime_col, admission_datetime_col):
        if coln not in ed_visits.columns:
            raise ValueError(
                f"arrived_and_admitted_in_window requires column {coln!r} on ed_visits"
            )
    moment = _prediction_moment(snapshot_date, prediction_time)
    arr = ed_visits[arrival_datetime_col]
    adm = ed_visits[admission_datetime_col]
    moment_arr = _align_tz(moment, arr)
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (arr > moment_arr)
        & (adm > moment_arr)
        & (adm <= moment_arr + prediction_window)
        & (ed_visits["is_admitted"].astype(bool))
    )
    if specialty is not None:
        mask = mask & (ed_visits["specialty"] == specialty)
    return int(mask.sum())


def count_observed(
    observation_mode: str,
    *,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    ed_visits: Optional[pd.DataFrame] = None,
    inpatient_visits: Optional[pd.DataFrame] = None,
    visits: Optional[pd.DataFrame] = None,
    specialty: Optional[str] = None,
    admission_datetime_col: str = "admission_datetime",
    departure_datetime_col: str = "departure_datetime",
    arrival_datetime_col: str = "arrival_datetime",
) -> int:
    """Dispatch to a counting strategy by name.

    Parameters
    ----------
    observation_mode : str
        One of the strings in OBSERVATION_MODES.
    snapshot_date : datetime.date
        Snapshot calendar date.
    prediction_time : tuple of (int, int)
        Hour and minute of the prediction moment.
    prediction_window : datetime.timedelta
        Prediction horizon passed through to the underlying counter.
    ed_visits : pandas.DataFrame, optional
        ED visit frame; required for modes that count from ED snapshots.
    inpatient_visits : pandas.DataFrame, optional
        Inpatient frame; required for *departed_in_window*.
    visits : pandas.DataFrame, optional
        Generic arrivals frame for *arrived_in_window* when not using
        *ed_visits*.
    specialty : str, optional
        Passed through to the underlying counter when supported.
    admission_datetime_col : str, default "admission_datetime"
        Passed to *admitted_in_window* and *arrived_and_admitted_in_window*.
    departure_datetime_col : str, default "departure_datetime"
        Passed to *departed_in_window*.
    arrival_datetime_col : str, default "arrival_datetime"
        Passed to *arrived_in_window* and *arrived_and_admitted_in_window*.

    Returns
    -------
    int
        Observed count. Returns 0 when a required frame is missing (None).

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
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unknown observation_mode {observation_mode!r}; "
            f"expected one of {OBSERVATION_MODES}"
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
            admission_datetime_col=admission_datetime_col,
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
            departure_datetime_col=departure_datetime_col,
        )
    if observation_mode == "arrived_in_window":
        frame = visits if visits is not None else ed_visits
        if frame is None:
            return 0
        return count_observed_arrived_in_window(
            frame,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            arrival_datetime_col=arrival_datetime_col,
        )
    if observation_mode == "arrived_and_admitted_in_window":
        if ed_visits is None:
            return 0
        return count_observed_arrived_and_admitted_in_window(
            ed_visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            specialty=specialty,
            arrival_datetime_col=arrival_datetime_col,
            admission_datetime_col=admission_datetime_col,
        )
    raise RuntimeError("unreachable")  # pragma: no cover
