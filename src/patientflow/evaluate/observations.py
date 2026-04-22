"""Observed-count computation for evaluation targets.

This module provides functions that count observed values for comparison
against predicted distributions.  The correct counting strategy depends on
the target's ``observation_mode``:

``admitted_at_some_point``
    Patients flagged as admitted at the snapshot, regardless of when they
    actually leave ED (uses the ``is_admitted`` flag).
``admitted_in_window``
    Patients already present whose admission to a bed falls within the
    prediction window.
``departed_in_window``
    Alias for the same counting logic as ``admitted_in_window``, used for
    departure targets to make intent clear.
``arrived_in_window``
    Patients who arrive after the prediction moment and before the end of
    the prediction window.  Only checks the arrival/start column.
``arrived_and_admitted_in_window``
    Patients who arrive after the prediction moment and whose admission to
    a bed (end event) falls within the prediction window.

The :func:`count_observed` dispatcher selects the appropriate strategy
from the ``observation_mode`` field of an :class:`EvaluationTarget`.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd


def _resolve_start_series(visits: pd.DataFrame, start_time_col: str) -> pd.Series:
    """Extract the start-time series from a DataFrame column or index."""
    if start_time_col in visits.columns:
        return visits[start_time_col]
    if visits.index.name == start_time_col:
        return visits.index.to_series()
    raise ValueError(f"'{start_time_col}' not found in DataFrame columns or index")


def _prediction_moment(
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    start_series: pd.Series,
) -> datetime:
    """Build a prediction moment, adding UTC if the data is tz-aware."""
    moment = datetime.combine(
        snapshot_date, time(prediction_time[0], prediction_time[1])
    )
    if start_series.dt.tz is not None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def count_observed_admitted_at_some_point(
    ed_visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    *,
    specialty: Optional[str] = None,
    label_col: str = "is_admitted",
    service_col: str = "specialty",
) -> int:
    """Count patients flagged as admitted at the snapshot (any-time semantics).

    Parameters
    ----------
    ed_visits
        Full ED visits dataframe with columns ``snapshot_date``,
        ``prediction_time``, *label_col*, and (when *specialty* is given)
        *service_col*.
    snapshot_date
        Date of the snapshot.
    prediction_time
        ``(hour, minute)`` of the prediction moment.
    specialty
        If provided, count only admissions to this specialty.
    label_col
        Boolean/flag column indicating admission.
    service_col
        Column used for service-level filtering.

    Returns
    -------
    int
        Number of admitted patients matching the criteria.
    """
    mask = (
        (ed_visits["snapshot_date"] == snapshot_date)
        & (ed_visits["prediction_time"] == prediction_time)
        & (ed_visits[label_col].astype(bool))
    )
    if specialty is not None:
        mask = mask & (ed_visits[service_col] == specialty)
    return int(mask.sum())


def count_observed_admitted_in_window(
    visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    start_time_col: str = "arrival_datetime",
    end_time_col: str = "departure_datetime",
    specialty: Optional[str] = None,
    service_col: str = "specialty",
) -> int:
    """Count patients already present whose end event falls in-window.

    An event is counted when the timestamp in *end_time_col* falls within
    ``(prediction_moment, prediction_moment + prediction_window]`` and the
    event started before or at the prediction moment.

    Also used as ``departed_in_window`` (same logic, different intent).

    Parameters
    ----------
    visits
        Visits dataframe.  *start_time_col* may be a column or the index.
    snapshot_date
        Date of the snapshot.
    prediction_time
        ``(hour, minute)`` of the prediction moment.
    prediction_window
        Duration of the forecast window.
    start_time_col
        Column (or index name) for event start timestamps.
    end_time_col
        Column for event end timestamps.
    specialty
        If provided, count only events for this specialty.
    service_col
        Column used for service-level filtering.

    Returns
    -------
    int
        Number of events within the window.
    """
    start_series = _resolve_start_series(visits, start_time_col)
    moment = _prediction_moment(snapshot_date, prediction_time, start_series)
    window_end = moment + prediction_window

    mask = (
        (start_series <= moment)
        & (visits[end_time_col] > moment)
        & (visits[end_time_col] <= window_end)
    )

    if specialty is not None:
        mask = mask & (visits[service_col] == specialty)

    return int(mask.sum())


def count_observed_arrived_in_window(
    visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    start_time_col: str = "arrival_datetime",
    specialty: Optional[str] = None,
    service_col: str = "specialty",
) -> int:
    """Count patients who arrive within the prediction window.

    Only checks the arrival/start column — does not check an end column.
    Used for direct-admission targets where arrival = admission
    (non-ED emergency, elective).

    Parameters
    ----------
    visits
        Visits dataframe.  *start_time_col* may be a column or the index.
    snapshot_date
        Date of the snapshot.
    prediction_time
        ``(hour, minute)`` of the prediction moment.
    prediction_window
        Duration of the forecast window.
    start_time_col
        Column (or index name) for arrival timestamps.
    specialty
        If provided, count only events for this specialty.
    service_col
        Column used for service-level filtering.

    Returns
    -------
    int
        Number of arrivals within the window.
    """
    start_series = _resolve_start_series(visits, start_time_col)
    moment = _prediction_moment(snapshot_date, prediction_time, start_series)
    window_end = moment + prediction_window

    mask = (start_series > moment) & (start_series <= window_end)

    if specialty is not None:
        mask = mask & (visits[service_col] == specialty)

    return int(mask.sum())


def count_observed_arrived_and_admitted_in_window(
    visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    *,
    start_time_col: str = "arrival_datetime",
    end_time_col: str = "departure_datetime",
    specialty: Optional[str] = None,
    service_col: str = "specialty",
) -> int:
    """Count patients who arrive after the prediction moment and whose
    end event (admission to bed) falls within the window.

    Parameters
    ----------
    visits
        Visits dataframe.  *start_time_col* may be a column or the index.
    snapshot_date
        Date of the snapshot.
    prediction_time
        ``(hour, minute)`` of the prediction moment.
    prediction_window
        Duration of the forecast window.
    start_time_col
        Column (or index name) for arrival timestamps.
    end_time_col
        Column for admission-to-bed / end-event timestamps.
    specialty
        If provided, count only events for this specialty.
    service_col
        Column used for service-level filtering.

    Returns
    -------
    int
        Number of events matching the criteria.
    """
    start_series = _resolve_start_series(visits, start_time_col)
    moment = _prediction_moment(snapshot_date, prediction_time, start_series)
    window_end = moment + prediction_window

    mask = (start_series > moment) & (visits[end_time_col] <= window_end)

    if specialty is not None:
        mask = mask & (visits[service_col] == specialty)

    return int(mask.sum())


def count_observed(
    observation_mode: str,
    *,
    visits: pd.DataFrame,
    snapshot_date: date,
    prediction_time: Tuple[int, int],
    prediction_window: timedelta,
    specialty: Optional[str] = None,
    label_col: str = "is_admitted",
    service_col: str = "specialty",
    start_time_col: str = "arrival_datetime",
    end_time_col: str = "departure_datetime",
) -> int:
    """Dispatch to the correct counting function based on *observation_mode*.

    Parameters
    ----------
    observation_mode
        One of ``"admitted_at_some_point"``, ``"admitted_in_window"``,
        ``"departed_in_window"``, ``"arrived_in_window"``, or
        ``"arrived_and_admitted_in_window"``.
        Other modes (``"not_applicable"``, ``"classifier_binary"``,
        ``"arrival_rates"``, ``"survival_comparison"``) do not produce
        per-snapshot counts and will raise :class:`ValueError`.
    visits
        Visits dataframe appropriate for the target being evaluated.
    snapshot_date
        Date of the snapshot.
    prediction_time
        ``(hour, minute)`` of the prediction moment.
    prediction_window
        Duration of the forecast window.
    specialty
        Optional service filter.
    label_col
        Boolean/flag column for ``admitted_at_some_point`` mode.
    service_col
        Column used for service-level filtering.
    start_time_col
        Start-time column for window-based counting modes.
    end_time_col
        End-time column for window-based counting modes.

    Returns
    -------
    int
        Observed count for the given snapshot.

    Raises
    ------
    ValueError
        If *observation_mode* does not support per-snapshot counting.
    """
    if observation_mode == "admitted_at_some_point":
        return count_observed_admitted_at_some_point(
            visits,
            snapshot_date,
            prediction_time,
            specialty=specialty,
            label_col=label_col,
            service_col=service_col,
        )

    if observation_mode in ("admitted_in_window", "departed_in_window"):
        return count_observed_admitted_in_window(
            visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            start_time_col=start_time_col,
            end_time_col=end_time_col,
            specialty=specialty,
            service_col=service_col,
        )

    if observation_mode == "arrived_in_window":
        return count_observed_arrived_in_window(
            visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            start_time_col=start_time_col,
            specialty=specialty,
            service_col=service_col,
        )

    if observation_mode == "arrived_and_admitted_in_window":
        return count_observed_arrived_and_admitted_in_window(
            visits,
            snapshot_date,
            prediction_time,
            prediction_window,
            start_time_col=start_time_col,
            end_time_col=end_time_col,
            specialty=specialty,
            service_col=service_col,
        )

    raise ValueError(
        f"observation_mode {observation_mode!r} does not support "
        "per-snapshot counting. Modes that produce per-snapshot counts "
        "are 'admitted_at_some_point', 'admitted_in_window', "
        "'departed_in_window', 'arrived_in_window', and "
        "'arrived_and_admitted_in_window'."
    )
