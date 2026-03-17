"""Observed-count computation for evaluation targets.

This module provides functions that count observed values for comparison
against predicted distributions.  The correct counting strategy depends on
the target's ``observation_mode``:

``count_at_some_point``
    Patients flagged as admitted at the snapshot, regardless of when they
    actually leave ED (uses the ``is_admitted`` flag).
``count_in_window``
    Events (admissions or departures) that occur within the prediction
    window after the prediction moment.

The :func:`count_observed` dispatcher selects the appropriate strategy
from the ``observation_mode`` field of an :class:`EvaluationTarget`.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd


def count_observed_at_some_point(
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


def count_observed_in_window(
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
    """Count events occurring within the prediction window.

    An event is counted when the timestamp in *end_time_col* falls within
    ``(prediction_moment, prediction_moment + prediction_window]`` and the
    event started before or at the prediction moment.

    This applies to both admission-in-window and departure-in-window
    targets; the caller supplies the appropriate dataframe and column
    names.

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
    prediction_moment = datetime.combine(
        snapshot_date, time(prediction_time[0], prediction_time[1])
    )

    if start_time_col in visits.columns:
        start_series = visits[start_time_col]
    elif visits.index.name == start_time_col:
        start_series = visits.index.to_series()
    else:
        raise ValueError(
            f"'{start_time_col}' not found in DataFrame columns or index"
        )

    if start_series.dt.tz is not None:
        prediction_moment = prediction_moment.replace(tzinfo=timezone.utc)

    window_end = prediction_moment + prediction_window

    mask = (start_series <= prediction_moment) & (
        visits[end_time_col] > prediction_moment
    ) & (visits[end_time_col] <= window_end)

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
        One of ``"count_at_some_point"`` or ``"count_in_window"``.
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
        Boolean/flag column for ``count_at_some_point`` mode.
    service_col
        Column used for service-level filtering.
    start_time_col
        Start-time column for ``count_in_window`` mode.
    end_time_col
        End-time column for ``count_in_window`` mode.

    Returns
    -------
    int
        Observed count for the given snapshot.

    Raises
    ------
    ValueError
        If *observation_mode* does not support per-snapshot counting.
    """
    if observation_mode == "count_at_some_point":
        return count_observed_at_some_point(
            visits,
            snapshot_date,
            prediction_time,
            specialty=specialty,
            label_col=label_col,
            service_col=service_col,
        )

    if observation_mode == "count_in_window":
        return count_observed_in_window(
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
        "are 'count_at_some_point' and 'count_in_window'."
    )
