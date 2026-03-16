"""Adapter functions between typed payloads and plotting input dictionaries.

This module provides boundary conversions for distribution-style evaluation:

- from plotting dictionaries into typed ``SnapshotResult`` mappings
- from typed ``SnapshotResult`` mappings into ``prob_dist_dict_all`` format

The plotting format is retained because existing visualisation functions
consume that dictionary structure.
"""

from datetime import date
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from patientflow.evaluation.types import SnapshotResult
from patientflow.load import get_model_key


def from_legacy_prob_dist_dict(
    legacy_prob_dist_dict: Mapping[date, Mapping[str, Any]],
) -> Dict[date, SnapshotResult]:
    """Convert a per-time plotting payload into typed snapshot results.

    Parameters
    ----------
    legacy_prob_dist_dict
        Mapping of snapshot date to payload with keys
        ``"agg_predicted"`` (DataFrame) and ``"agg_observed"`` (int).

    Returns
    -------
    Dict[date, SnapshotResult]
        Typed snapshot results keyed by snapshot date.

    Raises
    ------
    ValueError
        If required payload keys or probability columns are missing.
    TypeError
        If ``agg_predicted`` is not a pandas DataFrame.
    """
    converted: Dict[date, SnapshotResult] = {}
    for snapshot_date, payload in legacy_prob_dist_dict.items():
        if "agg_predicted" not in payload or "agg_observed" not in payload:
            raise ValueError(
                "Legacy payload must contain 'agg_predicted' and 'agg_observed'."
            )

        agg_predicted = payload["agg_predicted"]
        if not isinstance(agg_predicted, pd.DataFrame):
            raise TypeError(
                "Legacy payload field 'agg_predicted' must be a pandas DataFrame."
            )

        if "agg_proba" in agg_predicted.columns:
            proba_series = agg_predicted["agg_proba"]
        elif len(agg_predicted.columns) > 0:
            proba_series = agg_predicted.iloc[:, 0]
        else:
            raise ValueError(
                "Legacy payload DataFrame 'agg_predicted' has no probability column."
            )

        if len(agg_predicted.index) == 0:
            offset = 0
        else:
            offset = int(agg_predicted.index.min())

        converted[snapshot_date] = SnapshotResult(
            predicted_pmf=np.asarray(proba_series.to_numpy(), dtype=float),
            observed=int(payload["agg_observed"]),
            offset=offset,
        )
    return converted


def to_legacy_prob_dist_dict_all(
    snapshots_by_time: Mapping[Tuple[int, int], Mapping[date, SnapshotResult]],
    model_name: str,
) -> Dict[str, Dict[date, Dict[str, Any]]]:
    """Convert typed snapshots into ``prob_dist_dict_all`` plot input format.

    Parameters
    ----------
    snapshots_by_time
        Mapping of prediction time to typed per-date snapshot results.
    model_name
        Base model name used by ``get_model_key``.

    Returns
    -------
    Dict[str, Dict[date, Dict[str, Any]]]
        Nested dictionary keyed by model key with per-date entries containing:
        ``agg_predicted`` (DataFrame) and ``agg_observed`` (int).
    """
    legacy: Dict[str, Dict[date, Dict[str, Any]]] = {}
    for prediction_time, snapshots in snapshots_by_time.items():
        model_key = get_model_key(model_name, prediction_time)
        by_date: Dict[date, Dict[str, Any]] = {}

        for snapshot_date, result in snapshots.items():
            support = range(result.offset, result.offset + len(result.predicted_pmf))
            agg_predicted = pd.DataFrame(
                {"agg_proba": np.asarray(result.predicted_pmf, dtype=float)},
                index=support,
            )
            by_date[snapshot_date] = {
                "agg_predicted": agg_predicted,
                "agg_observed": int(result.observed),
            }

        legacy[model_key] = by_date

    return legacy
