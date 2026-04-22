"""Flat scalar output utilities.

This module standardises scalar output as a flat list of rows, suitable for
direct loading into pandas and straightforward cross-run comparisons.

It provides:

- metadata generation for scalar payloads
- a collector that accumulates row-wise scalar records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def default_scalars_meta(
    reliability_thresholds: Dict[str, int],
    schema_version: int = 4,
) -> Dict[str, Any]:
    """Build default metadata block for flat scalar outputs.

    Parameters
    ----------
    reliability_thresholds
        Reliability thresholds included in output metadata.
    schema_version
        Scalar schema version identifier.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with schema version, generation timestamp, and
        reliability thresholds.
    """
    return {
        "schema_version": schema_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reliability_thresholds": reliability_thresholds,
    }


@dataclass
class ScalarsCollector:
    """Collect scalar rows in a flat, DataFrame-friendly structure.

    Supports incremental evaluation: prior rows loaded from an existing
    ``scalars.json`` are preserved and merged with new rows at output
    time.  When a prior row and a new row share the same key
    ``(flow, service, component, prediction_time)``, the new row wins.

    Attributes
    ----------
    rows
        Scalar rows accumulated during the current evaluation run.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    _prior_rows: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _row_key(row: Dict[str, Any]) -> Tuple:
        return (
            row.get("flow"),
            row.get("service"),
            row.get("component"),
            row.get("prediction_time"),
            row.get("group_weekday", "all"),
        )

    def load_prior(self, rows: List[Dict[str, Any]]) -> None:
        """Load rows from a previous run for incremental merging.

        Parameters
        ----------
        rows
            Flat result rows from an earlier ``scalars.json``.  These are
            kept unless superseded by a new row with the same key.
        """
        self._prior_rows = list(rows)

    def merged_rows(self) -> List[Dict[str, Any]]:
        """Return prior rows merged with current rows (current wins).

        Returns
        -------
        List[Dict[str, Any]]
            Combined rows, deduplicated by
            ``(flow, service, component, prediction_time)``.
        """
        current_keys = {self._row_key(r) for r in self.rows}
        merged = [r for r in self._prior_rows if self._row_key(r) not in current_keys]
        merged.extend(self.rows)
        return merged

    def record(
        self,
        *,
        flow: str,
        service: Optional[str],
        component: str,
        prediction_time: str,
        flow_type: str,
        aspirational: bool,
        evaluated: bool,
        reason: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        group: Optional[Dict[str, str]] = None,
    ) -> None:
        """Append one scalar row.

        Parameters
        ----------
        flow
            Flow name.
        service
            Service name, or ``None`` for non-service-specific rows.
        component
            Component name for this row.
        prediction_time
            Prediction time key (typically ``HHMM``).
        flow_type
            Semantic flow type.
        aspirational
            Whether the flow is aspirational.
        evaluated
            Whether diagnostics were evaluated for this row.
        reason
            Optional reason when a row is not evaluated.
        metrics
            Optional mode-specific metric fields merged into the row.
        group
            Optional mapping of grouping dimensions for this row (for
            example ``{"weekday": "mon"}``).  Keys are prefixed with
            ``group_`` in the output row so that an ungrouped (``"all"``)
            row and a per-weekday row with the same flow/service/
            component/prediction_time coexist without colliding.  When
            omitted, a ``group_weekday`` column with value ``"all"`` is
            written so every row has a consistent shape.
        """
        row: Dict[str, Any] = {
            "flow": flow,
            "service": service,
            "component": component,
            "prediction_time": prediction_time,
            "flow_type": flow_type,
            "aspirational": aspirational,
            "evaluated": evaluated,
            "reason": reason,
            "group_weekday": "all",
        }
        if group:
            for dim, value in group.items():
                row[f"group_{dim}"] = value
        if metrics:
            row.update(metrics)
        self.rows.append(row)

    def service_summary(self) -> Dict[str, Any]:
        """Compute a summary of service coverage from all rows (merged).

        A service is *active* if any of its rows generated charts (i.e.
        does not have ``charts_generated`` set to ``False``).  A service
        is *inactive* only when every row that expresses an opinion marks
        it as ``charts_generated: false``.  Services that appear only in
        non-service-specific rows (``service is None``) are excluded.

        Returns
        -------
        Dict[str, Any]
            Summary with ``total_services``, ``active_services``,
            ``inactive_services``, and ``inactive_service_names``.
        """
        all_services: set = set()
        active_services: set = set()
        for row in self.merged_rows():
            svc = row.get("service")
            if svc is None:
                continue
            # Only consider the headline ("all") rows for service-level
            # activity classification.  Per-group rows (e.g. group_weekday
            # = "mon") would otherwise flip an inactive service to active.
            if row.get("group_weekday", "all") != "all":
                continue
            all_services.add(svc)
            if row.get("charts_generated") is not False:
                active_services.add(svc)
        inactive_services = all_services - active_services
        return {
            "total_services": len(all_services),
            "active_services": len(active_services),
            "inactive_services": len(inactive_services),
            "inactive_service_names": sorted(inactive_services),
        }

    def to_payload(
        self,
        meta: Dict[str, Any],
        *,
        include_service_summary: bool = False,
    ) -> Dict[str, Any]:
        """Build serialisable scalars payload.

        Prior rows and current rows are merged, with current rows taking
        precedence when both share the same
        ``(flow, service, component, prediction_time)`` key.

        Parameters
        ----------
        meta
            Metadata block for the output payload.
        include_service_summary
            When True, a ``_service_summary`` block is appended to the
            payload summarising active vs inactive service counts.

        Returns
        -------
        Dict[str, Any]
            Payload with ``_meta``, flat ``results`` rows, and
            optionally ``_service_summary``.
        """
        payload: Dict[str, Any] = {"_meta": meta, "results": self.merged_rows()}
        if include_service_summary:
            payload["_service_summary"] = self.service_summary()
        return payload
