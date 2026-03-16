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
from typing import Any, Dict, List, Optional


def default_scalars_meta(
    reliability_thresholds: Dict[str, int],
    schema_version: int = 3,
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

    Attributes
    ----------
    rows
        Accumulated scalar rows, each representing one
        ``(flow, service, component, prediction_time)`` record.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)

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
        }
        if metrics:
            row.update(metrics)
        self.rows.append(row)

    def to_payload(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Build serialisable scalars payload.

        Parameters
        ----------
        meta
            Metadata block for the output payload.

        Returns
        -------
        Dict[str, Any]
            Payload with ``_meta`` and flat ``results`` rows.
        """
        return {"_meta": meta, "results": self.rows}
