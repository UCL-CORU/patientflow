"""Scalar row collection, merge keys, and `scalars.json` serialisation.

Import from this module explicitly, for example::

    from patientflow.evaluate.scalars import ScalarsCollector

The JSON artefact contains `evaluation_rows` and `_service_summary`.
Handlers record service coverage under `_service_summary["by_slice"]`,
keyed by `evaluation_mode/flow_name/component` so multiple modes in one
run do not overwrite each other.

See Also
--------
patientflow.evaluate.runner.run_evaluation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from patientflow.evaluate.inputs import EvaluationTarget

# Reliability: minimum positive cases on the evaluated split for classifier headline metrics.
RELIABILITY_MIN_POSITIVE_CASES: int = 30

# Reliability: minimum departures for transition-matrix row calibration.
RELIABILITY_MIN_OBSERVATIONS_TRANSITION: int = 30

# Reliability / chart Gate A: minimum snapshot leaves (distribution) or histogram
# days (arrival deltas) for ``reliable`` and for drawing a panel.
RELIABILITY_MIN_OBSERVATIONS_DISTRIBUTION: int = 30

SERVICE_SENTINEL_ALL: str = "_all_"


def scalar_target_fields(target: EvaluationTarget) -> Dict[str, Any]:
    """Return identity fields shared by every ``evaluation_rows`` entry for *target*.

    Includes ``observation_mode`` so downstream tables can interpret distribution
    and benchmark scalars without joining back to ``EvaluationTarget`` definitions.
    """
    return {
        "evaluation_mode": target.evaluation_mode,
        "flow": target.flow_name,
        "flow_type": target.flow_type,
        "observation_mode": target.observation_mode,
    }


def scalar_merge_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return a stable tuple key used to merge and deduplicate scalar rows.

    Rows with different `evaluation_mode`, `component`, `prediction_time`, or
    `model_name` never share a key, so one grain cannot overwrite another.

    Parameters
    ----------
    row : mapping
        Scalar row dictionary (typically including `evaluation_mode`,
        `observation_mode`, `flow`, `service`, `component`, `prediction_time`,
        `model_name`).

    Returns
    -------
    tuple
        Components in order: `evaluation_mode`, `flow`, `service`,
        `component`, `prediction_time` (normalised to `(hour, minute)` or
        `None`), `model_name` (empty string if absent).

    Notes
    -----
    Survival rows use `prediction_time=None` and typically
    `service="_all_"`. Classifier model-diagnostics rows use a non-empty
    `model_name` (the clocked model key, e.g. ``admissions_0600``) and may
    include ``metrics_split`` (``"test"``, ``"valid"``, or ``"cv_train"``) for
    the population used at train time (from
    ``TrainedClassifier.selected_eval_metrics``). Plot cohort labels use
    run-level ``EvaluationInputs.eval_split`` instead.
    Classifier probability-quality rows use flow-level keys
    with ``model_name=""`` and ``prediction_time=None``.
    """
    return (
        row.get("evaluation_mode"),
        row.get("flow"),
        row.get("service"),
        row.get("component"),
        _json_key_prediction_time(row.get("prediction_time")),
        row.get("model_name") or "",
    )


def _json_key_prediction_time(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return value


@dataclass
class ScalarsCollector:
    """Accumulate scalar dicts, merge with prior runs, and write `scalars.json`.

    Rows are stored keyed by `scalar_merge_key`. Prior rows from an
    existing file can be loaded so incremental runs replace only matching keys.

    Attributes
    ----------
    rows : list of dict
        Present on the dataclass; new code should use `add_row`, which
        updates the internal merge index only.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    _by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = field(default_factory=dict)
    _service_summary: Dict[str, Any] = field(default_factory=dict)

    def load_prior(self, prior_rows: Iterable[Mapping[str, Any]]) -> None:
        """Merge in rows from a previous run; later `add_row` wins on key clash.

        Parameters
        ----------
        prior_rows : iterable of mapping
            Each mapping should be a scalar row compatible with
            `scalar_merge_key`.
        """
        for r in prior_rows:
            key = scalar_merge_key(r)
            self._by_key[key] = dict(r)

    def load_prior_from_path(self, path: Path) -> None:
        """Load `evaluation_rows` (and optional `_service_summary`) from JSON.

        Parameters
        ----------
        path : pathlib.Path
            Path to `scalars.json`. If the file does not exist, this is a no-op.

        Notes
        -----
        Accepts either `evaluation_rows` or legacy key `rows` in the payload.
        """
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("evaluation_rows") or payload.get("rows") or []
        if isinstance(rows, list):
            self.load_prior(rows)
        summary = payload.get("_service_summary")
        if isinstance(summary, dict):
            self._service_summary = dict(summary)

    def add_row(self, row: Mapping[str, Any]) -> None:
        """Insert or replace one scalar row keyed by `scalar_merge_key`.

        Parameters
        ----------
        row : mapping
            Row dictionary to store.
        """
        r = dict(row)
        self._by_key[scalar_merge_key(r)] = r

    def set_service_summary(self, summary: Mapping[str, Any]) -> None:
        """Replace the entire `_service_summary` payload.

        Parameters
        ----------
        summary : mapping
            Summary block written verbatim to JSON under `_service_summary`.
        """
        self._service_summary = dict(summary)

    def merge_service_summary_slice(
        self, slice_key: str, fragment: Mapping[str, Any]
    ) -> None:
        """Store one fragment under `_service_summary["by_slice"][slice_key]`.

        Parameters
        ----------
        slice_key : str
            Unique key for this handler slice (for example
            `distribution/ed_current_beds/bed_demand_ed_current`).
        fragment : mapping
            Service coverage counters and inactive service names for that slice.
        """
        by = self._service_summary.setdefault("by_slice", {})
        by[str(slice_key)] = dict(fragment)

    def service_summary(self) -> Dict[str, Any]:
        """Return a shallow copy of the accumulated `_service_summary` dict."""
        return dict(self._service_summary)

    def as_list(self) -> List[Dict[str, Any]]:
        """Return all scalar rows as a list (order not guaranteed)."""
        return list(self._by_key.values())

    def write_json(self, path: Path) -> None:
        """Serialise `evaluation_rows` and `_service_summary` to `path`.

        Parameters
        ----------
        path : pathlib.Path
            Output file path; parent directories are created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Any] = {
            "evaluation_rows": self.as_list(),
            "_service_summary": dict(self._service_summary),
        }
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def classifier_reliable(
    selected_eval_metrics: Mapping[str, Any],
    train_valid_test_positive_cases: Mapping[str, Any],
) -> bool:
    """Return whether positive-case counts meet the headline reliability bar.

    Parameters
    ----------
    selected_eval_metrics : mapping
        Must include `split` from trained models: ``"test"``, ``"valid"``, or
        ``"cv_train"``.
    train_valid_test_positive_cases : mapping
        Dataset metadata with `test_positive_cases` / `valid_positive_cases`
        (used for ``"test"`` / ``"valid"``). For ``"cv_train"``, the count is
        taken from ``selected_eval_metrics["n_positive_cases"]``.

    Returns
    -------
    bool
        `True` if the count for the active split is at least
        `RELIABILITY_MIN_POSITIVE_CASES`.
    """
    split = selected_eval_metrics.get("split")
    if split == "test":
        n = train_valid_test_positive_cases.get("test_positive_cases")
    elif split == "valid":
        n = train_valid_test_positive_cases.get("valid_positive_cases")
    elif split == "cv_train":
        n = selected_eval_metrics.get("n_positive_cases")
    else:
        return False
    if n is None:
        return False
    return int(n) >= RELIABILITY_MIN_POSITIVE_CASES
