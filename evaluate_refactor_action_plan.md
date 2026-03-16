# Evaluation refactor action plan

## Goal

Refactor the evaluation pipeline for clarity, extensibility, and maintainability while preserving backward compatibility for:

- `prob_dist_dict_all` shape
- existing visualisation functions and their inputs
- notebooks and code that rely on the legacy aggregate/viz interfaces

Back compatibility is **not required** for:

- `evaluate.py` implementation and API
- notebook `4d_Evaluate_demand_predictions.ipynb`
- current nested `scalars.json` structure

This plan adopts scalar schema option **(b): flat records**.

---

## Architecture overview

Move from one large `src/patientflow/evaluate.py` module to a staged `evaluation/` package (later cut over to `evaluate`) with focused modules and explicit boundaries:

```text
src/patientflow/evaluation/
    __init__.py      # re-exports public API
    types.py         # SnapshotResult, EvaluationTarget, input dataclasses
    targets.py       # get_default_evaluation_targets()
    scalars.py       # ScalarsCollector (flat records)
    handlers.py      # per-mode evaluation functions
    builder.py       # EvaluationInputsBuilder
    runner.py        # run_evaluation
    adapters.py      # typed <-> legacy prob_dist_dict_all conversions
    metrics.py       # classifier metric extraction (slim)
    legacy_api.py    # calculate_results, calc_mae_mpe (kept available)
```

Key principle: use typed structures internally, and adapt to legacy `prob_dist_dict_all` only at the viz boundary.

---

## Compatibility boundary

### Preserve exactly

- `prob_dist_dict_all` contract (`{model_key: {date: {"agg_predicted": DataFrame, "agg_observed": int}}}`)
- Viz APIs and behaviour:
  - `plot_epudd`
  - `plot_deltas`
  - `plot_arrival_deltas`
  - `plot_madcap`
  - `plot_calibration`
  - `plot_estimated_probabilities`
  - `plot_admission_time_survival_curve`
- Legacy helpers:
  - `calculate_results`
  - `calc_mae_mpe`
- `get_model_key` contract

### Allowed to change

- New evaluation orchestration API (`run_evaluation`, input schema)
- `EvaluationTarget` shape
- scalars output schema (`scalars.json`)
- notebook `4d` content and usage pattern

---

## Phase 1 — Foundation types and adapters

### Deliverables

Create typed internal dataclasses in `evaluation/types.py`:

- `SnapshotResult(predicted_pmf, observed, offset=0)`
- `ClassifierInput`
- `ArrivalDeltaPayload`
- `SurvivalCurvePayload`
- `EvaluationInputs`
- simplified `EvaluationTarget`

Refactor `EvaluationTarget`:

- keep `name`, `flow_type`, `evaluation_mode`, `flow_selection`
- replace `components: Optional[List[str]]` with `component: str` (singular, explicit)
- derive aspirational status from `evaluation_mode == "aspirational_skip"`

Create adapters in `evaluation/adapters.py`:

- `from_legacy_prob_dist_dict(...)` (legacy → typed)
- `to_legacy_prob_dist_dict_all(...)` (typed → legacy)

### Rationale

This enables cleaner internal code without breaking existing aggregate/viz consumers.

---

## Phase 2 — Flat scalars collector (option b)

### Deliverables

Replace `_upsert_scalar_metadata` with `ScalarsCollector` in `evaluation/scalars.py`:

- collect one flat record per `(flow, service, component, prediction_time)`
- include common fields on all rows:
  - `flow`, `service`, `component`, `prediction_time`
  - `flow_type`, `aspirational`, `evaluated`, `reason`
- add mode-specific metrics as extra columns (`mae`, `mpe`, `auroc`, etc.)
- flatten reliability fields:
  - `reliable`, `reliability_threshold`, `reliability_basis`

`scalars.json` target shape:

```json
{
  "_meta": { "...": "..." },
  "results": [
    { "flow": "...", "service": "...", "component": "...", "prediction_time": "...", "...": "..." }
  ]
}
```

### Rationale

Immediate load to DataFrame:

```python
df = pd.DataFrame(scalars["results"])
```

No nested traversal or special-casing classifier vs service flows.

---

## Phase 3 — Per-mode handlers (remove god-function pattern)

### Deliverables

Create handler functions in `evaluation/handlers.py`:

- `evaluate_classifier(...)`
- `evaluate_distribution(...)`
- `evaluate_arrival_deltas(...)`
- `evaluate_survival_curve(...)`
- `skip_aspirational(...)`

Add dispatch map:

```python
_MODE_HANDLERS = {
    "classifier": evaluate_classifier,
    "distribution": evaluate_distribution,
    "arrival_deltas": evaluate_arrival_deltas,
    "survival_curve": evaluate_survival_curve,
    "aspirational_skip": skip_aspirational,
}
```

Distribution handler approach:

1. use typed `SnapshotResult` internally
2. adapt to legacy `prob_dist_dict_all` via `to_legacy_prob_dist_dict_all(...)`
3. call `plot_epudd`, `calc_mae_mpe`, `plot_deltas` unchanged
4. emit flat scalar rows

### Rationale

New modes can be added independently without touching existing branches.

---

## Phase 4 — Evaluation input builder

### Deliverables

Create `EvaluationInputsBuilder` in `evaluation/builder.py` with methods:

- `add_classifier(...)`
- `add_distributions_from_service_dict(...)` (ingest output of `get_prob_dist_by_service`)
- `add_arrival_deltas(...)`
- `add_survival_curve(...)`
- `build()`

Builder should:

- centralise validation
- convert legacy dicts into typed payloads internally
- reduce notebook boilerplate substantially

### Rationale

Notebook users should not manually reshape multiple nested dicts to match orchestrator internals.

---

## Phase 5 — New runner

### Deliverables

Implement clean orchestrator in `evaluation/runner.py`:

```python
run_evaluation(output_root, inputs, *, config_path=None, run_label=None, services=None)
```

Responsibilities:

1. initialise output folders
2. iterate targets and services
3. dispatch to mode handlers
4. write flat `scalars.json`
5. return concise run summary

Expose via `evaluation/__init__.py` during migration, then re-export from `evaluate`.

---

## Phase 6 — Classifier metrics canonicalisation

### Deliverables

Move compatibility logic out of orchestration:

- add `ensure_canonical_metrics()` to `TrainedClassifier` in `model_artifacts.py`
- collapse extraction logic in `evaluation/metrics.py` to read canonical top-level metrics

### Rationale

Evaluation should consume a normalised artifact, not perform historical reconstruction inline.

---

## Phase 7 — Rewrite notebook 4d

### Deliverables

Update notebook to use:

- `EvaluationInputsBuilder`
- new `run_evaluation(...)`
- flat scalars inspection via DataFrame

Expected effects:

- less manual dictionary plumbing
- clearer expression of evaluation intent
- easier extension to new flows

---

## Phase 8 — Tests

### Deliverables

Rewrite/add tests for:

- adapter round-trips (typed ↔ legacy)
- `ScalarsCollector` output
- each handler in isolation
- builder validation and assembly
- runner integration and output writing
- flat scalars DataFrame compatibility

Retain tests for legacy helpers (`calculate_results`, `calc_mae_mpe`) to protect backwards compatibility.

---

## Phase 9 — Documentation updates

### Deliverables

Update documentation and skill guidance to reflect:

- new evaluation API
- new flat `scalars.json` schema
- recommended analysis workflow (`pd.DataFrame(scalars["results"])`)

---

## Implementation order

```text
Phase 1 (types + adapters)
  ├─ Phase 2 (flat scalars)
  ├─ Phase 4 (builder)
  └─ Phase 6 (model artifact canonicalisation)
        ↓
Phase 3 (handlers)
        ↓
Phase 5 (runner integration)
        ↓
Phase 7 (notebook rewrite)
Phase 8 (tests)
Phase 9 (docs)
```

Parallelisable after Phase 1: phases 2, 4, and 6.

---

## Risks and mitigations

### Risk 1: accidental breakage of legacy viz consumers

- **Mitigation:** keep legacy shape untouched and isolate conversion in adapters
- add adapter contract tests with real-ish sample payloads

### Risk 2: schema migration pain for scalar consumers

- **Mitigation:** version `scalars.json` schema in `_meta.schema_version`
- provide short migration note (nested → flat)

### Risk 3: hidden assumptions in notebook 4d data prep

- **Mitigation:** encode assumptions in builder validation and explicit error messages

---

## Success criteria

1. Existing aggregate/viz notebooks relying on `prob_dist_dict_all` still run unchanged.
2. New evaluation pipeline has no god-function style branch nest.
3. `scalars.json` is flat and immediately DataFrame-loadable.
4. Notebook 4d is shorter and clearer, with less manual reshaping.
5. Adding a new mode (e.g. transfer-matrix evaluation) requires adding one handler + target config, not editing core orchestration logic.

