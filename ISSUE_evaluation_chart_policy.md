# Evaluation chart policy: plot only when looking adds value

## Problem

`patientflow.evaluate` writes a PNG for almost every active service slice. With many services and several distribution / arrival flows, runs produce hundreds–thousands of charts that bloat disks and archives, while most add little beyond `scalars.json`.

Existing skips (`inactive_service`, `n_snapshots < 2`) are too weak. Conversely, dropping charts merely because observations are often zero would hide useful EPUDD signal when predicted mass is non-trivial.

## Policy (one-liner)

**Plot only when there is enough sample for a human to trust the picture, and either (a) there is non-trivial predicted or observed mass, or (b) a scalar flag asks for drill-down.**

Scalars always emit; charts are optional.

## Gates

| Gate | Rule | Skip / behaviour |
|------|------|------------------|
| **A — sample** | ≥ 30 snapshot leaves / histogram days for a panel (`CHART_GATE_A_MIN_OBSERVATIONS`; aligns with reliability bar). rPIT still computed from `MIN_DISTRIBUTION_SNAPSHOTS` (2). | No PNG; `insufficient_observations` |
| **B — mass** | Keep current inactive idea: obs≈0 **and** pred≈0 | `inactive_service` — no PNG |
| **C — worth inspecting** | Among A+B passers: plot if flagged, or (optional mode) always | `not_flagged` when in flagged mode |

**Do not** skip solely because observed counts are often zero while predictions are non-trivial.

### Flagged criteria (Gate C, distribution)

1. Prefer relative: any present `*_w2_reduction < 0` (model worse than that benchmark).
2. Else (no reduction fields): absolute fallback `rpit_cvm_mean_w2 >= 1.0`.

Arrival-delta figures are **not** flagged in v1 — they emit only under `charts="all"` (still Gates A/B).

## Chart modes

Expose on `run_evaluation`, e.g. `charts=`:

| Mode | Behaviour |
|------|-----------|
| `none` | Scalars only (`skip_reason: charts_disabled`) |
| `flagged` | **Default.** PNGs only for scalar-flagged slices |
| `all` | Every slice that passes A+B |

Classifier / survival charts: always on when mode ≠ `none` (few files, high value). Transition-matrix charts: stay deferred; if added later, same policy.

## Panelled charts

EPUDD and arrival-delta figures are **one file per service** with clocks as panels. Scalars stay per `service × clock`; the plot decision is at **figure** level.

| Layer | Rule |
|-------|------|
| **Gate B** | Inactive service → no figure |
| **Gate A** | Draw only panels (clocks) with enough sample; omit thin clocks |
| **Gate C (`flagged`)** | Write the figure if **any** Gate-A clock is flagged; otherwise skip |
| **Panels shown** | All Gate-A clocks for that service (context), not only the flagged panel |

Avoid one PNG per flagged clock (undoes panelisation). Avoid a lone flagged panel when sister clocks passed Gate A.

`charts_generated` on a clock row: `true` if that clock’s panel was drawn. Rows that did not justify or were not included use `insufficient_observations`, `not_flagged`, `charts_disabled`, or `inactive_service` as appropriate. Distribution / arrival rows also include `chart_flagged`.

## Related (separate issue)

Arrival-delta charts: include zero-arrival days; one histogram grid per service (clocks as panels).

## Acceptance

- [x] `ChartPolicy` / `charts= none|flagged|all` on `run_evaluation` (default `flagged`).
- [x] Handlers set `charts_generated` + `skip_reason` consistently; full scalars still written.
- [x] Panelled modes decide at figure level; include all Gate-A panels when the figure is written.
- [x] Zero-heavy / positive-prediction slices are not dropped by Gate B alone.
- [x] Recommended default for large runs is `flagged`.
- [x] Tests for skip reasons and mode behaviour.
