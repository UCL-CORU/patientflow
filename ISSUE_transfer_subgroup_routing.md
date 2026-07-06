# Feature: Subgroup-aware transfer routing in demand prediction

## Summary

Wire `TransferProbabilityEstimator` subgroup tables (`transfer_probabilities[cohort]["subgroups"]`) into `compute_transfer_arrivals` in a new `patientflow.predict.transfers` module (called from `service.py`), mirroring the **ED admissions** pattern: per-patient routing probabilities, then **one** aggregated PMF per flow via `pred_proba_to_agg_predicted(..., weights=...)`, without building separate PMFs per subgroup up front.

Production always uses subgroup routing. The cohort-pooled `["services"]` row remains fitted for diagnostics (`get_transition_matrix`) but is not the live prediction path.

## Motivation

### Fitted but unused

`TransferProbabilityEstimator.fit` trains:

- `["services"]` — routing pooled over all patients in the cohort
- `["subgroups"][g]` — routing among patients matching age/sex buckets (`paediatric`, `adult_male_young`, …)

The API supports `get_transfer_prob(source, cohort, subgroup=g)` and `get_destination_distribution(..., subgroup=g)`. `compute_transfer_arrivals` currently calls these with `subgroup=None`, reads the pooled row, and applies a **single** `compound_prob` to thin each source’s departure PMF. That can assign implausible destinations (e.g. male inpatients inheriting gynaecology mass from a pooled row).

### Clinical plausibility

Subgroup tables are learned from **male-only** or **female-only** movement histories (among others). Cohort `["services"]` rows mix them. A male departing cardiology should not use the same destination mix as the specialty-wide average if that average includes female-heavy transfers to gynaecology.

This is analogous to admissions, where `MultiSubgroupPredictor` returns a **per-patient** `specialty_prob` dict and `_process_ed_patients_for_specialty` applies **patient-level weights** when aggregating to one PMF per target specialty.

### Consistency with evaluation

Transition-matrix evaluation (separate spec — attach to this issue on GitHub) scores subgroup routing under the assumption that production uses the same per-patient resolution: per source, it builds patient-level expected destination counts `E_d = Σ_i p_i(d)` from each departing patient's subgroup row and runs a Monte Carlo Pearson X² test against observed counts. Evaluation and production must share the same rules for resolving (or excluding) patients who do not match a subgroup.

## Current behaviour (transfers)

```text
Per source S, cohort c:
  departure_pmf_S  ← pred_proba_to_agg_predicted(all inpatients at S, no subgroup split)
  compound_prob    ← get_transfer_prob(S, c) × get_destination_distribution(S, c)[T]  # subgroup=None
  contribution to T ← Distribution.thin(departure_pmf_S, compound_prob)
```

Documented assumption in `compute_transfer_arrivals`: transfer probabilities are **constant across patients** at a source.

## Reference pattern (ED admissions)

```text
Per patient i:
  specialty_prob[i]  ← MultiSubgroupPredictor.predict_dataframe(row)  # subgroup from row

Per target specialty spec:
  weight_i = P(admit to spec | i) × P(in window | i)
  agg_predicted_in_ed[spec] ← pred_proba_to_agg_predicted(P(admit | ED), weights=weight_i)
```

One PMF per specialty; heterogeneity lives in **weights**, not in five subgroup PMFs per specialty.

## Target behaviour (transfers)

### Per-patient routing

For each inpatient row `i` at source subspecialty `S` with cohort `c`:

1. Resolve subgroup `g(i)` with the same `subgroup_functions` as `TransferProbabilityEstimator` (shared `resolve_patient_subgroup` helper; reuse `transfer_model.subgroup_functions` or `create_subgroup_functions()`).
2. If no subgroup resolves, exclude the row from transfer routing (see **Subgroup resolution** below).
3. Patient-level probabilities:
   - `q_transfer_i = get_transfer_prob(S, c, subgroup=g(i))`
   - `q_dest_i(T) = get_destination_distribution(S, c, subgroup=g(i)).get(T, 0.0)`
   - `q_i(T) = q_transfer_i × q_dest_i(T)` (probability this departure, if it happens, goes to target `T`)

Combine with existing per-patient departure probability `p_depart_i` (from `_prepare_base_probabilities`).

### Aggregation per (target, source, cohort) — weighted, one PMF

For each target `T`, source `S`, and admission type / cohort `c`, among rows at `S` with that admission type:

```text
weight_i(T) = p_depart_i × q_i(T)    # 0 when row excluded from routing

transfer_count_pmf_{S→T} = pred_proba_to_agg_predicted(
    predictions_proba = p_depart_i as pred_proba column,
    weights = weight_i(T),
)
```

Then accumulate arrivals at `T` by convolving contributions across sources (same outer structure as today’s loop over `source_service`, but each contribution is a **patient-weighted** PMF instead of `thin(departure_pmf_S, scalar)`).

**No separate PMF per subgroup** is required up front; subgroup effects enter only through `weight_i(T)` via `g(i)`.

## Subgroup resolution

Shared rules for transfers and admissions (align `predict_dataframe` with the same helper where practical).

### Standard masks

Five mutually exclusive subgroups from `create_subgroup_functions()`:

- `paediatric` — `age < 18` (sex not used)
- `adult_male_young`, `adult_female_young`, `adult_male_senior`, `adult_female_senior` — require `sex == "M"` or `sex == "F"`

**Overlap:** raise `ValueError` if more than one mask matches a row (same contract as `MultiSubgroupPredictor.predict_dataframe`).

### Unmatched rows (missing or invalid sex on adults)

Prediction snapshots are **not** required to guarantee `sex ∈ {M, F}` for adults.

When a row matches **no** subgroup (typically an adult with missing, null, or non-`M`/`F` sex):

| Flow                 | Behaviour                                                                    |
| -------------------- | ---------------------------------------------------------------------------- |
| Transfer routing     | `weight_i(T) = 0` for all targets — row contributes **no** transfer arrivals |
| ED specialty routing | Row excluded from subgroup specialty masks (no `specialty_prob` for routing) |
| Inpatient departures | Unchanged — `p_depart_i` still applies via the departure classifier          |

Emit **one summary warning per predict call** (not per row), e.g. counts of excluded ED and inpatient rows. Optionally note breakdown by `sex` value where helpful.

**Do not** fall back to the pooled `["services"]` row for unmatched rows at prediction time. At fit time, those patients contribute only to `["services"]`, not to subgroup tables; excluding them at predict time is the conservative choice and avoids reintroducing sex-mixed destination mass.

**Paediatric** patients are unaffected by missing sex.

### ED specialty routing: empty consult vs missing sex

These are **orthogonal**. Subgroup resolution (age/sex bucket) happens before sequence lookup.

**Matched subgroup, empty consult** (`consultation_sequence` → `()`): **include** in ED specialty routing. `MultiSubgroupPredictor` routes the row to that subgroup’s `SequenceToOutcomePredictor`, which returns a per-subgroup marginal specialty distribution via `predict(())` — the `weights[tuple()]` entry fitted on that subgroup’s training data (prefix-aggregated mix over observed consultation paths). Male and female buckets can differ (e.g. gynae mass). This is existing admissions behaviour and is unchanged.

**Unmatched adult** (missing or invalid `sex`): **exclude** from ED specialty routing **regardless of consult sequence**. Do not assign any subgroup’s `predict(())` marginal, and do not pool across subgroups. Choosing one subgroup’s marginal without a resolved bucket would be arbitrary; pooling would reintroduce sex-mixed specialty mass.

At implementation time, `predict_dataframe` already leaves unmatched adults as `NaN` `specialty_prob`; the shared `resolve_patient_subgroup` helper should preserve that contract explicitly.

### Other edge cases

| Case                                                                     | Behaviour                                                                   |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Resolved `g(i)` not in `["subgroups"]` for cohort (empty training slice) | Exclude from transfer routing (`weight_i(T) = 0`); warn in the same summary |
| Subgroup row has no `source_service`                                     | Same as today: `get_transfer_prob` warns and returns `0.0`                  |
| Sparse subgroup at fit time                                              | No minimum-count threshold at prediction; use the fitted subgroup row as-is |

## What this does and does not guarantee

**Does:**

- Route each resolvable patient using the table trained on their age/sex bucket
- Avoid pooling male and female destination mixes at prediction time for patients with a resolved subgroup
- Reuse existing estimator fits and `pred_proba_to_agg_predicted` machinery
- Apply consistent exclude-and-warn behaviour for unmatched adults across transfer and ED specialty routing

**Does not:**

- Impose hard clinical rules (e.g. “male ⇒ never gynae”) — still empirical; bad training rows can still assign small mass
- Split beyond the five standard subgroups (not individual patient covariates beyond age/sex)
- Guarantee transfer routing for adults without valid sex — those rows are excluded
- Fix column-level / target inflow calibration without a separate design (covered by the transition-matrix evaluation spec)

**Note:** A patient excluded from transfer routing can still contribute to **departure** PMFs at their source. Document this in `compute_transfer_arrivals`.

## Implementation sketch

### 1. Shared subgroup resolution

Add `resolve_patient_subgroup` (name TBD) in `patientflow.predictors.subgroup_definitions` or a small shared module:

- Input: row, `subgroup_functions`
- Output: subgroup name, or `None` if no mask matches
- Raise on overlapping masks
- Used by transfer weight logic and, where practical, `MultiSubgroupPredictor.predict_dataframe` / `_prepare_base_probabilities`

### 2. New `patientflow.predict.transfers` module

Add `src/patientflow/predict/transfers.py` (same pattern as `distribution.py` and `flow_selection_checks.py`). `service.py` is already large (~1,500 lines); transfer routing logic does not belong inline there.

**Move** `compute_transfer_arrivals` from `service.py` into `transfers.py`. Keep a thin re-export in `service.py` so existing imports (`from patientflow.predict.service import compute_transfer_arrivals`) remain valid.

**New helpers** in `transfers.py`:

```python
def transfer_weight_to_target(
    row: pd.Series,
    source_service: str,
    target_service: str,
    cohort: str,
    transfer_model: TransferProbabilityEstimator,
) -> float:
    g = resolve_patient_subgroup(row, transfer_model.subgroup_functions)
    if g is None or g not in transfer_model.transfer_probabilities[cohort]["subgroups"]:
        return 0.0
    q_transfer = transfer_model.get_transfer_prob(source_service, cohort, subgroup=g)
    dest = transfer_model.get_destination_distribution(source_service, cohort, subgroup=g)
    return q_transfer * dest.get(target_service, 0.0)
```

Precompute subgroup masks on the full `inpatient_snapshots` frame once per call. Vectorize weights per `(source, cohort, target)` where practical. Emit the excluded-row summary warning from this module.

### 3. Replace scalar thinning in `compute_transfer_arrivals`

Implement in `transfers.py`:

- **Inputs:** `inpatient_snapshots`, `prob_departure_after_elective`, `prob_departure_after_emergency` (passed from `build_service_data` via `_finalise_service_data` — same `p_depart_i` as `_process_inpatients_for_specialty_by_admission_type`).
- **Inner loop:** for each `(target, source, cohort)`, build weighted PMF via `pred_proba_to_agg_predicted`, convolve into `arrival_dist`. Skip early when all weights are zero.
- **Docstring:** document per-patient routing, exclusion of unmatched rows, and the departure-vs-transfer asymmetry.

No optional flag: subgroup routing is the only path.

### 4. Wire through `build_service_data` / `_finalise_service_data`

`service.py` remains orchestration only. Thread `inpatient_snapshots` and departure probability series from `_prepare_base_probabilities` into `_finalise_service_data` → `transfers.compute_transfer_arrivals(...)`. Do not recompute departures inside `compute_transfer_arrivals`.

### 5. Tests

- Synthetic cohort: male and female rows at same source; pooled `["services"]` assigns mass to “gynae”; subgroup routing assigns ~0 for males, >0 only for females.
- `compute_transfer_arrivals` subgroup path ≠ former scalar-thin path on fixture.
- Adult with missing sex: zero transfer weight, warning emitted, departure PMF unchanged.
- Overlap in subgroup masks: raises.
- Regression: existing transfer predictor unit tests unchanged; primary coverage in new `tests/test_transfer_arrivals.py` (import from `patientflow.predict.transfers`); keep smoke tests in `tests/test_service.py` if needed.

### 6. Documentation

- `TransferProbabilityEstimator` docstring: `["subgroups"]` is the production prediction path via `compute_transfer_arrivals`; `["services"]` is marginal / diagnostic.
- `get_transition_matrix(cohort)` remains a pooled view of the `["services"]` row; document that live prediction uses patient-level subgroup tables.

### 7. Performance

Precompute subgroup masks once per snapshot frame. Nested loops over admission type × target × source are acceptable for v1; profile if a full ~511-service run is too slow. Most `(source, target)` pairs exit early when routing mass is zero.

## Decisions

| #   | Topic                   | Decision                                                                                                                                          |
| --- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Primary mechanism       | Patient-level `weight_i(T) = p_depart_i × q_i(T)` + `pred_proba_to_agg_predicted`, not five subgroup PMFs per source                              |
| D2  | Subgroup resolution     | Shared `resolve_patient_subgroup` using `transfer_model.subgroup_functions`; raise on overlapping masks                                           |
| D3  | Unmatched adults        | Exclude from transfer and ED specialty routing (`weight_i(T) = 0`); one summary warning per predict call; no pooled `["services"]` fallback       |
| D4  | Cohort                  | Keep `elective` / `emergency` split as today (`admission_type` in loop)                                                                           |
| D5  | API surface             | Subgroup routing only — no `use_subgroup_routing` flag                                                                                            |
| D6  | Departure probabilities | Pass `inpatient_snapshots` + `prob_departure_after_*` from `build_service_data`; single source of truth with inpatient outflows                   |
| D7  | Sparse training         | No minimum-count threshold at prediction                                                                                                          |
| D8  | Performance             | Vectorize masks once; nested loops OK for v1                                                                                                      |
| D9  | Evaluation              | Out of scope here; align transition-matrix evaluation (attached spec) to the same exclude rules for unmatched rows                                |
| D10 | Module layout           | New `patientflow.predict.transfers`; move `compute_transfer_arrivals` out of `service.py`; re-export from `service.py` for backward compatibility |

## Out of scope

- New subgroup definitions beyond age/sex buckets
- Hard-coded specialty eligibility maps (e.g. explicit gynae exclusion)
- Column / target-aggregated transfer evaluation
- Retraining `TransferProbabilityEstimator`
- uclhflow notebook wiring (separate issue)
- Changing how departure **classifiers** work (only how their outputs combine with routing)
- Old-vs-new transfer PMF comparison on production fixtures

## Acceptance criteria

- [ ] `compute_transfer_arrivals` uses `get_transfer_prob(..., subgroup=g(i))` and `get_destination_distribution(..., subgroup=g(i))` for patients with a resolved subgroup.
- [ ] Aggregation uses `pred_proba_to_agg_predicted` with per-patient weights (no scalar `thin(departure_pmf, compound_prob)` on pooled departures).
- [ ] Test demonstrates pooled `["services"]` routing can send mass to a destination for all patients at a source while subgroup routing does not for male patients on a female-skewed destination.
- [ ] Adult with missing/invalid sex: excluded from transfer routing, summary warning emitted, departure PMF unchanged.
- [ ] Overlapping subgroup masks raise `ValueError`.
- [ ] `uv run pytest` green.

## Related

- [`src/patientflow/predict/transfers.py`](src/patientflow/predict/transfers.py) — `compute_transfer_arrivals`, transfer weight helpers (new)
- [`src/patientflow/predict/service.py`](src/patientflow/predict/service.py) — `build_service_data`, `_finalise_service_data`, `_process_ed_patients_for_specialty`, `_process_inpatients_for_specialty_by_admission_type`
- [`src/patientflow/predictors/transfer_predictor.py`](src/patientflow/predictors/transfer_predictor.py) — `["services"]` vs `["subgroups"]`
- [`src/patientflow/predictors/subgroup_predictor.py`](src/patientflow/predictors/subgroup_predictor.py) — admissions reference pattern
- [`src/patientflow/predictors/subgroup_definitions.py`](src/patientflow/predictors/subgroup_definitions.py) — subgroup masks
- [`src/patientflow/aggregate.py`](src/patientflow/aggregate.py) — `pred_proba_to_agg_predicted(..., weights=...)`

**Attachment:** transition-matrix evaluation spec — patient-level Pearson X² calibration of subgroup routing rows.
