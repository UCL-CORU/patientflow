# Transfers: state of play

## What transfers represent

When a patient departs one subspecialty, they either leave the hospital (discharge) or move to another subspecialty (transfer). From the perspective of the *destination* service, a transfer is an inflow — another source of arrivals. From the perspective of the *source* service, it is part of departures (already covered by the departure predictions in rows 11/12).

## Prediction side — fully modelled

### TransferProbabilityEstimator (Row 13)

`src/patientflow/predictors/transfer_predictor.py` learns two things from historical inpatient movement data per source service and cohort (emergency/elective):

1. **P(transfer)** — the probability that a departure from this service is a transfer (vs a discharge)
2. **Destination distribution** — given a transfer occurs, the probability of going to each other service

Trained on a DataFrame with `current_subspecialty` (source), `next_subspecialty` (destination, NaN = discharge), and optionally `admission_type` (cohort).

### compute_transfer_arrivals (Rows 14/15)

`src/patientflow/predict/service.py` line 1231 computes transfer arrival PMFs for each target service by iterating over all source services:

- Takes the source's departure PMF (from the discharge classifier aggregation)
- Thins it by `P(transfer) × P(destination = target)` (compound probability)
- Convolves across all sources to get the total transfer arrival PMF for the target

This produces `elective_transfers` and `emergency_transfers` FlowInputs per service.

### FlowSelection

`FlowSelection.include_transfers_in` (default `True`) controls whether transfer arrivals are included in the prediction. When `True`, `DemandPredictor.predict_service` includes both `elective_transfers` and `emergency_transfers` as inflows, subject to cohort filtering.

Transfers are selectively includable:

- `FlowSelection.emergency_only()` — includes `emergency_transfers` (filtered by `cohort="emergency"`)
- `FlowSelection.elective_only()` — includes `elective_transfers` (filtered by `cohort="elective"`)
- `FlowSelection.custom(include_transfers_in=False)` — excludes both
- `FlowSelection.custom(include_transfers_in=True, cohort="all")` — includes both

## Evaluation side — not yet implemented

### No standalone evaluation targets for transfers

`src/patientflow/evaluate/targets.py` has no evaluation targets for rows 13, 14, or 15. Every currently defined target that has an explicit `FlowSelection` sets `include_transfers_in=False` (ed_current_beds, ed_yta_beds).

### Transfers appear only in combined targets

Transfers are included in the combined-flow evaluation targets, which use `FlowSelection.emergency_only()` or `FlowSelection.elective_only()` (both have `include_transfers_in=True`):

| Target | Evaluation plan row | Includes transfers | Observation mode |
|---|---|---|---|
| `combined_emergency_arrivals` | Row 16 | Yes (emergency) | aspirational_skip (no evaluation) |
| `combined_elective_arrivals` | Row 17 | Yes (elective) | arrived_in_window |
| `combined_net_emergency` | Row 18 | Yes (emergency) | aspirational_skip (no evaluation) |
| `combined_net_elective` | Row 19 | Yes (elective) | admitted_in_window |

So currently, emergency combined predictions that include transfers are aspirational (no observed-vs-predicted evaluation). Only the elective combined targets (rows 17, 19) are evaluated, and they include elective transfers in the prediction.

### Evaluation plan says "tbc" for Row 13

The transition matrix evaluation is listed as "Heatmap residuals (tbc)" — comparing training-period transition matrices against test-period transition matrices. No implementation exists yet. This is a matrix-level evaluation, not a per-service distribution evaluation — a different evaluation mode from anything currently in the framework.

### Evaluation plan says "—" for Rows 14/15

The plan does not define observations or evaluation for transfer arrivals as standalone flows. This makes sense: transfer arrivals are a derived quantity (a function of departure PMFs + the transition matrix), not a directly observed flow. You can't easily count "observed transfer arrivals to service X" without access to real-time inpatient movement data that tracks each patient's subspecialty changes within a visit.

## Observation counting implications

### For combined targets that include transfers

`combined_elective_arrivals` (Row 17) uses `arrived_in_window` and `combined_net_elective` (Row 19) uses `admitted_in_window`. For correct observed counts, these targets need:

- Elective admissions arriving within the window (new arrivals from outside the hospital)
- **Plus** transfers into the service from other services within the window

This means the observation dataset for combined targets needs to capture both external arrivals and internal transfers. A single inpatient arrivals DataFrame won't suffice unless it also includes intra-hospital transfer events.

### For standalone transfer evaluation (future)

If standalone evaluation of transfers is ever needed, it would require:

- **Row 13 (transition matrix):** A test-period transition matrix computed from inpatient movement data, compared against the training-period matrix. This is not a per-snapshot evaluation — it is a single comparison per cohort. Would need a new evaluation mode.
- **Rows 14/15 (transfer arrival counts):** Per-service, per-snapshot counts of patients who transferred in within the prediction window. Would need inpatient movement records with timestamps showing when each transfer occurred, filtered to transfers that happen within (prediction_moment, prediction_moment + window].

## What exists and what is missing

| Aspect | Status |
|---|---|
| Transfer probability model | Implemented (`TransferProbabilityEstimator`) |
| Transfer arrival PMF computation | Implemented (`compute_transfer_arrivals`) |
| FlowSelection toggle for transfers | Implemented (`include_transfers_in`) |
| Standalone evaluation targets (rows 13–15) | Not defined |
| Transition matrix comparison (row 13) | Not implemented; evaluation mode does not exist |
| Observation counting for transfer arrivals | Not implemented, but data is available (see below) |
AGG| Transfers in combined evaluated targets | Included in prediction (rows 17, 19); observation counting subject to the same issues as other window-based targets (see `observation_counting_requirement.md`) |

## Data available for transfer observation counting

The inpatient snapshots contain `current_subspecialty`, `next_subspecialty`, and a timestamp of when the patient moved to `next_subspecialty`. This is sufficient to count observed transfer arrivals for a target service within the prediction window:

- `next_subspecialty == target_service` (patient transferred to this service)
- Move timestamp falls within `(prediction_moment, prediction_moment + window]`
- `current_subspecialty != target_service` (genuine transfer, not a stay)

**Limitation:** we do not currently account for multiple transfers within the window. A patient who transfers A → B → C within the window would be counted as an arrival at B but not at C (from the snapshot at prediction time). This is consistent with the prediction model — `compute_transfer_arrivals` also models at most one transfer per patient per window (it thins the departure PMF by a single compound probability without chaining).
