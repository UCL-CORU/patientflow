# Observation counting: current behaviour and required fix

## How the predictors work

`src/patientflow/predictors/incoming_admission_predictors.py` implements three predictor classes. Each models a different arrival-to-admission pathway, and this determines what the corresponding observation should count.

### DirectAdmissionPredictor (non-ED emergency, elective)

Assumes every arrival is admitted immediately — arrival time equals admission time. Produces a single Poisson distribution with rate equal to the sum of arrival rates across time intervals. No survival curve or admission probability adjustment.

**Observation needed:** patients who *arrive* (= are admitted) within the prediction window. There is no separate departure/admission timestamp — the arrival is the event.

### ParametricIncomingAdmissionPredictor (ED YTA, aspirational)

Uses an aspirational curve to determine, for each time interval, the probability that a patient arriving in that interval will be admitted within the remaining prediction window. Arrival rates are thinned by these probabilities: total admitted ~ Poisson(Σ λ_t θ_t).

**Observation needed:** patients who arrive at ED *after* the prediction moment and whose admission (ED departure) falls *within* the prediction window.

### EmpiricalIncomingAdmissionPredictor (ED YTA, empirical)

Same structure as parametric, but uses an empirical survival curve (fitted from training data) to calculate admission probabilities per time interval.

**Observation needed:** same as parametric — ED arrivals after the prediction moment whose admission falls within the window.

## Observation modes

Five observation modes are defined in `src/patientflow/evaluate/observations.py`. Each name describes who is being counted and when.

### `admitted_at_some_point`

Previously `count_at_some_point`.

Filters the visits DataFrame to rows matching `snapshot_date` and `prediction_time`, then counts rows where `label_col` (default `is_admitted`) is true. Optionally filtered by `service_col`.

Semantics: "of patients present in ED at prediction time, how many are admitted (irrespective of when)."

### `admitted_in_window`

Previously `count_in_window`.

Counts rows where:

- `start_time_col` **<=** prediction moment (patient was already present)
- `end_time_col` **>** prediction moment
- `end_time_col` **<=** prediction moment + prediction window

Semantics: "of patients already present, how many were admitted to a bed within the window."

### `departed_in_window`

Alias for the same function as `admitted_in_window` (identical counting logic). Used for departure targets to make the intent clear.

Semantics: "of patients already present, how many departed within the window."

### `arrived_in_window`

Counts rows where:

- `start_time_col` **>** prediction moment (patient arrives after the prediction)
- `start_time_col` **<=** prediction moment + prediction window

Only checks the arrival/start column. Does not check `end_time_col`.

Semantics: "how many patients arrived within the window (we don't care when they were admitted to a bed)." Used for direct admission targets where arrival = admission (non-ED emergency, elective).

### `arrived_and_admitted_in_window`

Previously `count_arrived_in_window`.

Counts rows where:

- `start_time_col` **>** prediction moment (patient arrives after the prediction)
- `end_time_col` **<=** prediction moment + prediction window

Semantics: "how many patients arrived after the prediction moment and were admitted to a bed within the window." Used for ED YTA targets where there is a delay between arrival at ED and admission to a bed (the empirical and parametric predictors model this delay).

## What each target needs

| Target | Predictor type | Observation dataset | Observation mode |
|---|---|---|---|
| **ed_current_beds** | Classifier | ED snapshot | `admitted_at_some_point` |
| **ed_current_window_beds** | Classifier + survival curve | ED visits | `admitted_in_window` |
| **ed_yta_beds** | Parametric or Empirical | ED arrivals | `arrived_and_admitted_in_window` |
| **non_ed_yta_beds** | Direct | Non-ED emergency inpatient arrivals | `arrived_in_window` |
| **elective_yta_beds** | Direct | Elective inpatient arrivals | `arrived_in_window` |
| **discharge_emergency** | Classifier | Emergency inpatients | `departed_in_window` |
| **discharge_elective** | Classifier | Elective inpatients | `departed_in_window` |

Note: `arrived_in_window` and `arrived_and_admitted_in_window` give the same result when `arrival_datetime == departure_datetime` in the data, which is the case for non-ED/elective records in `inpatient_arrivals`. The separate modes exist for conceptual clarity about what each target is testing.

## Additional observations

- `arrival_rates` and `survival_comparison` observation modes are handled by their own evaluation handlers (`evaluate_arrival_deltas`, `evaluate_survival_curve`) and do not use `count_observed(...)`.
- In `inpatient_arrivals`, `departure_datetime` is interpreted as time admitted to ward bed.

## The problem in `get_prob_dist_by_service`

`get_prob_dist_by_service` (line ~883 of `src/patientflow/aggregate.py`) always passes the same `ed_visits` DataFrame to `count_observed` regardless of which flow is being evaluated.

This is wrong in two ways for YTA targets:

1. **Wrong dataset** — ED visits instead of the appropriate arrivals data
2. **Wrong semantics** — `admitted_in_window` instead of `arrived_and_admitted_in_window` or `arrived_in_window`

## The legacy `get_prob_dist*` functions in `aggregate.py`

Three functions produce the `{agg_predicted, agg_observed}` evaluation dict format. Only one has the problem.

### `get_prob_dist` (line 379) — no issue

Observation data (`y_test`) is passed in directly by the caller and summed per snapshot. Self-contained; no coupling to any visits DataFrame.

### `get_prob_dist_using_survival_curve` (line 495) — no issue

Observations counted inline from caller-supplied `test_visits` using correct `arrived_and_admitted_in_window` semantics:

```python
mask = (test_visits[start_time_col] > prediction_moment) & (
    test_visits[end_time_col] <= prediction_moment + prediction_window
)
```

### `get_prob_dist_by_service` (line 691) — the problem

Uses `ed_visits` for both building ED snapshots (correct) and counting observations (wrong for non-ed_current targets).

## Required fix

### 1. Pull observation counting out of `get_prob_dist_by_service`

That function's responsibility is aggregating predictions from patient-level models into service-level PMFs. It should return predictions only. Observation counting should move to the evaluation layer, where the caller knows which target is being evaluated, which dataset to count from, and which observation mode applies.

### 2. Add `arrived_in_window` and `arrived_and_admitted_in_window` modes

`observations.py` needs functions for these two modes. `get_prob_dist_using_survival_curve` already implements the `arrived_and_admitted_in_window` logic inline (line 588) and can serve as a reference. `arrived_in_window` is simpler — only one timestamp is checked.

### 3. Wire observation counting into the evaluation layer

The builder or handler needs to accept per-target observation data and apply the correct counting at evaluation time. The adapter (`from_legacy_prob_dist_dict`) and `SnapshotResult` currently require `agg_observed` — they will need updating to accept predictions-only input.

### Back-compatibility

- **`get_prob_dist`**: No change needed.
- **`get_prob_dist_using_survival_curve`**: No change needed.
- **`get_prob_dist_by_service`**: Return format changes (predictions only). Callers (notebook 4d, remote environment) updated to supply observations through the builder.
- **Notebook 4d**: Will need updating to supply observation data through the builder instead of relying on `get_prob_dist_by_service`.
