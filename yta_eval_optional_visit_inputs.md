# Proposal: make `ed_visits` / `inpatient_visits` optional in `get_prob_dist_by_service`

**Status:** proposal
**Affects:** `patientflow.aggregate.get_prob_dist_by_service`, `patientflow.predict.service.build_service_data`, `patientflow.predictors.incoming_admission_predictors.IncomingAdmissionPredictor.predict`
**Motivating consumer:** [`uclhflow/evaluate_module_plan.md`](https://github.com/UCL-CORU/uclhflow/blob/main/evaluate_module_plan.md)

## Problem

`get_prob_dist_by_service` already threads `prediction_date=dt` into `build_service_data` for each snapshot date, which is what weekday-stratified YTA predictors (`stratify_by_weekday=True`) need to pick the right rate profile.

It currently requires `ed_visits` and `inpatient_visits` even for `FlowSelection` choices that do not use them. Downstream evaluation pipelines that want to score a single YTA flow (e.g. `non_ed_yta` or `elective_yta`) therefore cannot use `get_prob_dist_by_service` and hand-roll the per-`prediction_time` loop. Hand-rolled loops tend to drop `prediction_date`, at which point weekday-stratified models silently fall back to pooled rates and emit a `UserWarning`.

## Proposed change

1. **Make `ed_visits` optional in `get_prob_dist_by_service` / `build_service_data`.** Required only when `flow_selection` includes `ed_current`. `ed_yta` does not need raw ED visits at predict time — the fitted model already encodes arrival rates. Raise a clear error when a selected flow needs visits that were not supplied:

   ```text
   ValueError: ed_visits is required when flow_selection.include_ed_current is True
   ```

2. **Make `inpatient_visits` optional symmetrically.** Required only when departures or transfers are included.

3. **Document the weekday contract on `IncomingAdmissionPredictor.predict`.** Add to the docstring:

   > When the model was fit with `stratify_by_weekday=True`, callers must pass `prediction_date` (or a `prediction_context` entry containing it) for each call. Calling without it falls back to pooled rates and emits `UserWarning`. For per-snapshot evaluation, prefer `get_prob_dist_by_service`, which threads `prediction_date` for you.

4. **Add `strict_prediction_date` to `build_service_data` / `get_prob_dist_by_service`.** Forwarded to the YTA predictors. When `True`, a weekday-stratified model missing weekday weights for the requested date raises instead of warning. This turns silent regressions in consumer code into loud failures.

5. **Thread `prediction_date` to `non_ed_yta` and `elective_yta` `predict_mean` calls inside `build_service_data`.** These paths currently omit `prediction_date`, so weekday-stratified Direct predictors silently use pooled rates even when the caller supplied a date. The fix mirrors the existing `ed_yta` path.

## Effect on consumers

With (1) and (2) in place, hand-rolled YTA blocks collapse to one call each:

```python
ed_yta_prob_dist_by_service[model_key] = get_prob_dist_by_service(
    snapshot_dates=test_snapshot_dates,
    prediction_time=pt,
    models=(None, None, bundle.spec_model, bundle.ed_yta_model, None, None, None),
    specialties=subspecialties,
    prediction_window=pred_window,
    x1=x1, y1=y1, x2=x2, y2=y2,
    flow_selection=FlowSelection.custom(include_ed_yta=True),
    strict_prediction_date=True,
)
```

`prediction_date` threading happens inside `build_service_data`. Same shape for `non_ed_yta` and `elective_yta`.

## Independent follow-up for evaluation diagnostics

Consumers wiring `EvaluationInputsBuilder.add_arrival_deltas(...)` should also pass `predictors_by_service` and `filter_keys_by_service` so the diagnostic baseline matches what the YTA model actually used. Otherwise the obs-vs-expected plot compares against pooled rates while the PMFs were built from weekday rates:

```python
builder.add_arrival_deltas(
    "ed_yta_arrival_rates",
    arrivals_by_service=obs_ed_yta_by_service,
    snapshot_dates=test_snapshot_dates,
    prediction_window=prediction_window,
    yta_time_interval=yta_time_interval,
    predictors_by_service={s: bundle.ed_yta_model for s in subspecialties},
    filter_keys_by_service={s: s for s in subspecialties},
    strict_prediction_date=True,
)
```

This is consumer-side and does not depend on the API change; both should land together to avoid silent baseline mismatch.

## Risks

- **Backwards compatibility**: existing callers that pass `ed_visits` / `inpatient_visits` continue to work unchanged. Only new code paths gain the ability to omit them.
- **Validation must be explicit**: silently treating a missing `ed_visits` as "no ED data" would be worse than the status quo. Error messages must name both the missing argument and the offending `flow_selection` flag.
