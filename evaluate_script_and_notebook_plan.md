# evaluate.py and notebook 4d plan (current)

## Purpose

`evaluate.py` is an orchestration layer that writes evaluation outputs (plots + scalar metadata)
for configured targets. Notebook `4d` demonstrates the full evaluation workflow: building
service-level probability distributions, preparing canonical inputs, calling `run_evaluation(...)`,
and reviewing the outputs.

## Agreed design decisions

- **Single canonical non-classifier input shape:** `flow_inputs_by_service` only.
- **No auto-building in `run_evaluation(...)`:** callers pass explicit prepared inputs.
- **Classifier scalar metrics are read-only in evaluation:** read from model artifacts, do not
  recompute from raw visits inside `evaluate.py`.
- **One global scalar file per run:** `scalars.json` at run root, including skipped entries.
- **Aspirational targets:** record metadata and skip invalid observed-vs-predicted diagnostics.
- **Prediction time convention:** tuples `(hour, minute)` throughout.
- **Multiple runs:** `run_evaluation(...)` writes to `output_root/<run_label_or_timestamp>/`.

## Current `run_evaluation(...)` contract

```python
run_evaluation(
    output_root,
    prediction_times,
    config_path=None,
    run_label=None,
    evaluation_targets=None,
    classifier_inputs=None,
    flow_inputs_by_service=None,
    services=None,
)
```

### Inputs

- `prediction_times`: required list of `(hour, minute)` tuples.
- `classifier_inputs`: `{target_name: {"trained_models": ..., "visits_df": ..., "label_col": ...}}`
  (`visits_df` is used for classifier plots; classifier scalars come from artifacts).
- `flow_inputs_by_service`: canonical map
  `service -> flow_name -> prediction_time -> payload`.
- `services`: optional explicit service list; otherwise inferred from `flow_inputs_by_service`.

### Explicitly out of scope in `run_evaluation(...)`

- Building flow inputs from model outputs internally.
- Alternate non-canonical input surfaces (for example `flow_inputs_by_target`).

## Evaluation target contract

`EvaluationTarget` defines what/how to evaluate:

- `name`
- `flow_type` (`classifier`, `pmf`, `poisson`, `special`)
- `evaluation_mode` (`classifier`, `distribution`, `arrival_deltas`, `survival_curve`, `aspirational_skip`)
- `aspirational` (boolean)
- optional `components`
- optional `flow_selection`

Graceful failure rule: missing input/model for a target/component/time records
`evaluated: false` with a clear `reason` in `scalars.json`; run continues.

## Classifier metrics source (read-only)

For each classifier model/timepoint, evaluation reads metrics in this order:

1. `TrainedClassifier.selected_eval_metrics` (canonical)
2. `TrainingResults.selected_eval_metrics` (legacy compatibility)
3. `TrainingResults.test_results` (legacy test metrics)
4. best CV validation metrics from `TrainingResults.training_info["cv_trials"]` (legacy)

If required metrics are still unavailable, mark that timepoint as skipped in `scalars.json`.

## Model artifact notes

- `selected_eval_metrics` now sits on **`TrainedClassifier`** (top-level), not nested in
  `TrainingResults`.
- Dataset metadata stores explicit positive-case counts:
  `train_valid_test_positive_cases`.

## Output structure

```text
eval-output/
└── <run_label_or_timestamp>/
    ├── config.yaml                  # copied when provided
    ├── scalars.json                 # one global scalars file
    ├── classifiers/
    │   └── <classifier_name>/
    │       ├── madcap.png
    │       ├── discrimination.png
    │       └── calibration.png
    └── services/
        └── <service_name_sanitized>/
            ├── <flow>_<component>_<time>_epudd.png
            ├── <flow>_<component>_<time>_obs_exp.png
            ├── <flow>_<component>_<time>_deltas.png
            └── <flow>_<component>_survival.png
```

Service directory names are sanitized for filesystem safety (for example `haem/onc`).

## Arrival-delta granularity (agreed)

`ed_yta_arrival_rates` should be evaluated at the same cohort granularity used by YTA model
training filters (service/specialty filter keys), not one pooled arrivals dataframe duplicated
under each service.

`arrival_deltas` payload per service/time must include:

- `df` (service-filtered inpatient arrivals dataframe)
- `snapshot_dates`
- `prediction_window`
- optional `yta_time_interval`

## Notebook 4d section structure

1. **Build evaluation dictionaries and visualise** — call `get_prob_dist_by_service` with an
   explicit `FlowSelection` for each prediction time, then show an EPUDD plot for one service.
2. **Prepare evaluation inputs** — reshape outputs into canonical `flow_inputs_by_service`
   (`service -> flow -> prediction_time -> payload`), map `ed_current_beds` from the
   `get_prob_dist_by_service` output, build service-specific YTA arrivals inputs using
   `yta_model_by_spec.filters`, and assemble `classifier_inputs`.
3. **Run evaluation** — call `run_evaluation(...)` once with the explicit inputs. The notebook
   exercises the ED-relevant target subset; the full default registry is documented but not run.
4. **Review outputs** — inspect the timestamped run directory and sample `scalars.json` entries.

No section should rely on auto-input builders or alternate non-canonical input APIs.

## Backward compatibility

- Keep `calc_mae_mpe` and `calculate_results` callable for existing notebook usage.
- Keep metric-reading fallbacks for previously saved classifier artifacts.

## Deferred items

- Explicit runtime switch for evaluating against validation vs test sets in orchestration.
  (Current classifier scalars depend on what was persisted during training.)
