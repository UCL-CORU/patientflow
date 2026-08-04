---
name: evaluation-review
description: >-
  Review the output of a patientflow/uclhflow evaluation run (a run directory
  containing scalars.json, evaluation_run.yaml, and optional chart PNGs written
  by patientflow.evaluate.runner.run_evaluation). Use when the user asks to
  review, triage, summarise, or interpret an evaluation run, an eval-output
  folder, scalars.json, rPIT/CvM results, or flagged charts.
---

# Reviewing an Evaluation Run

An evaluation run directory (under `notebooks/eval-output/` in patientflow, or
`eval-output/{valid|test}/{run_name}/` in uclhflow) looks like:

```
{run_dir}/
  evaluation_run.yaml        # run settings, targets, prediction_dict, charts mode
  scalars.json               # ALWAYS written - the primary review artefact
  classifiers/{flow}/        # feature importance, SHAP, discrimination, MADCAP, calibration PNGs
  distributions/{flow}/{service}/{component}.png   # EPUDD calibration panels
  arrivals/{flow}/{service}/{component}.png        # arrival-delta histograms
  survival/                  # train-vs-test admission-time survival (if present)
```

Scalars always emit; PNGs are gated by the chart policy (see below). **Base the
review on `scalars.json`; open charts only to drill into what the scalars
flag.**

## Review workflow

1. **Read `evaluation_run.yaml` first.** Note the `charts` mode
   (`none`/`flagged`/`all`), the eval split, and which targets ran. The charts
   mode changes how to interpret missing PNGs (see chart policy section).
   Older runs may lack the `charts` field; if arrival PNGs exist, treat the
   run as `all`.
2. **Load `scalars.json`** and group `evaluation_rows` by `evaluation_mode`.
   Prefer a short Python/jq pass over reading the raw file when there are many
   rows (uclhflow runs cover hundreds of subspecialties).
3. **Triage rows** using the per-mode guidance below.
4. **Check coverage** via `_service_summary.by_slice`: `n_inactive_services`
   and `inactive_service_names` show which services had nothing to evaluate.
5. **Open flagged charts only**, then write up findings.

## scalars.json structure

```json
{
  "evaluation_rows": [ { ...one dict per slice... } ],
  "_service_summary": { "by_slice": { "{mode}/{flow}/{component}": {...} } }
}
```

Every row carries identity fields: `evaluation_mode`, `flow`, `flow_type`,
`observation_mode`, `service`, `component`, `prediction_time` (`[hour, minute]`
or `null`), `model_name`. Row grain =
`(evaluation_mode, flow, service, component, prediction_time, model_name)`.
`service` is `"_all_"` for classifier and survival rows. Bookkeeping fields:
`charts_generated`, optional `skip_reason`, `reliable`, and (newer runs)
`chart_flagged`.

Do **not** look for MAE/MPE - the modern package uses rPIT/CvM for
distributions and Pearson X² for transition matrices.

## Triage by evaluation mode

### `classifier_model_diagnostics` (one row per clocked model)

- Metrics: `auroc`, `auprc`, `log_loss`, `n_samples`, `n_positive_cases`.
- `reliable` = `n_positive_cases >= 30` on the split named in `metrics_split`.
- There is no hard pass/fail cutoff. Compare clocks against each other and
  against previous runs; call out any clock notably worse than its siblings
  (e.g. AUROC ~0.05 below the rest) and any `reliable: false` row.
- Caveat: these metrics come from train-time scoring (`metrics_split`:
  typically `cv_train` or `test`), which may differ from the run's eval split.

### `classifier_probability_quality`

Chart bookkeeping only (no metrics on the row). If `charts_generated`, review
the discrimination/MADCAP/calibration PNGs under `classifiers/{flow}/`.

### `distribution` (one row per service x prediction time)

The headline metric is the randomised-PIT Cramér-von Mises statistic:

- `rpit_cvm_mean_w2` - model calibration (lower is better; under perfect
  calibration the CvM statistic averages ~0.17, so values near that are good
  and values >= 1.0 warrant a look).
- `rpit_cvm_benchmark_*` - a Binomial(n, p̄) baseline;
  `rpit_cvm_specialty_proportions_*` - a training-mix baseline (when present).
- **`*_w2_reduction` fields are the key triage signal**: benchmark W² minus
  model W². Positive = model beats that baseline; **negative = model is worse
  than the baseline - always report these**.
- `reliable` = `n_snapshots >= 30`. Note unreliable rows but don't over-read
  their W² values.
- Benchmarks only exist where a benchmark cohort was registered (in uclhflow:
  ED-current and departures flows, not YTA flows). Without reductions, fall
  back to absolute `rpit_cvm_mean_w2 >= 1.0` as the concern threshold.

Rank distribution problems by: negative reduction first (most negative =
worst), then high absolute W² among reliable rows.

### `arrival_deltas`

Rows carry only identity + `reliable` + `charts_generated` - **there are no
quality scalars for arrivals**. Assessment requires the PNGs (histograms of
observed-minus-expected arrivals per clock). If the charts are absent (normal
under `flagged` mode), state that arrivals were not visually reviewed and
suggest rerunning with `--charts all` if needed.

### `transition_matrix`

- Metrics: `pearson_x2`, `p_value` (Monte Carlo goodness of fit),
  `n_departures`, plus `destinations` / `expected_counts` / `observed_counts`.
- `reliable` = `n_departures >= 30`. Low `p_value` (e.g. < 0.05) with a
  reliable sample suggests miscalibrated routing - report the row and compare
  expected vs observed counts. Expect `skip_reason` values like
  `no_observed_departures` for thin rows.

## Chart policy: interpreting `all` vs `flagged`

The `charts` mode controls PNG emission, gated by:

- **Gate A** (sample): >= 30 observations to draw a panel.
- **Gate B** (inactive): nothing to plot -> `skip_reason: inactive_service`.
- **Gate C** (flagged, distributions only): any `*_w2_reduction < 0`, else
  `rpit_cvm_mean_w2 >= 1.0`.

| Mode                | What a missing/present PNG means                                                                                                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `none`              | No PNGs ever (`skip_reason: charts_disabled`). Review scalars only.                                                                                                                                                                                                                               |
| `flagged` (default) | A **present** distribution PNG means Gate C fired - treat every PNG in the run as a problem to inspect and explain. A **missing** PNG with `skip_reason: not_flagged` is good news, not missing data. Arrival-delta PNGs are never written in this mode; do not report them as missing artefacts. |
| `all`               | Every Gate-A slice is charted, so `charts_generated: true` carries **no signal**. Use the `chart_flagged` field (or recompute Gate C from the reductions) to prioritise which PNGs to open; do not attempt to review every chart in a large run.                                                  |

In all modes, `skip_reason` distinguishes `charts_disabled`,
`insufficient_observations` (Gate A fail), `not_flagged` (Gate C pass), and
`inactive_service`. Panelled figures (EPUDD, arrivals) are one file per
service with clocks as panels; if any clock flags, all Gate-A clocks are drawn,
so an in-figure panel is not itself evidence that that clock flagged.

## uclhflow specifics

- Runs are produced by `python -m predictor.evaluate` (see uclhflow README);
  flows are named `uclh_*`: ED admissions + discharges classifiers,
  distribution flows (`uclh_ed_current_beds`, `uclh_non_ed_yta_beds`,
  `uclh_elective_yta_beds`, `uclh_departures_elective_beds`,
  `uclh_departures_emergency_beds`), transition matrices
  (elective/emergency), and `uclh_ed_yta_arrival_rates` arrival deltas.
- Services are the full Clarity **subspecialty** list (hundreds), not the
  legacy medical/surgical/haem-onc/paediatric groups. Expect many
  `reliable: false` and inactive services; summarise these as counts, and
  reserve per-service detail for flagged or negative-reduction rows.
- `uclh_ed_yta_beds` (aspirational YTA bed demand) is deliberately excluded
  from evaluation; do not report its absence as a gap.

## Report format

Structure the write-up as:

1. **Run summary** - split, date range if available, charts mode, targets run,
   row counts per mode.
2. **Headline verdict** - is anything flagged? One or two sentences.
3. **Findings per mode** - classifier metrics table (per clock); distribution
   problems ranked by negative reduction / high W²; transition-matrix rows
   with low p-values; arrivals status.
4. **Coverage and reliability** - inactive services, unreliable slices,
   `skip_reason` counts.
5. **Chart review** - which PNGs were opened and what they showed; note
   anything (like arrivals under `flagged`) that could not be reviewed.

## Presentations

When the user asks for a PowerPoint / slide deck from an evaluation run, do
**not** format slides in this skill. Hand off to the uclhflow
`eval-presentation` skill (`.cursor/skills/eval-presentation/` in the uclhflow
repo), which builds a results-only deck via `scripts/build_deck.py`.

That skill also owns locating a fresh `eval_valid*` / `eval_test*` zip in
`~/Downloads`, confirming it with the user, and unzipping into
`/Users/zellaking/Google Drive/UCL(H)/UCLH Evaluation` before building the
deck.

Optionally write a short `commentary.json` (slide-key → list of bullet strings)
for pass-through onto slides; keep interpretation here, layout there.
