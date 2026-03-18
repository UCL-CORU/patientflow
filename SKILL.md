---
name: eval-review
description: >
  Review evaluation outputs from the bed demand prediction tool. Trigger when the user
  points you at an eval-output folder, asks to review model performance, or generate
  evaluation reports/presentations.
---

# Evaluation Review Skill

You are reviewing evaluation outputs from a hospital bed demand prediction system. The system predicts bed demand per service by convolving multiple patient flow components. Your job is to review what the evaluation has produced, surface problems, and help create clear reports.

For the full evaluation plan — including the complete list of flows, what each predicts, and how they relate — see `Evaluation_plan_v2b.docx` in the root folder.

## Core principle

**Calibration over classification.** Patient-level classifiers produce probabilities that feed into Bernoulli trials and convolutions to generate bed count distributions. The accuracy of the probability values matters more than correct classification of individual patients. The primary evaluation tools are visual — MADCAP plots for classifiers, EPUDD plots for distributions — supplemented by scalar metrics.

## Output folder structure

```
eval-output/
├── config.yaml
├── scalars.json
├── classifiers/
│   └── {classifier_name}/
│       ├── madcap.png
│       ├── madcap_{prediction_time}.png
│       ├── discrimination.png
│       ├── discrimination_{prediction_time}.png
│       ├── calibration.png
│       ├── calibration_{prediction_time}.png
│       ├── feature_importance.png
│       └── shap_plot_{prediction_time}.png  (when shap is installed)
└── services/{service_name}/
    ├── ed_current_beds_{prediction_time}_{diagnostic}.png
    ├── ed_current_window_beds_{prediction_time}_{diagnostic}.png
    ├── ed_current_window_prob_survival.png
    ├── ed_yta_arrival_rates_{prediction_time}_deltas.png
    ├── ed_yta_beds_{prediction_time}_{diagnostic}.png
    ├── non_ed_yta_beds_{prediction_time}_{diagnostic}.png
    ├── elective_yta_beds_{prediction_time}_{diagnostic}.png
    ├── discharge_emergency_{prediction_time}_{diagnostic}.png
    ├── discharge_elective_{prediction_time}_{diagnostic}.png
    ├── combined_elective_arrivals_{prediction_time}_{diagnostic}.png
    └── combined_net_elective_{prediction_time}_{diagnostic}.png
```

Where `{diagnostic}` is `epudd` or `obs_exp`. Classifier charts that show all prediction times on one figure omit `{prediction_time}`. Aspirational flows produce no files in the services folder.

### Inactive services

When `run_evaluation` is called with `skip_inactive_services=True`, services whose distribution snapshots show negligible activity (zero observed counts and near-zero predicted means) are **not given a service folder or chart files**. Their scalar rows are still recorded in `scalars.json` with `"charts_generated": false` and `"skip_reason": "inactive_service"`. In production environments with hundreds of subspecialties, the majority of services may be inactive.

When `skip_inactive_services` is enabled, `scalars.json` includes a `_service_summary` block:

```json
"_service_summary": {
  "total_services": 511,
  "active_services": 199,
  "inactive_services": 312,
  "inactive_service_names": ["svc_a", "svc_b", "..."]
}
```

## Flows

File name prefixes map to flow groups:

| Prefix | What it covers |
|--------|---------------|
| `ed_current` | Patients currently in ED: admission classifier, subspecialty mapping, bed counts (with and without prediction window) |
| `ed_yta` | Patients yet to arrive via ED: arrival rates, bed counts |
| `non_ed_yta` | Emergency admissions not via ED |
| `elective_yta` | Elective admissions |
| `discharge` | Discharge classifier, emergency and elective departures |
| `transfer` | Transfers between services (not yet in scope) |
| `combined` | Convolutions across flow groups: emergency/elective arrivals, net flow |

### Aspirational flows

Some flows assume operational targets (e.g. 4-hour ED targets) that don't match observed reality. These are marked `"aspirational": true` in scalars.json. Standard observed-vs-predicted diagnostics are not valid for them. Currently aspirational: `ed_current_window_beds_aspirational`, `ed_yta_beds_aspirational`, `combined_emergency_arrivals`, `combined_net_emergency`.

Combined predictions that include any aspirational component are themselves aspirational. Flag the aspirational mismatch once in your review, not per instance.

## How to interpret each diagnostic

### MADCAP plots (primary classifier diagnostic)

Cumulative predicted vs cumulative observed outcomes, patients ordered by increasing predicted probability. Well-calibrated: lines track each other. Good discrimination: curves bow to bottom-left.

- Predicted below observed = under-prediction. Flag prominently — this is dangerous for bed demand.
- Predicted above observed = over-prediction.
- Check across prediction times and by subgroup where available.

### Discrimination plots

Histograms of predicted probabilities for positive vs negative cases. Look for separation. After recalibration, a sparse extreme right tail is expected and not by itself a problem.

### Calibration plots

Predicted probability bins vs observed fraction. Points should follow the diagonal. Use both uniform and quantile views — uniform as primary, quantile as robustness check.

### Feature importance plots

Horizontal bar charts showing the top features by XGBoost importance, one panel per prediction time. Useful for sanity-checking that the model is using clinically plausible features. Flag any feature that dominates unexpectedly or that shouldn't be available at prediction time (data leakage).

### SHAP plots (optional — requires shap package)

SHAP summary plots showing feature impact on predictions, one file per prediction time. Each dot is a patient; colour encodes feature value, horizontal position encodes SHAP value. Look for features with large spread (high impact) and check directionality makes clinical sense. These are only generated when the `shap` package is installed.

### EPUDD plots (primary distribution diagnostic)

Grey points = predicted CDF. Coloured points = where observations fall. Well-calibrated: coloured tracks grey. Grey may not follow y=x diagonal — this is expected for discrete distributions with stepped CDFs. Review across prediction times.

### Obs-exp histograms

Distribution of (observed − expected) across test snapshots. Should centre around zero. Systematic shift = bias. Wide spread = high variance.

### Survival curve comparison

Training/test survival curves overlaid. Similar curves = no temporal drift. Clear gaps suggest process drift (e.g. ED becoming slower/faster).

### Arrival rate delta plots

Cumulative observed arrivals vs cumulative mean arrival rate from training, overlaid per test date, average delta in red. Systematic positive/negative average indicates arrival rates have shifted between training and test.

## Scalar metrics

One `scalars.json` at the output root, keyed by flow name and prediction time. Each entry includes `flow_type` and `aspirational`.

| Metric | Applies to | Watch for |
|--------|-----------|-----------|
| Log loss | Classifiers | Primary classifier metric. Lower is better. |
| AUROC, AUPRC | Classifiers | Secondary. Useful for comparison across times/folds. |
| MAE | Distributions | Average error magnitude. Scale-dependent — interpret relative to typical counts. |
| MPE | Distributions | Systematic bias. **Positive = under-prediction. Flag prominently.** |
| n_snapshots | Distributions | Sample size for reliability. |

### Minimum sample sizes

Flag metrics as unreliable below these thresholds: 50 positive cases (classifiers), 30 snapshots (distributions), 10 transfers (transition matrix).

## Review workflow

1. **Inventory**: List folders present. Note which flows and services are covered. Gaps are themselves a finding.
2. **Scan scalars**: Load scalars.json. Flag any positive MPE (under-prediction) prominently. Note MAE relative to typical counts for each service. If a `_service_summary` block is present, report the headline counts (e.g. "199 of 511 services had sufficient activity for chart generation; 312 inactive services are recorded in scalars only") and move on — do not examine inactive services individually.
3. **Examine visual diagnostics**: MADCAP for classifiers. EPUDD for distribution flows (coloured points should track grey — not the diagonal). Describe what you see. Only active services (those with chart folders) need visual review.
4. **Cross-flow checks**: Do component flows explain the combined views? If combined is poor but components look fine, the convolution or a specific flow may be the issue.
5. **Prioritise findings**: Under-prediction of demand and poor calibration of feeding classifiers are higher priority than slight over-prediction or noisy metrics at low-volume services.

## Outputs

Produce what the user asks for:

- **Quick review summary**: Concise markdown. What was evaluated, headline metrics, flagged issues, next steps.
- **Technical evaluation report**: Detailed Word document (use docx skill). Executive summary → flow-by-flow → cross-cutting issues → recommendations.
- **Presentation**: Slide deck (use pptx skill). Ask who the audience is.
- **Comparison report**: Across evaluation runs — what improved, what regressed.
- **Issue tracker**: Spreadsheet (use xlsx skill). Columns: flow, service, issue, severity, status, date, notes.

## Things to watch out for

- **Don't over-interpret noisy metrics.** Small services produce volatile results. Say so.
- **Chart interpretation.** If working from image files, describe carefully. If working from data, be precise.
- **Suggest, don't prescribe model changes.** Frame as questions or suggestions.
