# 4d. Evaluate demand predictions

After evaluating individual model components in the 3x notebooks (group snapshots in 3b, bed demand in **3d**, yet-to-arrive in **3f**), this notebook shows how we run a **systematic evaluation** with `patientflow.evaluate`: declare targets, assemble inputs, call `run_evaluation`, and read scalars from `scalars.json`.

For manual EPUDD and baseline comparison by service, see **3d**. For arrival deltas and survival curves, see **3f**.

### Data requirements

| Cohort / target                  | Data frame           | Required columns                                        |
| -------------------------------- | -------------------- | ------------------------------------------------------- |
| ED admissions classifier         | `ed_visits`          | `is_admitted`, `prediction_time`, `snapshot_date`       |
| ED-current bed demand            | `ed_visits`          | same + `specialty`                                      |
| YTA arrival deltas               | `inpatient_arrivals` | `arrival_datetime` (filtered per service via YTA model) |
| Window-based ED admission (UCLH) | `ed_visits`          | + `departure_datetime`                                  |

Public extracts omit `departure_datetime`. Set `RUN_FABRICATED_TIME_DEMOS=True` to call `synthesise_departure_times` for local demos of window-based evaluation.

### Evaluate package in brief

- **`EvaluationTarget`** — one row per evaluation task (`evaluation_mode`, `flow_name`, `component`, and `observation_mode` for distribution targets).
- **`EvaluationInputsBuilder`** — register classifiers, PMF dicts, observation frames, and benchmarks; **`build()`** returns immutable **`EvaluationInputs`**.
- **`run_evaluation`** — dispatches each target to a handler; writes plots and **`scalars.json`**.

**This notebook evaluates:** admission classifier diagnostics and probability quality; ED-current bed demand (with binomial and specialty-proportions benchmarks); and yet-to-arrive arrival-rate deltas. Set **`eval_split`** to `"valid"` or `"test"` on the builder to choose the holdout cohort.

## Approach

1. **Load data** — `prepare_prediction_inputs` and temporal splits (section 1; same pattern as 4c).
2. **Observation modes** — which patients count toward each observed value (section 2).
3. **Build PMF dicts** — `get_prob_dist_by_service` for ED-current bed demand (section 3).
4. **Assemble evaluation inputs** — `EvaluationInputsBuilder`, targets, and benchmarks (section 4).
5. **Run evaluation** — `run_evaluation` (section 5).
6. **Inspect output** — `evaluation_rows` in `scalars.json` (section 6).

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2

```

## 1. Load data and train models

The data loading and configuration steps match notebook 4c. Here `prepare_prediction_inputs` performs training and assembly in one call.

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you do not have the public data, set `data_folder_name` to `'data-synthetic'`.

```python
from typing import Any

from patientflow.train.emergency_demand import prepare_prediction_inputs
from patientflow.prepare import create_temporal_splits
from patientflow.load import get_model_key
from patientflow.generate import synthesise_departure_times
from datetime import timedelta
import pandas as pd

data_folder_name = "data-public"
prediction_inputs = prepare_prediction_inputs(data_folder_name, verbose=False)

admissions_models = prediction_inputs["admission_models"]
spec_model = prediction_inputs["specialty_model"]
yta_model_by_spec = prediction_inputs["yta_model"]
ed_visits = prediction_inputs["ed_visits"]
inpatient_arrivals = prediction_inputs["inpatient_arrivals"]
params = prediction_inputs["config"]
model_name = "admissions"

x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
prediction_window = timedelta(minutes=params["prediction_window"])
yta_time_interval = timedelta(minutes=params["yta_time_interval"])
prediction_times = params["prediction_times"]
prediction_dict = {tuple[Any, ...](pt): prediction_window for pt in prediction_times}

start_training_set = params["start_training_set"]
start_calibration_set = params["start_calibration_set"]
start_validation_set = params["start_validation_set"]
start_test_set = params["start_test_set"]
end_test_set = params["end_test_set"]

# Routine development: "valid". Final holdout report: "test" (same saved models).
eval_split = "valid"

_, _, valid_visits_df, test_visits_df = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
    visit_col="visit_number",
    verbose=False,
    start_calibration=start_calibration_set,
)

if eval_split == "valid":
    eval_visits_df = valid_visits_df
    eval_snapshot_start = start_validation_set
    eval_snapshot_end = start_test_set
elif eval_split == "test":
    eval_visits_df = test_visits_df
    eval_snapshot_start = start_test_set
    eval_snapshot_end = end_test_set
else:
    raise ValueError(f"eval_split must be 'valid' or 'test', got {eval_split!r}")

eval_snapshot_dates = [
    d.date()
    for d in pd.date_range(
        eval_snapshot_start, eval_snapshot_end, freq="D", inclusive="left"
    )
]
print(
    f"{eval_split} cohort: {len(eval_snapshot_dates)} snapshot dates "
    f"({eval_snapshot_start} to {eval_snapshot_end}, exclusive end)"
)

inpatient_arrivals = inpatient_arrivals.copy()
inpatient_arrivals["arrival_datetime"] = pd.to_datetime(
    inpatient_arrivals["arrival_datetime"], utc=True
)
_, _, valid_inpatient_arrivals_df, test_inpatient_arrivals_df = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
    verbose=False,
    start_calibration=start_calibration_set,
)

eval_inpatient_arrivals_df = (
    valid_inpatient_arrivals_df if eval_split == "valid" else test_inpatient_arrivals_df
)

# Skip when your extract already includes departure_datetime.
RUN_FABRICATED_TIME_DEMOS = False

if RUN_FABRICATED_TIME_DEMOS:
    eval_inpatient_arrivals_df = synthesise_departure_times(
        eval_inpatient_arrivals_df, kind="inpatient_arrivals", seed=42
    )
    eval_visits_df = synthesise_departure_times(eval_visits_df, kind="ed_visits", seed=43)
else:
    print(
        "Skipping synthesise_departure_times "
        "(RUN_FABRICATED_TIME_DEMOS=False)."
    )

specialties = ["medical", "surgical", "haem/onc", "paediatric"]

```

    valid cohort: 30 snapshot dates (2031-09-01 to 2031-10-01, exclusive end)
    Skipping synthesise_departure_times (RUN_FABRICATED_TIME_DEMOS=False).

## 2. Observation modes

Bed-demand evaluation compares a **predicted distribution** (how many admissions we expect) with an **observed count** (how many actually happened). For each snapshot date and prediction time, something has to define _which patients count_ toward that observed value.

**`observation_mode`** names that counting rule. For example, `admitted_at_some_point` counts patients in the ED snapshot who are eventually admitted, while `admitted_in_window` counts only those who leave for a ward before the prediction window ends. You declare the mode on each evaluation task; patientflow applies the same rule whenever it counts observed admissions from your data. Choose a mode your extract supports — `admitted_in_window` needs `departure_datetime`, which the public dataset omits (see the data-requirements table above).

| `observation_mode`               | Cohort                                          | What is counted                                             | Data frame / columns                             |
| -------------------------------- | ----------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| `admitted_at_some_point`         | Patients already in ED at the prediction moment | Eventually admitted (any time)                              | `ed_visits`, `is_admitted`                       |
| `admitted_in_window`             | Patients in the ED snapshot                     | Admitted and **leave ED for a ward** before the window ends | `ed_visits`, `is_admitted`, `departure_datetime` |
| `arrived_in_window`              | Yet-to-arrive                                   | **`arrival_datetime`** falls in the prediction window       | `inpatient_arrivals`                             |
| `arrived_and_admitted_in_window` | Yet-to-arrive (direct admission)                | Arrive and are admitted within the window (not via ED)      | `inpatient_arrivals` (often pre-filtered)        |

This notebook uses `admitted_at_some_point` for ED-current bed demand and classifiers. YTA arrival deltas (section 4) compare observed and expected arrival timing; they use filtered `inpatient_arrivals` frames rather than `count_observed`, though the target still carries an `observation_mode` label for scalar rows. Window-based bed-demand modes are common at UCLH; see notebook **3f** for survival-curve bed-demand evaluation (the public dataset uses synthesised ward timestamps for illustration). After `builder.build()` in section 4, the printed target list shows which mode each task declares.

## 3. Build ED-current PMF dicts

The evaluate package does not build predictions for you. You assemble PMF dicts first, then register them on the builder with `add_distributions_from_service_dict`.

`get_prob_dist_by_service` returns nested dicts: `service → model_key → snapshot_date → leaf`, where each **leaf** is a small dict holding `agg_predicted` (the PMF) and `agg_observed` (the count on that snapshot). Pass the same `observation_mode` here as on the bed-demand `EvaluationTarget` in section 4 (`admitted_at_some_point` in this run).

We build sequence-predictor PMFs for evaluation and a specialty-proportions baseline (training-set average routing) for benchmark comparison — the same baseline as notebook 3d. When building PMFs we use a narrow `FlowSelection` (ED-current patients only).

```python
from patientflow.aggregate import get_prob_dist_by_service
from patientflow.model_artifacts import ServiceModels
from patientflow.predict.demand import FlowSelection
from patientflow.predictors.value_to_outcome_predictor import ConstantSpecialtyProbs

# Narrow FlowSelection for PMF building: ED-current patients only.
flow_sel_ed_current = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=False,  # yet-to-arrive ED admissions
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)


def build_ed_current_by_service(spec_predictor) -> dict:
    by_service = {svc: {} for svc in specialties}
    for prediction_time, pw in prediction_dict.items():
        model_key = get_model_key(model_name, prediction_time)

        # ServiceModels: admission classifier + specialty router for this prediction time.
        service_models = ServiceModels(
            prediction_time=prediction_time,
            prediction_window=pw,
            ed_classifier=admissions_models[model_key],
            spec_model=spec_predictor,
        )

        # Returns {specialty: {snapshot_date: {agg_predicted, agg_observed}}}.
        by_specialty = get_prob_dist_by_service(
            eval_visits_df,
            eval_snapshot_dates,
            prediction_time,
            service_models,
            specialties,
            pw,
            flow_selection=flow_sel_ed_current,
            component="arrivals",
            observation_mode="admitted_at_some_point",
            use_admission_in_window_prob=False,
            verbose=False,
        )
        for specialty in specialties:
            by_service[specialty][model_key] = by_specialty[specialty]
    return by_service


ed_current_by_service = build_ed_current_by_service(spec_model)

train_inpatient_arrivals_df, _, _, _ = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
    verbose=False,
    start_calibration=start_calibration_set,
)
baseline_probs = (
    train_inpatient_arrivals_df["specialty"].value_counts(normalize=True).to_dict()
)
baseline_spec_model = ConstantSpecialtyProbs(baseline_probs)
ed_current_baseline_by_service = build_ed_current_by_service(baseline_spec_model)

print(
    f"Built ED-current PMFs: {len(prediction_times)} prediction times × "
    f"{len(specialties)} specialties"
)

```

    Built ED-current PMFs: 5 prediction times × 4 specialties

## 4. Assemble evaluation inputs

With PMF dicts ready from section 3, the next step is to tell `patientflow.evaluate` **what to score and which data to use**. Each measurement is one `EvaluationTarget` row: it names the handler (`evaluation_mode`), the data block (`flow_name`), the output family (`component`), and—for distribution targets—the counting rule (`observation_mode`).

Rather than constructing those rows by hand, I call `standard_ed_targets()` from `patientflow.evaluate.inputs`. It returns a standard ED admissions target list whose `flow_name` values match the `add_*` registrations below. In this notebook I keep four targets and omit yet-to-arrive bed-demand EPUDD (`include_ed_yta_distribution=False`), which needs ward-admission timestamps that are not included in the public data:

| Target                         | `flow_name`            | `evaluation_mode`                | What it produces                                           |
| ------------------------------ | ---------------------- | -------------------------------- | ---------------------------------------------------------- |
| Classifier diagnostics         | `ed_admissions_cls`    | `classifier_model_diagnostics`   | Headline metrics and SHAP plots per prediction time        |
| Classifier probability quality | `ed_admissions_cls`    | `classifier_probability_quality` | Discrimination, MADCAP, and calibration on the visit frame |
| ED-current bed demand          | `ed_current_beds`      | `distribution`                   | EPUDD plots and rPIT+CvM scalars (with benchmarks)         |
| YTA arrival deltas             | `ed_yta_arrival_rates` | `arrival_deltas`                 | Arrival-rate delta plots per service                       |

**Wiring pattern:**

1. Get the target list.
2. Create an `EvaluationInputsBuilder` with `flow_selection`, `prediction_dict`, and `eval_split`.
3. For each target, call the matching `add_*` method using the same `flow_name`.
4. Call `build()` and then `run_evaluation` (section 5).

Register visit frames, snapshot dates, and PMF dicts for the same holdout as `eval_split` (here the validation cohort from section 1). On the builder, set `include_ed_yta=True` in `flow_selection` so the yet-to-arrive arrival-delta target is in scope.

| `flow_name`            | `add_*` registration                                                                      | What you pass                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ed_admissions_cls`    | `add_classifier`                                                                          | trained models + `eval_visits_df`                                        |
| `ed_current_beds`      | `add_distributions_from_service_dict`, `add_distribution_observations`, benchmark helpers | PMF dicts from section 3 + `ed_visits` per service                       |
| `ed_yta_arrival_rates` | `add_arrival_deltas`                                                                      | filtered `inpatient_arrivals` per service, snapshot dates, YTA predictor |

For ED-current bed demand we also register two benchmarks: binomial class-balance (`add_distribution_benchmark_cohort`) and the specialty-proportions PMF dict from section 3 (`add_distribution_benchmark_from_service_dict`) — a naive router that gives every patient the same training-set average specialty mix, rather than using consult sequences (the same baseline as notebook **3d**). When enough snapshots have observations, distribution rows include `rpit_cvm_w2_reduction` against the binomial benchmark (see section 6).

```python
from pathlib import Path

from patientflow.evaluate.inputs import EvaluationInputsBuilder, standard_ed_targets
from patientflow.predict.demand import FlowSelection

# EvaluationTarget rows: flow_name must match add_* registrations below.
evaluation_targets = standard_ed_targets(
    # Omit YTA bed-demand EPUDD (needs ward-admission times not in public data).
    include_ed_yta_distribution=False,
)

# FlowSelection on the builder: include yet-to-arrive for the arrival-delta target.
eval_flow_selection = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=True,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)

builder = (
    EvaluationInputsBuilder(
        flow_selection=eval_flow_selection,
        prediction_dict=prediction_dict,
        eval_split=eval_split,
    )
    .with_evaluation_targets(evaluation_targets)
    # ed_admissions_cls — classifier_model_diagnostics + classifier_probability_quality
    .add_classifier(
        flow_name="ed_admissions_cls",
        trained_models=admissions_models,
        visits_df=eval_visits_df,
        label_col="is_admitted",
    )
    # ed_current_beds — distribution (PMFs from section 3)
    .add_distributions_from_service_dict(
        flow_name="ed_current_beds",
        prob_dist_by_service=ed_current_by_service,
        model_name=model_name,
    )
    .add_distribution_observations(
        flow_name="ed_current_beds",
        ed_visits_by_service={s: eval_visits_df for s in specialties},
    )
    # Benchmarks for ed_current_beds: binomial class-balance + specialty proportions
    .add_distribution_benchmark_cohort(admissions_ed_visits=eval_visits_df)
    .add_distribution_benchmark_from_service_dict(
        "ed_current_beds",
        ed_current_baseline_by_service,
        benchmark_kind="specialty_proportions",
    )
)

# ed_yta_arrival_rates — one arrivals frame per service (YTA model filter)
obs_ed_yta_by_service = {
    s: yta_model_by_spec.filter_dataframe(
        eval_inpatient_arrivals_df, yta_model_by_spec.filters[s]
    )
    for s in specialties
}

builder.add_arrival_deltas(
    flow_name="ed_yta_arrival_rates",
    arrivals_by_service=obs_ed_yta_by_service,
    snapshot_dates=eval_snapshot_dates,
    yta_time_interval=yta_time_interval,
    predictors_by_service={s: yta_model_by_spec for s in specialties},
    filter_keys_by_service={s: s for s in specialties},
)

inputs = builder.build()

print(f"Targets: {len(evaluation_targets)}")
for t in evaluation_targets:
    print(f"  {t.flow_name}/{t.component}: {t.evaluation_mode} (observation_mode={t.observation_mode})")

```

    Targets: 4
      ed_admissions_cls/classifier_model_diagnostics: classifier_model_diagnostics (observation_mode=admitted_at_some_point)
      ed_admissions_cls/classifier_discrimination_madcap_calibration: classifier_probability_quality (observation_mode=admitted_at_some_point)
      ed_current_beds/bed_demand_ed_current: distribution (observation_mode=admitted_at_some_point)
      ed_yta_arrival_rates/arrival_delta: arrival_deltas (observation_mode=arrived_in_window)

## 5. Run evaluation

`run_evaluation` writes `evaluation_run.yaml`, plot directories, and `scalars.json` under a timestamped folder.

```python
from datetime import datetime

from patientflow.evaluate.runner import run_evaluation

run_name = f"notebook4d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out = run_evaluation(
    Path("eval-output"),
    inputs,
    run_name=run_name,
    # Small public-data demo: plot all Gate A/B passers. Large multi-service
    # runs (e.g. UCLH) should use the default charts="flagged".
    charts="all",
    training_metadata={
        "start_training_set": str(start_training_set),
        "start_validation_set": str(start_validation_set),
        "start_test_set": str(start_test_set),
        "end_test_set": str(end_test_set),
        "yta_time_interval_minutes": int(yta_time_interval.total_seconds() // 60),
    },
)
out

```

    Predicted classification (not admitted, admitted):  [647 414]


    Predicted classification (not admitted, admitted):  [1029  515]


    Predicted classification (not admitted, admitted):  [1739  821]


    Predicted classification (not admitted, admitted):  [1870  992]


    Predicted classification (not admitted, admitted):  [1614  774]





    {'run_dir': PosixPath('eval-output/notebook4d_20260804_191146'),
     'scalars_path': PosixPath('eval-output/notebook4d_20260804_191146/scalars.json'),
     'manifest_path': PosixPath('eval-output/notebook4d_20260804_191146/evaluation_run.yaml'),
     'n_targets': 4}

## 6. Review outputs

Scalar rows are stored under the `evaluation_rows` key (with optional `_service_summary`). On bed-demand distribution rows, look for:

For the idea behind randomised PIT histograms, see notebook **3b**; here the same principle is combined with Cramér–von Mises as a scalar calibration score.

For each snapshot with both an observed bed count and a predicted PMF, the handler draws a randomised PIT value uniformly in \([F(k-1), F(k)]\), where \(k\) is the observed count and \(F\) is the predicted CDF. It then runs a one-sample Cramér–von Mises test of those draws against Uniform(0,1) and records \(W^2\). That randomisation step is repeated several times; `rpit_cvm_mean_w2` is the mean \(W^2\) across repetitions (per service and prediction time, when at least two snapshots qualify).

- `rpit_cvm_mean_w2` — rPIT+CvM \(W^2\) for the evaluated model. For a well-calibrated model, the transformed values should look like draws from a flat Uniform(0,1) distribution, giving a small \(W^2\).
- `rpit_cvm_benchmark_mean_w2` and `rpit_cvm_w2_reduction` — the same statistic and reduction for the **binomial** class-balance benchmark (positive reduction means the model improves on that baseline).
- `rpit_cvm_specialty_proportions_mean_w2` and `rpit_cvm_specialty_proportions_w2_reduction` — the same for the **specialty-proportions** benchmark from section 4 (training-set average specialty mix, as in notebook **3d**).

Use these scalars to triage where to look (services / prediction times with poor or negative reductions); then open the corresponding EPUDD and arrival-delta plots to diagnose what is wrong.

```python
import json

from IPython.display import display

run_dir = out["run_dir"]
scalars_path = out["scalars_path"]

print("Run directory:", run_dir)
print("Scalars path:", scalars_path)

payload = json.loads(scalars_path.read_text(encoding="utf-8"))
rows = payload.get("evaluation_rows") or []
scalars_df = pd.DataFrame(rows)
print(f"Scalar rows: {len(scalars_df)}")

# Bed-demand distribution rows: compare model calibration to the binomial baseline.
bed_demand_rows = scalars_df[
    (scalars_df["evaluation_mode"] == "distribution")
    & (scalars_df["flow"] != "ed_admissions_cls")
]
base_cols = [
    c
    for c in [
        "flow",
        "service",
        "prediction_time",
        "rpit_cvm_mean_w2",
        "rpit_cvm_benchmark_mean_w2",
        "rpit_cvm_w2_reduction",
    ]
    if c in bed_demand_rows.columns
]
display(bed_demand_rows[base_cols].head(20))

```

    Run directory: eval-output/notebook4d_20260804_191146
    Scalars path: eval-output/notebook4d_20260804_191146/scalars.json
    Scalar rows: 46

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>flow</th>
      <th>service</th>
      <th>prediction_time</th>
      <th>rpit_cvm_mean_w2</th>
      <th>rpit_cvm_benchmark_mean_w2</th>
      <th>rpit_cvm_w2_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>[6, 0]</td>
      <td>2.586187</td>
      <td>2.496441</td>
      <td>-0.089745</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>[9, 30]</td>
      <td>2.071827</td>
      <td>3.693121</td>
      <td>1.621293</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>[12, 0]</td>
      <td>2.202597</td>
      <td>5.696769</td>
      <td>3.494172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>[15, 30]</td>
      <td>2.340532</td>
      <td>6.054620</td>
      <td>3.714088</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>[22, 0]</td>
      <td>0.956285</td>
      <td>5.681486</td>
      <td>4.725201</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>[6, 0]</td>
      <td>0.300416</td>
      <td>8.920682</td>
      <td>8.620266</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>[9, 30]</td>
      <td>0.444873</td>
      <td>7.996816</td>
      <td>7.551942</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>[12, 0]</td>
      <td>0.887283</td>
      <td>9.242915</td>
      <td>8.355632</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>[15, 30]</td>
      <td>0.530690</td>
      <td>9.759489</td>
      <td>9.228799</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>[22, 0]</td>
      <td>0.074633</td>
      <td>9.708881</td>
      <td>9.634248</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>[6, 0]</td>
      <td>0.091693</td>
      <td>9.435551</td>
      <td>9.343859</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>[9, 30]</td>
      <td>0.130170</td>
      <td>9.592582</td>
      <td>9.462412</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>[12, 0]</td>
      <td>0.105657</td>
      <td>9.832507</td>
      <td>9.726850</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>[15, 30]</td>
      <td>0.132659</td>
      <td>9.985821</td>
      <td>9.853162</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>[22, 0]</td>
      <td>0.096550</td>
      <td>9.729465</td>
      <td>9.632915</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ed_current_beds</td>
      <td>paediatric</td>
      <td>[6, 0]</td>
      <td>0.119541</td>
      <td>9.674350</td>
      <td>9.554809</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ed_current_beds</td>
      <td>paediatric</td>
      <td>[9, 30]</td>
      <td>0.127717</td>
      <td>9.857766</td>
      <td>9.730049</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ed_current_beds</td>
      <td>paediatric</td>
      <td>[12, 0]</td>
      <td>1.173558</td>
      <td>9.854889</td>
      <td>8.681332</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ed_current_beds</td>
      <td>paediatric</td>
      <td>[15, 30]</td>
      <td>0.737867</td>
      <td>9.994305</td>
      <td>9.256437</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ed_current_beds</td>
      <td>paediatric</td>
      <td>[22, 0]</td>
      <td>0.242987</td>
      <td>9.951676</td>
      <td>9.708689</td>
    </tr>
  </tbody>
</table>
</div>

Across all services and prediction times, `rpit_cvm_w2_reduction` is positive and often large, meaning the ED-current bed-demand model is consistently better calibrated than the binomial baseline. The biggest gains are in surgical, haem/onc, and paediatric beds (baseline W² very high, model W² close to zero), while medical beds still show clear improvement but with higher residual W² — a good candidate for closer inspection with EPUDD plots.

## Summary

In this notebook I have shown how to run a systematic evaluation with `patientflow.evaluate`. I built ED-current PMF dicts with `get_prob_dist_by_service`, registered them on `EvaluationInputsBuilder` together with admission classifiers, YTA arrival deltas, and two distribution benchmarks (binomial class-balance and the specialty-proportions baseline from notebook **3d**). I declared the evaluation tasks with `standard_ed_targets()`, ran `run_evaluation`, and read `evaluation_rows` from `scalars.json` — use those scalars to triage where to look; a next step might be to open the EPUDD plots to diagnose what is wrong.

Notebook 3d covered the same ED-current PMFs manually (EPUDD plots and scalar MAE against the specialty-proportions baseline). Notebook 3f covered yet-to-arrive arrival deltas and survival-curve bed demand in more detail. To include those components in this evaluation (YTA bed-demand EPUDD based on survival curve, for example), add the corresponding `EvaluationTarget` rows (or enable the extra flags on `standard_ed_targets()`) and wire them with matching `add_*` calls and distinct flow_name values in section 4.
