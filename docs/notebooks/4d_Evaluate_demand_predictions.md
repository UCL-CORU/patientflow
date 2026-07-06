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

1. **Load data** — `prepare_prediction_inputs` and temporal splits (same pattern as 4c).
2. **Observation modes** — which patients count toward each observed value (section 2).
3. **Build PMF dicts** — `get_prob_dist_by_service` for ED-current bed demand.
4. **Run evaluation** — `EvaluationInputsBuilder`, benchmarks, `run_evaluation`.
5. **Inspect output** — `evaluation_rows` in `scalars.json`.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2

import sklearn

sklearn.set_config(display="text")

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
start_validation_set = params["start_validation_set"]
start_test_set = params["start_test_set"]
end_test_set = params["end_test_set"]

# Routine development: "valid". Final holdout report: "test" (same saved models).
eval_split = "valid"

_, valid_visits_df, test_visits_df = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
    visit_col="visit_number",
    verbose=False,
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
_, valid_inpatient_arrivals_df, test_inpatient_arrivals_df = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
    verbose=False,
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
| `departed_in_window`             | Current inpatients on a snapshot                | **Leave their subspecialty** within the window              | Inpatient snapshots, departure label column      |

This notebook uses `admitted_at_some_point` for ED-current bed demand and classifiers. YTA arrival deltas (section 4) compare observed and expected arrival timing; they use filtered `inpatient_arrivals` frames rather than `count_observed`, though the target still carries an `observation_mode` label for scalar rows. Window-based bed-demand modes are common at UCLH; see notebook 3f for survival-curve evaluation with real ward timestamps. After `builder.build()` in section 4, the printed target list shows which mode each task declares.

## 3. Build ED-current PMF dicts

The evaluate package does not build predictions for you. You assemble PMF dicts first, then register them on the builder with `add_distributions_from_service_dict`.

`get_prob_dist_by_service` returns nested dicts: `service → model_key → snapshot_date → leaf` (`agg_predicted`, `agg_observed`). Pass the same `observation_mode` here as on the bed-demand `EvaluationTarget` in section 4 (`admitted_at_some_point` in this run).

We build sequence-predictor PMFs for evaluation and a specialty-proportions baseline (training-set average routing) for benchmark comparison — the same baseline as notebook 3d.

```python
from patientflow.aggregate import get_prob_dist_by_service
from patientflow.model_artifacts import ServiceModels
from patientflow.predict.demand import FlowSelection
from patientflow.predictors.value_to_outcome_predictor import ConstantSpecialtyProbs

flow_sel_ed_current = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=False,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)


def build_ed_current_by_service(spec_predictor) -> dict:
    by_service = {svc: {} for svc in specialties}
    for prediction_time, pw in prediction_dict.items():
        model_key = get_model_key(model_name, prediction_time)
        service_models = ServiceModels(
            prediction_time=prediction_time,
            prediction_window=pw,
            ed_classifier=admissions_models[model_key],
            inpatient_classifier=None,
            spec_model=spec_predictor,
            ed_yta_model=None,
            non_ed_yta_model=None,
            elective_yta_model=None,
            transfer_model=None,
        )
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

train_inpatient_arrivals_df, _, _ = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
    verbose=False,
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

With PMF dicts ready from section 3, the next step is to tell `patientflow.evaluate` what to measure and which data to use. Each measurement is one `EvaluationTarget` row: it names the handler (`evaluation_mode`), the data block (`flow_name`), the output family (`component`), and—for distribution targets—the counting rule (`observation_mode`).

Rather than constructing those rows by hand, we call `standard_ed_targets()` from `patientflow.evaluate.inputs`. It returns the usual ED admissions evaluation list with `flow_name` values that match the `add_*` registrations below. In this notebook we keep four targets and omit yet-to-arrive bed-demand EPUDD (`include_ed_yta_distribution=False`), which needs ward-admission timestamps that are not included in the public data:

| Target                         | `flow_name`            | `evaluation_mode`                | What it produces                                           |
| ------------------------------ | ---------------------- | -------------------------------- | ---------------------------------------------------------- |
| Classifier diagnostics         | `ed_admissions_cls`    | `classifier_model_diagnostics`   | Headline metrics and SHAP plots per prediction time        |
| Classifier probability quality | `ed_admissions_cls`    | `classifier_probability_quality` | Discrimination, MADCAP, and calibration on the visit frame |
| ED-current bed demand          | `ed_current_beds`      | `distribution`                   | EPUDD plots and rPIT+CvM scalars (with benchmarks)         |
| YTA arrival deltas             | `ed_yta_arrival_rates` | `arrival_deltas`                 | Cumulative arrival-timing plots per service                |

You can trim or extend the list with the helper's boolean flags, or build `EvaluationTarget` rows manually for flows this helper does not cover.

Recipe: (1) get the target list; (2) create an `EvaluationInputsBuilder` with `flow_selection`, `prediction_dict`, and `eval_split`; (3) call one `add_*` method per target, using the same `flow_name` on the target and the registration; (4) `build()` then `run_evaluation`. Register visit frames, snapshot dates, and PMF dicts for the same holdout as `eval_split` (here the validation cohort from section 1).

| `flow_name`            | `add_*` registration                                                                      | What you pass                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ed_admissions_cls`    | `add_classifier`                                                                          | trained models + `eval_visits_df`                                        |
| `ed_current_beds`      | `add_distributions_from_service_dict`, `add_distribution_observations`, benchmark helpers | PMF dicts from section 3 + `ed_visits` per service                       |
| `ed_yta_arrival_rates` | `add_arrival_deltas`                                                                      | filtered `inpatient_arrivals` per service, snapshot dates, YTA predictor |

For ED-current bed demand we also register two benchmarks: binomial class-balance (`add_distribution_benchmark_cohort`) and the specialty-proportions PMF dict from section 3 (`add_distribution_benchmark_from_service_dict`). When enough snapshots have observations, distribution rows include `rpit_cvm_*_w2_reduction` scalars against each benchmark.

```python
from pathlib import Path

from patientflow.evaluate.inputs import EvaluationInputsBuilder, standard_ed_targets
from patientflow.predict.demand import FlowSelection

evaluation_targets = standard_ed_targets(
    include_ed_yta_distribution=False,
)

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
    .add_classifier(
        flow_name="ed_admissions_cls",
        trained_models=admissions_models,
        visits_df=eval_visits_df,
        label_col="is_admitted",
    )
    .add_distributions_from_service_dict(
        flow_name="ed_current_beds",
        prob_dist_by_service=ed_current_by_service,
        model_name=model_name,
    )
    .add_distribution_observations(
        flow_name="ed_current_beds",
        ed_visits_by_service={s: eval_visits_df for s in specialties},
    )
    .add_distribution_benchmark_cohort(admissions_ed_visits=eval_visits_df)
    .add_distribution_benchmark_from_service_dict(
        "ed_current_beds",
        ed_current_baseline_by_service,
        benchmark_kind="specialty_proportions",
    )
)

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
      ed_yta_arrival_rates/arrival_delta_cumulative: arrival_deltas (observation_mode=arrived_in_window)

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

    /Users/zellaking/miniconda3/envs/patientflow/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    Predicted classification (not admitted, admitted):  [662 399]
    Predicted classification (not admitted, admitted):  [1039  505]
    Predicted classification (not admitted, admitted):  [1751  809]
    Predicted classification (not admitted, admitted):  [1918  944]
    Predicted classification (not admitted, admitted):  [1549  839]





    {'run_dir': PosixPath('eval-output/notebook4d_20260706_163545'),
     'scalars_path': PosixPath('eval-output/notebook4d_20260706_163545/scalars.json'),
     'manifest_path': PosixPath('eval-output/notebook4d_20260706_163545/evaluation_run.yaml'),
     'n_targets': 4}

## 6. Review outputs

Scalar rows are stored under the `evaluation_rows` key (with optional `_service_summary`). Look for `rpit_cvm_mean_w2`, `rpit_cvm_benchmark_mean_w2`, and `rpit_cvm_specialty_proportions_mean_w2` on distribution rows.

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

base_cols = [
    c
    for c in [
        "evaluation_mode",
        "flow",
        "service",
        "component",
        "prediction_time",
        "charts_generated",
        "skip_reason",
        "rpit_cvm_mean_w2",
        "rpit_cvm_benchmark_mean_w2",
        "rpit_cvm_specialty_proportions_mean_w2",
    ]
    if c in scalars_df.columns
]
display(scalars_df[base_cols].head(20))

```

    Run directory: eval-output/notebook4d_20260706_163545
    Scalars path: eval-output/notebook4d_20260706_163545/scalars.json
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
      <th>evaluation_mode</th>
      <th>flow</th>
      <th>service</th>
      <th>component</th>
      <th>prediction_time</th>
      <th>charts_generated</th>
      <th>rpit_cvm_mean_w2</th>
      <th>rpit_cvm_benchmark_mean_w2</th>
      <th>rpit_cvm_specialty_proportions_mean_w2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[6, 0]</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[9, 30]</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[12, 0]</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[15, 30]</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[22, 0]</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>classifier_probability_quality</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_discrimination_madcap_calibration</td>
      <td>None</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[6, 0]</td>
      <td>True</td>
      <td>0.773730</td>
      <td>2.492335</td>
      <td>1.213580</td>
    </tr>
    <tr>
      <th>7</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[9, 30]</td>
      <td>True</td>
      <td>1.126916</td>
      <td>3.686872</td>
      <td>1.441604</td>
    </tr>
    <tr>
      <th>8</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[12, 0]</td>
      <td>True</td>
      <td>1.771353</td>
      <td>5.691530</td>
      <td>1.944947</td>
    </tr>
    <tr>
      <th>9</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[15, 30]</td>
      <td>True</td>
      <td>1.861555</td>
      <td>6.052490</td>
      <td>2.717361</td>
    </tr>
    <tr>
      <th>10</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[22, 0]</td>
      <td>True</td>
      <td>2.600397</td>
      <td>5.671083</td>
      <td>2.457058</td>
    </tr>
    <tr>
      <th>11</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[6, 0]</td>
      <td>True</td>
      <td>0.050033</td>
      <td>8.913299</td>
      <td>0.053610</td>
    </tr>
    <tr>
      <th>12</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[9, 30]</td>
      <td>True</td>
      <td>0.199328</td>
      <td>8.011692</td>
      <td>0.535820</td>
    </tr>
    <tr>
      <th>13</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[12, 0]</td>
      <td>True</td>
      <td>0.403116</td>
      <td>9.242185</td>
      <td>0.972275</td>
    </tr>
    <tr>
      <th>14</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[15, 30]</td>
      <td>True</td>
      <td>0.680136</td>
      <td>9.759980</td>
      <td>0.972317</td>
    </tr>
    <tr>
      <th>15</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[22, 0]</td>
      <td>True</td>
      <td>0.586499</td>
      <td>9.709997</td>
      <td>0.557722</td>
    </tr>
    <tr>
      <th>16</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>bed_demand_ed_current</td>
      <td>[6, 0]</td>
      <td>True</td>
      <td>0.247034</td>
      <td>9.436285</td>
      <td>0.337690</td>
    </tr>
    <tr>
      <th>17</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>bed_demand_ed_current</td>
      <td>[9, 30]</td>
      <td>True</td>
      <td>0.155640</td>
      <td>9.594669</td>
      <td>0.435843</td>
    </tr>
    <tr>
      <th>18</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>bed_demand_ed_current</td>
      <td>[12, 0]</td>
      <td>True</td>
      <td>0.106256</td>
      <td>9.831979</td>
      <td>0.304898</td>
    </tr>
    <tr>
      <th>19</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>haem/onc</td>
      <td>bed_demand_ed_current</td>
      <td>[15, 30]</td>
      <td>True</td>
      <td>0.182181</td>
      <td>9.985909</td>
      <td>0.582547</td>
    </tr>
  </tbody>
</table>
</div>

## Summary

In this notebook I have shown how to run a systematic evaluation with `patientflow.evaluate`. I built ED-current PMF dicts with `get_prob_dist_by_service`, registered them on `EvaluationInputsBuilder` together with admission classifiers, YTA arrival deltas, and two distribution benchmarks (binomial class-balance and the specialty-proportions baseline from notebook **3d**). I declared the evaluation tasks with `standard_ed_targets()`, ran `run_evaluation`, and read `evaluation_rows` from `scalars.json` — use those scalars to triage where to look; open the EPUDD and arrival-delta plots to diagnose what is wrong.

Notebook **3d** covers the same ED-current PMFs manually (EPUDD plots and scalar MAE against the specialty-proportions baseline). Notebook **3f** covers yet-to-arrive arrival deltas and survival-curve bed demand in more detail. To extend this run (YTA bed-demand EPUDD, departures, survival), add `EvaluationTarget` rows and matching `add_*` registrations with distinct `flow_name` values.

In the notebooks that follow, prefixed with 4, I demonstrate how these functions are assembled into a production system at University College London Hospital to predict emergency demand.
