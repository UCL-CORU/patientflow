# 4d. Evaluate demand predictions

In the 3x\_ notebooks, I evaluated individual model components in isolation (group snapshots in 3b, bed demand by service in 3d, yet-to-arrive demand in 3f). This notebook shows the `patientflow.evaluate`. At UCLH we use it to describe what to measure with `EvaluationTarget` rows, assemble data with `EvaluationInputsBuilder` (including a single `FlowSelection` for the run), then call `run_evaluation` to write charts plus `scalars.json` under a timestamped directory.

A runner dispatches evaluation activity in five `evaluation_mode` scenarios. Classifier evaluation is split into **model-level diagnostics** (`classifier_model_diagnostics`, A) and **flow-level probability quality** (`classifier_probability_quality`, B: discrimination, MADCAP, and calibration on all visits in the run cohort). Bed-demand PMF quality uses **`distribution`** (C). Yet-to-arrive rate diagnostics use **`arrival_deltas`**. A single global **`survival_curve`** target is optional (not shown here).

Set **`eval_split`** on `EvaluationInputsBuilder` to `"valid"` (default) or `"test"`. Register visit frames and snapshot dates for that holdout only; plot titles and `evaluation_run.yaml` record the same cohort. Train-time `selected_eval_metrics` on saved classifiers stay on validation unless you retrained with `evaluate_on_test=True`.

### How the `evaluate` package is designed

Three objects carry the design:

- **`EvaluationTarget`** — One row for each thing to be evaluated: it names which **runner branch** to use (`evaluation_mode`), how outputs should be **grouped and labelled** (`flow_name`, `flow_type`, `component`), and which **observation strategy** pairs with predicted distributions (`observation_mode`). See the table below for more detail.
- **`EvaluationInputsBuilder` → `EvaluationInputs`** — The builder is a **mutable staging area**: you set `flow_selection`, `prediction_times`, **`eval_split`**, and `evaluation_targets`, then call `add_classifier`, `add_distributions_*`, `add_arrival_deltas`, and so on. **`build()`** returns an **`EvaluationInputs`** instance: an **immutable** bundle of all registered tables and dicts the runner reads (classifier blocks, distribution blocks, arrival blocks, observation contexts, optional survival).
- **`run_evaluation`** — Takes an output root plus **`EvaluationInputs`**, walks through the **`inputs.evaluation_targets`**, and **dispatches each target by `evaluation_mode`** to the right handler (plots under `classifiers/`, `distributions/`, `arrivals/`, … and scalar rows in `scalars.json`).

The components of an EvaluationTarget are shown below.

| `EvaluationTarget` field | Role                                                                                                                                                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_name`              | Must match the `flow_name` passed to the corresponding **`add_*`** call so the target receives the right data block.                                                                                                                                     |
| `flow_type`              | Logical pathway label for scalars and reporting (for example admissions vs departures).                                                                                                                                                                  |
| `evaluation_mode`        | Selects the **handler**: `classifier_model_diagnostics`, `classifier_probability_quality`, `distribution`, `arrival_deltas`, or `survival_curve`.                                                                                                        |
| `component`              | Names the **chart/output family** for this target (`scalars.json`; for distribution mode, the PNG basename `{component}.png`). Unlike `evaluation_mode`, several targets can share a mode but differ by `component` (e.g. ED-current vs YTA bed demand). |
| `observation_mode`       | Names the `count_observed` strategy in `patientflow.evaluate.observations`; used when distribution evaluation recomputes observed counts from builder frames.                                                                                            |

## Approach

In this notebook I show the following:

1. **Load data** — same pattern as notebook 4c: `prepare_prediction_inputs`, temporal splits, synthetic `departure_datetime` on evaluation ED visits and inpatient arrivals where extracts omit those timestamps.
2. **Demonstrate pairing** of observed values with predicted distributions, under different values of `observation_mode`
3. **Demonstrate the evaluate package** — `EvaluationInputsBuilder` with `flow_selection`, `prediction_times`, `eval_split`, and a custom `evaluation_targets` list whose `flow_name` keys match `add_classifier` / `add_distributions_*` / `add_arrival_deltas`. Distribution targets pair `add_distributions_from_service_dict` with `add_distribution_observations`, and **`add_distribution_benchmark_cohort`** for the Binomial-benchmark on admission-style modes. For arrival deltas with a multi-service predictor, pass `predictors_by_service` and `filter_keys_by_service` so the baseline matches each service.
4. **Run evaluation** — `run_evaluation(output_root, inputs, run_name=...)`.
5. **Inspect output** — load `evaluation_rows` from `scalars.json` (and optional `_service_summary`).

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2

```

## 1. Load data and train models

The data loading and configuration steps match notebook 4c. Here `prepare_prediction_inputs` performs training and assembly in one call.

Public ED extracts omit `departure_datetime`; this notebook synthesises leave-ED times on the evaluation visit frame (and ward admission times on inpatient arrivals) for demonstration purposes.

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you do not have the public data, set `data_folder_name` to `'data-synthetic'`.

```python
from typing import Any


from patientflow.train.emergency_demand import prepare_prediction_inputs
from patientflow.prepare import create_temporal_splits
from patientflow.load import get_model_key
from datetime import timedelta, datetime, time, timezone
import numpy as np
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

# Generate the valid and test visits datasets
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

# Filter the visits dataframe to the valid or test split according to the value of eval_split
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

# Prepare an array of snapshot dates for the evaluation cohort
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

# Synthetic ward-admission timestamps for notebook demonstrations (see 4c).
np_rng = np.random.default_rng(42)
ward_delay_hours = np_rng.gamma(
    shape=4.0, scale=1.5, size=len(eval_inpatient_arrivals_df)
)
ward_delay_hours = ward_delay_hours.clip(min=0.25)

eval_inpatient_arrivals_df = eval_inpatient_arrivals_df.copy()
eval_inpatient_arrivals_df["departure_datetime"] = (
    eval_inpatient_arrivals_df["arrival_datetime"]
    + pd.to_timedelta(ward_delay_hours, unit="h")
)

# Create fake ED departure times for demonstration purposes
eval_visits_df = eval_visits_df.copy()
if "departure_datetime" not in eval_visits_df.columns:
    np_rng_ed = np.random.default_rng(43)
    leave_delay_hours = np_rng_ed.gamma(shape=3.0, scale=2.0, size=len(eval_visits_df))
    leave_delay_hours = leave_delay_hours.clip(min=0.25)
    snapshot_moment = pd.to_datetime(
        eval_visits_df["snapshot_date"].astype(str)
        + " "
        + eval_visits_df["prediction_time"].map(lambda t: f"{t[0]:02d}:{t[1]:02d}:00")
    )
    eval_visits_df["departure_datetime"] = (
        snapshot_moment + pd.to_timedelta(leave_delay_hours, unit="h")
    ).dt.tz_localize("UTC")
    eval_visits_df.loc[
        ~eval_visits_df["is_admitted"].astype(bool), "departure_datetime"
    ] = pd.NaT

specialties = ["medical", "surgical", "haem/onc", "paediatric"]


```

    valid cohort: 30 snapshot dates (2031-09-01 to 2031-10-01, exclusive end)

## 2. Pairing predicted distributions with their observed values

Evaluation is organised around pairings of predicted distributions for a given moment in time and patient cohort, with the observed values associated with each snapshot - ie what actually happened to the visits in that cohort.

In this section we first explain how the observed values are counted, and then show how the predicted distributions are prepared.

### 2a. Observation modes and their use in evaluation.

`OBSERVATION_MODES` control what is counted within the observed value. The various observation modes available in patientflow are:

- **`admitted_at_some_point`** — Patients **already in ED** at the prediction moment who are marked admitted (uses **`ed_visits`** dataframe and defaults to `is_admitted` column). The observation mode tells the code to count eventual admissions from among the snapshot, without reference to the time they were admitted.
- **`admitted_in_window`** — Patients **in that ED snapshot** who are admitted and **leave ED for a ward** before the window ends (uses **`ed_visits`** dataframe and the **`departure_datetime`** column). The observation mode tells the code to count only visits that ended in admission, and for which the departure_datetime (leaving the ED) was within the prediction window.
- **`arrived_in_window`** — **Yet-to-arrive** visits whose **`arrival_datetime`** falls in the prediction window. Uses the `inpatient_arrivals` dataset and the `arrival_datetime` column.
- **`arrived_and_admitted_in_window`** (not used in this notebook, but used at UCLH) — Arrivals who **are admitted directly (not via ED)** within the window, using the `inpatient_arrivals` dataset. (Where more than one arrivals cohort is included in the `inpatient_arrivals` dataset, the calling function pre-filters the dataset before calling the observation counting functions.)
- **`departed_in_window`** — **Current inpatients** on a snapshot who **leave their subspecialty** within the window (not used in this notebook, but used at UCLH) - Uses an inpatient snapshots dataframe with a label indicating the patient left the subspecialty within a prediction window.

```python
from patientflow.evaluate.observations import OBSERVATION_MODES, count_observed
print("Modes:", ", ".join(OBSERVATION_MODES))

```

    Modes: admitted_at_some_point, admitted_in_window, departed_in_window, arrived_in_window, arrived_and_admitted_in_window

Below we'll pick a random snapshot date to demonstrate the use of observation modes. We'll show both the use of the `count_observed` function with a specified `observation_mode`, and for comparison a cross-check against the relevant dataframe to confirm the observed values concur.

```python
demo_snapshot_date = eval_snapshot_dates[len(eval_snapshot_dates) // 2]
demo_prediction_time = prediction_times[0]
demo_prediction_moment = datetime.combine(
    demo_snapshot_date, time(*demo_prediction_time), tzinfo=timezone.utc
)
demo_window_end = demo_prediction_moment + prediction_window

demo_specialty = "surgical"
demo_arrivals = eval_inpatient_arrivals_df
if demo_specialty in yta_model_by_spec.filters:
    demo_arrivals = yta_model_by_spec.filter_dataframe(
        eval_inpatient_arrivals_df, yta_model_by_spec.filters[demo_specialty]
    )

print(
    f"Demonstrating using the {eval_split!r} dataset:\n"
    f"  snapshot={demo_snapshot_date}\n"
    f"  time={demo_prediction_time}\n"
    f"  specialty={demo_specialty!r}"
)

```

    Demonstrating using the 'valid' dataset:
      snapshot=2031-09-16
      time=(6, 0)
      specialty='surgical'

```python
# Counting the number of ed_visits that ended in admission at some point
n = count_observed(
    "admitted_at_some_point",
    snapshot_date=demo_snapshot_date,
    prediction_time=demo_prediction_time,
    prediction_window=prediction_window,
    ed_visits=eval_visits_df,
    specialty=demo_specialty,
)
print(f'Observed value using "admitted_at_some_point" observation mode: {n}')

# Checking against the original data source
print(f'Observed value using dataframe filtering: {len(eval_visits_df[(eval_visits_df.specialty == demo_specialty) &
    (eval_visits_df.is_admitted) &
    (eval_visits_df.snapshot_date == demo_snapshot_date) &
    (eval_visits_df.prediction_time == demo_prediction_time)])}')

```

    Observed value using "admitted_at_some_point" observation mode: 1
    Observed value using dataframe filtering: 1

```python
# Counting the number of ed_visits that ended in admission within the prediction window
n = count_observed(
    "admitted_in_window",
    snapshot_date=demo_snapshot_date,
    prediction_time=demo_prediction_time,
    prediction_window=prediction_window,
    ed_visits=eval_visits_df,
    specialty=demo_specialty,
)
print(f'Observed value using "admitted_in_window" observation mode: {n}')


# Checking against the original data source
print(f'Observed value using dataframe filtering: {len(eval_visits_df[(eval_visits_df.specialty == demo_specialty) &
    (eval_visits_df["snapshot_date"] == demo_snapshot_date)
    & (eval_visits_df["prediction_time"] == demo_prediction_time)
    & (eval_visits_df["is_admitted"].astype(bool))
    & (eval_visits_df["departure_datetime"] > demo_prediction_moment)
    & (eval_visits_df["departure_datetime"] <= demo_prediction_moment + prediction_window)

    ])}')


```

    Observed value using "admitted_in_window" observation mode: 1
    Observed value using dataframe filtering: 1

```python
# Counting inpatient arrivals within the prediction window (yet-to-arrive cohort)
n = count_observed(
    "arrived_in_window",
    snapshot_date=demo_snapshot_date,
    prediction_time=demo_prediction_time,
    prediction_window=prediction_window,
    inpatient_arrivals=demo_arrivals,
)
print(f'Observed value using "arrived_in_window" observation mode: {n}')

print(
    "Observed value using dataframe filtering:",
    len(
        demo_arrivals[
            (demo_arrivals.arrival_datetime > demo_prediction_moment)
            & (demo_arrivals.arrival_datetime <= demo_window_end)
        ]
    ),
)

```

    Observed value using "arrived_in_window" observation mode: 3
    Observed value using dataframe filtering: 3

```python
# Counting inpatient arrivals admitted within the prediction window
n = count_observed(
    "arrived_and_admitted_in_window",
    snapshot_date=demo_snapshot_date,
    prediction_time=demo_prediction_time,
    prediction_window=prediction_window,
    inpatient_arrivals=demo_arrivals,
)
print(f'Observed value using "arrived_and_admitted_in_window" observation mode: {n}')

print(
    "Observed value using dataframe filtering:",
    len(
        demo_arrivals[
            (demo_arrivals.arrival_datetime > demo_prediction_moment)
            & (demo_arrivals.arrival_datetime <= demo_window_end)
            & (demo_arrivals.departure_datetime > demo_prediction_moment)
            & (demo_arrivals.departure_datetime <= demo_window_end)
        ]
    ),
)

```

    Observed value using "arrived_and_admitted_in_window" observation mode: 0
    Observed value using dataframe filtering: 0

### 2b. Prepare dictionaries of predicted distributions, plus their observed values.

`get_prob_dist_by_service` returns, for each service, a snapshot date mapped to a 'leaf' that contains `agg_predicted` (a distinct PMF for each prediction time and snapshot date) and an `agg_observed` integer count for one `prediction_time` and one `flow_selection`.

For ED current patients, the code loops `snapshot_date` over the evaluation calendar (`eval_snapshot_dates`, driven by `eval_split`), finds the patients in ED at that prediction moment, and generates a prediction of the number of beds needed.

For ED yet-to-arrive PMFs, the code loops `snapshot_date` over the evaluation calendar (`eval_snapshot_dates`, driven by `eval_split`) and calls `yta_model_by_spec.predict(..., prediction_date=snapshot_date)`, matching weekday-stratified arrival profiles to that day.

When calling `get_prob_dist_by_service`, certain parameters need to be specified:

- the prediction time
- the prediction window
- the trained classifiers and models used for prediction; these are stored in the ServiceModels object
- a flow selection; ED current and ED yet-to-arrive use different `FlowSelection` flags
- the dataset containing snapshots to be used for prediction (if relevant to the requested flow selection)
- the aspirational curve parameters, (x1, y1, x2, y2) if relevant
- the observation mode against which the distribution will be evaluated. As the implementation at UCLH is aspirational (ie they assume four-hour targets are met), we don't evaluate whether visits ended in admission **within the prediction window**. We simply evaluate the number who were admitted eventually. We use the `observation_mode` "admitted_at_some_point" below

```python
from patientflow.aggregate import get_prob_dist_by_service
from patientflow.model_artifacts import ServiceModels
from patientflow.predict.demand import FlowSelection

flow_sel_ed_current = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=False,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)

# Per service: model_key -> snapshot_date -> leaf (distinct PMF per prediction_time)
ed_current_by_service = {svc: {} for svc in specialties}
ed_yta_by_service = {svc: {} for svc in specialties}

for prediction_time, prediction_window in prediction_dict.items():
    model_key = get_model_key(model_name, prediction_time)
    admission_model = admissions_models[model_key]
    service_models = ServiceModels(
        prediction_time=prediction_time,
        prediction_window=prediction_window,
        ed_classifier=admission_model,
        inpatient_classifier=None,
        spec_model=spec_model,
        ed_yta_model=yta_model_by_spec,
        non_ed_yta_model=None,
        elective_yta_model=None,
        transfer_model=None,
    )

    ed_current_probability_distributions_by_specialty = get_prob_dist_by_service(
        eval_visits_df,
        eval_snapshot_dates,
        prediction_time,
        service_models,
        specialties,
        prediction_window,
        flow_selection=flow_sel_ed_current,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        component="arrivals",
        observation_mode="admitted_at_some_point",
        verbose=False,
    )
    for specialty in specialties:
        ed_current_by_service[specialty][model_key] = (
            ed_current_probability_distributions_by_specialty[specialty]
        )

    for specialty in specialties:
        yet_to_arrive_leaf_by_snapshot_date = {}
        for snapshot_date in eval_snapshot_dates:
            if specialty in yta_model_by_spec.filters:
                yet_to_arrive_prediction_frame = yta_model_by_spec.predict(
                    prediction_time=prediction_time,
                    prediction_window=prediction_window,
                    filter_keys=specialty,
                    prediction_date=snapshot_date,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )[specialty]
                predicted_probability_mass = yet_to_arrive_prediction_frame[
                    "agg_proba"
                ].copy(deep=True)
                filtered_arrivals = yta_model_by_spec.filter_dataframe(
                    eval_inpatient_arrivals_df,
                    yta_model_by_spec.filters[specialty],
                )
                agg_observed = count_observed(
                    "arrived_in_window",
                    snapshot_date=snapshot_date,
                    prediction_time=prediction_time,
                    prediction_window=prediction_window,
                    inpatient_arrivals=filtered_arrivals,
                )
            else:
                predicted_probability_mass = pd.Series(
                    [1.0], index=[0], name="agg_proba"
                )
                agg_observed = 0
            yet_to_arrive_leaf_by_snapshot_date[snapshot_date] = {
                "agg_predicted": predicted_probability_mass,
                "agg_observed": agg_observed,
            }
        ed_yta_by_service[specialty][model_key] = yet_to_arrive_leaf_by_snapshot_date

number_of_prediction_times = len(prediction_times)
number_of_specialties = len(specialties)
print(
    "Built nested distribution dictionaries:\n"
    f"  • {number_of_prediction_times} prediction times (one model_key per time-of-day).\n"
    f"  • {number_of_specialties} specialties.\n"
)

```

    Built nested distribution dictionaries:
      • 5 prediction times (one model_key per time-of-day).
      • 4 specialties.

Examining the output from this process, we can see a predicted distribution with observed values for each of the ED current and ED yet-to-arrive elements.

```python
demo_model_key = get_model_key(model_name, demo_prediction_time)

print(f'ED current snapshots predicted distribution for {demo_specialty} service on {demo_snapshot_date}:')
agg_pred_curr = ed_current_by_service[demo_specialty][demo_model_key][demo_snapshot_date]['agg_predicted']
print(agg_pred_curr.head(10))  # Print only the first 10 values
if len(agg_pred_curr) > 10:
    print(f"... ({len(agg_pred_curr)} total)")

print(f'\nED current snapshots observed values for number admitted at some point to {demo_specialty} service on {demo_snapshot_date}:')
print(ed_current_by_service[demo_specialty][demo_model_key][demo_snapshot_date]['agg_observed'])

print(f'\nED yet-to-arrive predicted distribution for {demo_specialty} service on {demo_snapshot_date}:')
agg_pred_yta = ed_yta_by_service[demo_specialty][demo_model_key][demo_snapshot_date]['agg_predicted']
print(agg_pred_yta.head(10))  # Print only the first 10 values
if len(agg_pred_yta) > 10:
    print(f"... ({len(agg_pred_yta)} total)")

print(f'\nED yet-to-arrive observed values for {demo_specialty} service on {demo_snapshot_date}:')
print(ed_yta_by_service[demo_specialty][demo_model_key][demo_snapshot_date]['agg_observed'])
```

    ED current snapshots predicted distribution for surgical service on 2031-09-16:
          agg_proba
    0  1.272620e-01
    1  2.217735e-01
    2  2.928779e-01
    3  2.256789e-01
    4  1.014280e-01
    5  2.655857e-02
    6  4.044790e-03
    7  3.575432e-04
    8  1.830299e-05
    9  5.413717e-07
    ... (44 total)

    ED current snapshots observed values for number admitted at some point to surgical service on 2031-09-16:
    1

    ED yet-to-arrive predicted distribution for surgical service on 2031-09-16:
    sum
    0    0.143060
    1    0.278179
    2    0.270458
    3    0.175301
    4    0.085218
    5    0.033141
    6    0.010740
    7    0.002984
    8    0.000725
    9    0.000157
    Name: agg_proba, dtype: float64
    ... (20 total)

    ED yet-to-arrive observed values for surgical service on 2031-09-16:
    3

## 3. Prepare evaluation inputs

The cells below step through the same workflow, but this time using the evaluate package. The benefits of the package is that is does the pairing of aggregate distributions shown above, and generates evaluation outputs for all services in a single run, and saves them to a directory that can later be used for a systematic review.

The steps are to prepare per-service tables the arrival handler needs, declare **`EvaluationTarget`** rows (each names an `evaluation_mode` and links to builder data via **`flow_name`**), then call the builder’s **`add_*`** methods to add evaluations we wish to test. Finally **`build()`** returns immutable **`EvaluationInputs`** for **`run_evaluation`**.

The **evaluation run** carries one combined `FlowSelection` on `EvaluationInputs` (here: current + ED yet-to-arrive only). `evaluation_run.yaml` takes a copy of project `config.yaml` (to record training dates, etc.) plus evaluation-only settings for this run.

with **`flow_name`** keys that match each **`add_*`** call—e.g. `ed_current_beds`, `ed_yta_beds`—so outputs and **`scalars.json`** stay separated.

### 3a. Declare `EvaluationTarget` rows

Each target picks a runner branch via **`evaluation_mode`**. Options are:

- model-level classifier diagnostics (**A**)
- flow-level probability quality (**B**: discrimination, MADCAP, and calibration)
- bed-demand **distribution** (**C**)
- **arrival_deltas**.

If the evaluation target is a predicted distribution, **`observation_mode`** names the **`count_observed`** strategy paired with that target. (Note that the **`evaluate_distribution`** function recomputes **`agg_observed`** on each leaf and raises an error if a pre-built leaf disagrees with the recomputed count.)

**`flow_name`** is a key that links the evaluation target to the data that will be used: the same key must be passed to **`add_classifier`**, **`add_distributions_from_service_dict`**, **`add_distribution_observations`**, or **`add_arrival_deltas`**. This is not the same as **`FlowSelection`**. Several targets may share one `flow_name` when they use the same registered tables (below, both classifier targets use **`ed_admissions_cls`**).

**`flow_type`** is a label for reporting (for example `"admissions"` or `"departures"`); used in plot titles and **`scalars.json`**, not for handler dispatch or builder wiring.

- **`component`** names the chart and scalar family for this target. It does not select the handler (**`evaluation_mode`** does). It distinguishes outputs when several targets share the same mode or `flow_name` (for example **`bed_demand_ed_current`** vs **`bed_demand_ed_yta`**, both `evaluation_mode="distribution"`). For distribution targets, the `component` string names which PMF–observed pairing is evaluated, not the chart type. It also sets the PNG basename under each service folder (`distributions/<flow_name>/<service>/{component}.png`).

For **inpatient departures** (not run in this notebook), `component` additionally selects the **admission route** when recomputing observed counts: the handler maps `target.component` through **`DEPARTURES_DISTRIBUTION_ADMISSION_TYPE`** (for example `departures_elective` → `"elective"`). The value must match a dict key **exactly**; otherwise no route filter is applied and observed counts can mix elective and emergency patients on the same **`inpatient_visits`** frame—misaligned with a route-specific PMF. Use separate **`flow_name`** values per route, the same **`inpatient_visits_by_service`** snapshot per service, and **`observation_mode="departed_in_window"`**.

Pairing for distribution targets in this notebook: **ED current** uses **`admitted_at_some_point`** with **`ed_visits_by_service`**; **ED yet-to-arrive** uses **`arrived_in_window`** with **`inpatient_arrivals_by_service`**.

**`component` values in this notebook**:

| `component`                                    | Charts / outputs                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `classifier_model_diagnostics`                 | Headline metrics, feature importances, SHAP                                                             |
| `classifier_discrimination_madcap_calibration` | Discrimination, MADCAP, MADCAP-by-age, calibration                                                      |
| `bed_demand_ed_current`                        | EPUDD plot; rPIT+CvM summary scalars (`rpit_cvm_mean_w2`, …); Binomial benchmark when cohort registered |
| `bed_demand_ed_yta`                            | EPUDD plot; rPIT+CvM summary scalars only (no binomial benchmark for `arrived_in_window`)               |
| `arrival_delta_cumulative`                     | Cumulative arrival delta PNG per service and prediction time                                            |

```python
from pathlib import Path
from patientflow.evaluate.inputs import EvaluationInputsBuilder, EvaluationTarget

evaluation_targets = [
    # (A) Model-level classifier diagnostics: headline metrics and plots for each trained
    # admissions model (not repeated per hospital service).
    EvaluationTarget(
        flow_name="ed_admissions_cls",
        flow_type="admissions",
        evaluation_mode="classifier_model_diagnostics",
        component="classifier_model_diagnostics",
        observation_mode="admitted_at_some_point",
    ),
    # (B) Discrimination, MADCAP, and calibration on the full evaluation visit frame (same classifiers as A).
    EvaluationTarget(
        flow_name="ed_admissions_cls",
        flow_type="admissions",
        evaluation_mode="classifier_probability_quality",
        component="classifier_discrimination_madcap_calibration",
        observation_mode="admitted_at_some_point",
    ),
    # (C) ED-current bed-demand PMF vs observed count (EPUDD chart; component names the pairing).
    EvaluationTarget(
        flow_name="ed_current_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_current",
        observation_mode="admitted_at_some_point",
    ),
    # (C) ED yet-to-arrive bed-demand PMF vs observed count (EPUDD; inpatient_arrivals_by_service).
    EvaluationTarget(
        flow_name="ed_yta_beds",
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_yta",
        observation_mode="arrived_in_window",
    ),
    # Cumulative arrival-time diagnostics: observed vs expected admission timing in the window,
    # one PNG per (service, prediction_time); uses the arrival-deltas handler (not count_observed).
    EvaluationTarget(
        flow_name="ed_yta_arrival_rates",
        flow_type="admissions",
        evaluation_mode="arrival_deltas",
        component="arrival_delta_cumulative",
        observation_mode="arrived_in_window",
    ),
]

```

### 3b. Initialise the builder

**EvaluationInputsBuilder** is a mutable (ie changeable) object used to register everything an evaluation run needs: flow_selection, prediction_times, evaluation_targets, trained classifiers, distribution PMFs, dataframes. Once all of these have been define, build() returns a single immutable **EvaluationInputs** snapshot for running the evaluation.

**`EvaluationInputs`** carries exactly **one** **`FlowSelection`** for the run (written to **`evaluation_run.yaml`**). It should reflect the broad scenario you want to evaluate, even when individual prediction dicts were built with narrower selections in section 1. Here we enable ED current and ED yet-to-arrive only.

The builder also needs the global **`prediction_times`** list before any **`add_*`** call. We attach the target list with **`with_evaluation_targets`**.

```python
eval_flow_selection = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=True,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)

builder = EvaluationInputsBuilder(
    flow_selection=eval_flow_selection,
    prediction_dict=prediction_dict,
    eval_split=eval_split,
).with_evaluation_targets(evaluation_targets)

```

### 3c. Add admission classifier evaluation task to the builder

**`add_classifier`** registers the trained models per prediction time and the ED visits for the run cohort (`eval_visits_df`). When add_classifier is called, discrimination, MADCAP, calibration, and SHAP plots will be generated. The **`flow_name`** argument must match the classifier targets’ **`flow_name`** (`ed_admissions_cls`).

```python
builder.add_classifier(
    flow_name="ed_admissions_cls", # note the use of the same flow_name used in the classifer EvaluationTarget above
    trained_models=admissions_models,
    visits_df=eval_visits_df,
    label_col="is_admitted",
)

```

    <patientflow.evaluate.inputs.EvaluationInputsBuilder at 0x117cb2b40>

### 3d. Add ED current bed demand evaluation task to the builder

Below we add nested PMFs from section 1, then attach **observation** frames: one ED visits dataframe per service (here the same **`eval_visits_df`** for each) via **`ed_visits_by_service`**. Distribution evaluation recomputes **`agg_observed`** for **`observation_mode="admitted_at_some_point"`** from those frames and generates EPUDD plots.

```python
# add the predicted distribution to the builder
builder.add_distributions_from_service_dict(
    flow_name="ed_current_beds",
    prob_dist_by_service=ed_current_by_service,
    model_name=model_name,
)

# select the data that will be used to compute the observed values
obs_ed_current_by_service = {service: eval_visits_df for service in specialties}

# add the observed values to the builder
builder.add_distribution_observations(
    flow_name="ed_current_beds",
    ed_visits_by_service=obs_ed_current_by_service,
)

```

    <patientflow.evaluate.inputs.EvaluationInputsBuilder at 0x117cb2b40>

### 3e. Add ED yet-to-arrive bed demand evaluation task to the builder.

Same pattern for **`ed_yta_beds`**: predicted PMFs from section 1, then **`inpatient_arrivals_by_service`** with one inpatient arrivals dataframe per specialty. Note the use of flow_name specified in Section 3a for the yet-to-arrive evaluation task. This evaluation will use the **`observation_mode="arrived_in_window"`**. Charts are saved as **`bed_demand_ed_yta.png`** per service (EPUDD).

```python
# add the predicted distribution to the builder
builder.add_distributions_from_service_dict(
    flow_name="ed_yta_beds",
    prob_dist_by_service=ed_yta_by_service,
    model_name=model_name,
)

# select the data that will be used to compute the observed values
obs_ed_yta_by_service = {
    s: yta_model_by_spec.filter_dataframe(
        eval_inpatient_arrivals_df, yta_model_by_spec.filters[s]
    )
    for s in specialties
}

# add the observed values to the builder
builder.add_distribution_observations(
    "ed_yta_beds",
    inpatient_arrivals_by_service=obs_ed_yta_by_service,
)

```

    <patientflow.evaluate.inputs.EvaluationInputsBuilder at 0x117cb2b40>

### 3f. Add ED yet-to-arrive arrival deltas to the evaluation task

**`add_arrival_deltas`** takes snapshot dates, the prediction window, and optional **fitted predictors**. Because **`yta_model_by_spec`** has one weight entry per service, pass **`filter_keys_by_service`** so each service’s baseline matches the correct fitted profile.

```python
builder.add_arrival_deltas(
    flow_name="ed_yta_arrival_rates",
    arrivals_by_service=obs_ed_yta_by_service,
    snapshot_dates=eval_snapshot_dates,
    yta_time_interval=yta_time_interval,
    predictors_by_service={s: yta_model_by_spec for s in specialties},
    filter_keys_by_service={s: s for s in specialties},
)

```

    <patientflow.evaluate.inputs.EvaluationInputsBuilder at 0x117cb2b40>

### 3g. Register benchmark cohort for rPIT + CvM

Distribution evaluation now writes **randomised PIT + Cramér–von Mises** summary scalars on each active service × prediction-time row when there are at least two snapshots with observations (`rpit_cvm_mean_w2`, `rpit_cvm_std_w2`, `rpit_cvm_n_observations`, …). EPUDD plots use the same minimum-snapshot gate.

For **`admitted_at_some_point`** targets (ED current in this notebook), you can register a **Binomial(n, p̄)** benchmark: global class balance **p̄** per `prediction_time` on the full eval-split cohort, with **n** inferred from each snapshot PMF. That adds `rpit_cvm_benchmark_mean_w2` and `rpit_cvm_w2_reduction` on the same scalar rows. **Yet-to-arrive** (`arrived_in_window`) still gets rPIT+CvM but no binomial benchmark.

Call **`add_distribution_benchmark_cohort`** once before **`build()`**, passing the same eval-split visit frame used for observation counts (here **`eval_visits_df`**).

```python
builder.add_distribution_benchmark_cohort(
    admissions_ed_visits=eval_visits_df,
    admissions_label_col="is_admitted",  # optional; default is "is_admitted"
)

```

    Targets: 5
      ed_admissions_cls/classifier_model_diagnostics: classifier_model_diagnostics (observation_mode=admitted_at_some_point)
      ed_admissions_cls/classifier_discrimination_madcap_calibration: classifier_probability_quality (observation_mode=admitted_at_some_point)
      ed_current_beds/bed_demand_ed_current: distribution (observation_mode=admitted_at_some_point)
      ed_yta_beds/bed_demand_ed_yta: distribution (observation_mode=arrived_in_window)
      ed_yta_arrival_rates/arrival_delta_cumulative: arrival_deltas (observation_mode=arrived_in_window)
    Prediction times: [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)]

### 3h. Build `EvaluationInputs`

**`build()`** checks that **`flow_selection`** and **`prediction_times`** are set, then returns the immutable object consumed by **`run_evaluation`**. The printout lists each target for a quick sanity check before the (longer) evaluation run.

```python
inputs = builder.build()

print(f"Targets: {len(evaluation_targets)}")
for t in evaluation_targets:
    print(f"  {t.flow_name}/{t.component}: {t.evaluation_mode} (observation_mode={t.observation_mode})")
print(f"Prediction times: {prediction_times}")
```

## 4. Run evaluation

`run_evaluation` creates `run_dir / evaluation_run.yaml`, `run_dir / scalars.json`, and subfolders (`classifiers/`, `distributions/`, `arrivals/`, …) when those modes produce output. It returns `run_dir`, `scalars_path`, `manifest_path`, and `n_targets`.

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

    Predicted classification (not admitted, admitted):  [662 399]


    Predicted classification (not admitted, admitted):  [1039  505]


    Predicted classification (not admitted, admitted):  [1751  809]


    Predicted classification (not admitted, admitted):  [1918  944]


    Predicted classification (not admitted, admitted):  [1549  839]





    {'run_dir': PosixPath('eval-output/notebook4d_20260610_151049'),
     'scalars_path': PosixPath('eval-output/notebook4d_20260610_151049/scalars.json'),
     'manifest_path': PosixPath('eval-output/notebook4d_20260610_151049/evaluation_run.yaml'),
     'n_targets': 5}

## 5. Review outputs

Scalar rows are stored under the `evaluation_rows` key (with optional `_service_summary` for inactive-service bookkeeping). Distribution rows include **`rpit_cvm_mean_w2`** when a service × clock has at least two observed snapshots; **`rpit_cvm_benchmark_mean_w2`** and **`rpit_cvm_w2_reduction`** appear for **`bed_demand_ed_current`** when the benchmark cohort was registered. Below we print the run layout and show key columns from the scalar table.

```python
import json

from IPython.display import display

run_dir = out["run_dir"]
scalars_path = out["scalars_path"]

print("Run directory:", run_dir)
print("Scalars path:", scalars_path)

print("\nDirectory structure (depth <= 2):")
for p in sorted(run_dir.rglob("*")):
    depth = len(p.relative_to(run_dir).parts)
    if depth <= 2:
        indent = "  " * (depth - 1)
        print(f"{indent}{p.name}{'/' if p.is_dir() else ''}")

payload = json.loads(scalars_path.read_text(encoding="utf-8"))
rows = payload.get("evaluation_rows") or []
scalars_df = pd.DataFrame(rows)
print(f"\nScalar rows: {len(scalars_df)}")
if "_service_summary" in payload:
    print("_service_summary keys:", list(payload["_service_summary"].keys()))

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
        "rpit_cvm_w2_reduction",
        "rpit_cvm_n_observations",
    ]
    if c in scalars_df.columns
]
display(scalars_df[base_cols].head(16))

```

    Run directory: eval-output/notebook4d_20260610_151049
    Scalars path: eval-output/notebook4d_20260610_151049/scalars.json

    Directory structure (depth <= 2):
    arrivals/
      ed_yta_arrival_rates/
    classifiers/
      ed_admissions_cls/
    distributions/
      ed_current_beds/
      ed_yta_beds/
    evaluation_run.yaml
    scalars.json

    Scalar rows: 66
    _service_summary keys: ['by_slice']

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
    </tr>
    <tr>
      <th>1</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[9, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[12, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[15, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>classifier_model_diagnostics</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_model_diagnostics</td>
      <td>[22, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>classifier_probability_quality</td>
      <td>ed_admissions_cls</td>
      <td>_all_</td>
      <td>classifier_discrimination_madcap_calibration</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[6, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[9, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[12, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[15, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>medical</td>
      <td>bed_demand_ed_current</td>
      <td>[22, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[6, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[9, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[12, 0]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[15, 30]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>distribution</td>
      <td>ed_current_beds</td>
      <td>surgical</td>
      <td>bed_demand_ed_current</td>
      <td>[22, 0]</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

## Summary

1. Built **service-level prediction dicts** with a **per-clock** nested layout for EPUDD (`get_model_key` as the middle key).
2. Declared **`EvaluationTarget` instances** and wired **`EvaluationInputsBuilder`** with one **`flow_selection`** and **`eval_split`**, matching `flow_name` keys on targets to builder registrations, including **`add_distribution_benchmark_cohort`** for ED admissions.
3. Ran **`run_evaluation`** to emit plots and **`scalars.json`** (including rPIT+CvM summary scalars on distribution rows).
4. Loaded **`evaluation_rows`** for a tabular overview.

For a wider evaluation matrix (inpatient departures by route, non-ED yet-to-arrive, survival), add **`EvaluationTarget`** rows and matching builder registrations—use distinct **`flow_name`** values per departures route (e.g. `departures_elective`) with **`component="departures_elective"`** (must match **`DEPARTURES_DISTRIBUTION_ADMISSION_TYPE`**), **`inpatient_visits_by_service`** on the shared snapshot frame, and route-specific PMFs.
