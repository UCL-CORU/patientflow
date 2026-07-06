# 3d. Evaluate bed demand predictions by hospital service

In notebook 3c I predicted bed counts by hospital service for one group snapshot. Here I evaluate those predictions across the test set using the approaches from notebook 3b (histograms of observed minus expected, and EPUDD plots).

I also ask whether routing patients to specialties using consult sequences (the model from 3c) beats a naive alternative: give every patient the same specialty mix, taken from training-set averages (for example 45% medical, 30% surgical, …). A table of mean absolute errors (MAE) summarises that comparison across services; EPUDD plots show _how_ predictions differ from observations for individual services.

### Data requirements

| Column / dataset                                                                | Used for                                             |
| ------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `ed_visits` with `snapshot_date`, `prediction_time`, `is_admitted`, `specialty` | ED-current bed demand (`admitted_at_some_point`)     |
| `inpatient_arrivals` with `specialty`                                           | Baseline specialty proportions from the training set |

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you do not have the public data, set `data_folder_name` to `'data-synthetic'`.

For the same PMFs through `patientflow.evaluate` (`EvaluationInputsBuilder` and `run_evaluation`), see notebook **4d**.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2

```

## Load data and train models

`prepare_prediction_inputs` trains admission models at each prediction time and a hospital service model, as in notebooks 3b and 3c.

```python
from datetime import timedelta

import pandas as pd

from patientflow.load import get_model_key
from patientflow.model_artifacts import ServiceModels
from patientflow.prepare import create_temporal_splits
from patientflow.predict.demand import FlowSelection
from patientflow.train.emergency_demand import prepare_prediction_inputs

data_folder_name = "data-public"
prediction_inputs = prepare_prediction_inputs(data_folder_name, verbose=False)

admissions_models = prediction_inputs["admission_models"]
spec_model = prediction_inputs["specialty_model"]
ed_visits = prediction_inputs["ed_visits"]
inpatient_arrivals = prediction_inputs["inpatient_arrivals"]
specialties = prediction_inputs["specialties"]
params = prediction_inputs["config"]

model_name = "admissions"
prediction_window = timedelta(minutes=params["prediction_window"])
prediction_times = list(params["prediction_times"])

start_training_set = params["start_training_set"]
start_validation_set = params["start_validation_set"]
start_test_set = params["start_test_set"]
end_test_set = params["end_test_set"]

_, _, test_visits_df = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
    visit_col="visit_number",
    verbose=False,
)

test_snapshot_dates = [
    d.date()
    for d in pd.date_range(start_test_set, end_test_set, freq="D", inclusive="left")
]

```

## Generate predicted distributions by hospital service

Notebook **3c** called `get_prob_dist` for **one** group snapshot at a time, weighting admission probabilities by the specialty model. Here I use **`get_prob_dist_by_service`** to build the same specialty-weighted PMFs for **every** prediction time and snapshot date in the test set, and store results as `{model_key: {specialty: {snapshot_date: leaf}}}`.

This is also the first use in the 3x notebooks of **`FlowSelection`** (which flows to include) and **`ServiceModels`** (bundling the admission classifier and specialty router for each prediction time). Both are explained for production use in notebook **4a**.

Each **leaf** holds `agg_predicted` (the PMF) and `agg_observed` (the count on that snapshot). The plotting helpers used in the next sections read that structure:

- **`calc_mae_mpe`** — mean absolute error and mean percentage error across snapshot dates (scalar summary per prediction time).
- **`plot_deltas`** — histograms of observed minus expected values from those scalars.
- **`plot_epudd`** — EPUDD charts comparing the full predicted distribution to observed counts at each snapshot.

```python
from patientflow.aggregate import get_prob_dist_by_service
from patientflow.predictors.value_to_outcome_predictor import ConstantSpecialtyProbs

# ED current patients only (no yet-to-arrive, transfers, or departures).
flow_sel_ed_current = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=False,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)


def build_ed_current_distributions(spec_predictor) -> dict:
    """Build {model_key: {specialty: {snapshot_date: leaf}}} for one routing model."""
    by_model_key: dict = {}
    for prediction_time in prediction_times:
        model_key = get_model_key(model_name, prediction_time)
        service_models = ServiceModels(
            prediction_time=prediction_time,
            prediction_window=prediction_window,
            ed_classifier=admissions_models[model_key],
            inpatient_classifier=None,
            spec_model=spec_predictor,
            ed_yta_model=None,
            non_ed_yta_model=None,
            elective_yta_model=None,
            transfer_model=None,
        )
        by_specialty = get_prob_dist_by_service(
            test_visits_df,
            test_snapshot_dates,
            prediction_time,
            service_models,
            specialties,
            prediction_window,
            flow_selection=flow_sel_ed_current,
            component="arrivals",
            observation_mode="admitted_at_some_point",
            use_admission_in_window_prob=False,
            verbose=False,
        )
        by_model_key[model_key] = by_specialty
    return by_model_key


prob_dist_dict_all = build_ed_current_distributions(spec_model)

```

## Evaluate predictions by hospital service

Use `calc_mae_mpe` for scalar summaries across snapshot dates, and EPUDD plots to inspect distribution shape. I show histograms for one service and EPUDD for all services.

```python
from patientflow.evaluate import calc_mae_mpe
from patientflow.viz.observed_against_expected import plot_deltas

for specialty in specialties:
    specialty_prob_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all.items()
    }
    results = calc_mae_mpe(specialty_prob_dist)
    plot_deltas(
        results,
        suptitle=f"Histograms of observed - expected values for {specialty} service",
    )

```

```python
from patientflow.viz.epudd import plot_epudd

for specialty in specialties:
    specialty_prob_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all.items()
    }
    plot_epudd(
        prediction_times,
        specialty_prob_dist,
        model_name=model_name,
        suptitle=f"EPUDD plots for {specialty} service (sequence predictor)",
    )

```

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_8_0.png)

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_8_1.png)

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_8_2.png)

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_8_3.png)

## Compare with a baseline: average specialty proportions

The baseline gives every patient the same probability of admission to each hospital service, based on training-set averages from `inpatient_arrivals`. Positive `mae_reduction` means the sequence predictor has lower mean absolute error than the baseline.

```python
inpatient_arrivals = inpatient_arrivals.copy()
inpatient_arrivals["arrival_datetime"] = pd.to_datetime(
    inpatient_arrivals["arrival_datetime"], utc=True
)
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
prob_dist_dict_all_baseline = build_ed_current_distributions(baseline_spec_model)

comparison_rows = []
for specialty in specialties:
    model_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all.items()
    }
    baseline_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all_baseline.items()
    }
    model_results = calc_mae_mpe(model_dist)
    baseline_results = calc_mae_mpe(baseline_dist)
    for model_key in model_results:
        model_mae = model_results[model_key]["mae"]
        baseline_mae = baseline_results[model_key]["mae"]
        comparison_rows.append(
            {
                "specialty": specialty,
                "model_key": model_key,
                "mae_sequence": model_mae,
                "mae_baseline": baseline_mae,
                "mae_reduction": baseline_mae - model_mae,
            }
        )

comparison_df = pd.DataFrame(comparison_rows)
from IPython.display import display

display(comparison_df)

```

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
      <th>specialty</th>
      <th>model_key</th>
      <th>mae_sequence</th>
      <th>mae_baseline</th>
      <th>mae_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>surgical</td>
      <td>admissions_0600</td>
      <td>0.840294</td>
      <td>0.867202</td>
      <td>0.026908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>surgical</td>
      <td>admissions_0930</td>
      <td>0.833684</td>
      <td>0.860290</td>
      <td>0.026606</td>
    </tr>
    <tr>
      <th>2</th>
      <td>surgical</td>
      <td>admissions_1200</td>
      <td>1.210593</td>
      <td>1.387282</td>
      <td>0.176688</td>
    </tr>
    <tr>
      <th>3</th>
      <td>surgical</td>
      <td>admissions_1530</td>
      <td>1.472382</td>
      <td>1.633026</td>
      <td>0.160645</td>
    </tr>
    <tr>
      <th>4</th>
      <td>surgical</td>
      <td>admissions_2200</td>
      <td>1.469115</td>
      <td>1.498436</td>
      <td>0.029322</td>
    </tr>
    <tr>
      <th>5</th>
      <td>haem/onc</td>
      <td>admissions_0600</td>
      <td>0.416914</td>
      <td>0.538630</td>
      <td>0.121716</td>
    </tr>
    <tr>
      <th>6</th>
      <td>haem/onc</td>
      <td>admissions_0930</td>
      <td>0.448767</td>
      <td>0.539597</td>
      <td>0.090830</td>
    </tr>
    <tr>
      <th>7</th>
      <td>haem/onc</td>
      <td>admissions_1200</td>
      <td>0.575750</td>
      <td>0.645621</td>
      <td>0.069871</td>
    </tr>
    <tr>
      <th>8</th>
      <td>haem/onc</td>
      <td>admissions_1530</td>
      <td>0.798887</td>
      <td>0.882070</td>
      <td>0.083183</td>
    </tr>
    <tr>
      <th>9</th>
      <td>haem/onc</td>
      <td>admissions_2200</td>
      <td>0.739284</td>
      <td>0.825518</td>
      <td>0.086235</td>
    </tr>
    <tr>
      <th>10</th>
      <td>medical</td>
      <td>admissions_0600</td>
      <td>1.481626</td>
      <td>1.606857</td>
      <td>0.125230</td>
    </tr>
    <tr>
      <th>11</th>
      <td>medical</td>
      <td>admissions_0930</td>
      <td>1.359773</td>
      <td>1.492533</td>
      <td>0.132760</td>
    </tr>
    <tr>
      <th>12</th>
      <td>medical</td>
      <td>admissions_1200</td>
      <td>1.669823</td>
      <td>1.751949</td>
      <td>0.082125</td>
    </tr>
    <tr>
      <th>13</th>
      <td>medical</td>
      <td>admissions_1530</td>
      <td>2.489543</td>
      <td>2.748100</td>
      <td>0.258558</td>
    </tr>
    <tr>
      <th>14</th>
      <td>medical</td>
      <td>admissions_2200</td>
      <td>3.574783</td>
      <td>3.930862</td>
      <td>0.356079</td>
    </tr>
    <tr>
      <th>15</th>
      <td>paediatric</td>
      <td>admissions_0600</td>
      <td>0.318168</td>
      <td>0.536598</td>
      <td>0.218430</td>
    </tr>
    <tr>
      <th>16</th>
      <td>paediatric</td>
      <td>admissions_0930</td>
      <td>0.319004</td>
      <td>0.482169</td>
      <td>0.163165</td>
    </tr>
    <tr>
      <th>17</th>
      <td>paediatric</td>
      <td>admissions_1200</td>
      <td>0.512457</td>
      <td>0.565690</td>
      <td>0.053233</td>
    </tr>
    <tr>
      <th>18</th>
      <td>paediatric</td>
      <td>admissions_1530</td>
      <td>0.696109</td>
      <td>0.795939</td>
      <td>0.099830</td>
    </tr>
    <tr>
      <th>19</th>
      <td>paediatric</td>
      <td>admissions_2200</td>
      <td>0.648586</td>
      <td>0.724046</td>
      <td>0.075460</td>
    </tr>
  </tbody>
</table>
</div>

### Illustrative EPUDD: baseline vs sequence predictor

Scalars above summarise all services. Below I show EPUDD pairs for **haem/onc** and **paediatric**, where the baseline tends to over-predict most clearly.

```python
from IPython.display import display

for specialty in ["haem/onc", "paediatric"]:
    model_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all.items()
    }
    baseline_dist = {
        model_key: dist_dict[specialty]
        for model_key, dist_dict in prob_dist_dict_all_baseline.items()
    }
    print(f"\nEPUDD for {specialty}: baseline (historical proportions)")
    plot_epudd(
        prediction_times,
        baseline_dist,
        model_name=model_name,
        suptitle=f"{specialty} — baseline specialty proportions",
    )
    print(f"EPUDD for {specialty}: sequence predictor")
    plot_epudd(
        prediction_times,
        model_dist,
        model_name=model_name,
        suptitle=f"{specialty} — sequence predictor",
    )

```

    EPUDD for haem/onc: baseline (historical proportions)

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_12_1.png)

    EPUDD for haem/onc: sequence predictor

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_12_3.png)

    EPUDD for paediatric: baseline (historical proportions)

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_12_5.png)

    EPUDD for paediatric: sequence predictor

![png](3d_Evaluate_bed_demand_by_hospital_service_files/3d_Evaluate_bed_demand_by_hospital_service_12_7.png)

## Summary

In this notebook I have shown how to evaluate predicted bed count distributions by hospital service, using the evaluation approaches introduced in notebook 3b. I built specialty-weighted PMFs across the test set with `get_prob_dist_by_service`, summarised calibration with `calc_mae_mpe`, and used EPUDD plots to diagnose where predicted and observed distributions differ.

I also compared the sequence specialty predictor from notebook 3c against a baseline that gives every patient the same specialty mix, based on average admission proportions from the training set. Scalar MAE reductions across services support triage; illustrative EPUDD pairs show _how_ the baseline and the sequence predictor diverge for individual services.

For the same PMFs run through `patientflow.evaluate` (`EvaluationInputsBuilder` and `run_evaluation`), see notebook **4d**.

In the notebooks that follow, prefixed with 4, I demonstrate how these functions are assembled into a production system at University College London Hospital to predict emergency demand.
