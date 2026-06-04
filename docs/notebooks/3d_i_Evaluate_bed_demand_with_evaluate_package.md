# 3d_i. Evaluate bed demand with `patientflow.evaluate`

In notebook **3d**, I evaluated specialty-weighted bed demand predictions with manual plots (`calc_mae_mpe`, `plot_deltas`, `plot_epudd`).

This notebook shows the same **distribution** evaluation through the `patientflow.evaluate` package: declare an `EvaluationTarget`, register predicted PMFs on `EvaluationInputsBuilder`, and call `run_evaluation` to write a timestamped output directory (`scalars.json`, EPUDD charts under `distributions/`).

### Why use the evaluate package?

Notebook **3d** is a good way to learn the evaluation logic: you build PMFs, compute scalar summaries, and call plotting functions directly.

The evaluate package is preferable when you want evaluation to behave more like a **repeatable pipeline**:

- **Scale across many services** — in **3d**, each hospital service gets its own histogram and EPUDD figure. That is manageable for a handful of specialties, but visual inspection does not scale when you evaluate many services (or many prediction times). A run directory with `scalars.json` lets you scan and compare results in tabular form, and open only the charts you need.
- **Standard output layout** — each run writes a timestamped folder with `evaluation_run.yaml`, `scalars.json`, and charts in predictable subfolders, rather than ad hoc figures in the notebook.
- **Explicit evaluation contract** — an `EvaluationTarget` records what you are measuring (`evaluation_mode`, `observation_mode`, `component`), which makes runs easier to compare and audit.
- **Centralised observation counting** — observed values are recomputed from registered visit frames using `patientflow.evaluate.observations`, so the same counting rules apply across services and prediction times.
- **Less notebook glue code** — once PMFs are registered on the builder, `run_evaluation` handles dispatch, file naming, and scalar collection for you.
- **Room to grow** — you can add further evaluation targets later without rewriting the manual plot loops from **3d**.
- **Future evaluation options** — later `patientflow` releases will add new evaluation capabilities on top of this package (for example, additional metrics for assessing predicted distributions). Adopting `run_evaluation` now means those options can plug into the same run layout rather than requiring another one-off notebook workflow.

You still use the same prediction method as **3d** (specialty-weighted `get_prob_dist` PMFs). This notebook only changes how those PMFs are evaluated and stored.

### Prerequisites

- Notebooks **3b**, **3c**, and **3d** (same models and evaluation concepts)
- `patientflow` 1.7 or later (evaluate package and backward-compatible imports)

### About the data

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you do not have the public data, change `data_folder_name` from `'data-public'` to `'data-synthetic'`.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Load data and train models

This section matches notebook **3d**: `prepare_prediction_inputs` trains admission models at each prediction time and a hospital service model, then I take the **test** holdout for evaluation.

```python
from datetime import datetime, timedelta

from patientflow.prepare import create_temporal_splits
from patientflow.train.emergency_demand import prepare_prediction_inputs

data_folder_name = "data-public"
prediction_inputs = prepare_prediction_inputs(data_folder_name)

admissions_models = prediction_inputs["admission_models"]
spec_model = prediction_inputs["specialty_model"]
ed_visits = prediction_inputs["ed_visits"]
specialties = prediction_inputs["specialties"]
params = prediction_inputs["config"]

model_name = "admissions"
prediction_window = timedelta(minutes=params["prediction_window"])
prediction_times = list(params["prediction_times"])
prediction_dict = {tuple(pt): prediction_window for pt in prediction_times}

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
)
```

    Split sizes: [62071, 10415, 29134]
    Split sizes: [7716, 1285, 3898]

    Processing: (6, 0)



    Processing: (9, 30)



    Processing: (12, 0)



    Processing: (15, 30)



    Processing: (22, 0)


    Split sizes: [62071, 10415, 29134]

## Generate predicted distributions by hospital service

I reuse the helper from notebook **3d** to build specialty-weighted PMFs with `get_prob_dist` for every prediction time and hospital service.

```python
from patientflow.aggregate import get_prob_dist
from patientflow.load import get_model_key
from patientflow.prepare import prepare_group_snapshot_dict, prepare_patient_snapshots


def get_specialty_probability_distributions(
    test_visits_df,
    spec_model,
    admissions_models,
    model_name,
    specialties=None,
):
    if specialties is None:
        specialties = ["medical", "surgical", "haem/onc", "paediatric"]

    if hasattr(spec_model, "predict_dataframe"):
        test_visits_df = test_visits_df.copy()
        test_visits_df.loc[:, "specialty_prob"] = spec_model.predict_dataframe(
            test_visits_df
        )
    else:

        def determine_specialty(row):
            return spec_model.predict(row["consultation_sequence"])

        test_visits_df = test_visits_df.copy()
        test_visits_df.loc[:, "specialty_prob"] = test_visits_df.apply(
            determine_specialty, axis=1
        )

    prob_dist_dict_all = {}

    for _prediction_time in test_visits_df.prediction_time.unique():
        prob_dist_dict_for_pats_in_ED = {}
        print("\nProcessing :" + str(_prediction_time))
        model_key = get_model_key(model_name, _prediction_time)

        for specialty in specialties:
            print(
                f"Predicting bed counts for {specialty} service, "
                "for all snapshots in the test set"
            )

            prob_admission_to_specialty = test_visits_df["specialty_prob"].apply(
                lambda x: x.get(specialty, 0.0) if isinstance(x, dict) else 0.0
            )

            X_test, y_test = prepare_patient_snapshots(
                df=test_visits_df,
                prediction_time=_prediction_time,
                single_snapshot_per_visit=False,
                visit_col="visit_number",
            )

            group_snapshots_dict = prepare_group_snapshot_dict(
                test_visits_df[test_visits_df.prediction_time == _prediction_time]
            )

            admitted_to_specialty = test_visits_df["specialty"] == specialty

            prob_dist_dict_for_pats_in_ED[specialty] = get_prob_dist(
                group_snapshots_dict,
                X_test,
                y_test,
                admissions_models[model_key],
                weights=prob_admission_to_specialty,
                category_filter=admitted_to_specialty,
                normal_approx_threshold=30,
            )

        prob_dist_dict_all[model_key] = prob_dist_dict_for_pats_in_ED

    return prob_dist_dict_all


prob_dist_dict_all = get_specialty_probability_distributions(
    test_visits_df,
    spec_model,
    admissions_models,
    model_name,
    specialties=specialties,
)
```

    Processing :(22, 0)
    Predicting bed counts for surgical service, for all snapshots in the test set


    Predicting bed counts for haem/onc service, for all snapshots in the test set


    Predicting bed counts for medical service, for all snapshots in the test set


    Predicting bed counts for paediatric service, for all snapshots in the test set



    Processing :(6, 0)
    Predicting bed counts for surgical service, for all snapshots in the test set


    Predicting bed counts for haem/onc service, for all snapshots in the test set


    Predicting bed counts for medical service, for all snapshots in the test set


    Predicting bed counts for paediatric service, for all snapshots in the test set



    Processing :(15, 30)
    Predicting bed counts for surgical service, for all snapshots in the test set


    Predicting bed counts for haem/onc service, for all snapshots in the test set


    Predicting bed counts for medical service, for all snapshots in the test set


    Predicting bed counts for paediatric service, for all snapshots in the test set



    Processing :(9, 30)
    Predicting bed counts for surgical service, for all snapshots in the test set


    Predicting bed counts for haem/onc service, for all snapshots in the test set


    Predicting bed counts for medical service, for all snapshots in the test set


    Predicting bed counts for paediatric service, for all snapshots in the test set



    Processing :(12, 0)
    Predicting bed counts for surgical service, for all snapshots in the test set


    Predicting bed counts for haem/onc service, for all snapshots in the test set


    Predicting bed counts for medical service, for all snapshots in the test set


    Predicting bed counts for paediatric service, for all snapshots in the test set

## Register inputs on `EvaluationInputsBuilder`

Notebook **3d** stores PMFs as `{model_key → specialty → snapshot_date → leaf}`. The evaluate package expects `{specialty → model_key → snapshot_date → leaf}` for `add_distributions_from_service_dict`.

The steps match notebook **4d**: declare **`EvaluationTarget`** rows, initialise **`EvaluationInputsBuilder`**, register predicted PMFs and observation frames with matching **`flow_name`** keys, then call **`build()`**.

### `EvaluationTarget` fields (one distribution target here)

| Field              | Role                                                                                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_name`        | Links the target to builder data. The same string must be passed to **`add_distributions_from_service_dict`** and **`add_distribution_observations`**. Not the same as **`FlowSelection`**.                                           |
| `flow_type`        | Logical pathway label for scalars and reporting (here `"admissions"`).                                                                                                                                                                |
| `evaluation_mode`  | Selects the runner handler. Here `"distribution"` runs EPUDD charts and distribution scalars.                                                                                                                                         |
| `component`        | Names the chart and scalar family for this target. For distribution mode, it sets the PNG basename under each service folder: `distributions/<flow_name>/<service>/{component}.png`.                                                  |
| `observation_mode` | Names the **`count_observed`** strategy when the runner recomputes **`agg_observed`** on each leaf (here `"admitted_at_some_point"` — admitted patients at the prediction moment, filtered by specialty when evaluating per service). |

### Builder inputs set on this run

| Input                                     | Value in this notebook                                                                                                                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`flow_selection`**                      | ED current patients only (`FlowSelection.custom` with inflows off except **`include_ed_current=True`**). Written to **`evaluation_run.yaml`**; reflects the broad scenario for this run. |
| **`prediction_dict`**                     | Maps each `(hour, minute)` to **`prediction_window`** (from `config.yaml`). Required before any **`add_*`** call.                                                                        |
| **`eval_split`**                          | `"test"` — visit frames and plot titles refer to the test holdout (notebook **3d** uses the same split).                                                                                 |
| **`evaluation_targets`**                  | One distribution target: **`flow_name="ed_current_beds"`**, **`component="bed_demand_ed_current"`**.                                                                                     |
| **`add_distributions_from_service_dict`** | Registers reshaped specialty PMFs under **`flow_name="ed_current_beds"`**.                                                                                                               |
| **`add_distribution_observations`**       | Registers ED visit frames for observed counting via **`ed_visits_by_service`** (same **`test_visits_df`** per service, as in **4d**).                                                    |

Distribution evaluation **recomputes** **`agg_observed`** from those frames and raises an error if a pre-built leaf disagrees with the recomputed count.

```python
from patientflow.evaluate.inputs import EvaluationInputsBuilder, EvaluationTarget
from patientflow.predict.types import FlowSelection

FLOW_NAME = "ed_current_beds"

prob_dist_by_service = {
    specialty: {
        model_key: prob_dist_dict_all[model_key][specialty]
        for model_key in prob_dist_dict_all
    }
    for specialty in specialties
}

evaluation_targets = [
    EvaluationTarget(
        flow_name=FLOW_NAME,
        flow_type="admissions",
        evaluation_mode="distribution",
        component="bed_demand_ed_current",
        observation_mode="admitted_at_some_point",
    ),
]

eval_flow_selection = FlowSelection.custom(
    include_ed_current=True,
    include_ed_yta=False,
    include_non_ed_yta=False,
    include_elective_yta=False,
    include_transfers_in=False,
    include_departures=False,
)

builder = EvaluationInputsBuilder(
    flow_selection=eval_flow_selection,
    prediction_dict=prediction_dict,
    eval_split="test",
).with_evaluation_targets(evaluation_targets)

builder.add_distributions_from_service_dict(
    flow_name=FLOW_NAME,
    prob_dist_by_service=prob_dist_by_service,
    model_name=model_name,
)

ed_visits_by_service = {service: test_visits_df for service in specialties}
builder.add_distribution_observations(
    flow_name=FLOW_NAME,
    ed_visits_by_service=ed_visits_by_service,
)

inputs = builder.build()
for target in inputs.evaluation_targets:
    print(
        f"  {target.flow_name}/{target.component}: {target.evaluation_mode} "
        f"(observation_mode={target.observation_mode})"
    )
```

      ed_current_beds/bed_demand_ed_current: distribution (observation_mode=admitted_at_some_point)

## Run evaluation

`run_evaluation` writes:

- `evaluation_run.yaml` — run settings, **`flow_selection`**, **`prediction_dict`**, and **`eval_split`**
- `scalars.json` — summary metrics per target, service, and prediction time
- `distributions/ed_current_beds/<service>/bed_demand_ed_current.png` — EPUDD charts (one PNG per active service)

```python
from pathlib import Path

from patientflow.evaluate.runner import run_evaluation

output_root = Path("eval-output")
run_name = f"notebook3g_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

out = run_evaluation(
    output_root=output_root,
    inputs=inputs,
    run_name=run_name,
)

print(f"Run directory: {out['run_dir']}")
print(f"Scalars path: {out['scalars_path']}")
print(f"Manifest path: {out['manifest_path']}")
print(f"Targets evaluated: {out['n_targets']}")
```

    Run directory: eval-output/notebook3g_20260527_211917
    Scalars path: eval-output/notebook3g_20260527_211917/scalars.json
    Manifest path: eval-output/notebook3g_20260527_211917/evaluation_run.yaml
    Targets evaluated: 1

## Summary

- I generated the same specialty-weighted PMFs as notebook **3d**, then passed them to `EvaluationInputsBuilder` and `run_evaluation`.
- For manual histograms and EPUDD plots in the notebook, continue to use notebook **3d**.
- If `run_evaluation` raises an error about `agg_observed` not matching a recomputed count, compare one `(service, snapshot_date, prediction_time)` between the PMF leaf from `get_prob_dist` and `count_observed(..., observation_mode="admitted_at_some_point", specialty=...)` on `test_visits_df`. Notebook **3d** counts observed admissions via `category_filter` on snapshot subsets; the package counts admitted rows on the full ED frame at each prediction moment.
