# 4c. Evaluate predictions of emergency demand

The previous notebook demonstrated the full implementation in code of training models for use at UCLH using the functions provided in `patientflow`.

The final step is to evaluate the predicted distributions against observed numbers. I will evaluate the predictions by specialty using the approaches covered in a [previous notebook](3b_Evaluate_group_snapshots.md).

- histograms of observed versus expected values
- Adjusted qq plots

As the predictions for yet-to-arrive patients are aspirational, these cannot be directed evaluated against observed numbers of admissions in the prediction window. In reality, due to poor ED performance, few may have been admitted within the window. Similarly for the group of patients in the ED, we calculate the predicted number of beds needed within the prediction window, but the observed numbers will not reflect the targets.

We can, however, evaluate the predictions in a slightly different way.

- For the patients in ED, we can compare the predicted bed counts needed for each specialty against observed numbers admitted to each specialty from among patients comprising each group snapshot in the test set period, without taking into account how long it takes each patient to be admitted.
- For the yet-to-arrive patients, we can compare the predicted with the observed arrival rates within the prediction window.

## Set up the notebook environment

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

```python
from patientflow.load import set_project_root
project_root = set_project_root()

```

    Inferred project root: /Users/zellaking/Repos/patientflow

## Set file paths and load data

I'm going to use real patient data from UCLH to demonstrate the implementation.

As noted previously, you can request the datasets that are used here on [Zenodo](https://zenodo.org/records/14866057). Alternatively you can use the synthetic data that has been created from the distributions of real patient data. If you don't have the public data, change the argument in the cell below from `data_folder_name='data-public'` to `data_folder_name='data-synthetic'`.

```python
from patientflow.load import set_file_paths

# set file paths
data_folder_name = 'data-public'
data_file_path = project_root / data_folder_name

data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
    project_root,
    data_folder_name=data_folder_name,
    config_file = 'config.yaml', verbose=False)
```

```python
import pandas as pd
from patientflow.load import load_data

# load ED snapshots data
ed_visits = load_data(data_file_path,
                    file_name='ed_visits.csv',
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"],
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])
ed_visits.snapshot_date = pd.to_datetime(ed_visits.snapshot_date).dt.date

# load data on inpatient arrivals
inpatient_arrivals = inpatient_arrivals = load_data(data_file_path,
                    file_name='inpatient_arrivals.csv')
inpatient_arrivals['arrival_datetime'] = pd.to_datetime(inpatient_arrivals['arrival_datetime'], utc = True)

```

## Set modelling parameters

The parameters are used in training or inference. They are set in config.json in the root of the repository and loaded by `load_config_file()`

```python
# load params
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

```

## Apply temporal splits

```python
from patientflow.prepare import create_temporal_splits

train_visits_df, valid_visits_df, test_visits_df = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
)

train_inpatient_arrivals_df, _, test_inpatient_arrivals_df = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
)
```

    Split sizes: [62071, 10415, 29134]
    Split sizes: [7716, 1285, 3898]

## Train models to predict bed count distributions for patients currently in the ED

This time I'll use a larger parameter grid, while still limiting the search space to a few hyperparameters for expediency.

```python

from patientflow.train.classifiers import train_classifier
from patientflow.load import get_model_key

grid = { # Current parameters
    'n_estimators': [30, 40, 50],  # Number of trees
    'subsample': [0.7, 0.8, 0.9],  # Sample ratio of training instances
    'colsample_bytree': [0.7, 0.8, 0.9],  # Sample ratio of columns for each tree
   }

exclude_from_training_data = [ 'snapshot_date', 'prediction_time','visit_number', 'consultation_sequence', 'specialty', 'final_sequence', ]

ordinal_mappings = {
    "latest_acvpu": ["A", "C", "V", "P", "U"],
    "latest_obs_manchester_triage_acuity": [
        "Blue",
        "Green",
        "Yellow",
        "Orange",
        "Red",
    ],
    "latest_obs_objective_pain_score": [
        "Nil",
        "Mild",
        "Moderate",
        "Severe\\E\\Very Severe",
    ],
    "latest_obs_level_of_consciousness": ["A", "C", "V", "P", "U"],
}

# create a dictionary to store the trained models
admissions_models = {}
model_name = 'admissions'

# Loop through each prediction time
for prediction_time in ed_visits.prediction_time.unique():
    print(f"Training model for {prediction_time}")
    model = train_classifier(
        train_visits=train_visits_df,
        valid_visits=valid_visits_df,
        test_visits=test_visits_df,
        grid=grid,
        exclude_from_training_data=exclude_from_training_data,
        ordinal_mappings=ordinal_mappings,
        prediction_time=prediction_time,
        visit_col="visit_number",
        calibrate_probabilities=True,
        calibration_method="isotonic",
        use_balanced_training=True,
    )
    model_key = get_model_key(model_name, prediction_time)

    admissions_models[model_key] = model
```

    Training model for (22, 0)
    Training model for (15, 30)
    Training model for (6, 0)
    Training model for (12, 0)
    Training model for (9, 30)

## Train specialty model

Here, when training the model predicting specialty of admission, the `apply_special_category_filtering` parameter has been set to True, so it will be assumed that all patients under 18 on arrival will be admitted to a paediatric specialty.

```python
from patientflow.predictors.sequence_predictor import SequencePredictor

spec_model = SequencePredictor(
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=True,
)

spec_model = spec_model.fit(train_visits_df)
```

## Train models for yet-to-arrive patients

```python
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor
from patientflow.prepare import create_yta_filters

x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
prediction_window = params["prediction_window"]
yta_time_interval = params["yta_time_interval"]

specialty_filters = create_yta_filters(ed_visits)
yta_model_by_spec =  WeightedPoissonPredictor(filters = specialty_filters, verbose=False)

# calculate the number of days between the start of the training and validation sets; used for working out daily arrival rates
num_days = (start_validation_set - start_training_set).days

if 'arrival_datetime' in train_inpatient_arrivals_df.columns:
    train_inpatient_arrivals_df.set_index('arrival_datetime', inplace=True)

yta_model_by_spec =yta_model_by_spec.fit(train_inpatient_arrivals_df,
              prediction_window=prediction_window,
              yta_time_interval=yta_time_interval,
              prediction_times=ed_visits.prediction_time.unique(),
              num_days=num_days )
```

## Generate predicted distributions for each specialty and prediction time for patients in ED

As we are treating paediatric patients differently from adults, the code below includes logic to identify eligible snapshots when caclculating for paediatric versus adult subgroups. When evaluating the predictions for adult destinations (medical, surgical and haem/onc), patients under 18 will be excluded. When evaluating the predictions for paediatric patients, adults will be excluded.

```python
from patientflow.prepare import prepare_patient_snapshots, prepare_group_snapshot_dict
from patientflow.aggregate import get_prob_dist
from patientflow.predict.emergency_demand import get_specialty_probs

prob_dist_dict_all = {}
specialties = ['medical', 'surgical', 'haem/onc', 'paediatric']

special_params = spec_model.special_params

special_category_func = special_params["special_category_func"]
special_category_dict = special_params["special_category_dict"]
special_func_map = special_params["special_func_map"]

# Get predictions of admission to specialty
test_visits_df.loc[:, "specialty_prob"] = get_specialty_probs(
    specialties,
    spec_model,
    test_visits_df,
    special_category_func=special_category_func,
    special_category_dict=special_category_dict,
)

# initialise a dictionary to store the probability distributions for each specialty and prediction time
prob_dist_dict_all = {}

# Process each time of day
for _prediction_time in ed_visits.prediction_time.unique():

    prob_dist_dict_for_pats_in_ED = {}

    print("\nProcessing :" + str(_prediction_time))
    model_key = get_model_key(model_name, _prediction_time)

    for specialty in specialties:
        print(f"Predicting bed counts for {specialty} specialty, for all snapshots in the test set")
        prob_dist_dict_for_specialty = {}

        # Get indices of patients who are eligible for this specialty
        func = special_func_map.get(specialty, special_func_map["default"])
        non_zero_indices = test_visits_df[
            test_visits_df.apply(func, axis=1)
        ].index

        test_df_eligible = test_visits_df.copy()
        test_df_eligible = test_df_eligible.loc[non_zero_indices]

        # get the probability of admission to the specialty for eligible patients
        # for patients under 18, the probability of admission to paediatric is 1
        prob_admission_to_specialty = test_df_eligible["specialty_prob"].apply(
            lambda x: x[specialty]
        )

        # Prepare patient snapshots for the eligible patients
        X_test, y_test = prepare_patient_snapshots(
            df=test_df_eligible,
            prediction_time=_prediction_time,
            single_snapshot_per_visit=False,
            exclude_columns=exclude_from_training_data,
            visit_col='visit_number'
        )

        # filter the probability of admission to the specialty for the eligible patients
        filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[
            non_zero_indices
        ]

        group_snapshots_dict = prepare_group_snapshot_dict(
            test_df_eligible[test_df_eligible.prediction_time == _prediction_time]
        )

        admitted_to_specialty = test_df_eligible['specialty'] == specialty

        # get probability distribution for this time of day, for this specialty, for patients in ED
        prob_dist_dict_for_pats_in_ED[specialty] = get_prob_dist(
            group_snapshots_dict, X_test, y_test, admissions_models[model_key],
            weights=filtered_prob_admission_to_specialty,
            category_filter=admitted_to_specialty,
            normal_approx_threshold=30
            )


    prob_dist_dict_all[f'{model_key}'] = prob_dist_dict_for_pats_in_ED

```

    Processing :(22, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(15, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(6, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(12, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(9, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

## Visualise the performance of emergency demand prediction models for patients in the ED

Below I show various ways to evaluate the predicted distributions.

- histograms of observed versus expected values
- QQ plots
- Adjusted QQ plots

I have included the unadjusted QQ plots to illustrate that they are not useful when making predictions over small ranges.

See [a previous notebook](3b_Evaluate_group_snapshots.md) for more on these approaches.

```python
from patientflow.evaluate import calc_mae_mpe
from patientflow.viz.evaluation_plots import plot_observed_against_expected

for specialty in specialties:

    specialty_prob_dist = {time: dist_dict[specialty] for time, dist_dict in prob_dist_dict_all.items()}
    results = calc_mae_mpe(specialty_prob_dist)
    plot_observed_against_expected(results,
                                   main_title=f"Histograms of Observed - Expected Values for {specialty} specialty",)

```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_20_0.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_20_1.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_20_2.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_20_3.png)

```python
from patientflow.viz.qq_plot import qq_plot

for specialty in specialties:

    specialty_prob_dist = {time: dist_dict[specialty] for time, dist_dict in prob_dist_dict_all.items()}

    qq_plot(ed_visits.prediction_time.unique(),
            specialty_prob_dist,
            model_name="admissions",
            suptitle=f"QQ plot for {specialty} specialty")
```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_21_0.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_21_1.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_21_2.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_21_3.png)

These QQ Plots demonstrate the problems mentioned [previously](3b_Evaluate_group_snapshots.md) for small ranges of observed values. In particular, the haematology/oncology and paediatric models show a very marked pattern of sharp vertical jumps. That is becuase on roughly half of the dates in the test set, the actual number of patients admitted to haematology/oncology and paediatric specialties was zero. A QQ plot, being based on the assumption of continuous distributions, cannot reveal what a correct model would show.

For these reasons, the Adjusted QQ Plots, which are shown below, are more useful.

```python
from patientflow.viz.adjusted_qq_plot import adjusted_qq_plot

for specialty in specialties:

    specialty_prob_dist = {time: dist_dict[specialty] for time, dist_dict in prob_dist_dict_all.items()}

    adjusted_qq_plot(ed_visits.prediction_time.unique(),
            specialty_prob_dist,
            model_name="admissions",
            suptitle=f"Adjusted QQ plot for {specialty} specialty")
```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_23_0.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_23_1.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_23_2.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_23_3.png)

In medical specialties (which have the highest numbers of admissions, accounting for significant majority) the model performance is similar to that seen when not sub-divided by specialty. (See [a previous notebook](3b_Evaluate_group_snapshots.md) for more on this.) The model underestimates beds needed for patients in the ED at 22:00 and overestimates at 12:00 and 15:30.

In plots for all other specialties it looks as if the model lines begin part way up the x axis, but there are points stacked below this, representing the high proportion of instances where the most likely number of admissions is zero. For haematology/oncology and paediatrics, the observed values track the model predictions very well, for all times of day. This is also true for surgery, but with greater variation.

## Evaluate predictions for patients yet-to-arrive to the ED

As the predictions for yet-to-arrive patients are aspirational, these cannot be directly compared with observed numbers of patients who arrived after the moment of prediction, and were admitted within the prediction window. Due to slower than target processing of patients through the ED, these future arrivals would have a smaller likelihood of being admitted within the window than suggested by the aspirational target.

We can, however, compare the predictions based on arrival rates at the front door of ED, that were learned from the training set, against observed arrival rates at the front door during the test set.

To illustrate, I start by plotting the cumulative arrivals of patients later admitted within a prediction window on one date. The blue line shows the cumulative number of arrivals. The orange lines shows the cumulative mean arrival rate.

```python
from patientflow.viz.arrival_comparison import plot_arrival_comparison
plot_arrival_comparison(test_inpatient_arrivals_df, (22,0), start_test_set, prediction_window, show_delta=False, yta_time_interval = 15)


# plt.show()
```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_26_0.png)

The chart below show the delta between the two lines

```python
plot_arrival_comparison(test_inpatient_arrivals_df, (22,0), start_test_set, prediction_window, show_only_delta=True, yta_time_interval = 60)

```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_28_0.png)

The chart below shows multiple version of the delta for each date in the test set, for each prediction time, with the average delta shown in red.

```python
from patientflow.viz.arrival_comparison import plot_multiple_deltas
from datetime import timedelta


start_date = start_test_set
end_date = end_test_set
snapshot_dates = []

current_date = start_date
while current_date < end_date:
    snapshot_dates.append(current_date)
    current_date += timedelta(days=1)

        # Sort prediction times by converting to minutes since midnight
prediction_times_sorted = sorted(
    ed_visits.prediction_time.unique(),
    key=lambda x: x[0] * 60 + x[1],  # Convert (hour, minute) to minutes since midnight
)

for prediction_time in prediction_times_sorted:
    plot_multiple_deltas(test_inpatient_arrivals_df, prediction_time, snapshot_dates, prediction_window, yta_time_interval=15)
```

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_30_0.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_30_1.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_30_2.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_30_3.png)

![png](4c_Evaluate_emergency_demand_predictions_files/4c_Evaluate_emergency_demand_predictions_30_4.png)

## Conclusion

In this notebook I have shown how to evaluate predicted bed counts for the patients in ED, by specialty, and for the patients yet-to-arrive. Both approaches required adjustments for the fact that the predicted distributions are aspirational.

These models are based on a relatively small dataset (six months of training data, a one month validation set and a 3 month test set). In the real-time application at UCLH, we use much more training data, and we also have the benefit of some additional features (such as age as a continuous variable) which improves model performance.

Nonetheless, the models perform well on the relatively small datasets made available here.
