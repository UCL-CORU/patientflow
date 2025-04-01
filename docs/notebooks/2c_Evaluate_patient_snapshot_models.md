# Evaluate models trained on patient-level snapshots

## Things to consider

In the last notebook, I showed how to train models on patient snapshots using `patientflow`. Now, let's think about how to evaluate those models.

When evaluating patient snapshots, we focus on:

- How well-calibrated the predicted probabilities are
- The distribution of these probabilities

We don't focus as much on typical classification metrics like Area under the ROC curve, accuracy or precision/recall.

### Why don't we focus on typical classification metrics?

The ultimate goal is to predict bed count distributions for groups of patients. Bed count distributions will be calculated in two steps

1. First, we predict the probability of the outcome we are interested in (admission or discharge) for each individual patient, as shown in previous notebooks.
2. Then, we use these probabilities in Bernoulli trials to get bed count distributions. The Bernouill trials step will be shown in later notebooks.

Because of this approach, the accuracy of the probability values matters more than correct classification. That is why we use log loss to optimise our classifiers.

### About the data used in this notebook

I'm going to use real patient data from visits to the Emergency Department (ED) and Same Day Emergency Care (SDEC) unit at UCLH to demonstrate the evaluation. The methods shown will work on any data in the same structure.

You can request the datasets that are used here on [Zenodo](https://zenodo.org/records/14866057). Alternatively you can use the synthetic data that has been created from the distributions of real patient data. If you don't have the public data, change the argument in the cell below from `data_folder_name='data-public'` to `data_folder_name='data-synthetic'`.

## Loading real patient data

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

The function below identifies the root of the patientflow repository, in order to locate the folders containing data.

```python
import pandas as pd
from patientflow.load import set_file_paths, load_data

# set project root
from patientflow.load import set_project_root
project_root = set_project_root()

# set file paths
data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
        project_root,
        data_folder_name='data-public', # change this to data-synthetic if you don't have the public dataset
        verbose=False)

# load the data
ed_visits = load_data(data_file_path,
                    file_name='ed_visits.csv',
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"],
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])


```

    Inferred project root: /Users/zellaking/Repos/patientflow

Inspecting the data that has been loaded, we can see that it is similar in structure to the fake data that was generated on the fly in the previous notebooks. The dates have been pushed into the future, to minimise the likelihood of re-identifcation of patients.

```python
ed_visits.head()
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
      <th>snapshot_date</th>
      <th>prediction_time</th>
      <th>visit_number</th>
      <th>elapsed_los</th>
      <th>sex</th>
      <th>age_group</th>
      <th>arrival_method</th>
      <th>current_location_type</th>
      <th>total_locations_visited</th>
      <th>num_obs</th>
      <th>num_obs_events</th>
      <th>num_obs_types</th>
      <th>num_lab_batteries_ordered</th>
      <th>has_consultation</th>
      <th>consultation_sequence</th>
      <th>visited_majors</th>
      <th>visited_otf</th>
      <th>visited_paeds</th>
      <th>visited_rat</th>
      <th>visited_resus</th>
      <th>visited_sdec</th>
      <th>visited_sdec_waiting</th>
      <th>visited_unknown</th>
      <th>visited_utc</th>
      <th>visited_waiting</th>
      <th>num_obs_blood_pressure</th>
      <th>num_obs_pulse</th>
      <th>num_obs_air_or_oxygen</th>
      <th>num_obs_glasgow_coma_scale_best_motor_response</th>
      <th>num_obs_level_of_consciousness</th>
      <th>num_obs_news_score_result</th>
      <th>num_obs_manchester_triage_acuity</th>
      <th>num_obs_objective_pain_score</th>
      <th>num_obs_subjective_pain_score</th>
      <th>num_obs_temperature</th>
      <th>num_obs_oxygen_delivery_method</th>
      <th>num_obs_pupil_reaction_right</th>
      <th>num_obs_oxygen_flow_rate</th>
      <th>num_obs_uclh_sskin_areas_observed</th>
      <th>latest_obs_pulse</th>
      <th>latest_obs_respirations</th>
      <th>latest_obs_level_of_consciousness</th>
      <th>latest_obs_news_score_result</th>
      <th>latest_obs_manchester_triage_acuity</th>
      <th>latest_obs_objective_pain_score</th>
      <th>latest_obs_temperature</th>
      <th>lab_orders_bc</th>
      <th>lab_orders_bon</th>
      <th>lab_orders_crp</th>
      <th>lab_orders_csnf</th>
      <th>lab_orders_ddit</th>
      <th>lab_orders_ncov</th>
      <th>lab_orders_rflu</th>
      <th>lab_orders_xcov</th>
      <th>latest_lab_results_crea</th>
      <th>latest_lab_results_hctu</th>
      <th>latest_lab_results_k</th>
      <th>latest_lab_results_lac</th>
      <th>latest_lab_results_na</th>
      <th>latest_lab_results_pco2</th>
      <th>latest_lab_results_ph</th>
      <th>latest_lab_results_wcc</th>
      <th>latest_lab_results_alb</th>
      <th>latest_lab_results_htrt</th>
      <th>training_validation_test</th>
      <th>final_sequence</th>
      <th>is_admitted</th>
      <th>random_number</th>
      <th>specialty</th>
    </tr>
    <tr>
      <th>snapshot_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/17/2031</td>
      <td>(12, 0)</td>
      <td>30767</td>
      <td>1920</td>
      <td>F</td>
      <td>55-64</td>
      <td>Ambulance</td>
      <td>majors</td>
      <td>4</td>
      <td>107</td>
      <td>34</td>
      <td>34</td>
      <td>4</td>
      <td>False</td>
      <td>[]</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>71.0</td>
      <td>16.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>Yellow</td>
      <td>Nil</td>
      <td>98.2</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>train</td>
      <td>[]</td>
      <td>False</td>
      <td>15795</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/17/2031</td>
      <td>(15, 30)</td>
      <td>30767</td>
      <td>14520</td>
      <td>F</td>
      <td>55-64</td>
      <td>Ambulance</td>
      <td>majors</td>
      <td>5</td>
      <td>138</td>
      <td>39</td>
      <td>34</td>
      <td>6</td>
      <td>False</td>
      <td>[]</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48.0</td>
      <td>16.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>Yellow</td>
      <td>Nil</td>
      <td>98.1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>57.0</td>
      <td>0.422</td>
      <td>3.8</td>
      <td>1.0</td>
      <td>138.0</td>
      <td>4.61</td>
      <td>7.474</td>
      <td>8.77</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>train</td>
      <td>[]</td>
      <td>False</td>
      <td>860</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12/10/2031</td>
      <td>(15, 30)</td>
      <td>36297</td>
      <td>9180</td>
      <td>M</td>
      <td>75-102</td>
      <td>NaN</td>
      <td>majors</td>
      <td>4</td>
      <td>127</td>
      <td>12</td>
      <td>37</td>
      <td>8</td>
      <td>False</td>
      <td>[]</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>63.0</td>
      <td>22.0</td>
      <td>A</td>
      <td>2.0</td>
      <td>Orange</td>
      <td>Nil</td>
      <td>97.5</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>97.0</td>
      <td>0.483</td>
      <td>4.1</td>
      <td>1.2</td>
      <td>140.0</td>
      <td>4.82</td>
      <td>7.433</td>
      <td>6.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>[]</td>
      <td>False</td>
      <td>76820</td>
      <td>surgical</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3/28/2031</td>
      <td>(6, 0)</td>
      <td>53554</td>
      <td>2220</td>
      <td>F</td>
      <td>35-44</td>
      <td>Public Trans</td>
      <td>rat</td>
      <td>3</td>
      <td>356</td>
      <td>101</td>
      <td>57</td>
      <td>5</td>
      <td>False</td>
      <td>[]</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>15</td>
      <td>16</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70.0</td>
      <td>17.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>Green</td>
      <td>Mild</td>
      <td>97.7</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>train</td>
      <td>[]</td>
      <td>False</td>
      <td>54886</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3/28/2031</td>
      <td>(9, 30)</td>
      <td>53554</td>
      <td>14820</td>
      <td>F</td>
      <td>35-44</td>
      <td>Public Trans</td>
      <td>majors</td>
      <td>4</td>
      <td>375</td>
      <td>107</td>
      <td>57</td>
      <td>7</td>
      <td>False</td>
      <td>[]</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>16</td>
      <td>17</td>
      <td>10</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65.0</td>
      <td>17.0</td>
      <td>A</td>
      <td>0.0</td>
      <td>Green</td>
      <td>Mild</td>
      <td>97.7</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>68.0</td>
      <td>0.379</td>
      <td>4.1</td>
      <td>1.6</td>
      <td>139.0</td>
      <td>4.00</td>
      <td>7.536</td>
      <td>13.03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>train</td>
      <td>[]</td>
      <td>False</td>
      <td>6265</td>
      <td>medical</td>
    </tr>
  </tbody>
</table>
</div>

The dates for training, validation and test sets that match this dataset are defined in the config file in the root directory of `patientflow`.

```python
#
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set = params["start_training_set"]
print(f"Training set starts: {start_training_set}")

start_validation_set = params["start_validation_set"]
print(f"Validation set starts: {start_validation_set}")

start_test_set = params["start_test_set"]
print(f"Test set starts: {start_test_set}")

end_test_set = params["end_test_set"]
print(f"Test set ends: {end_test_set}")

```

    Training set starts: 2031-03-01
    Validation set starts: 2031-09-01
    Test set starts: 2031-10-01
    Test set ends: 2032-01-01

## Train one model for each prediction time

First, we apply the temporal splits as shown in the previous notebook.

```python


from datetime import date
from patientflow.prepare import create_temporal_splits

# create the temporal splits
train_visits, valid_visits, test_visits = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date", # states which column contains the date to use when making the splits
    visit_col="visit_number", # states which column contains the visit number to use when making the splits

)

```

    Split sizes: [53801, 6519, 19494]

Next we train a model for each prediction time.

```python
prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)]

print("\nNumber of observations for each prediction time")
print(ed_visits.prediction_time.value_counts())
```

    Number of observations for each prediction time
    prediction_time
    (15, 30)    22279
    (12, 0)     19075
    (22, 0)     18842
    (9, 30)     11421
    (6, 0)       8197
    Name: count, dtype: int64

As shown in the previous notebook, we define ordinal mappings where appropriate. These include:

- `age_group` - Age on arrival at the ED, defined in groups
- `latest_obs_manchester_triage_acuity` - Manchester Triage Score (where blue is the lowest acuity and red the highest)
- `latest_obs_objective_pain_score` - ranging from nil to very severe
- `latest_obs_level_of_consciousness` the ACVPU measure of consciousness, where A (aware) and U (unconscious) at are the extremes.

```python
ordinal_mappings = {
    "age_group": [
        "0-17",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75-102",
    ],
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
        "Severe_Very Severe",
    ],
    "latest_obs_level_of_consciousness": [
        "A", #alert
        "C", #confused
        "V", #voice - responds to voice stimulus
        "P", #pain - responds to pain stimulus
        "U" #unconscious - no response to pain or voice stimulus
    ]    }

```

In the real data, there are some columns that will be used for predicting admission to specialty, if admitted, that we don't use here.

```python
exclude_from_training_data = [ 'snapshot_date', 'prediction_time','visit_number', 'consultation_sequence', 'specialty', 'final_sequence', ]
```

We loop through each prediction time, training a model. To start with, we will not balance the dataset.

```python
from patientflow.train.classifiers import train_classifier

trained_models = []

# Loop through each prediction time
for prediction_time in prediction_times:
    print(f"Training model for {prediction_time}")
    model = train_classifier(
        train_visits=train_visits,
        valid_visits=valid_visits,
        test_visits=test_visits,
        grid={"n_estimators": [20, 30, 40]},
        exclude_from_training_data=exclude_from_training_data,
        ordinal_mappings=ordinal_mappings,
        prediction_time=prediction_time,
        visit_col="visit_number",
        calibrate_probabilities=False,
        use_balanced_training=False,
    )

    trained_models.append(model)
```

    (6, 0)
    (9, 30)
    (12, 0)
    (15, 30)
    (22, 0)

## Inspecting the base model

Below I show three different charts, all showing the calibration and distribution of the models, in slightly different ways.

### Distribution plots

A distribution plot shows the spread of predicted probabilities for positive and negative cases.

- X-axis (Predicted Probability): Represents the model's predicted probabilities from 0 to 1.
- Y-axis (Density): Shows the relative frequency of each probability value.

The plot displays two histograms:

- Blue line/area: Distribution of predicted probabilities for negative cases (patients who weren't admitted)
- Orange line/area: Distribution of predicted probabilities for positive cases (patients who were admitted)

Ideal separation between these distributions indicates a well-performing model:

- Negative cases (blue) should cluster toward lower probabilities (left side)
- Positive cases (orange) should cluster toward higher probabilities (right side)

The degree of overlap between distributions helps assess model discrimination ability. Less overlap suggests the model effectively distinguishes between positive and negative cases, while significant overlap indicates areas where the model struggles to differentiate between outcomes.

From the plot below, we see that the model is discriminating poorly, with a high degree of overlap, and very few positive cases at the higher end.

```python
# without balanced training
from patientflow.viz.distribution_plots import plot_prediction_distributions
plot_prediction_distributions(
    trained_models=trained_models,  # Convert dict values to list
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)

```

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_20_0.png)

### Calibration plots

A calibration plot shows how well a model's predicted probabilities match actual outcomes.

- X-axis (Mean Predicted Probability): The model's predicted probabilities, ordered from 0 to 1, grouped into bins, either using the uniform or the quantile strategy (see below).
- Y-axis (Fraction of Positives): The observed proportion of admissions for visits in that group.

A perfectly calibrated model would align its points along the diagonal line, meaning a 70% predicted probability means the event happens 70% of the time.

Uniform vs Quantile Strategies:

- Uniform: Divides predictions into equal-width probability bins (e.g., 0.0-0.1, 0.1-0.2), so some bins may have few or many points.
- Quantile: Ensures each bin has the same number of predictions, regardless of how wide or narrow each bin's probability range is.

Below, we see reasonable calibration at the lower end, but deteriorating towards the higher end.

```python
# without balanced training
from patientflow.viz.calibration_plot import plot_calibration

plot_calibration(
    trained_models=trained_models,  # Convert dict values to list
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    strategy="quantile",  # optional
    suptitle="Base model with imbalanced training data"  # optional
)
```

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_22_0.png)

### MADCAP (Model Accuracy Diagnostic Calibration Plot)

A MADCAP (Model Accuracy Diagnostic Calibration Plot) visually compares the predicted probabilities from a model with the actual outcomes (e.g., admissions or events) in a dataset. This plot helps to assess how well the model's predicted probabilities align with the observed values.

The blue line represents the cumulative predicted outcomes, which are derived by summing the predicted probabilities as we move through the test set, ordered by increasing probability.
The orange line represents the cumulative observed outcomes, calculated based on the actual labels in the dataset, averaged over the same sorted order of predicted probabilities.

If the model is well calibrated, these two lines will closely follow each other, and the curves will bow to the bottom left.

Below, we see that the models under-predict the likelihood of admissions, as the blue line (predicted outcomes) consistently falls below the orange line (actual outcomes). The models are systematically assigning lower probabilities than it should, meaning that (later) we will under-predict the number of beds needed for these patients.

```python
## without balanced training
from patientflow.viz.madcap_plot import generate_madcap_plots
generate_madcap_plots(
    trained_models=trained_models,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_24_0.png)

## Inspecting a balanced and calibrated model

In the previous notebook I showed that the `train_classifier()` function will balance the training set, by under-sampling the negative class, and then re-calibrate the data using the validation set. Below I train the models with these arguments set to true, and re-run the plots.

```python
from patientflow.train.classifiers import train_classifier

trained_models = []

# Loop through each prediction time
for prediction_time in prediction_times:
    print(f"Training model for {prediction_time}")
    model = train_classifier(
        train_visits=train_visits,
        valid_visits=valid_visits,
        test_visits=test_visits,
        grid={"n_estimators": [20, 30, 40]},
        exclude_from_training_data=exclude_from_training_data,
        ordinal_mappings=ordinal_mappings,
        prediction_time=prediction_time,
        visit_col="visit_number",
        calibrate_probabilities=True,
        calibration_method="sigmoid",
        use_balanced_training=True,
    )

    trained_models.append(model)


```

    Training model for (6, 0)
    Training model for (9, 30)
    Training model for (12, 0)
    Training model for (15, 30)
    Training model for (22, 0)

From the plots below, we see improved discrimination. There are positive cases clustered at the right hand end of the distribution plot, and the MADCAP lines are closer. The model slightly underpredicts at 06:00, 09:30 and 22:00, and slightly overpredicts at 12:00 and 15:30. These improvements have been achieved while maintaining good calibration.

```python
plot_prediction_distributions(
    trained_models=trained_models,  # Convert dict values to list
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
plot_calibration(
    trained_models=trained_models,  # Convert dict values to list
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    strategy="quantile",  # optional
    suptitle="Base model with imbalanced training data"  # optional
)

generate_madcap_plots(
    trained_models=trained_models,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_28_0.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_28_1.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_28_2.png)

## MADCAP plots by age

It can be useful to look at sub-categories of patients, to understand whether models perform better for some groups. Here we show MADCAP plots by age group.

The performance is worse for children over all. There are fewer of them in the data, which can be seen by comparing the y axis limits. The y axis maximum is the total number of snapshots in the test that were in at the prediction time. In general, there are twice as many adults as over 65s (except at 22:00), and very few children. The models perform poorly for children, and best for adults under 65. They tend to under-predict for older people, especially at 22:00 and 06:00.

Analysis like this helps understand the limitations of the modelling, and consider alternative approaches. For example, we might consider training a different model for older people, if there was enough data, or gathering more training data before deployment.

```python
from patientflow.viz.madcap_plot import generate_madcap_plots_by_group
generate_madcap_plots_by_group(
    trained_models=list(trained_models),
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    grouping_var="age_group",
    grouping_var_name="Age Group",
    plot_difference=False
)
```

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_30_0.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_30_1.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_30_2.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_30_3.png)

![png](2c_Evaluate_patient_snapshot_models_files/2c_Evaluate_patient_snapshot_models_30_4.png)

## Conclusion

Here I have shown how visualations within `patientflow` can help you

- assess the discrimination and calibration of your models
- identify areas of weakness in your models by comparing predictions across different patient groups

I have also shown how using balanced training set, and re-calibrating using the validation set, can help to improve the discrimination of models where you start with imbalanced data. This is common in healthcare data.

This notebook concludes the set covering patient snapshots. We have created predicted probabilities for each patient, based on what is known about them at the time of the snapshot. However, bed managers really want predictions for the whole cohort of patients at a time. This is where `patientflow` comes into its own. In the next notebook, I show how to create group snapshots.
