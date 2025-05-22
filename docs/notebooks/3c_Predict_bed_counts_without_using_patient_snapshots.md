# 3c. Predict bed counts without using patient snapshots

There are situations where we might want to predict bed count distributions without having details of the patients, for example when predicting the number of beds needed:

- for patients yet-to-arrive to the Emergency Department, who will need a bed within a prediction window
- for emergency patients who arrive via other routes than the ED, and become inpatients (such as emergency transfers from other hospitals)
- for elective admissions of patients. Elective patients may be on a 'To Come In' list, but often their encounter for the elective procedure begins at the moment they arrive. In a simple case without making use of any data on TCI lists, we might want to predict based on past patterns of such arrivals between a prediction time (eg 09:30) and the end of a prediction window (eg 8 hours later).

For these situations, you can use `patientflow` to learn patterns from past data, and use these to predict a bed count distribution at the aggregate level.

In this notebook, I'll use the example of predicting the number of beds needed for patients yet to arrive to the Emergency Department. I'll start by creating a very simple model trained on past data to predict the number of patients.

I will also include an example of a custom function developed to predict number of patients yet to arrive, who will be admitted within a prediction window assuming ED targets are met.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Create fake arrival times

I will generate some fake data on patients in an Emergency Department (ED) using the same method as in previous notebooks.

```python
from patientflow.generate import create_fake_finished_visits
visits_df, _, _ = create_fake_finished_visits('2023-01-01', '2023-04-01', 25)
visits_df.head()
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
      <th>patient_id</th>
      <th>visit_number</th>
      <th>arrival_datetime</th>
      <th>departure_datetime</th>
      <th>is_admitted</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1658</td>
      <td>14</td>
      <td>2023-01-01 03:31:47</td>
      <td>2023-01-01 08:00:47</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>238</td>
      <td>20</td>
      <td>2023-01-01 04:25:57</td>
      <td>2023-01-01 07:43:57</td>
      <td>1</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>354</td>
      <td>1</td>
      <td>2023-01-01 05:21:43</td>
      <td>2023-01-01 08:52:43</td>
      <td>1</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>114</td>
      <td>3</td>
      <td>2023-01-01 08:01:26</td>
      <td>2023-01-01 09:38:26</td>
      <td>0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>497</td>
      <td>10</td>
      <td>2023-01-01 08:20:52</td>
      <td>2023-01-01 11:20:52</td>
      <td>0</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>

For this analysis I only want to make predictions for patienets who are later admitted, so I will delete the non-admitted.

```python
import pandas as pd

inpatient_arrivals = visits_df[visits_df.is_admitted == 1].rename(columns = {'departure_datetime': 'admitted_to_ward_datetime'}).drop(columns = 'is_admitted')
inpatient_arrivals['arrival_datetime'] = pd.to_datetime(inpatient_arrivals['arrival_datetime'])
```

I will generate an array of dates covered by the data I've loaded. I'm calling these `snapshot_dates` for consistency.

```python
from datetime import datetime, time, timedelta, date

# Create date range
snapshot_dates = []
start_date = date(2023, 1, 1)
end_date = date(2023, 4, 1)

current_date = start_date
while current_date < end_date:
    snapshot_dates.append(current_date)
    current_date += timedelta(days=1)

print('First ten snapshot dates')
snapshot_dates[0:10]
```

    First ten snapshot dates





    [datetime.date(2023, 1, 1),
     datetime.date(2023, 1, 2),
     datetime.date(2023, 1, 3),
     datetime.date(2023, 1, 4),
     datetime.date(2023, 1, 5),
     datetime.date(2023, 1, 6),
     datetime.date(2023, 1, 7),
     datetime.date(2023, 1, 8),
     datetime.date(2023, 1, 9),
     datetime.date(2023, 1, 10)]

## Train a simple Poisson model to predict the patients who are yet-to-arrive

The function below generates counts by snapshot date for the number of patients who arrived after the prediction time and were admitted before the end of the prediction window.

```python

def count_yet_to_arrive(df, snapshot_dates, prediction_times, prediction_window_hours):
    """
    Count patients who arrived after a prediction time and were admitted to a ward
    within a specified window.
    """
    # Create an empty list to store results
    results = []

    # For each combination of date and time
    for date_val in snapshot_dates:
        for hour, minute in prediction_times:
            # Create the prediction datetime
            prediction_datetime = pd.Timestamp(datetime.combine(date_val, time(hour=hour, minute=minute)))

            # Calculate the end of the prediction window
            prediction_window_end = prediction_datetime + pd.Timedelta(hours=prediction_window_hours)

            # Count patients who arrived after prediction time and were admitted within the window
            admitted_within_window = df[
                (df['arrival_datetime'] > prediction_datetime) &
                (df['admitted_to_ward_datetime'] <= prediction_window_end)
            ]['patient_id'].nunique()

            # Store the result
            results.append({
                'snapshot_date': date_val,
                'prediction_time': (hour, minute),
                'count': admitted_within_window
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
```

As in previous notebooks, I'll apply a temporal split to the data.

```python
from datetime import date
from patientflow.prepare import create_temporal_splits

# set the temporal split
start_training_set = date(2023, 1, 1)
start_validation_set = date(2023, 2, 15) # 6 week training set
start_test_set = date(2023, 3, 1) # 2 week validation set
end_test_set = date(2023, 4, 1) # 1 month test set

# create the temporal splits
train_visits, _, _ = create_temporal_splits(
    inpatient_arrivals,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime", # states which column contains the date to use when making the splits

)
```

    Split sizes: [313, 109, 206]

After applying the function, the count data is shown below.

```python
prediction_times = [(9, 30)]

yet_to_arrive_counts = count_yet_to_arrive(train_visits, snapshot_dates, prediction_times, prediction_window_hours=8)
yet_to_arrive_counts.head(10)
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-02</td>
      <td>(9, 30)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-03</td>
      <td>(9, 30)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-04</td>
      <td>(9, 30)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-05</td>
      <td>(9, 30)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-01-06</td>
      <td>(9, 30)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-01-07</td>
      <td>(9, 30)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-01-08</td>
      <td>(9, 30)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-01-09</td>
      <td>(9, 30)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-01-10</td>
      <td>(9, 30)</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

Here I use the mean daily count as the mean of a Poisson distribution.

```python
from scipy import stats
poisson_mean = yet_to_arrive_counts['count'].mean()
poisson_model = stats.poisson(poisson_mean)
```

I use the Poisson model to predict a bed count distribution for the patients yet-to-arrive.

```python
prob_dist_data = [poisson_model.pmf(k) for k in range(20)]

from patientflow.viz.prob_dist_plot import prob_dist_plot
from patientflow.viz.utils import format_prediction_time
title = (
    f'Probability distribution for number of beds needed for patients'
    f'\nwho will arrive after {format_prediction_time((9,30))} on {snapshot_dates[0]} and need a bed within 8 hours'
)
prob_dist_plot(prob_dist_data, title,
    include_titles=True)
```

![png](3c_Predict_bed_counts_without_using_patient_snapshots_files/3c_Predict_bed_counts_without_using_patient_snapshots_17_0.png)

## Train a weighted Poisson model

The model above has learned the rates of arrivals of patients who are later admitted within a prediction window from past data.

A problem with this approach is that rates are learned from periods of poor performance. Currently, in England Emergency Departments have a target of processing all patients within four hours of their arrival time. However, EDs across the country have not hit targets since the end of the Covid pandemic.

The poor performance is illustrated by the survival curve below, which shows that only 59% of admitted patients left the ED to go to the ward within four hours.

```python
from patientflow.viz.survival_curves import plot_admission_time_survival_curve
title = 'Survival curve showing probability of still being in the ED after a given elapsed time since arrival'
plot_admission_time_survival_curve(inpatient_arrivals, title)
```

    Proportion of patients admitted within 4 hours: 58.76%

![png](3c_Predict_bed_counts_without_using_patient_snapshots_files/3c_Predict_bed_counts_without_using_patient_snapshots_19_1.png)

`patientflow` offers a weighted Poisson model, that will calculate each patient's probability of being admitted from their arrival time, if targets are met. Targets are set using the parameters set in config.yaml

```python
from patientflow.load import load_config_file, set_file_paths, set_project_root
project_root = set_project_root()

_, _, _, config_path = set_file_paths(project_root, data_folder_name = 'data-public', verbose = False)
params = load_config_file(config_path)

x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]

print(f'The aspiration is that within {str(x1)} hours of arrival, {str(y1*100)}% of patients will have been admitted, and that witin {str(x2)} hours of arrival, {str(y2*100)}% of patients will have been admitted')
```

    Inferred project root: /Users/zellaking/Repos/patientflow
    The aspiration is that within 4.0 hours of arrival, 76.0% of patients will have been admitted, and that witin 12.0 hours of arrival, 99.0% of patients will have been admitted

The aspiration can be plotted as an inverted survival curve, as shown below.

```python
from patientflow.viz.aspirational_curve_plot import plot_curve

figsize = (6,3)

plot_curve(
    title = 'Aspirational curve reflecting a ' + str(int(x1)) + ' hour target for ' + str(int(y1*100)) + \
        '% of patients\nand a '+ str(int(x2)) + ' hour target for ' + str(int(y2*100)) + '% of patients',
    x1 = x1,
    y1 = y1,
    x2 = x2,
    y2 = y2,
    include_titles=True,
    annotate_points=True,
)

```

![png](3c_Predict_bed_counts_without_using_patient_snapshots_files/3c_Predict_bed_counts_without_using_patient_snapshots_23_0.png)

Below I demonstrate the use of a Weighted Poisson predictor.

Its `fit()` method will, for each prediction time:

- filter the dataframe if a filtering criteria is given (more detail below)
- calculate arrival rates for a series of discrete time intervals (where the duration of each time interval is specified as `yta_time_interval` minutes) within a 24 hour period; `yta_time_interval` must divide evenly into a 24 hour period (ie be a factor of 24\* 60)
- return the arrival rates for the intervals between the prediction time and the end of the prediction window in a dictionary; if the data is unfiltered it will use a generic key of 'unfiltered'; if the data is filtered, it will use the filters as keys

The `predict()` method will:

- retrieve the arrival rates saved for the prediction window
- for each discrete time interval, using the aspirational curve introduced above, and taking into account the time remaining before end of window, calculate a probability of admission in prediction window
- weight the arrival rates for each time interval by this probability
- generate a Poisson distribution for each time interval
- convolute the distributions to return a single distribution for arrivals over all time intervals

```python
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor

yta_model =  WeightedPoissonPredictor(verbose=True)
num_days = (start_validation_set - start_training_set).days
if 'arrival_datetime' in train_visits.columns:
    train_visits.set_index('arrival_datetime', inplace=True)

yta_model.fit(train_visits, prediction_window=8*60, yta_time_interval=15, prediction_times=[(9,30)], num_days=num_days)

```

    Calculating time-varying arrival rates for data provided, which spans 45 unique dates
    Weighted Poisson Predictor trained for these times: [(9, 30)]
    using prediction window of 480 minutes after the time of prediction
    and time interval of 15 minutes within the prediction window.
    The error value for prediction will be 1e-07
    To see the weights saved by this model, used the get_weights() method

<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>WeightedPoissonPredictor(filters={}, verbose=True)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>WeightedPoissonPredictor</div></div><div><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></div></label><div class="sk-toggleable__content "><pre>WeightedPoissonPredictor(filters={}, verbose=True)</pre></div> </div></div></div></div>

Below we view the results of the fit method for the 09:30 prediction time.

```python
arrival_rates_by_time_interval = yta_model.weights['unfiltered'][(9,30)]['arrival_rates']
print(
    f'The calculated arrival rates for the first 10 discrete time intervals '
    f'for the 09:30 prediction time are: {[round(v, 3) for v in arrival_rates_by_time_interval[0:10]]}')
```

    The calculated arrival rates for the first 10 discrete time intervals for the 09:30 prediction time are: [0.067, 0.111, 0.111, 0.133, 0.133, 0.133, 0.333, 0.133, 0.178, 0.111]

To use the weighted poisson for prediction, a `prediction_context` argument specifies the required prediction time and filtering.

```python
from patientflow.viz.prob_dist_plot import prob_dist_plot
from patientflow.viz.utils import format_prediction_time

prediction_context = {
    'unfiltered': {
        'prediction_time': tuple([9,30])
    }
}

weighted_poisson_prediction = yta_model.predict(prediction_context, x1, y1, x2, y2)

```

The chart below show the results of using this weighted predictor to generate an unfettered distribution for patients yet-to-arrive. The numbers are higher than the equivalent chart above.

```python
title = (
    f'Probability distribution for number of beds needed for patients '
    f'who will arrive after {format_prediction_time((9,30))} on {snapshot_dates[0]} '
    f'\nand need a bed within 8 hours '
    f'if the ED is meeting the target of {int(x1)} hours for {y1*100}% of patients'
)
prob_dist_plot(weighted_poisson_prediction['unfiltered'], title,
    include_titles=True,
    truncate_at_beds=20)
```

![png](3c_Predict_bed_counts_without_using_patient_snapshots_files/3c_Predict_bed_counts_without_using_patient_snapshots_31_0.png)

## Conclusion

Here I have demonstrated the use of `patientflow` to generate bed counts for groups of patients, without using patient snapshots.

If you have count data on past visits that approximate to a statistical distribution, preparing a model to predict a bed count distribution is simple to do with standard libraries like scipy. You don't need `patientflow` functions for that.

However there might be cases where the historical data don't reflect the desired performance of the ED, as in the example shown here. In that case, the users of your predictions might be more interested in understanding their unfettered demand. `patientflow` provides functions that enable you to produce such predictions.
