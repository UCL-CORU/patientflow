# 2d. Explore the datasets provided

Two datasets have been provided with this repository.

- `ed_visits.csv`
- `inpatient_arrivals.csv`

These accompany my fully worked example of the modelling of emergency demand for beds. And they are also useful to illustrate what patient snapshots might be made up of.

This notebook does some data exploration by plotting charts of all relevant variables in each dataset.

The `inpatient_arrivals` dataset contains arrival times of all patients who visited the UCLH Emergency Department (ED) and the Same Day Emergency Care (SDEC) unit, over the period of the data, and were later admitted. It includes their sex, child status (whether adult or child), and which specialty they were admitted to.

The `ed_visits` database contains a set of snapshots of patients who visited the ED and SDEC over the period of the data, including both admitted and discharged patients. Each snapshot includes information known at the time of the snapshot, and excludes anything that was recorded later, except the variables that serve a 'labels' for model training. These are:

- `is_admitted` - whether the visit ended in admission to a ward
- `final_sequence` - the sequence of consultations the patient had during the visit
- `specialty` - the specialty of the admission, if the patient was admitted

See the [data dictionaries](https://github.com/UCL-CORU/patientflow/tree/main/data-dictionaries) for detailed information about the variables in the data provided.

## Learn more about the data

I recorded a webinar to demonstrate how we converted data from the UCLH Electronic Health Record in a form suitable for this modelling. If you click on the image below, the video will open at the point where I provide detail about the datasets

<a href="https://www.youtube.com/watch?v=ha_zckz3_rU&t=262s" target="_blank">
    <img src="img/thumbnail_NHSR_webinar.jpg" alt="Link to webinar on how to turn your EHR data into predictions of demand for emergency beds" width="600"/>
</a>

## Set up the notebook environment

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

```python
from patientflow.load import set_project_root
project_root = set_project_root()
```

    Inferred project root: /Users/zellaking/Repos/patientflow

## Load parameters and set file paths

Parameters are set in config.json and (for UCLH implementation in config-uclh.yaml). You can change these for your own purposes. I'll talk more about the role of each parameter as it becomes relevant. Here we are loading the pre-defned training, validation and test set dates.

```python
from patientflow.load import set_file_paths

# set file paths
data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
        project_root,
        data_folder_name='data-public',  # change this to data-synthetic if you don't have the public dataset
        verbose=False
        )

```

## Load data

This notebook has been run using real data which you can download from [Zenodo](https://zenodo.org/records/14866057) on request.

Alternatively you can use the synthetic dataset that was generated using a stratified sampling approach based on the distributions reported in the [data dictionaries](https://github.com/UCL-CORU/patientflow/tree/main/data-dictionaries). Two relationships from the original data were preserved: the proportion of admitted versus non-admitted patients and the hourly arrival time patterns. All other variables were sampled independently using summary statistics stratified by admission status. This approach maintains relevant dependencies for admission outcomes while treating other variables as independent of each other.

If you don't have the public data, change the argument in the cell above from `data_folder_name='data-public'` to `data_folder_name='data-synthetic'`.

```python
import pandas as pd
from patientflow.load import load_data

ed_visits = load_data(data_file_path,
                    file_name='ed_visits.csv',
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"],
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])

inpatient_arrivals = load_data(data_file_path,
                    file_name='inpatient_arrivals.csv',
                    index_column = 'arrival_datetime',)

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
      <th>elapsed_los</th>
      <th>sex</th>
      <th>arrival_method</th>
      <th>num_obs</th>
      <th>num_obs_events</th>
      <th>num_obs_types</th>
      <th>num_lab_batteries_ordered</th>
      <th>has_consultation</th>
      <th>...</th>
      <th>visited_waiting</th>
      <th>visited_unknown</th>
      <th>latest_obs_respirations</th>
      <th>latest_obs_temperature</th>
      <th>latest_obs_news_score_result</th>
      <th>latest_obs_objective_pain_score</th>
      <th>visit_number</th>
      <th>is_admitted</th>
      <th>specialty</th>
      <th>final_sequence</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>98242</th>
      <td>2031-01-14</td>
      <td>(22, 0)</td>
      <td>20740.0</td>
      <td>M</td>
      <td>Amb no medic</td>
      <td>74</td>
      <td>6</td>
      <td>23</td>
      <td>8</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>19.0</td>
      <td>96.8</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>000019a46d7c</td>
      <td>True</td>
      <td>surgical</td>
      <td>['surgical']</td>
    </tr>
    <tr>
      <th>100119</th>
      <td>2031-01-19</td>
      <td>(15, 30)</td>
      <td>3780.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>29</td>
      <td>3</td>
      <td>23</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>16.0</td>
      <td>98.4</td>
      <td>1.0</td>
      <td>Mild</td>
      <td>00015db18883</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>189750</th>
      <td>2031-09-29</td>
      <td>(22, 0)</td>
      <td>10466.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>12</td>
      <td>2</td>
      <td>12</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>32.0</td>
      <td>97.9</td>
      <td>NaN</td>
      <td>Nil</td>
      <td>0001fbabb70e</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>192732</th>
      <td>2031-10-07</td>
      <td>(22, 0)</td>
      <td>13729.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>14</td>
      <td>1</td>
      <td>14</td>
      <td>7</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>97.7</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>00021c715ac7</td>
      <td>False</td>
      <td>NaN</td>
      <td>['surgical']</td>
    </tr>
    <tr>
      <th>119891</th>
      <td>2031-03-12</td>
      <td>(6, 0)</td>
      <td>2504.0</td>
      <td>M</td>
      <td>Walk-in</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>0002af190380</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>

## Explore ED visits dataset

Note that each snapshot has a date and a prediction time formatted separately, with the prediction time as a tuple of (hour, minute). All functions in `patientflow` expect prediction times in this format. Each record in the snapshots dataframe is indexed by a unique snapshot_id.

```python
ed_visits.head(10)
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
      <th>elapsed_los</th>
      <th>sex</th>
      <th>arrival_method</th>
      <th>num_obs</th>
      <th>num_obs_events</th>
      <th>num_obs_types</th>
      <th>num_lab_batteries_ordered</th>
      <th>has_consultation</th>
      <th>...</th>
      <th>visited_waiting</th>
      <th>visited_unknown</th>
      <th>latest_obs_respirations</th>
      <th>latest_obs_temperature</th>
      <th>latest_obs_news_score_result</th>
      <th>latest_obs_objective_pain_score</th>
      <th>visit_number</th>
      <th>is_admitted</th>
      <th>specialty</th>
      <th>final_sequence</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>98242</th>
      <td>2031-01-14</td>
      <td>(22, 0)</td>
      <td>20740.0</td>
      <td>M</td>
      <td>Amb no medic</td>
      <td>74</td>
      <td>6</td>
      <td>23</td>
      <td>8</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>19.0</td>
      <td>96.8</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>000019a46d7c</td>
      <td>True</td>
      <td>surgical</td>
      <td>['surgical']</td>
    </tr>
    <tr>
      <th>100119</th>
      <td>2031-01-19</td>
      <td>(15, 30)</td>
      <td>3780.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>29</td>
      <td>3</td>
      <td>23</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>16.0</td>
      <td>98.4</td>
      <td>1.0</td>
      <td>Mild</td>
      <td>00015db18883</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>189750</th>
      <td>2031-09-29</td>
      <td>(22, 0)</td>
      <td>10466.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>12</td>
      <td>2</td>
      <td>12</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>32.0</td>
      <td>97.9</td>
      <td>NaN</td>
      <td>Nil</td>
      <td>0001fbabb70e</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>192732</th>
      <td>2031-10-07</td>
      <td>(22, 0)</td>
      <td>13729.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>14</td>
      <td>1</td>
      <td>14</td>
      <td>7</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>97.7</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>00021c715ac7</td>
      <td>False</td>
      <td>NaN</td>
      <td>['surgical']</td>
    </tr>
    <tr>
      <th>119891</th>
      <td>2031-03-12</td>
      <td>(6, 0)</td>
      <td>2504.0</td>
      <td>M</td>
      <td>Walk-in</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>0002af190380</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>157035</th>
      <td>2031-06-27</td>
      <td>(15, 30)</td>
      <td>1548.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>14</td>
      <td>1</td>
      <td>14</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>18.0</td>
      <td>97.7</td>
      <td>NaN</td>
      <td>Moderate</td>
      <td>00033228d206</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>209659</th>
      <td>2031-11-30</td>
      <td>(6, 0)</td>
      <td>15020.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>33</td>
      <td>3</td>
      <td>24</td>
      <td>9</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>17.0</td>
      <td>98.2</td>
      <td>0.0</td>
      <td>Nil</td>
      <td>0003d8c503cf</td>
      <td>False</td>
      <td>NaN</td>
      <td>['obs_gyn']</td>
    </tr>
    <tr>
      <th>105648</th>
      <td>2031-02-03</td>
      <td>(12, 0)</td>
      <td>3502.0</td>
      <td>M</td>
      <td>Walk-in</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>00043419ec6b</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>172620</th>
      <td>2031-08-12</td>
      <td>(22, 0)</td>
      <td>7274.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>14</td>
      <td>2</td>
      <td>14</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>97.5</td>
      <td>NaN</td>
      <td>Mild</td>
      <td>0004c73468a6</td>
      <td>False</td>
      <td>NaN</td>
      <td>['obs_gyn']</td>
    </tr>
    <tr>
      <th>96560</th>
      <td>2031-01-09</td>
      <td>(22, 0)</td>
      <td>19768.0</td>
      <td>F</td>
      <td>Walk-in</td>
      <td>30</td>
      <td>3</td>
      <td>21</td>
      <td>0</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>17.0</td>
      <td>98.2</td>
      <td>0.0</td>
      <td>Moderate</td>
      <td>0004eacbae15</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 68 columns</p>
</div>

### Grouping of columns in ED visits dataset

The ED visits dataset contains variables of different types.

For convenience, I use a function called `get_dict_cols()` to organise the charts below into different sections.

```python
from patientflow.load import get_dict_cols
dict_cols = get_dict_cols(ed_visits)

for key, value in dict_cols.items():
    print(f"\nColumns in group called {key}:")
    print(value)


```

    Columns in group called not used in training:
    ['snapshot_id', 'snapshot_date', 'prediction_time', 'visit_number', 'training_validation_test', 'random_number']

    Columns in group called arrival and demographic:
    ['elapsed_los', 'sex', 'age_group', 'age_on_arrival', 'arrival_method']

    Columns in group called summary:
    ['num_obs', 'num_obs_events', 'num_obs_types', 'num_lab_batteries_ordered']

    Columns in group called location:
    ['current_location_type', 'total_locations_visited', 'visited_majors', 'visited_otf', 'visited_paeds', 'visited_rat', 'visited_resus', 'visited_sdec', 'visited_sdec_waiting', 'visited_taf', 'visited_utc', 'visited_waiting', 'visited_unknown']

    Columns in group called observations:
    ['num_obs_blood_pressure', 'num_obs_pulse', 'num_obs_air_or_oxygen', 'num_obs_level_of_consciousness', 'num_obs_news_score_result', 'num_obs_temperature', 'num_obs_manchester_triage_acuity', 'num_obs_objective_pain_score', 'num_obs_subjective_pain_score', 'num_obs_glasgow_coma_scale_best_motor_response', 'num_obs_oxygen_delivery_method', 'num_obs_oxygen_flow_rate', 'num_obs_pupil_reaction_right', 'num_obs_uclh_sskin_areas_observed', 'latest_obs_pulse', 'latest_obs_level_of_consciousness', 'latest_obs_manchester_triage_acuity', 'latest_obs_respirations', 'latest_obs_temperature', 'latest_obs_news_score_result', 'latest_obs_objective_pain_score']

    Columns in group called lab orders and results:
    ['lab_orders_bc', 'lab_orders_crp', 'lab_orders_csnf', 'lab_orders_ddit', 'lab_orders_rflu', 'latest_lab_results_crea', 'latest_lab_results_hctu', 'latest_lab_results_k', 'latest_lab_results_lac', 'latest_lab_results_na', 'latest_lab_results_pco2', 'latest_lab_results_ph', 'latest_lab_results_wcc', 'latest_lab_results_htrt', 'latest_lab_results_alb', 'lab_orders_bon', 'lab_orders_ncov', 'lab_orders_xcov']

    Columns in group called consults:
    ['has_consultation', 'consultation_sequence', 'final_sequence', 'specialty']

    Columns in group called outcome:
    ['is_admitted']

Also for the plots, I convert the boolean columns to text values.

```python
# Function to convert boolean columns to text values "true" or "false" - used for plotting format
def bool_to_text(df):
    bool_cols = df.select_dtypes(include='bool').columns.drop('is_admitted')
    for col in bool_cols:
        df[col] = df[col].apply(lambda x: 'true' if x else 'false')
    return df

# Apply the function
ed_visits = bool_to_text(ed_visits)

# temporarily add a is_admitted column to arrivals
inpatient_arrivals['is_admitted'] = True
inpatient_arrivals = bool_to_text(inpatient_arrivals)
```

As some variables are ordinal, I create a dictionary to record the ordering of the values.

```python
ordinal_mappings = {
    # age group
    "age_group": [
        "0-17",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75-115",
    ],
    # triage score
    "latest_obs_manchester_triage_acuity": ["Blue", "Green", "Yellow", "Orange", "Red"],
    # pain score
    "latest_obs_objective_pain_score": [
        r"Nil",
        r"Mild",
        r"Moderate",
        r"Severe\E\Very Severe",
    ],
    # level of consciousness
    "latest_obs_level_of_consciousness": [
        "A", #alert
        "C", #confused
        "V", #voice - responds to voice stimulus
        "P", #pain - responds to pain stimulus
        "U" #unconscious - no response to pain or voice stimulus
    ]
}
```

### Arrival and demographic variables

Here I import a function called `plot_data_distribution` to provide a convenient way of requesting each plot without multiple lines of code for each.

```python
from patientflow.viz.data_distribution import plot_data_distribution

```

#### Elapsed Length of Stay

Both admitted and not admitted visits appear to have a long tail of visits lasting more than 24 hours. Any snapshots where the ED visit has lasted more than 72 hours are excluded.

```python
ed_visits['elapsed_los_hrs'] = ed_visits['elapsed_los']/3600
plot_data_distribution(df=ed_visits, col_name='elapsed_los_hrs', grouping_var='is_admitted', grouping_var_name='whether patient admitted', plot_type='both',
                        title = 'Distribution of elapsed length of stay by whether patient admitted', truncate_outliers=False)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_20_0.png)

Below, I plot the snapshots where the elapsed visit duration is greater than 24 hours. We can see that the long tail of longer visits is more numerous for discharged than for admitted patients. This could be because patients leave the ED without being recorded as discharged on the system.

```python
if ed_visits[ed_visits.elapsed_los_hrs >= 24].shape[0] > 0:
    plot_data_distribution(ed_visits[ed_visits.elapsed_los_hrs >= 24], 'elapsed_los_hrs', 'is_admitted', 'whether patient admitted', plot_type='both',
                        title = 'Distribution of elapsed length of stay by whether patient admitted (where elapsed length of stay >= 24 hours)', truncate_outliers=False)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_22_0.png)

#### Sex, age group and arrival method

The charts below show distributions between admitted and not admitted patients for sex, age group and arrival method. More older people are admitted. Most walk-ins are discharged.

```python
plot_data_distribution(ed_visits, 'sex', 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_24_0.png)

```python
if 'age_group' in ed_visits.columns:
    plot_data_distribution(ed_visits, 'age_group', 'is_admitted', 'whether patient admitted', plot_type='hist', ordinal_order=ordinal_mappings['age_group'], rotate_x_labels = True)
else:
    plot_data_distribution(ed_visits, 'age_on_arrival', 'is_admitted', 'whether patient admitted', plot_type='hist')

```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_25_0.png)

```python
plot_data_distribution(ed_visits, 'arrival_method', 'is_admitted', 'whether patient admitted', plot_type='hist', rotate_x_labels = True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_26_0.png)

### Count variables

The counts variables record the following, up to the moment of the snapshot

- the number of observations recorded
- the number of events at which observations were recorded (if heart rate and respiratory rate have the same timestamp in the original data, this is one event)
- the number of different types of observations (heart rate and respiratory would be two types)
- the number of lab test batteries ordered

```python
for col_name in dict_cols['summary']:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist', is_discrete = False, truncate_outliers=True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_28_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_28_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_28_2.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_28_3.png)

The plots above have been set to exlude outliers, for better readability. However, there are some extreme values of num_obs and num_obs_events. In such cases, you might consider removing the outliers, depending on what model is to be applied to the data

```python
print(ed_visits.num_obs.max())
print(ed_visits.num_obs_events.max())
```

    989
    266

### Location variables

The variable `current_location_type` records the location of the patient at the time of the snapshot. Refer to the [data dictionary](https://github.com/UCL-CORU/patientflow/tree/main/data-dictionaries/ed_visits_data_dictionary.csv) for more information about what each location type means. Patients who visit the UTC (Urgent Treatment Centre) are more likely to be discharged than admitted. The UTC provides care for patients with minor injuries and illnesses.

```python
plot_data_distribution(ed_visits, 'current_location_type', 'is_admitted', 'whether patient admitted', plot_type='hist', rotate_x_labels = True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_32_0.png)

```python
for col_name in dict_cols['location'][1:]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_2.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_3.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_4.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_5.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_6.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_7.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_8.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_9.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_10.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_33_11.png)

### Observations variables

The variables in the observations group record vital signs, triage scores, and also the number of times certain observations have been recorded, up to the moment of the snapshot.

```python
dict_cols['observations']
```

    ['num_obs_blood_pressure',
     'num_obs_pulse',
     'num_obs_air_or_oxygen',
     'num_obs_level_of_consciousness',
     'num_obs_news_score_result',
     'num_obs_temperature',
     'num_obs_manchester_triage_acuity',
     'num_obs_objective_pain_score',
     'num_obs_subjective_pain_score',
     'num_obs_glasgow_coma_scale_best_motor_response',
     'num_obs_oxygen_delivery_method',
     'num_obs_oxygen_flow_rate',
     'num_obs_pupil_reaction_right',
     'num_obs_uclh_sskin_areas_observed',
     'latest_obs_pulse',
     'latest_obs_level_of_consciousness',
     'latest_obs_manchester_triage_acuity',
     'latest_obs_respirations',
     'latest_obs_temperature',
     'latest_obs_news_score_result',
     'latest_obs_objective_pain_score']

I first plot the variables that count the number of times something was recorded.

#### Count variables

```python
for col_name in [item for item in dict_cols['observations'] if str(item).startswith('num')]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted',
                            plot_type='hist',
                            is_discrete = True,
                            truncate_outliers=True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_2.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_3.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_4.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_5.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_6.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_7.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_8.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_9.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_10.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_11.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_12.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_38_13.png)

#### News Scores and Manchester Triage score values

News Scores are commonly used to track the acuity of a patient, and Manchester Triage scores are used at the door of the ED to prioritise patients

```python
for col_name in [item for item in dict_cols['observations'] if ('manchester' in str(item) ) and str(item).startswith('latest')]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist', rotate_x_labels = True, ordinal_order=ordinal_mappings['latest_obs_manchester_triage_acuity'])
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_40_0.png)

```python
plot_data_distribution(ed_visits, 'latest_obs_objective_pain_score', 'is_admitted', 'whether patient admitted',
                        plot_type='hist',
                        rotate_x_labels = True,
                        ordinal_order=ordinal_mappings['latest_obs_objective_pain_score'])

```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_41_0.png)

```python

for col_name in [item for item in dict_cols['observations'] if 'news' in str(item) and str(item).startswith('latest')]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist', is_discrete = True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_42_0.png)

The ACVPU score is commonly used to track states of consciousness

```python
for col_name in [item for item in dict_cols['observations'] if 'consciousness' in str(item) and str(item).startswith('latest')]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted',
                            plot_type='hist',
                            ordinal_order=ordinal_mappings['latest_obs_level_of_consciousness'])
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_44_0.png)

Temporarily excluding the most common value of A from the ACVPU score, we can see the spread of other values

```python
for col_name in [item for item in dict_cols['observations'] if 'consciousness' in str(item) and str(item).startswith('latest')]:
    plot_data_distribution(ed_visits[~(ed_visits.latest_obs_level_of_consciousness == 'A')].copy(), col_name, 'is_admitted', 'whether patient admitted',
                            plot_type='hist',
                            ordinal_order=ordinal_mappings['latest_obs_level_of_consciousness'])
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_46_0.png)

#### Vital signs values

I now plot the distributions of the vital signs values.

```python
for col_name in [item for item in dict_cols['observations'] if str(item).startswith('latest') and ('pulse' in str(item) or 'resp' in str(item) or 'temp' in str(item))]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted',
                            plot_type='hist',
                            is_discrete = False,
                            truncate_outliers=True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_48_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_48_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_48_2.png)

### Lab variables

The lab variables include boolean values for whether a lab battery was ordered, and the results of certain lab test. The data include only a small a subset of the lab battery orders and test results that might be requested for a patient in the ED.

```python
dict_cols['lab orders and results']
```

    ['lab_orders_bc',
     'lab_orders_crp',
     'lab_orders_csnf',
     'lab_orders_ddit',
     'lab_orders_rflu',
     'latest_lab_results_crea',
     'latest_lab_results_hctu',
     'latest_lab_results_k',
     'latest_lab_results_lac',
     'latest_lab_results_na',
     'latest_lab_results_pco2',
     'latest_lab_results_ph',
     'latest_lab_results_wcc',
     'latest_lab_results_htrt',
     'latest_lab_results_alb',
     'lab_orders_bon',
     'lab_orders_ncov',
     'lab_orders_xcov']

#### Lab orders

It is notable in the charts below, which show whether a lab battery was ordered, that battery CRP (for markers of inflammation) is very commonly ordered for admitted patients; among the patients later admitted the majority have a CRP battery ordered whereas among the non-admitted patients only a minority have it. This difference between admitted and non-admitted (where the majority of admitted have something while the majority of discharged patients do not) only applies to this lab battery order. It will show up later as a strong predictor of admission.

```python
for col_name in [item for item in dict_cols['lab orders and results'] if str(item).startswith('lab') ]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_2.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_3.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_4.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_5.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_6.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_52_7.png)

#### Lab results

The plots below show the latest lab values.

```python
for col_name in [item for item in dict_cols['lab orders and results'] if str(item).startswith('latest') ]:
    plot_data_distribution(ed_visits, col_name, 'is_admitted', 'whether patient admitted', plot_type='hist', outlier_threshold=3, truncate_outliers=True)
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_0.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_1.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_2.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_3.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_4.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_5.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_6.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_7.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_8.png)

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_54_9.png)

### Consults variables

The `has_consultation` variable records whether a referral request was made to another service or specialty up to the point of the snapshot. The sequence of referrals up to that point is recorded in `consultation_sequence` and the final sequence, at the end of the ED visit in `final_sequence`. `specialty` records the specialty that the patient was admitted under, if admitted.

The first plot shows that the number of admitted patients with consult requests at the time of the snapshots is about the same as those without. The group without consult requests will have their later in the visit, after the snapshot was recorded.

```python
plot_data_distribution(ed_visits, 'has_consultation', 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_56_0.png)

A very small number of non-admitted patients have a specialty of admission recorded. These are most likely patients referred from ED to SDEC, which we don't include in the admitted patients.

```python
plot_data_distribution(ed_visits, 'specialty', 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_58_0.png)

## Explore inpatient arrivals dataset

The inpatient_arrivals dataset records all of the arrival dates and and times of patients who were later admitted to a ward. Other information is also recorded, such as sex and child status, as will as specialty of admission. This dataset will be used to predict the number of patients yet-to-arrive at the time of prediction.

```python
inpatient_arrivals.head()
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
      <th>sex</th>
      <th>is_child</th>
      <th>is_admitted</th>
    </tr>
    <tr>
      <th>arrival_datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2031-02-26 11:28:00+00:00</th>
      <td>medical</td>
      <td>M</td>
      <td>false</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2031-01-14 16:51:00+00:00</th>
      <td>surgical</td>
      <td>F</td>
      <td>false</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2031-02-06 20:51:00+00:00</th>
      <td>surgical</td>
      <td>M</td>
      <td>false</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2031-01-10 13:43:00+00:00</th>
      <td>haem/onc</td>
      <td>M</td>
      <td>false</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2031-01-02 13:55:00+00:00</th>
      <td>haem/onc</td>
      <td>F</td>
      <td>false</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

```python
# temporarily add is_admitted column to arrivals dataset, to be able to use the plot_data_distribution function
inpatient_arrivals['is_admitted'] = True
plot_data_distribution(inpatient_arrivals.reset_index().copy(), 'specialty', 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_61_0.png)

```python
plot_data_distribution(inpatient_arrivals, 'is_child', 'is_admitted', 'whether patient admitted', plot_type='hist')
```

![png](2d_Explore_the_datasets_provided_files/2d_Explore_the_datasets_provided_62_0.png)

## Summary

This notebook has shown how to load files that have been provided, and shows some plots of the variables included. This is an illustrative dataset, showing the type of variables that were used for the analysis at UCLH. Other sites will have different data.
