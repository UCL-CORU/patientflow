# Create group snapshots

## About snapshots

Functions in `patientflow` help you convert patient-level predictions into a predicted bed count distributions for groups of patients at a point in time. As a reminder, the package is organised around the following concepts:

- Prediction time: A moment in the day at which predictions are to be made, for example 09:30.
- Patient snapshot: A summary of data from the EHR capturing is known about a single patient at the prediction time. Each patient snapshot has a date and a prediction time associated with it.
- Group snaphot: A set of patients snapshots. Each group snapshot has a date and a prediction time associated with it.
- Prediction window: A period of hours that begins at the prediction time.

In this notebook, I show how to prepare a group snapshot, using fake data that resembles visits to the Emergency Department (ED). In this example, a group snapshot comprises all patients who were in the ED at a point in time.  

I then demonstrate the use of a simple model that predicts admission for each patient, and show how those predictions can be aggregated into a prediction for the number of beds needed for the patients comprising the group snapshot.


```python
# Reload functions every time
%load_ext autoreload 
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## Set up prediction times

The first step is to specify the times of day at which we want to create predictions. 


```python
prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)] # each time is expressed as a tuple of (hour, minute)
```

## Create patient snapshots 

We'll create some fake patient shapshots, using an example of patients in an Emergency Department (ED). See the [2a_Create_patient_snapshots](2a_Create_patient_snapshots.md) notebook for more information about to convert finished hospital visits into snapshots. 


```python
from patientflow.generate import patient_visits, create_snapshots
start_date = '2023-01-01'
end_date = '2023-04-01'

# Create fake patient data
visits_df, observations_df, lab_orders_df = patient_visits(start_date, end_date, 100)

# Convert the data in snapshots
snapshots_df = create_snapshots(visits_df, observations_df, lab_orders_df, prediction_times, start_date, end_date)
snapshots_df.head(10)
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
      <th>is_admitted</th>
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_troponin_orders</th>
      <th>num_bmp_orders</th>
      <th>num_urinalysis_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>45</td>
      <td>0</td>
      <td>51</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>72</td>
      <td>0</td>
      <td>21</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>38</td>
      <td>0</td>
      <td>50</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>19</td>
      <td>0</td>
      <td>23</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>94</td>
      <td>0</td>
      <td>24</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>52</td>
      <td>0</td>
      <td>63</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>22</td>
      <td>1</td>
      <td>100</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>88</td>
      <td>1</td>
      <td>46</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>41</td>
      <td>0</td>
      <td>27</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>17</td>
      <td>1</td>
      <td>50</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Note that each record in the snapshots dataframe is indexed by a unique snapshot_id. 

## Create group snapshots

For each combination of snapshot_date and prediction_time, we can identify the group of patients in the ED at that moment. These patients comprise the group snapshot. `patientflow` includes a `prepare_group_snapshot_dict()` function. As input, it requires a pandas dataframe with a `snapshot_date` column. If a start date and end date are provided, the function will check for any intervening snapshot dates that are missing, and create an empty group snapshot for this date

The keys of the dictionary are the snapshot_date. The values are a list of patients in the ED at that time, identified by their unique `snapshot_id`.

Here I create a group snapshot dictionary for patients in the ED at 09.30



```python
from patientflow.prepare import prepare_group_snapshot_dict

# select the snapshots to include in the probability distribution, 
group_snapshots_dict = prepare_group_snapshot_dict(
    snapshots_df[snapshots_df.prediction_time == (9,30)]
    )

print("First 10 keys in the snapshots dictionary")
print(list(group_snapshots_dict.keys())[0:10])

```

    First 10 keys in the snapshots dictionary
    [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3), datetime.date(2023, 1, 4), datetime.date(2023, 1, 5), datetime.date(2023, 1, 6), datetime.date(2023, 1, 7), datetime.date(2023, 1, 8), datetime.date(2023, 1, 9), datetime.date(2023, 1, 10)]


From the first key in the dictionary, we can see the patients belonging to this first snapshot. 


```python
first_group_snapshot_key = list(group_snapshots_dict.keys())[0]
first_group_snapshot_values = group_snapshots_dict[first_group_snapshot_key]

print("\nUnique snapshot_ids in the first group snapshot:")
print(first_group_snapshot_values)

print(f"\nThere are {len(first_group_snapshot_values)} patients in the first group snapshot")

print("\nPatient snapshots belonging to the first group snapshot:")
snapshots_df.loc[first_group_snapshot_values]
```

    
    Unique snapshot_ids in the first group snapshot:
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    There are 12 patients in the first group snapshot
    
    Patient snapshots belonging to the first group snapshot:





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
      <th>is_admitted</th>
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_troponin_orders</th>
      <th>num_bmp_orders</th>
      <th>num_urinalysis_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>22</td>
      <td>1</td>
      <td>100</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>88</td>
      <td>1</td>
      <td>46</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>41</td>
      <td>0</td>
      <td>27</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>17</td>
      <td>1</td>
      <td>50</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>68</td>
      <td>0</td>
      <td>53</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>43</td>
      <td>0</td>
      <td>39</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>27</td>
      <td>0</td>
      <td>24</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>71</td>
      <td>0</td>
      <td>100</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>59</td>
      <td>0</td>
      <td>73</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>33</td>
      <td>1</td>
      <td>40</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>1</td>
      <td>0</td>
      <td>55</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>63</td>
      <td>0</td>
      <td>43</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Prepare a model that can make predictions for each patient snapshot

In the cell below, I'm using some functions provided within `patientflow`to prepare a XGBoost classifier. This model will be used to make predictions. 


```python
from datetime import date   
from patientflow.prepare import create_temporal_splits
from patientflow.train.classifiers import train_classifier

# set the temporal split
start_training_set = date(2023, 1, 1)
start_validation_set = date(2023, 2, 15)
start_test_set = date(2023, 3, 1)
end_test_set = date(2023, 4, 1)

# create the temporal splits
train_visits, valid_visits, test_visits = create_temporal_splits(
    snapshots_df,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
)

# minimal hyperparameter grid for xgboost
grid = {"n_estimators": [30], 
        "subsample": [0.7], 
        "colsample_bytree": [0.7],
}

prediction_time = (9, 30)

# exclude columns that are not needed for training
exclude_from_training_data=['visit_number', 'snapshot_date', 'prediction_time']
visit_col='visit_number'

ordinal_mappings={'latest_triage_score': [1, 2, 3, 4, 5]}

# train the patient-level model
model = train_classifier(
    train_visits,
    valid_visits,
    test_visits,
    grid=grid,   
    prediction_time=prediction_time,
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings=ordinal_mappings,
    visit_col=visit_col,
    use_balanced_training=True,
    calibrate_probabilities=False
)

```

    Split sizes: [3798, 1276, 2683]


## Using the trained model, get bed count probability for one snapshot

The snapshot date and prediction time should not be included in the classifier


```python
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.aggregate import get_prob_dist_for_prediction_moment

first_snapshot_prepared_for_model, _ = get_snapshots_at_prediction_time(
    snapshots_df.loc[first_group_snapshot_values], prediction_time, exclude_from_training_data, visit_col=visit_col
)

first_snapshot_prepared_for_model

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
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_troponin_orders</th>
      <th>num_bmp_orders</th>
      <th>num_urinalysis_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>55</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>50</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>24</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>40</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>27</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>73</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>43</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>53</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>100</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



From the output below, we can see a probability distribution for the number of beds needed for the 12 patients in the ED at the snapshot date and prediction time


```python
bed_count_prob_dist = get_prob_dist_for_prediction_moment(
    first_snapshot_prepared_for_model, model, weights=None, inference_time=True, y_test=None
)

bed_count_prob_dist['agg_predicted'].rename(columns={'agg_proba': 'prob_of_bed_count'})

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
      <th>prob_of_bed_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.56949241724012e-5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000448979697629399</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00480678488865520</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0267934114212809</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0886450190019757</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.185854921843942</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.255285376420707</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.232373816377793</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.139055356247333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0530641348292273</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0121210012947725</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.00146475916286634</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.07438896455642e-5</td>
    </tr>
  </tbody>
</table>
</div>




```python
from patientflow.viz.prob_dist_plot import prob_dist_plot
from patientflow.viz.utils import format_prediction_time
title = (
    f'Probability distribution for number of beds needed by the '
    f'{len(first_snapshot_prepared_for_model)} patients\n'
    f'in the ED at {format_prediction_time(prediction_time)} '
    f'on {first_group_snapshot_key}'
)
prob_dist_plot(bed_count_prob_dist['agg_predicted'], title,  
    include_titles=True)

```


    
![png](2b_Create_group_snapshots_files/2b_Create_group_snapshots_18_0.png)
    


## Make predictions for group snapshots

Here we'll first split the data into training, validation and test sets using a temporal split, and create a patient-level model predicting probability of admission for each patient snapshot using a `train_classifier()` function provided within `patientflow`.

We'll make predictions for 09.30 


```python

X_test, y_test = get_snapshots_at_prediction_time(
    test_visits, prediction_time, exclude_from_training_data, visit_col=visit_col
)



```


```python
X_test.head()
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
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_troponin_orders</th>
      <th>num_bmp_orders</th>
      <th>num_urinalysis_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5082</th>
      <td>41</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5081</th>
      <td>38</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5084</th>
      <td>13</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5087</th>
      <td>28</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5076</th>
      <td>61</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from patientflow.aggregate import get_prob_dist

group_snapshots_dict = prepare_group_snapshot_dict(
    test_visits[test_visits.prediction_time == (9,30)]
    )
# get probability distribution for this time of day
prob_dists_for_group_snapshots = get_prob_dist(
        group_snapshots_dict, X_test, y_test, pipeline
    )
```

    Calculating probability distributions for 31 snapshot dates
    This may take a minute or more



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[90], line 7
          3 group_snapshots_dict = prepare_snapshots_dict(
          4     test_visits[test_visits.prediction_time == (9,30)]
          5     )
          6 # get probability distribution for this time of day
    ----> 7 prob_dists_for_group_snapshots = get_prob_dist(
          8         group_snapshots_dict, X_test, y_test, pipeline
          9     )


    File ~/Repos/patientflow/src/patientflow/aggregate.py:487, in get_prob_dist(snapshots_dict, X_test, y_test, model, weights)
        484         prediction_moment_weights = weights.loc[snapshots_to_include].values
        486     # Compute the predicted and observed valuesfor the current snapshot date
    --> 487     prob_dist_dict[dt] = get_prob_dist_for_prediction_moment(
        488         X_test=X_test.loc[snapshots_to_include],
        489         y_test=y_test.loc[snapshots_to_include],
        490         model=model,
        491         weights=prediction_moment_weights,
        492     )
        494 # Increment the counter and notify the user every 10 snapshot dates processed
        495 count += 1


    File ~/Repos/patientflow/src/patientflow/aggregate.py:404, in get_prob_dist_for_prediction_moment(X_test, model, weights, inference_time, y_test)
        401 prediction_moment_dict = {}
        403 if len(X_test) > 0:
    --> 404     pred_proba = model_input_to_pred_proba(X_test, model)
        405     agg_predicted = pred_proba_to_agg_predicted(pred_proba, weights)
        406     prediction_moment_dict["agg_predicted"] = agg_predicted


    File ~/Repos/patientflow/src/patientflow/aggregate.py:314, in model_input_to_pred_proba(model_input, model)
        312     return pd.DataFrame(columns=["pred_proba"])
        313 else:
    --> 314     predictions = model.predict_proba(model_input)[:, 1]
        315     return pd.DataFrame(
        316         predictions, index=model_input.index, columns=["pred_proba"]
        317     )


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/sklearn/pipeline.py:722, in Pipeline.predict_proba(self, X, **params)
        720 if not _routing_enabled():
        721     for _, name, transform in self._iter(with_final=False):
    --> 722         Xt = transform.transform(Xt)
        723     return self.steps[-1][1].predict_proba(Xt, **params)
        725 # metadata routing enabled


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/sklearn/utils/_set_output.py:295, in _wrap_method_output.<locals>.wrapped(self, X, *args, **kwargs)
        293 @wraps(f)
        294 def wrapped(self, X, *args, **kwargs):
    --> 295     data_to_wrap = f(self, X, *args, **kwargs)
        296     if isinstance(data_to_wrap, tuple):
        297         # only wrap the first output for cross decomposition
        298         return_tuple = (
        299             _wrap_data_with_container(method, data_to_wrap[0], X, self),
        300             *data_to_wrap[1:],
        301         )


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/sklearn/compose/_column_transformer.py:1003, in ColumnTransformer.transform(self, X, **params)
       1001     diff = all_names - set(column_names)
       1002     if diff:
    -> 1003         raise ValueError(f"columns are missing: {diff}")
       1004 else:
       1005     # ndarray was used for fitting or transforming, thus we only
       1006     # check that n_features_in_ is consistent
       1007     self._check_n_features(X, reset=False)


    ValueError: columns are missing: {'snapshot_datetime'}



```python
type(prob_dists_for_group_snapshots)
```




    dict




```python
list(prob_dists_for_group_snapshots.keys())[0:5]
```




    [datetime.date(2023, 3, 1),
     datetime.date(2023, 3, 2),
     datetime.date(2023, 3, 3),
     datetime.date(2023, 3, 4),
     datetime.date(2023, 3, 5)]




```python

```
