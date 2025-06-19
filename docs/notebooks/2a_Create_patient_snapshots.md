# 2a. Create patient-level snapshots

## About snapshots

`patientflow` is organised around the following concepts:

- Prediction time: A moment in the day at which predictions are to be made, for example 09:30.
- Patient snapshot: A summary of data from the EHR capturing is known about a single patient at the prediction time. Each patient snapshot has a date and a prediction time associated with it.
- Group snapshot: A set of patient snapshots. Each group snapshot has a date and a prediction time associated with it.
- Prediction window: A period of hours that begins at the prediction time.

To use `patientflow` your data should be in snapshot form.

In this notebook I suggest how to you might prepare your data, starting from data on finished hospital visits. I start with fake data on Emergency Department visits, and demonstrate how to convert it into snapshots. There are two examples

- A simple example of creating snapshots assuming you have one flat table of hospital visits
- An example of creating snapshots from data structured as a relational database.

## A note on creating your own shapshots

The snapshot creation shown here is designed to work with fake data generated below. You would need to create your own version of this process, to handle the data you have.

In practice, determining from data _whether a patient was admitted after the ED visit_, and _when they were ready to be admitted_, can be tricky. How do you account for the fact that the patient may wait in the ED for a bed, due to lack of available beds? Likewise, if you are trying to predict discharge at the end of a hospital visit, should that that be the time they were ready to leave, or the time they actually left? Discharge delays are common, due to waiting for medication or transport, or waiting for onward care provision to become available.

The outcome that you are aiming for will depend on your setting, and the information needs of the bed managers you are looking to support. You may have to infer when a patient was ready from available data. Suffice to say, think carefully about what it is you are trying to predict, and how you will identify that outcome in data.

## Creating fake finished visits

I'll start by loading some fake data resembling the structure of EHR data on Emergency Department (ED) visits, using a function called `create_fake_finished_visits`. In my fake data, each visit has one row, with an arrival time at the ED, a discharge time from the ED, the patient's age and an outcome of whether they were admitted after the ED visit.

The `is_admitted` column is our label, indicating the outcome in this imaginary case.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

```python
from patientflow.generate import create_fake_finished_visits
visits_df, _, _ = create_fake_finished_visits('2023-01-01', '2023-04-01', 25)

print(f'There are {len(visits_df)} visits in the fake dataset, with arrivals between {visits_df.arrival_datetime.min().date()} and {visits_df.arrival_datetime.max().date()} inclusive.')
visits_df.head()
```

    There are 2253 visits in the fake dataset, with arrivals between 2023-01-01 and 2023-03-31 inclusive.

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
      <th>age</th>
      <th>is_admitted</th>
      <th>specialty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>354</td>
      <td>1</td>
      <td>2023-01-01 05:21:43</td>
      <td>2023-01-01 12:35:43</td>
      <td>31</td>
      <td>1</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1281</td>
      <td>7</td>
      <td>2023-01-01 07:22:18</td>
      <td>2023-01-01 22:46:18</td>
      <td>31</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113</td>
      <td>15</td>
      <td>2023-01-01 07:31:29</td>
      <td>2023-01-01 16:12:29</td>
      <td>41</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>114</td>
      <td>3</td>
      <td>2023-01-01 08:01:26</td>
      <td>2023-01-01 10:34:26</td>
      <td>33</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1937</td>
      <td>18</td>
      <td>2023-01-01 08:45:38</td>
      <td>2023-01-01 16:30:38</td>
      <td>14</td>
      <td>0</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

## Example 1: Create snapshots from fake data - a simple example

My goal is to create snapshots of these visits. First, I define the times of day I will be issuing predictions at. Each time is expressed as a tuple of (hour, minute)

```python
prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)] # each time is expressed as a tuple of (hour, minute)
```

Then using the code below I create an array of all the snapshot dates in some date range that my data covers.

```python
from datetime import datetime, time, timedelta, date

# Create date range
snapshot_dates = []
start_date = date(2023, 1, 1)
end_date = date(2023, 4, 1)

# Iterate to create an array of dates
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

Next I iterate through the date array, using the arrival and departure times from the hospital visits table to identify any patients who were in the ED at each prediction time (eg 09:30 or 12.00) on each date.

```python
import pandas as pd


# Create empty list to store results for each snapshot date
patient_shapshot_list = []

# For each combination of date and time
for date_val in snapshot_dates:
    for hour, minute in prediction_times:
        snapshot_datetime = datetime.combine(date_val, time(hour=hour, minute=minute))

        # Filter dataframe for this snapshot
        mask = (visits_df["arrival_datetime"] <= snapshot_datetime) & (
            visits_df["departure_datetime"] > snapshot_datetime
        )
        snapshot_df = visits_df[mask].copy()

        # Skip if no patients at this time
        if len(snapshot_df) == 0:
            continue

        # Add snapshot information columns
        snapshot_df["snapshot_date"] = date_val
        snapshot_df["prediction_time"] = [(hour, minute)] * len(snapshot_df)

        patient_shapshot_list.append(snapshot_df)

# Combine all results into single dataframe
snapshots_df = pd.concat(patient_shapshot_list, ignore_index=True)

# Name the index snapshot_id
snapshots_df.index.name = "snapshot_id"
```

Note that each record in the snapshots dataframe is indexed by a unique snapshot_id.

```python
snapshots_df.head()
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
      <th>age</th>
      <th>is_admitted</th>
      <th>specialty</th>
      <th>snapshot_date</th>
      <th>prediction_time</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>354</td>
      <td>1</td>
      <td>2023-01-01 05:21:43</td>
      <td>2023-01-01 12:35:43</td>
      <td>31</td>
      <td>1</td>
      <td>medical</td>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>354</td>
      <td>1</td>
      <td>2023-01-01 05:21:43</td>
      <td>2023-01-01 12:35:43</td>
      <td>31</td>
      <td>1</td>
      <td>medical</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1281</td>
      <td>7</td>
      <td>2023-01-01 07:22:18</td>
      <td>2023-01-01 22:46:18</td>
      <td>31</td>
      <td>0</td>
      <td>None</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113</td>
      <td>15</td>
      <td>2023-01-01 07:31:29</td>
      <td>2023-01-01 16:12:29</td>
      <td>41</td>
      <td>0</td>
      <td>None</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>114</td>
      <td>3</td>
      <td>2023-01-01 08:01:26</td>
      <td>2023-01-01 10:34:26</td>
      <td>33</td>
      <td>0</td>
      <td>None</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
  </tbody>
</table>
</div>

Some patients are present at more than one of the prediction times, given them more than one entry in snapshots_df

```python
# Count the number of snapshots per visit and show top five
snapshots_df.visit_number.value_counts().head()
```

    visit_number
    1940    7
    375     7
    1812    7
    1733    7
    1736    7
    Name: count, dtype: int64

Below I show one example of a patient who was in the ED long enough to have multiple snapshots, captured at the various prediction times during their visit.

```python
# Displaying the snapshots for a visit with multiple snapshots
example_visit_number = snapshots_df.visit_number.value_counts().index[0]
snapshots_df[snapshots_df.visit_number == example_visit_number]

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
      <th>age</th>
      <th>is_admitted</th>
      <th>specialty</th>
      <th>snapshot_date</th>
      <th>prediction_time</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2959</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-19</td>
      <td>(15, 30)</td>
    </tr>
    <tr>
      <th>2965</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-19</td>
      <td>(22, 0)</td>
    </tr>
    <tr>
      <th>2977</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-20</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>2983</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-20</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>2989</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-20</td>
      <td>(12, 0)</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-20</td>
      <td>(15, 30)</td>
    </tr>
    <tr>
      <th>3014</th>
      <td>358</td>
      <td>1940</td>
      <td>2023-03-19 14:29:26</td>
      <td>2023-03-21 03:27:26</td>
      <td>79</td>
      <td>0</td>
      <td>None</td>
      <td>2023-03-20</td>
      <td>(22, 0)</td>
    </tr>
  </tbody>
</table>
</div>

## Example 2: Creating fake finished visits from a relational database

Electronic Health Record systems and their data warehouses are often structured as relational databases, with information stored on multiple linked tables. Timestamps are used to capture how information about a patient accumulates as the ED visit progresses. Patients may visit various locations in the ED, such as triage, where their acuity is recorded, and different activities related to their care are carried out, like measurements of vital signs or lab tests.

The function below returns three fake dataframes, meant to resemble EHR data.

- hospital visit dataframe - already seen above
- observations dataframe - with a single measurement, a triage score, plus a timestamp for when that was recorded
- lab orders dataframe - with five types of lab orders plus a timestamp for when these tests were requested

The function that creates the fake data returns one triage score for each visit, within 10 minutes of arrival

```python
visits_df, observations_df, lab_orders_df = create_fake_finished_visits('2023-01-01', '2023-04-01', 25)

print(f'There are {len(observations_df)} triage scores in the observations_df dataframe, for {len(observations_df.visit_number.unique())} visits')
observations_df.head()
```

    There are 2253 triage scores in the observations_df dataframe, for 2253 visits

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
      <th>visit_number</th>
      <th>observation_datetime</th>
      <th>triage_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-01-01 05:25:48.686712</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2023-01-01 07:24:04.659833</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>2023-01-01 07:39:02.025157</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2023-01-01 08:10:51.432211</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>2023-01-01 08:50:52.495502</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

The function that creates the fake data returns a random number of lab tests for each patient, for visits over 2 hours. Not all visits will have lab orders in this fake data.

```python
print(f'There are {len(lab_orders_df)} lab orders in the dataset, for {len(lab_orders_df.visit_number.unique())} visits')
lab_orders_df.head()
```

    There are 5754 lab orders in the dataset, for 2091 visits

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
      <th>visit_number</th>
      <th>order_datetime</th>
      <th>lab_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-01-01 05:51:39.377886</td>
      <td>BMP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2023-01-01 05:58:40.347001</td>
      <td>D-dimer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2023-01-01 06:36:24.534586</td>
      <td>CBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2023-01-01 06:49:29.836402</td>
      <td>Urinalysis</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>2023-01-01 07:43:33.443262</td>
      <td>Troponin</td>
    </tr>
  </tbody>
</table>
</div>

The `create_fake_snapshots()` function will pull information from the three fake tables, and prepare snapshots.

```python
from datetime import date
start_date = date(2023, 1, 1)
end_date = date(2023, 4, 1)

from patientflow.generate import create_fake_snapshots

# Create snapshots
new_snapshots_df = create_fake_snapshots(df=visits_df, observations_df=observations_df, lab_orders_df=lab_orders_df, prediction_times=prediction_times, start_date=start_date, end_date=end_date)
new_snapshots_df.head()
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
      <th>patient_id</th>
      <th>visit_number</th>
      <th>is_admitted</th>
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_bmp_orders</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_urinalysis_orders</th>
      <th>num_troponin_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>354</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>354</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>1281</td>
      <td>7</td>
      <td>0</td>
      <td>31</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>113</td>
      <td>15</td>
      <td>0</td>
      <td>41</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>114</td>
      <td>3</td>
      <td>0</td>
      <td>33</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

Returning to the example visit above, we can see that at 09:30 on 2023-01-10, the first snapshot for this patient, the triage score had not yet been recorded. This, and the lab orders, were placed between 09:30 and 12:00, so they appear first in the 12:00 snapshot.

```python
new_snapshots_df[new_snapshots_df.visit_number==example_visit_number]
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
      <th>patient_id</th>
      <th>visit_number</th>
      <th>is_admitted</th>
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_bmp_orders</th>
      <th>num_d-dimer_orders</th>
      <th>num_cbc_orders</th>
      <th>num_urinalysis_orders</th>
      <th>num_troponin_orders</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2959</th>
      <td>2023-03-19</td>
      <td>(15, 30)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2965</th>
      <td>2023-03-19</td>
      <td>(22, 0)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2977</th>
      <td>2023-03-20</td>
      <td>(6, 0)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2983</th>
      <td>2023-03-20</td>
      <td>(9, 30)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2989</th>
      <td>2023-03-20</td>
      <td>(12, 0)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>2023-03-20</td>
      <td>(15, 30)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3014</th>
      <td>2023-03-20</td>
      <td>(22, 0)</td>
      <td>358</td>
      <td>1940</td>
      <td>0</td>
      <td>79</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

## Summary

Here I have shown how to create patient snapshots from finished patient visits. Note that there is a discarding of some information, or summarisation involved. The lab orders have been reduced to counts, and only the latest triage score has been taken. In the same vein, you might just take the last recorded heart rate or oxygen saturation level, or the latest value of a lab result. A snapshot loses some of the richness of the full data in an EHR, but with the benefit that you get data that replicate unfinished visits.

Note that ED visit data can be patchy in ways that are meaningful. For example, a severely ill patient might have many heart rate values recorded and many lab orders, while a patient with a sprained ankle might have zero heart rate measurements or lab orders. For predicting probability of admission after ED, such variation in data completeness is revealing. By summarising to counts, snapshots allow us to capture that variation in data completeness without having to discard observations that have missing data.

In the next notebook I'll show how to make predictions using patient snapshots.
