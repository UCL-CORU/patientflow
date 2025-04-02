# Create patient-level snapshots

## About snapshots

`patientflow` is organised around the following concepts:

- Prediction time: A moment in the day at which predictions are to be made, for example 09:30.
- Patient snapshot: A summary of data from the EHR capturing is known about a single patient at the prediction time. Each patient snapshot has a date and a prediction time associated with it.
- Group snaphot: A set of patients snapshots. Each group snapshot has a date and a prediction time associated with it.
- Prediction window: A period of hours that begins at the prediction time.

Its intended use is with data on hospital visits that are unfinished, to predict whether some outcome (admission from A&E, discharge from hospital, or transfer to another clinical specialty) will happen within the prediction window. What the outcome is doesn't really matter; the same methods can be used. 

The utility of our approach - and the thing that makes it very generalisable - is that we then build up from the patient-level predictions into a predictions for a whole cohort of patients at a point in time. That step is what creates useful information for bed managers. I show this in later notebooks.

To use `patientflow` your data should be in snapshot form. Here I suggest how to you might prepare your data, starting from past hospital visits that have already finished. I'm going to use some fake data on Emergency Department visits, and imagine that we want to predict whether a patient will be admitted to a ward after they leave the ED, or discharged from hospital. 

NOTE: In practice, how to *whether a patient was admitted after the ED visit*, and *when they were ready to be admitted*, can be tricky. How do you account for the fact that the patient may wait in the ED for a bed, due to lack of available beds? Likewise, if you are trying to predict discharge at the end of a hospital visit, should that that be the time they were ready to leave, or the time they actually left? Discharge delays are common, due to waiting for medication or transport, or waiting for onward care provision to become available. 

The outcome that you are aiming for will depend on your setting. You may have to infer when a patient was ready from available data. Suffice to say, think carefully about what it is you are trying to predict, and how you will identify that outcome in data. 

## Creating fake finished visits

I'll start by loading some fake data resembling structure of EHR data on Emergency Department (ED) visits. In my fake data, each visit has one row, with an arrival time at the ED, a discharge time from the ED, the patient's age and an outcome of whether they were admitted after the ED visit. 

The `is_admitted` column is our label, indicating the outcome in this imaginary case. 


```python
# Reload functions every time
%load_ext autoreload 
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



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



## Create snapshots from fake data

My goal is to create snapshots of these visits. First, I define the times of day I will be issuing predictions at. 


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



Next I iterate through the date array, using the arrival and departure times from the hospital visits table to identify any patients who were in the ED at the prediction time (eg 09:30 or 12.00 on each date). 


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
      <th>is_admitted</th>
      <th>age</th>
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
      <td>2023-01-01</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>238</td>
      <td>20</td>
      <td>2023-01-01 04:25:57</td>
      <td>2023-01-01 07:43:57</td>
      <td>1</td>
      <td>61</td>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>354</td>
      <td>1</td>
      <td>2023-01-01 05:21:43</td>
      <td>2023-01-01 08:52:43</td>
      <td>1</td>
      <td>86</td>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>114</td>
      <td>3</td>
      <td>2023-01-01 08:01:26</td>
      <td>2023-01-01 09:38:26</td>
      <td>0</td>
      <td>33</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>497</td>
      <td>10</td>
      <td>2023-01-01 08:20:52</td>
      <td>2023-01-01 11:20:52</td>
      <td>0</td>
      <td>59</td>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
    </tr>
  </tbody>
</table>
</div>



Some patients are present at more than one of the prediction times, given them more than one entry in snapshots_df


```python
snapshots_df.visit_number.value_counts()
```




    visit_number
    1784    4
    1342    3
    1432    3
    1292    3
    1604    3
           ..
    777     1
    787     1
    774     1
    770     1
    2238    1
    Name: count, Length: 1675, dtype: int64




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
      <th>is_admitted</th>
      <th>age</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1512</th>
      <td>459</td>
      <td>1784</td>
      <td>2023-03-13 05:47:42</td>
      <td>2023-03-13 18:59:42</td>
      <td>0</td>
      <td>48</td>
      <td>2023-03-13</td>
      <td>(6, 0)</td>
    </tr>
    <tr>
      <th>1513</th>
      <td>459</td>
      <td>1784</td>
      <td>2023-03-13 05:47:42</td>
      <td>2023-03-13 18:59:42</td>
      <td>0</td>
      <td>48</td>
      <td>2023-03-13</td>
      <td>(9, 30)</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>459</td>
      <td>1784</td>
      <td>2023-03-13 05:47:42</td>
      <td>2023-03-13 18:59:42</td>
      <td>0</td>
      <td>48</td>
      <td>2023-03-13</td>
      <td>(12, 0)</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>459</td>
      <td>1784</td>
      <td>2023-03-13 05:47:42</td>
      <td>2023-03-13 18:59:42</td>
      <td>0</td>
      <td>48</td>
      <td>2023-03-13</td>
      <td>(15, 30)</td>
    </tr>
  </tbody>
</table>
</div>



## Creating fake finished visits - a more complicated example 

In an EHR, information about the patient accumulates as the ED visit progresses. Patients may visit various locations in the ED, such as triage, where their acuity is recorded, and they have various different things done to them, like measurements of vital signs or lab tests. 

The function below returns three fake dataframes, meant to resemble EHR data. 

- hospital visit
- observations - with a single measurement - a triage score - plus a timestamp for when that was recorded
- lab orders - with five types of lab orders plus a timestamp for when that test was requested

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
      <td>14</td>
      <td>2023-01-01 03:35:36.163784</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>2023-01-01 04:29:37.764451</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2023-01-01 05:23:25.286763</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2023-01-01 08:10:51.432211</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>2023-01-01 08:26:20.775693</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



The function that creates the fake data returns a random number of lab tests for each patient, for visits over 2 hours 



```python
print(f'There are {len(lab_orders_df)} lab orders in the dataset, for {len(lab_orders_df.visit_number.unique())} visits')
lab_orders_df.head()
```

    There are 5177 lab orders in the dataset, for 1874 visits





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
      <td>14</td>
      <td>2023-01-01 03:46:03.833071</td>
      <td>CBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>2023-01-01 04:42:28.890734</td>
      <td>CBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>2023-01-01 05:08:21.224579</td>
      <td>D-dimer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2023-01-01 05:23:31.838338</td>
      <td>BMP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>2023-01-01 05:45:18.145241</td>
      <td>BMP</td>
    </tr>
  </tbody>
</table>
</div>



## Create snapshots from fake data - a more complicated example

A function called `create_fake_snapshots()` will pull information from all three tables. Note that this function has been designed to work with the fake data generated above. You would need to create your own version of this function, to handle the data you have. 


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
      <th>num_cbc_orders</th>
      <th>num_d-dimer_orders</th>
      <th>num_bmp_orders</th>
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
      <td>1658</td>
      <td>14</td>
      <td>0</td>
      <td>30</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>238</td>
      <td>20</td>
      <td>1</td>
      <td>61</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01</td>
      <td>(6, 0)</td>
      <td>354</td>
      <td>1</td>
      <td>1</td>
      <td>86</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>114</td>
      <td>3</td>
      <td>0</td>
      <td>33</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>497</td>
      <td>10</td>
      <td>0</td>
      <td>59</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <th>num_cbc_orders</th>
      <th>num_d-dimer_orders</th>
      <th>num_bmp_orders</th>
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
      <th>1512</th>
      <td>2023-03-13</td>
      <td>(6, 0)</td>
      <td>459</td>
      <td>1784</td>
      <td>0</td>
      <td>48</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1513</th>
      <td>2023-03-13</td>
      <td>(9, 30)</td>
      <td>459</td>
      <td>1784</td>
      <td>0</td>
      <td>48</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>2023-03-13</td>
      <td>(12, 0)</td>
      <td>459</td>
      <td>1784</td>
      <td>0</td>
      <td>48</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>2023-03-13</td>
      <td>(15, 30)</td>
      <td>459</td>
      <td>1784</td>
      <td>0</td>
      <td>48</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

Here I have shown how to create snapshots from finished patient visits. Note that there is a summarisation element going on. Above, there are counts of the number of lab orders, and the latest triage score. In the same vein, you might just take the last recorded heart rate or oxygen saturation level, or the latest value of a lab result. A snapshot loses some of the richness of the full data in an EHR, but with the benefit that you get data that replicates unfinished visits. 

You might ask why we don't use time series data, to hang on to that richness. The main reason is that hospital visit data can be very patchy, with a lot of missingness. For example, in the ED, a severely ill patient might have enough heart rate values recorded to constitute a time series, while a non-acute patient (say someone with a sprained ankle) might have one or no heart rate measurements. In the case of predicting probability of admission after ED, the absence of data is revealing in itself. By summarising, snapshots allow us to capture that variation in data completeness. 

In the next notebook I'll show how to make predictions using patient snapshots.





