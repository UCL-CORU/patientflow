# Create patient-level snapshots

## About snapshots

I'm [Zella King](https://github.com/zmek/), a health data scientist in the Clinical Operational Research Unit (CORU) at University College London. Since 2020, I have worked with University College London Hospital (UCLH) on practical tools to improve patient flow through the hospital. With a team from UCLH, I developed a predictive tool that is now in daily use by bed managers at the hospital. 

The tool we built for UCLH takes a 'snapshot' of patients in the hospital at a point in time, and using data from the hospital's electronic record system, predicts the number of emergency admissions in the next 8 or 12 hours. We are working on predicting discharges in the same way. 

The key principle is that we take data on hospital visits that are unfinished, and predict whether some outcome (admission from A&E, discharge from hospital, or transfer to another clinical specialty) will happen to each of those patients in a window of time. What the outcome is doesn't really matter; the same methods can be used. 

The utility of our approach - and the thing that makes it very generalisable - is that we then build up from the patient-level predictions into a predictions for a whole cohort of patients at a point in time. That step is what creates useful information for bed managers.

Here I show what I mean by a snapshot, and suggest how to prepare them. 

## How to create patient level snapshots

Below is some fake data resembling the structure of data on ED visits that is typical of the data warehouse of an Electronic Health Record (EHR) system. Each visit has one row in visits_df, with the patient's age and an outcome of whether they were admitted after the ED visit. 


```python
# Reload functions every time
%load_ext autoreload 
%autoreload 2
```


```python
from patientflow.generate import patient_visits
visits_df, observations_df, lab_orders_df = patient_visits('2023-01-01', '2023-01-31', 25)

print(f'There are {len(visits_df)} visits in the dataset, between {visits_df.arrival_datetime.min()} and {visits_df.arrival_datetime.max()}')
visits_df.head()
```

    There are 736 visits in the dataset, between 2023-01-01 07:06:59 and 2023-01-31 21:21:27





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
      <th>arrival_datetime</th>
      <th>departure_datetime</th>
      <th>is_admitted</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>2023-01-01 07:06:59</td>
      <td>2023-01-01 11:39:59</td>
      <td>0</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>2023-01-01 08:10:28</td>
      <td>2023-01-01 15:52:28</td>
      <td>1</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2023-01-01 08:34:49</td>
      <td>2023-01-01 11:37:49</td>
      <td>0</td>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2023-01-01 09:14:25</td>
      <td>2023-01-01 11:48:25</td>
      <td>1</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>2023-01-01 09:22:21</td>
      <td>2023-01-01 12:06:21</td>
      <td>0</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>



In an EHR, information about the patient accumulates as the ED visit progresses. Patients may visit various locations in the ED, such as triage, where their acuity is recorded, and they have various different things done to them, like measurements of vital signs or lab tests. 

The function above returns a observations_df table, with a single measurement - a triage score - plus a timestamp for when that was recorded. In the observations_df dataframe, every visit has a triage score within 10 minutes of arrival.


```python
print(f'There are {len(observations_df)} triage scores in the observations_df dataframe, for {len(observations_df.visit_number.unique())} visits')
observations_df.head()
```

    There are 736 triage scores in the observations_df dataframe, for 736 visits





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
      <td>8</td>
      <td>2023-01-01 07:16:56.752220</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>2023-01-01 08:14:42.082565</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>2023-01-01 08:42:47.977107</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2023-01-01 09:22:44.300841</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2023-01-01 09:23:29.539884</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Some patients might have a lab test ordered. In the fake data, this has been set up so that orders are placed within 90 minutes of arrival. 


```python
print(f'There are {len(lab_orders_df)} lab orders in the dataset, for {len(lab_orders_df.visit_number.unique())} visits')
lab_orders_df.head()
```

    There are 1811 lab orders in the dataset, for 621 visits





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
      <td>8</td>
      <td>2023-01-01 07:39:15.961554</td>
      <td>Troponin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>2023-01-01 08:25:39.163370</td>
      <td>BMP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>2023-01-01 08:27:35.390640</td>
      <td>BMP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>2023-01-01 08:35:09.882306</td>
      <td>Urinalysis</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>2023-01-01 09:04:05.742878</td>
      <td>CBC</td>
    </tr>
  </tbody>
</table>
</div>



Our goal is to create snapshots of these visits at a point in time. First, we define the times of day we will be issuing predictions at. 


```python
prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)] # each time is expressed as a tuple of (hour, minute)
```

Then we iterate through the dataset at these times, to create a series of snapshots. I'm deliberately exposing the code here so that you can see how this is done. Each snapshot summarises what is know about the patient at the time of the snapshot. The latest triage score is recorded, and a count of each type of lab orders.






```python
from datetime import date
start_date = date(2023, 1, 1)
end_date = date(2023, 1, 31)

from patientflow.generate import create_snapshots

# Create snapshots
snapshots_df = create_snapshots(visits_df, observations_df, lab_orders_df, prediction_times, start_date, end_date)
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
      <th>snapshot_date</th>
      <th>prediction_time</th>
      <th>snapshot_datetime</th>
      <th>visit_number</th>
      <th>arrival_datetime</th>
      <th>departure_datetime</th>
      <th>is_admitted</th>
      <th>latest_triage_score</th>
      <th>num_troponin_orders</th>
      <th>num_bmp_orders</th>
      <th>num_urinalysis_orders</th>
      <th>num_cbc_orders</th>
      <th>num_d-dimer_orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>2023-01-01 09:30:00</td>
      <td>8</td>
      <td>2023-01-01 07:06:59</td>
      <td>2023-01-01 11:39:59</td>
      <td>0</td>
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
      <td>2023-01-01 09:30:00</td>
      <td>21</td>
      <td>2023-01-01 08:10:28</td>
      <td>2023-01-01 15:52:28</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>2023-01-01 09:30:00</td>
      <td>14</td>
      <td>2023-01-01 08:34:49</td>
      <td>2023-01-01 11:37:49</td>
      <td>0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>2023-01-01 09:30:00</td>
      <td>2</td>
      <td>2023-01-01 09:14:25</td>
      <td>2023-01-01 11:48:25</td>
      <td>1</td>
      <td>1.0</td>
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
      <td>2023-01-01 09:30:00</td>
      <td>15</td>
      <td>2023-01-01 09:22:21</td>
      <td>2023-01-01 12:06:21</td>
      <td>0</td>
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



## Train a model to predict the outcome of each snapshot



```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, List, Tuple, Any

def train_admission_model(
    snapshots_df: pd.DataFrame,
    prediction_time: Tuple[int, int],
    exclude_from_training_data: List[str],
    ordinal_mappings: Dict[str, List[Any]]
):
    """
    Train an XGBoost model to predict patient admission based on filtered data.
    
    Parameters:
    -----------
    snapshots_df : pandas.DataFrame
        DataFrame containing patient snapshot data
    prediction_time : Tuple[int, int]
        The specific (hour, minute) tuple to filter training data by
    exclude_from_training_data : List[str]
        List of column names to exclude from model training
    ordinal_mappings : Dict[str, List[Any]]
        Dictionary mapping column names to ordered categories for ordinal encoding
    
    Returns:
    --------
    tuple
        (trained_model, X_test, y_test, accuracy, feature_importances)
    """
    # Filter data for the specific prediction time
    filtered_df = snapshots_df[snapshots_df['prediction_time'].apply(lambda x: x == prediction_time)]
    
    if filtered_df.empty:
        raise ValueError(f"No data found for prediction time {prediction_time}")
    
    # Prepare feature columns - exclude specified columns and target variable
    all_columns = filtered_df.columns.tolist()
    exclude_cols = exclude_from_training_data + ['is_admitted', 'prediction_time', 'snapshot_date', 'snapshot_datetime']
    feature_cols = [col for col in all_columns if col not in exclude_cols]
    
    # Create feature matrix
    X = filtered_df[feature_cols].copy()
    y = filtered_df['is_admitted']
    
    # Apply ordinal encoding to categorical features
    for col, categories in ordinal_mappings.items():
        if col in X.columns:
            # Create an ordinal encoder with the specified categories
            encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=np.nan)
            # Reshape the data for encoding and back
            X[col] = encoder.fit_transform(X[[col]])
    
    # One-hot encode any remaining categorical columns
    X = pd.get_dummies(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and train the XGBoost model with default settings
    model = XGBClassifier(
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Return the model, test data, and feature importances
    return model, X_test, y_test, accuracy, feature_importances
```

Let's train a model to predict admission for the 9:30 prediction time. We will specify that the triage scores are ordinal, and make use of sklearn's OrdinalEncoder to maintain the natural order of categories. We also need to include columns that are not relevant to the snapshot. 


```python

model, X_test, y_test, accuracy, feature_importances = train_admission_model(
    snapshots_df,
    prediction_time=(9, 30),
    exclude_from_training_data=['visit_number', 'arrival_datetime', 'departure_datetime'],
    ordinal_mappings={'latest_triage_score': [1, 2, 3, 4, 5]}
)
```


```python
feature_importances
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
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>latest_triage_score</td>
      <td>0.513054</td>
    </tr>
    <tr>
      <th>3</th>
      <td>num_urinalysis_orders</td>
      <td>0.179355</td>
    </tr>
    <tr>
      <th>5</th>
      <td>num_d-dimer_orders</td>
      <td>0.102420</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num_troponin_orders</td>
      <td>0.084515</td>
    </tr>
    <tr>
      <th>2</th>
      <td>num_bmp_orders</td>
      <td>0.076452</td>
    </tr>
    <tr>
      <th>4</th>
      <td>num_cbc_orders</td>
      <td>0.044204</td>
    </tr>
  </tbody>
</table>
</div>



The output below shows the predicted probability for the first patient in the test set


```python
print(f"The predicted probability of admission for the first patient is {model.predict_proba(X_test)[0][1]:.2f}")
```

    The predicted probability of admission for the first patient is 0.13


## Conclusion

Here I have shown 

* how to create snapshots from finished patient visits
* how to train a very simple model to predict admission at the end of the snapshot. 

This creates a predicted probability of admission for each patient, based on what is known about them at the time of the snapshot. However, bed managers really want predictions for the whole cohort of patients in the ED at a point in time. This is where `patientflow` comes into its own. In the next notebook, I show how to do this. 


