# Turning individual patient-level predictions into predictions of how many beds will be needed

In the previous notebook [4a_Predict_probability_of_admission_from_ED.md](4a_Predict_probability_of_admission_from_ED.md), we created a model that will generate a probability of admission for each individual patient in the ED or SDEC. Since the objective of this modelling is to predict number of beds needed for a group of patients in the ED at a prediction moment, this notebook shows how to convert the individual-level probabilities into aggregate predictions.

The patientflow python package includes a function called `get_prob_dist()` in a script called `aggregate.py` that will do the aggregating step. For each prediction moment, at which a set of patients were in ED or SDEC, this function generates a probability distribution showing how many beds will be needed by those patients. 

The function expects several inputs, including a dictionary - here called `snapshots_dict` - in which each key is a snapshot datetime and the values are an array of `snapshot_id` for each patient in the ED/SDEC at that snapshot datetime. 


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


## Load parameters and set file paths, and load data


```python
import pandas as pd
from patientflow.load import load_data, set_file_paths
from patientflow.prepare import create_temporal_splits


# set file paths
data_folder_name = 'data-public'
data_file_path = project_root / data_folder_name

data_file_path, media_file_path, model_file_path, config_path = set_file_paths(project_root, 
               data_folder_name=data_folder_name)

# load data
ed_visits = load_data(data_file_path, 
                    file_name='ed_visits.csv', 
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"], 
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])

# load params
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

# load params
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

# get test set from the original data
_, _, test_visits = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
)
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-public
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public/media
    Split sizes: [53801, 6519, 19494]


## Generate aggregate predictions for one time of day 15:30 

In the previous step, shown in notebook [4a_Predict_probability_of_admission_from_ED.md](4a_Predict_probability_of_admission_from_ED.md), we trained five models, one for each prediction time. Here we'll focus on the 15:30 prediction time. 

The first step is to retrieve data on patients in the ED at that time of day, and use a saved model to make a prediction for each of them. 

The function in the cell below formats the name of the model based on the time of day


```python
from patientflow.load import get_model_key
prediction_time = tuple([15,30])
model_name = 'admissions_balanced_calibrated'
model_key = get_model_key(model_name, prediction_time)
print(f'The name of the model to be loaded is {model_key}. It will be loaded from {model_file_path}')
```

    The name of the model to be loaded is admissions_balanced_calibrated_1530. It will be loaded from /Users/zellaking/Repos/patientflow/trained-models/public


Next we use the `prepare_for_inference()` function. This does several things:

* loads the trained model into a variable called `model`
* reloads the original data, selects the test set records only and takes the subset of snaphots at 15:30 in the afternoon
* prepares this subset for input into the trained model, which is returned in a variable called `X_test`. (Note that here we are not using X_test for training the model, but for inference - inference means that we are asking atrained   model to make predictions. The model will expect the input data to be in the same format as it received when it was trained)
* returns an array of values `y_test` which is a binary variable whether each patient was actually admitted. This will be used to evaluate the model. 


```python
from patientflow.prepare import prepare_for_inference

exclude_from_training_data = [ 'snapshot_date', 'prediction_time','consultation_sequence', 'visit_number', 'specialty', 'final_sequence', 'training_validation_test']

X_test, y_test, pipeline = prepare_for_inference(
    model_file_path, 
    model_name,
    prediction_time,
    model_only=False,
    df=test_visits,
    single_snapshot_per_visit=False,
    exclude_from_training_data=exclude_from_training_data)
```

We can also view the data that was loaded for running inference on the model


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
      <th>...</th>
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
      <th>2</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>59741</th>
      <td>25080</td>
      <td>F</td>
      <td>18-24</td>
      <td>NaN</td>
      <td>sdec</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>10</td>
      <td>5</td>
      <td>...</td>
      <td>79.0</td>
      <td>0.375</td>
      <td>3.9</td>
      <td>NaN</td>
      <td>140.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.41</td>
      <td>39.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60002</th>
      <td>11160</td>
      <td>F</td>
      <td>65-74</td>
      <td>NaN</td>
      <td>sdec_waiting</td>
      <td>4</td>
      <td>14</td>
      <td>2</td>
      <td>14</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>60261</th>
      <td>16680</td>
      <td>F</td>
      <td>35-44</td>
      <td>NaN</td>
      <td>sdec_waiting</td>
      <td>3</td>
      <td>20</td>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>60278</th>
      <td>24060</td>
      <td>F</td>
      <td>75-102</td>
      <td>Ambulance</td>
      <td>majors</td>
      <td>5</td>
      <td>67</td>
      <td>7</td>
      <td>27</td>
      <td>8</td>
      <td>...</td>
      <td>71.0</td>
      <td>0.421</td>
      <td>4.4</td>
      <td>1.2</td>
      <td>138.0</td>
      <td>6.44</td>
      <td>7.394</td>
      <td>5.48</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>



Next we prepare a dictionary of snapshots (which is the term used here to refer to moments in time when we want to make predictions). The key for each dictionary entry is a snapshot datetime. The values are snapshot_ids for patients in the ED or SDEC at that time.

The output of the following cell shows the first 10 keys, and the first set of values - simply a list of snapshot_ids


```python
from patientflow.prepare import prepare_group_snapshot_dict

# select the snapshots to include in the probability distribution, 
snapshots_dict = prepare_group_snapshot_dict(
    test_visits[test_visits.prediction_time == prediction_time]
    )

print("First 10 keys in the snapshots dictionary")
print(list(snapshots_dict.keys())[0:10])

first_record_key = list(snapshots_dict.keys())[0]
first_record_values = snapshots_dict[first_record_key]

print("\nRecord associated with the first key")
print(first_record_values)
```

    First 10 keys in the snapshots dictionary
    ['10/1/2031', '10/10/2031', '10/11/2031', '10/12/2031', '10/13/2031', '10/14/2031', '10/15/2031', '10/16/2031', '10/17/2031', '10/18/2031']
    
    Record associated with the first key
    [59741, 60002, 60261, 60278, 60317, 60324, 60335, 60351, 60358, 60363, 60375, 60380, 60383, 60385, 60387, 60389, 60400, 60405, 60408, 60409, 60410, 60411, 60412, 60413, 60414, 60415, 60416, 60417, 60418, 60419, 60420, 60421, 60422, 60423, 60424, 60425, 60426, 60427, 60428, 60429, 60430, 60431, 60432, 60433, 60434, 60435, 60436, 60437, 60438, 60443, 60444, 60445, 60448, 60449, 60451, 60452, 60453, 60454, 60455, 60456]


To see the snapshots associated with this first key in the snapshots dictionary, use the values it returns to retrieve the relevant rows in the original visits dataset.


```python
ed_visits.loc[first_record_values].head()
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
      <th>...</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59741</th>
      <td>10/1/2031</td>
      <td>(15, 30)</td>
      <td>135639</td>
      <td>25080</td>
      <td>F</td>
      <td>18-24</td>
      <td>NaN</td>
      <td>sdec</td>
      <td>5</td>
      <td>10</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.41</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>test</td>
      <td>['medical']</td>
      <td>True</td>
      <td>15383</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>60002</th>
      <td>10/1/2031</td>
      <td>(15, 30)</td>
      <td>135923</td>
      <td>11160</td>
      <td>F</td>
      <td>65-74</td>
      <td>NaN</td>
      <td>sdec_waiting</td>
      <td>4</td>
      <td>14</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['ambulatory']</td>
      <td>False</td>
      <td>71795</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60261</th>
      <td>10/1/2031</td>
      <td>(15, 30)</td>
      <td>136201</td>
      <td>16680</td>
      <td>F</td>
      <td>35-44</td>
      <td>NaN</td>
      <td>sdec_waiting</td>
      <td>3</td>
      <td>20</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['ambulatory']</td>
      <td>False</td>
      <td>200</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60278</th>
      <td>10/1/2031</td>
      <td>(15, 30)</td>
      <td>136221</td>
      <td>24060</td>
      <td>F</td>
      <td>75-102</td>
      <td>Ambulance</td>
      <td>majors</td>
      <td>5</td>
      <td>67</td>
      <td>...</td>
      <td>6.44</td>
      <td>7.394</td>
      <td>5.48</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['acute']</td>
      <td>True</td>
      <td>37734</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>60317</th>
      <td>10/1/2031</td>
      <td>(15, 30)</td>
      <td>136275</td>
      <td>51810</td>
      <td>M</td>
      <td>45-54</td>
      <td>NaN</td>
      <td>sdec</td>
      <td>12</td>
      <td>84</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['surgical']</td>
      <td>False</td>
      <td>21593</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 69 columns</p>
</div>



The following cell shows the number of patients who were in the ED at that snapshot datetime


```python
print(len(ed_visits.loc[first_record_values]))
```

    60


With the model, the snapshot dictionary, X_test and y_test as inputs, the get_prob_dist() function is called. It returns a dictionary, with the same keys as before, and with a probability distribution for each snapshot date in the test set. As this processes each snapshot date separately, it may take some time. 


```python
from patientflow.aggregate import get_prob_dist
# get probability distribution for this time of day
prob_dist = get_prob_dist(
        snapshots_dict, X_test, y_test, pipeline
    )
```

    Calculating probability distributions for 92 snapshot dates
    This may take a minute or more
    Processed 10 snapshot dates
    Processed 20 snapshot dates
    Processed 30 snapshot dates
    Processed 40 snapshot dates
    Processed 50 snapshot dates
    Processed 60 snapshot dates
    Processed 70 snapshot dates
    Processed 80 snapshot dates
    Processed 90 snapshot dates
    Processed 92 snapshot dates


The cell below shows the entry in the `prob_dist` dictionary (which is the first snapshot date in the test set) and the probability distribution associated with that date. 


```python

print("First key in the prob dist dictionary")
print(list(prob_dist.keys())[0])

print("Probability distribution for first snapshot datetime")
prob_dist[first_record_key]
```

    First key in the prob dist dictionary
    10/1/2031
    Probability distribution for first snapshot datetime





    {'agg_predicted':                agg_proba
     0    3.83000095709416e-8
     1    8.91447105034246e-7
     2    9.96474713480853e-6
     3    7.12986776280389e-5
     4   0.000367215548706615
     ..                   ...
     56  2.70016724906273e-41
     57  2.17495414369041e-43
     58  1.27411141069909e-45
     59  4.82803167661050e-48
     60  8.88078644945381e-51
     
     [61 rows x 1 columns],
     'agg_observed': 11}



To make this output more readable, we can redisplay it like this


```python
first_date_prob_dist = prob_dist[first_record_key]['agg_predicted'].rename(columns = {'agg_proba': 'probability'})
first_date_prob_dist.index.name = 'number of beds'

print(f"Probability of needing this number of beds on {first_record_key} at {prediction_time} based on EHR data from patients in the ED at that time")
display(first_date_prob_dist.head(15))

```

    Probability of needing this number of beds on 10/1/2031 at (15, 30) based on EHR data from patients in the ED at that time



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
      <th>probability</th>
    </tr>
    <tr>
      <th>number of beds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.83000095709416e-8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.91447105034246e-7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.96474713480853e-6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.12986776280389e-5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000367215548706615</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00145156182973333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.00458531140812794</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0119004455059500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0258924729749449</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0479546989485299</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0765094618936776</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.106161370669358</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.129109543666885</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.138509425956049</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.131784140162024</td>
    </tr>
  </tbody>
</table>
</div>


We can plot this probability distribution using a function from the patientflow package


```python
from patientflow.viz.prob_dist_plot import prob_dist_plot

title_ = f'Probability distribution for number of beds needed by patients in ED at {first_record_key} {prediction_time[0]:02d}:{prediction_time[1]:02d}'
prob_dist_plot(prob_dist_data=prob_dist[first_record_key]['agg_predicted'], 
    title=title_,  
    include_titles=True)
```


    
![png](4b_Predict_demand_from_patients_in_ED_files/4b_Predict_demand_from_patients_in_ED_25_0.png)
    


From the output above, we can see how many beds are predicted to be needed by the patients in the ED/SDEC at this particular snapshot. It gives a range of probability, rather than a single estimate. In a later notebook [5_Evaluate_model_performance.md](5_Evaluate_model_performance.md) we show how to evaluate the model's predictions. 

### Reading a minimum number of beds needed from the probability distribution

Our input for the UCLH output does not show a probability distribution like this. It shows 'at least this number of beds needed' with 90% and 70% probability. (Check back to notebook 2 for this). To calculate this from the distribution given, we illustrate first by added a cumulative probability to the dataframe created earlier.


```python
first_date_prob_dist['cumulative probability'] = first_date_prob_dist['probability'].cumsum()
first_date_prob_dist['probability of needing at least this number or beds'] = first_date_prob_dist['cumulative probability'].apply(lambda x: 1 -x)
display(first_date_prob_dist.head(20))
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
      <th>probability</th>
      <th>cumulative probability</th>
      <th>probability of needing at least this number or beds</th>
    </tr>
    <tr>
      <th>number of beds</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.83000095709416e-8</td>
      <td>3.83000095709416e-8</td>
      <td>0.999999961699990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.91447105034246e-7</td>
      <td>9.29747114605188e-7</td>
      <td>0.999999070252885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.96474713480853e-6</td>
      <td>1.08944942494137e-5</td>
      <td>0.999989105505751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.12986776280389e-5</td>
      <td>8.21931718774526e-5</td>
      <td>0.999917806828123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000367215548706615</td>
      <td>0.000449408720584068</td>
      <td>0.999550591279416</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00145156182973333</td>
      <td>0.00190097055031739</td>
      <td>0.998099029449683</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.00458531140812794</td>
      <td>0.00648628195844533</td>
      <td>0.993513718041555</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0119004455059500</td>
      <td>0.0183867274643953</td>
      <td>0.981613272535605</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0258924729749449</td>
      <td>0.0442792004393402</td>
      <td>0.955720799560660</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0479546989485299</td>
      <td>0.0922338993878701</td>
      <td>0.907766100612130</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0765094618936776</td>
      <td>0.168743361281548</td>
      <td>0.831256638718452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.106161370669358</td>
      <td>0.274904731950905</td>
      <td>0.725095268049095</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.129109543666885</td>
      <td>0.404014275617790</td>
      <td>0.595985724382210</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.138509425956049</td>
      <td>0.542523701573839</td>
      <td>0.457476298426161</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.131784140162024</td>
      <td>0.674307841735864</td>
      <td>0.325692158264136</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.111707503010854</td>
      <td>0.786015344746718</td>
      <td>0.213984655253282</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0846872810065416</td>
      <td>0.870702625753259</td>
      <td>0.129297374246741</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0576121112605319</td>
      <td>0.928314737013791</td>
      <td>0.0716852629862087</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0352707916903104</td>
      <td>0.963585528704102</td>
      <td>0.0364144712958984</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0194806642771849</td>
      <td>0.983066192981287</td>
      <td>0.0169338070187135</td>
    </tr>
  </tbody>
</table>
</div>


From this cumulative probability we can read off the number of beds where there is a 90% chance of needing at least this number



```python
row_indicating_at_least_90_pc = first_date_prob_dist[first_date_prob_dist['probability of needing at least this number or beds'] < 0.9].index[0]
print(f"There is a 90% chance of needing at least {row_indicating_at_least_90_pc} beds")
```

    There is a 90% chance of needing at least 10 beds


The predict module has a function for reading off the cdf in this way 


```python
from patientflow.predict.emergency_demand import index_of_sum
??index_of_sum
```

    [0;31mSignature:[0m [0mindex_of_sum[0m[0;34m([0m[0msequence[0m[0;34m:[0m [0mList[0m[0;34m[[0m[0mfloat[0m[0;34m][0m[0;34m,[0m [0mmax_sum[0m[0;34m:[0m [0mfloat[0m[0;34m)[0m [0;34m->[0m [0mint[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m   
    [0;32mdef[0m [0mindex_of_sum[0m[0;34m([0m[0msequence[0m[0;34m:[0m [0mList[0m[0;34m[[0m[0mfloat[0m[0;34m][0m[0;34m,[0m [0mmax_sum[0m[0;34m:[0m [0mfloat[0m[0;34m)[0m [0;34m->[0m [0mint[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"""Returns the index where the cumulative sum of a sequence of probabilities exceeds max_sum."""[0m[0;34m[0m
    [0;34m[0m    [0mcumulative_sum[0m [0;34m=[0m [0;36m0.0[0m[0;34m[0m
    [0;34m[0m    [0;32mfor[0m [0mi[0m[0;34m,[0m [0mvalue[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0msequence[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0mcumulative_sum[0m [0;34m+=[0m [0mvalue[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mcumulative_sum[0m [0;34m>=[0m [0;36m1[0m [0;34m-[0m [0mmax_sum[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0;32mreturn[0m [0mi[0m[0;34m[0m
    [0;34m[0m    [0;32mreturn[0m [0mlen[0m[0;34m([0m[0msequence[0m[0;34m)[0m [0;34m-[0m [0;36m1[0m  [0;31m# Return the last index if the sum doesn't exceed max_sum[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      ~/Repos/patientflow/src/patientflow/predict/emergency_demand.py
    [0;31mType:[0m      function


```python
sequence = first_date_prob_dist['probability'].values
print(f"There is a 90% chance of needing at least {index_of_sum(sequence, 0.9)} beds, using the index_of_sum function")

```

    There is a 90% chance of needing at least 10 beds, using the index_of_sum function


And this can be done from the original probability distribution


```python
sequence = prob_dist[first_record_key]['agg_predicted']['agg_proba'].values
cdf_cut_points = [0.9, 0.7]
for cut_point in cdf_cut_points:
    num_beds = index_of_sum(sequence, cut_point)
    print(f"At least {num_beds} beds needed with {int(cut_point*100)}% probability")

```

    At least 10 beds needed with 90% probability
    At least 12 beds needed with 70% probability



```python

```


```python

```
