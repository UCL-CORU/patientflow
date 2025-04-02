# Putting it all together

This notebook shows an example of how to convert data from the saved datasets, plus models trained earlier, into the output show below in columns D-G. 
[WORK IN PROGRESS] 


```python
from IPython.display import Image
Image(filename='img/UCLH application with annotation.jpg')
```




    
![jpeg](4f_Bring_it_all_together_files/4f_Bring_it_all_together_1_0.jpg)
    



In order to recreate the output this notebook, prior steps are

* train a model predicting admission to ED (this is done in notebook [4a_Predict_probability_of_admission_from_ED.md](4a_Predict_probability_of_admission_from_ED.md))
* train a model predicting admisson to each specialty if admitted (this is done in notebook [4c_Predict_probability_of_admission_to_specialty.md](4c_Predict_probability_of_admission_to_specialty.md))
* train a model predicting demand from patients yet-to-arrive (this is done in notebook [4d_Predict_demand_from_patients_yet_to_arrive.md](4d_Predict_demand_from_patients_yet_to_arrive.md))


## Set up notebook environment


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


```python
import pandas as pd
from patientflow.load import load_data
from patientflow.load import set_file_paths

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
prediction_times = params["prediction_times"]

x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
prediction_window = params["prediction_window"]
epsilon = float(params["epsilon"])
yta_time_interval = params["yta_time_interval"]

print(f'\nTraining set starts {start_training_set} and ends on {start_validation_set - pd.Timedelta(days=1)} inclusive')
print(f'Validation set starts on {start_validation_set} and ends on {start_test_set - pd.Timedelta(days=1)} inclusive' )
print(f'Test set starts on {start_test_set} and ends on {end_test_set- pd.Timedelta(days=1)} inclusive' )

print(f'\nThe coordinates used to derive the aspirational curve are ({int(x1)},{y1}) and ({int(x2)},{y2})')
print(f'The prediction window over which prediction will be made is {prediction_window/60} hours')
print(f'In order to calculate yet-to-arrive rates of arrival, the prediction window will be divied into intervals of {yta_time_interval} minutes')
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-public
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public/media
    
    Training set starts 2031-03-01 and ends on 2031-08-31 inclusive
    Validation set starts on 2031-09-01 and ends on 2031-09-30 inclusive
    Test set starts on 2031-10-01 and ends on 2031-12-31 inclusive
    
    The coordinates used to derive the aspirational curve are (4,0.76) and (12,0.99)
    The prediction window over which prediction will be made is 8.0 hours
    In order to calculate yet-to-arrive rates of arrival, the prediction window will be divied into intervals of 15 minutes


## Pick a random row to simulate the real-time environment


```python
from datetime import datetime, time

# Set seed
import numpy as np
np.random.seed(2404)

# Randomly pick a prediction moment to do inference on
random_row = ed_visits[ed_visits.training_validation_test == 'test'].sample(n=1)
prediction_time = random_row.prediction_time.values[0]
prediction_date = random_row.snapshot_date.values[0]
prediction_moment = datetime.combine(pd.to_datetime(prediction_date).date(), datetime.min.time()).replace(hour=prediction_time[0], minute=prediction_time[1])

prediction_snapshots = ed_visits[(ed_visits.prediction_time == prediction_time) & \
            (ed_visits.snapshot_date == prediction_date)]
prediction_snapshots
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
      <th>77829</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153328</td>
      <td>80793</td>
      <td>M</td>
      <td>75-102</td>
      <td>Ambulance</td>
      <td>sdec</td>
      <td>7</td>
      <td>112</td>
      <td>...</td>
      <td>5.78</td>
      <td>7.392</td>
      <td>10.31</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>test</td>
      <td>['acute']</td>
      <td>True</td>
      <td>63866</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>77935</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153425</td>
      <td>33893</td>
      <td>F</td>
      <td>25-34</td>
      <td>Public Trans</td>
      <td>sdec</td>
      <td>5</td>
      <td>34</td>
      <td>...</td>
      <td>4.84</td>
      <td>7.394</td>
      <td>9.71</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>test</td>
      <td>['obs_gyn']</td>
      <td>False</td>
      <td>25439</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77947</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153435</td>
      <td>9900</td>
      <td>F</td>
      <td>35-44</td>
      <td>NaN</td>
      <td>sdec</td>
      <td>4</td>
      <td>16</td>
      <td>...</td>
      <td>6.34</td>
      <td>7.353</td>
      <td>9.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['obs_gyn']</td>
      <td>False</td>
      <td>36175</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77951</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153437</td>
      <td>28398</td>
      <td>F</td>
      <td>25-34</td>
      <td>Walk-in</td>
      <td>majors</td>
      <td>4</td>
      <td>79</td>
      <td>...</td>
      <td>5.39</td>
      <td>7.402</td>
      <td>15.27</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>test</td>
      <td>['obs_gyn']</td>
      <td>False</td>
      <td>65532</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>77960</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153445</td>
      <td>26421</td>
      <td>F</td>
      <td>75-102</td>
      <td>Ambulance</td>
      <td>majors</td>
      <td>4</td>
      <td>56</td>
      <td>...</td>
      <td>5.31</td>
      <td>7.410</td>
      <td>3.96</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>test</td>
      <td>['acute']</td>
      <td>True</td>
      <td>57328</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78129</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153611</td>
      <td>3973</td>
      <td>M</td>
      <td>35-44</td>
      <td>Walk-in</td>
      <td>waiting</td>
      <td>2</td>
      <td>7</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>[]</td>
      <td>False</td>
      <td>57945</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78130</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153612</td>
      <td>3839</td>
      <td>F</td>
      <td>25-34</td>
      <td>Walk-in</td>
      <td>waiting</td>
      <td>2</td>
      <td>16</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>[]</td>
      <td>False</td>
      <td>78438</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78131</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153613</td>
      <td>3733</td>
      <td>M</td>
      <td>35-44</td>
      <td>Walk-in</td>
      <td>waiting</td>
      <td>2</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>[]</td>
      <td>False</td>
      <td>75014</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78132</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153614</td>
      <td>3644</td>
      <td>F</td>
      <td>55-64</td>
      <td>Walk-in</td>
      <td>waiting</td>
      <td>1</td>
      <td>16</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>['surgical']</td>
      <td>False</td>
      <td>4122</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78133</th>
      <td>12/23/2031</td>
      <td>(15, 30)</td>
      <td>153615</td>
      <td>3629</td>
      <td>F</td>
      <td>0-17</td>
      <td>Walk-in</td>
      <td>waiting</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>test</td>
      <td>[]</td>
      <td>False</td>
      <td>74758</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>92 rows × 69 columns</p>
</div>



## Generate predictions

The predictions for input into the spreadsheet output are generated by the create_predictions() function


```python
from patientflow.predict.emergency_demand import create_predictions

```

We will load previously created models from disk and save to a dictionary of models


```python
yta_model_name = f"ed_yet_to_arrive_by_spec_{int(prediction_window/60)}_hours"


model_names = {
    "admissions": "admissions",
    "specialty": "ed_specialty",
    "yet_to_arrive": yta_model_name
}

models = dict.fromkeys(model_names)



```


```python
model_file_path
```




    PosixPath('/Users/zellaking/Repos/patientflow/trained-models/public')




```python
from patientflow.load import load_saved_model, get_model_key

# as the admissions models are a dictionary of models, we need to load each one
models["admissions"] = {}
for prediction_time in ed_visits.prediction_time.unique():

    model_name_for_prediction_time = get_model_key("admissions", prediction_time)
    models["admissions"][model_name_for_prediction_time]  = load_saved_model(model_file_path, "admissions", prediction_time)

models["ed_specialty"] = load_saved_model(model_file_path, "specialty")
models[model_names["yet_to_arrive"]] = load_saved_model(model_file_path, yta_model_name)

```


```python
type(models["admissions"][model_name_for_prediction_time])
```




    patientflow.train.emergency_demand.ModelResults



In the cell below we create the predictions for this randomly chosen moment in time: 


```python
from patientflow.predict.emergency_demand import create_predictions


create_predictions(
    models = models,
    model_names=model_names,
    prediction_time = prediction_time,
    prediction_snapshots = prediction_snapshots,
    specialties = ['surgical', 'haem/onc', 'medical', 'paediatric'],
    prediction_window_hrs = prediction_window/60,
    cdf_cut_points =  [0.9, 0.7], 
    x1 = x1,
    y1 = y1,
    x2 = x2, 
    y2 = y2)
```




    {'surgical': {'in_ed': [2, 3], 'yet_to_arrive': [0, 0]},
     'haem/onc': {'in_ed': [0, 1], 'yet_to_arrive': [0, 0]},
     'medical': {'in_ed': [6, 8], 'yet_to_arrive': [0, 1]},
     'paediatric': {'in_ed': [0, 0], 'yet_to_arrive': [0, 0]}}




```python

```
