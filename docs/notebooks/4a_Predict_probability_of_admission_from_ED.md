# Predict ED admission probability

This notebook demonstrates the first stage of prediction, to generate a probability of admission for each patient in the ED. 

As one of the modelling decisions is to send predictions at specified times of day, we tailor the models to these times and train one model for each time. The dataset used for this modelling is derived from snapshots of visits at each time of day. The times of day are set in config.json file in the root directory of this repo. 

A patient episode (visit) may well span more than one of these times, so we need to consider how we will deal with the occurence of multiple snapshots per episode. At each of these times of day, we will use only one training sample from each hospital episode.

Separation of the visits into training, validation and test sets will be done chronologically into a training, validation and test set.

Here the logic for training a model is hidden in a function which uses an XGBoost classifier. We show how to call the function, and how to interrogate the results. You may have your own trained models, in which case skip this notebook and move onto to the one showing the aggregation to bed level [4b_Predict_demand_from_patients_in_ED.md](4b_Predict_demand_from_patients_in_ED.md)

Evaluation will be done separately in [5_Evaluate_model_performance.md](5_Evaluate_model_performance.md). 


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

File paths are defined in `set_file_paths()`. Here, the files are loaded from `data-public` in the root of the repository. When you first download or install this repository, this folder will be empty. You have two options: 

* Copy the two files from the `data-synthetic` folder into the `data-public` folder. Note that, if you run these notebooks with synthetic data, you will get artificially good performance of the models
* Request files that contain real patient data to put in the `data-public` folder; contact the owner of the repository; see the README in the root of the repository


```python
from patientflow.load import set_file_paths

# set file paths
data_folder_name = 'data-public'
data_file_path = project_root / data_folder_name

data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
    project_root, 
    data_folder_name=data_folder_name,
    config_file = 'config.yaml')
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-public
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public/media



```python
import pandas as pd
from patientflow.load import load_data

# load data
ed_visits = load_data(data_file_path, 
                    file_name='ed_visits.csv', 
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"], 
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])
```

Note that, in the output below, each row has a snapshot_id as an index. In the notebooks that follow, we retain the same values of snapshot_id throughout, meaning that they are consistent across the original dataset visits and the training, validation and test subsets of visits. At any point, when looking at output (eg the predicted probability of admission for a patient), you should be able to track that output back to the original row in the ed_visits dataset.


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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
<p>5 rows × 69 columns</p>
</div>



If you are looking at synthetic or public data, the date have been shifted into the future, as shown below. This is to minimise any risk of a patient being identifiable. 


```python
# print start and end dates
print(ed_visits.snapshot_date.min())
print(ed_visits.snapshot_date.max())
```

    10/1/2031
    9/9/2031


## Set modelling parameters

The parameters are used in training or inference. They are set in config.json in the root of the repository and loaded by `load_config_file()`


```python
# load params
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

print(f'\nTraining set starts {start_training_set} and ends on {start_validation_set - pd.Timedelta(days=1)} inclusive')
print(f'Validation set starts on {start_validation_set} and ends on {start_test_set - pd.Timedelta(days=1)} inclusive' )
print(f'Test set starts on {start_test_set} and ends on {end_test_set- pd.Timedelta(days=1)} inclusive' )
```

    
    Training set starts 2031-03-01 and ends on 2031-08-31 inclusive
    Validation set starts on 2031-09-01 and ends on 2031-09-30 inclusive
    Test set starts on 2031-10-01 and ends on 2031-12-31 inclusive


## Prediction times

The data has been prepared as a series of snapshots of each patient's data at five moments during the day. These five moments are the times when the bed managers wish to receive predictive models of emergency demand. If a patient arrives in the ED at 4 am, and leaves at 11 am, they will be represented in the 06:00 and 09:30 prediction times. Everything known about a patient is included up until that moment is included in that snapshot.

The predition times are presented as tuples in the form (hour, minute). 

From the output below we can see that there are most snapshots at 15:30 - since afternoons are typically the busiest times in the ED - and least at 06:00. 


```python
print("\nTimes of day at which predictions will be made")
print(ed_visits.prediction_time.unique())
```

    
    Times of day at which predictions will be made
    [(12, 0) (15, 30) (6, 0) (9, 30) (22, 0)]



```python
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


If you use different data, check that the prediction times in your dataset aligns with the specified times of day set in the parameters file config.yaml. That is because, later, we will use these times of day to evaluate the predictions. The evaluation will fail if the data loaded does not match. 

## Separate into training, validation and test sets

The first task in model development is to split the dataset into a training, validation and test set using a temporal split. Using a using a temporal split (rather than randomly assigning visits to each set) is appropriate for tasks where the model needs to be validated on unseen, future data.

The training set is used to fit the model parameters, the validation set helps tune hyperparameters and evaluate the model during development to prevent overfitting, and the test set provides an unbiased evaluation of the final model's performance on completely unseen data.

I first load the training, validation and test set dates from the configuration file


```python
# load params
from patientflow.load import load_config_file
params = load_config_file(config_path)

start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

print(f'\nTraining set starts {start_training_set} and ends on {start_validation_set - pd.Timedelta(days=1)} inclusive')
print(f'Validation set starts on {start_validation_set} and ends on {start_test_set - pd.Timedelta(days=1)} inclusive' )
print(f'Test set starts on {start_test_set} and ends on {end_test_set- pd.Timedelta(days=1)} inclusive' )
```

    
    Training set starts 2031-03-01 and ends on 2031-08-31 inclusive
    Validation set starts on 2031-09-01 and ends on 2031-09-30 inclusive
    Test set starts on 2031-10-01 and ends on 2031-12-31 inclusive



```python
from patientflow.prepare import create_temporal_splits

train_visits, valid_visits, test_visits = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
)
```

    Split sizes: [53801, 6519, 19494]


## Train a XGBoost Classifier for each time of day, and save the best model

### About the approach to model training

The first step is to load a transformer for the training data to turn it into a format that our ML classifier can read. This is done using a custom function (written for this package) called `create_column_transformer()` which in turn calls `ColumnTransfomer()`, a standard method in scikit-learn. 

The `ColumnTransformer` in scikit-learn is a tool that applies different transformations or preprocessing steps to different columns of a dataset in a single operation. OneHotEncoder converts categorical data into a format that can be provided to machine learning algorithms; without this, the model might interpret the categorical data as numerical, which would lead to incorrect results. OrdinalEncoder converts categorical data into ordered numerical values to reflect the inherent order in the age groups. It is the job of the modeller to indicate to the model how to handle each variables, based on your knowledge of what they represent. Here, `age_group`, `latest_obs_manchester_triage_acuity`, `latest_obs_objective_pain_score` and `latest_obs_level_of_consciousness` are all marked as ordered categories. 


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



```

Certain columns in the dataset provided are not used in training the admissions model. I specify them here


```python
exclude_from_training_data = [ 'snapshot_date', 'prediction_time','consultation_sequence', 'visit_number', 'specialty', 'final_sequence', 'training_validation_test']
```

I also specify a grid of hyperparameters; the classifier will iterate though them to find the best fitting model. 



```python
# # minimal grid for expediency
# grid = {"n_estimators": [30], 
#         "subsample": [0.7], 
#         "colsample_bytree": [0.7],
# }


# grid for hyperparameter tuning
grid = {
    'n_estimators':[30, 40, 50],
    'subsample':[0.7,0.8,0.9],
    'colsample_bytree': [0.7,0.8,0.9],

}
```

We are interested in predictions at different times of day. So we will train a model for each time of day. We will filter each visit so that it only appears once in the training data. 

We iterate through the hyperparameter grid defined above to find the best model for each time of day, keeping track of the best model and its results. When evaluating the best configuration from among the range of hyperparameter options, we will save common ML metrics (AUC, AUPRC and logloss) and compare each configuration for the best (lowest) logloss. This is done by looking at performance on a validation set. 

The best model is saved, as is a dictionary of its metadata, including

* the parameters used in this version of training
* how many visits were in training, validation and test sets
* class balance in each set - the proportion of positive (ie visit ended in admission) to negative (visit ended in discharge) 
* area under ROC curve and log loss (performance metrics) for training (based on 5-fold cross validation), validation and test sets
* List of features and their importances in the model


```python
train_visits.prediction_time.value_counts()
```




    prediction_time
    (15, 30)    15054
    (12, 0)     12764
    (22, 0)     12642
    (9, 30)      7671
    (6, 0)       5670
    Name: count, dtype: int64



Note that the classes are imbalanced. The admitted patients range from 12% to 17% of the total visits.  


```python
train_visits.groupby('prediction_time')['is_admitted'].mean()
```




    prediction_time
    (6, 0)      0.172134
    (9, 30)     0.130752
    (12, 0)     0.124021
    (15, 30)    0.154975
    (22, 0)     0.164610
    Name: is_admitted, dtype: float64




```python
# train admissions model
from patientflow.train.classifiers import train_multiple_classifiers
from patientflow.train.utils import save_model

prediction_times = ed_visits.prediction_time.unique()
model_name = 'admissions'

trained_models = train_multiple_classifiers(
    train_visits=train_visits,
    valid_visits=valid_visits,
    test_visits=test_visits,
    grid=grid,
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings=ordinal_mappings,
    prediction_times=prediction_times,
    model_name=model_name,
    calibrate_probabilities=False,
    calibration_method='sigmoid',
    use_balanced_training=False,
    visit_col='visit_number' # visit_col is needed to ensure we get only one snapshot for each visit in the training set; snapshots are randomly sampled
)

# save models 
save_model(trained_models, "admissions", model_file_path)
print(f"Models have been saved to {model_file_path}")
```

    
    Processing: (12, 0)
    
    Processing: (15, 30)
    
    Processing: (6, 0)
    
    Processing: (9, 30)
    
    Processing: (22, 0)
    Models have been saved to /Users/zellaking/Repos/patientflow/trained-models/public


## Inspecting the output from model training 

Five models have been trained, each saved under a key which is the name of the model passed into train_all_models() (default value is 'admissions'), with the prediction time appended


```python
print(trained_models.keys())

```

    dict_keys(['admissions_1200', 'admissions_1530', 'admissions_0600', 'admissions_0930', 'admissions_2200'])


If we display the values for one of the items in the dictionary of trained_models, we see that a trained classifer has been returned by the function. 




```python
type(trained_models['admissions_1530'])
```




    patientflow.metrics.TrainedClassifier



Within the object that is returned, various metrics have been saved, including
- how many observations were in th training, validation and test sets
- the class balance (proportion of admitted patients in the whole dataset)
- the best parameters of all combinations tried
- results on training and validation sets
- results on test set
- the model features


```python
trained_models['admissions_1530'].training_results
```




    TrainingResults(prediction_time=(15, 30), training_info={'best_hyperparameters': {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}, 'cv_trials': [HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9580444574749587), 'train_logloss': np.float64(0.17128741291716607), 'train_auprc': np.float64(0.8715806541831641), 'valid_auc': np.float64(0.8185838772831474), 'valid_logloss': np.float64(0.35776512326952187), 'valid_auprc': np.float64(0.49339818684816883)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9608349568241314), 'train_logloss': np.float64(0.16742416684695283), 'train_auprc': np.float64(0.882931124248984), 'valid_auc': np.float64(0.822116547223759), 'valid_logloss': np.float64(0.35205162513739036), 'valid_auprc': np.float64(0.5043328103662739)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9598064524398104), 'train_logloss': np.float64(0.17022840144891377), 'train_auprc': np.float64(0.87777765815176), 'valid_auc': np.float64(0.8247641678471854), 'valid_logloss': np.float64(0.34924519090362005), 'valid_auprc': np.float64(0.5074360419229299)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 40, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9692443638232101), 'train_logloss': np.float64(0.14865393562155715), 'train_auprc': np.float64(0.9087734492031064), 'valid_auc': np.float64(0.8121986406595069), 'valid_logloss': np.float64(0.3684489567791951), 'valid_auprc': np.float64(0.4849689907988184)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 40, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9710228713540963), 'train_logloss': np.float64(0.14699313553863402), 'train_auprc': np.float64(0.9147567021018306), 'valid_auc': np.float64(0.8193077419226252), 'valid_logloss': np.float64(0.3592349996978693), 'valid_auprc': np.float64(0.49806947859320305)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 40, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9691513680511268), 'train_logloss': np.float64(0.15104868908420951), 'train_auprc': np.float64(0.909170981419507), 'valid_auc': np.float64(0.8209588586801381), 'valid_logloss': np.float64(0.356456334988685), 'valid_auprc': np.float64(0.5012065512221315)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 50, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9754563248111626), 'train_logloss': np.float64(0.13288738885594306), 'train_auprc': np.float64(0.9281149470128899), 'valid_auc': np.float64(0.8096349174456348), 'valid_logloss': np.float64(0.377019522300244), 'valid_auprc': np.float64(0.479819468091331)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 50, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9774629975386757), 'train_logloss': np.float64(0.13016884797762782), 'train_auprc': np.float64(0.9347452833694614), 'valid_auc': np.float64(0.8172446924602526), 'valid_logloss': np.float64(0.3662449650267632), 'valid_auprc': np.float64(0.49704858829900667)}), HyperParameterTrial(parameters={'colsample_bytree': 0.7, 'n_estimators': 50, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9761153362492024), 'train_logloss': np.float64(0.13452344588064163), 'train_auprc': np.float64(0.9310445097310222), 'valid_auc': np.float64(0.8190570465706817), 'valid_logloss': np.float64(0.36242036948112594), 'valid_auprc': np.float64(0.49848185066078765)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 30, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9575966336851891), 'train_logloss': np.float64(0.17221388353993197), 'train_auprc': np.float64(0.8697242486589554), 'valid_auc': np.float64(0.8215121382050729), 'valid_logloss': np.float64(0.35355391464605573), 'valid_auprc': np.float64(0.5025811558888195)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 30, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9613572848523688), 'train_logloss': np.float64(0.16609720661125424), 'train_auprc': np.float64(0.8839334990368076), 'valid_auc': np.float64(0.8211516096065401), 'valid_logloss': np.float64(0.35352202514808706), 'valid_auprc': np.float64(0.5037399330017573)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 30, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9612197695738036), 'train_logloss': np.float64(0.16786813558249167), 'train_auprc': np.float64(0.8819262315667638), 'valid_auc': np.float64(0.8229450570916768), 'valid_logloss': np.float64(0.3498415351977358), 'valid_auprc': np.float64(0.510734679488184)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 40, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.969381536323513), 'train_logloss': np.float64(0.14913349612326793), 'train_auprc': np.float64(0.9066869770558401), 'valid_auc': np.float64(0.817453330373526), 'valid_logloss': np.float64(0.36259422267672653), 'valid_auprc': np.float64(0.495675611613409)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 40, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9723492479266348), 'train_logloss': np.float64(0.1444252786169537), 'train_auprc': np.float64(0.9177285397857797), 'valid_auc': np.float64(0.8180471650866726), 'valid_logloss': np.float64(0.3591630393393653), 'valid_auprc': np.float64(0.5017773287085623)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 40, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9708461000402444), 'train_logloss': np.float64(0.14846540852173842), 'train_auprc': np.float64(0.9122144963405245), 'valid_auc': np.float64(0.8186979549340385), 'valid_logloss': np.float64(0.3563048237918468), 'valid_auprc': np.float64(0.5058499039908038)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 50, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9760730709584294), 'train_logloss': np.float64(0.13215909902301967), 'train_auprc': np.float64(0.9281286653342111), 'valid_auc': np.float64(0.8155560696782997), 'valid_logloss': np.float64(0.3691008860710845), 'valid_auprc': np.float64(0.49521959174136754)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 50, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9778727029069616), 'train_logloss': np.float64(0.1281988760192599), 'train_auprc': np.float64(0.9358940020395858), 'valid_auc': np.float64(0.8164993603588429), 'valid_logloss': np.float64(0.3663614510899075), 'valid_auprc': np.float64(0.5003884377800455)}), HyperParameterTrial(parameters={'colsample_bytree': 0.8, 'n_estimators': 50, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9770524273204056), 'train_logloss': np.float64(0.1326286731781076), 'train_auprc': np.float64(0.9326235819201699), 'valid_auc': np.float64(0.8166351586602891), 'valid_logloss': np.float64(0.362835187448866), 'valid_auprc': np.float64(0.49938972574190477)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 30, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9587195588023907), 'train_logloss': np.float64(0.17032587443038633), 'train_auprc': np.float64(0.8736492871060602), 'valid_auc': np.float64(0.8223120263384989), 'valid_logloss': np.float64(0.35403002043463105), 'valid_auprc': np.float64(0.5002372008506849)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 30, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9616915338990724), 'train_logloss': np.float64(0.16669441740964946), 'train_auprc': np.float64(0.8818769205248029), 'valid_auc': np.float64(0.8210045591086317), 'valid_logloss': np.float64(0.35333147730144987), 'valid_auprc': np.float64(0.50126181865915)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 30, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9630450675445494), 'train_logloss': np.float64(0.16495354959691733), 'train_auprc': np.float64(0.8842299626908368), 'valid_auc': np.float64(0.8241810528407454), 'valid_logloss': np.float64(0.3504599607812959), 'valid_auprc': np.float64(0.5080164834038613)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 40, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9701352465081156), 'train_logloss': np.float64(0.14806973718414185), 'train_auprc': np.float64(0.9087807651605763), 'valid_auc': np.float64(0.8169164756443813), 'valid_logloss': np.float64(0.364023632546598), 'valid_auprc': np.float64(0.4916922092906617)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 40, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9725269487052615), 'train_logloss': np.float64(0.14466950788358884), 'train_auprc': np.float64(0.9164418710374193), 'valid_auc': np.float64(0.816365888636766), 'valid_logloss': np.float64(0.3624987777657173), 'valid_auprc': np.float64(0.49368484958032954)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 40, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9723265688132766), 'train_logloss': np.float64(0.14520084746470563), 'train_auprc': np.float64(0.9147273368169351), 'valid_auc': np.float64(0.8180853921714757), 'valid_logloss': np.float64(0.35982675392316926), 'valid_auprc': np.float64(0.49955569744201006)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 50, 'subsample': 0.7}, cv_results={'train_auc': np.float64(0.9770441304896635), 'train_logloss': np.float64(0.13011561597637503), 'train_auprc': np.float64(0.9318968412342847), 'valid_auc': np.float64(0.8145408316189766), 'valid_logloss': np.float64(0.3713820723635733), 'valid_auprc': np.float64(0.4905006082084631)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 50, 'subsample': 0.8}, cv_results={'train_auc': np.float64(0.9780830536636665), 'train_logloss': np.float64(0.129304555259721), 'train_auprc': np.float64(0.9344493917029153), 'valid_auc': np.float64(0.8140243107652976), 'valid_logloss': np.float64(0.3700875125835403), 'valid_auprc': np.float64(0.49198646174320376)}), HyperParameterTrial(parameters={'colsample_bytree': 0.9, 'n_estimators': 50, 'subsample': 0.9}, cv_results={'train_auc': np.float64(0.9775949270091541), 'train_logloss': np.float64(0.13112836217597798), 'train_auprc': np.float64(0.9321150905413849), 'valid_auc': np.float64(0.8162584044026705), 'valid_logloss': np.float64(0.3655153671708106), 'valid_auprc': np.float64(0.4995022293772326)})], 'features': {'names': ['elapsed_los', 'sex_F', 'sex_M', 'age_group', 'arrival_method_Amb no medic', 'arrival_method_Ambulance', 'arrival_method_Custodial se', 'arrival_method_Non-emergenc', 'arrival_method_Police', 'arrival_method_Public Trans', 'arrival_method_Walk-in', 'arrival_method_nan', 'current_location_type_majors', 'current_location_type_otf', 'current_location_type_paeds', 'current_location_type_rat', 'current_location_type_resus', 'current_location_type_sdec', 'current_location_type_sdec_waiting', 'current_location_type_utc', 'current_location_type_waiting', 'total_locations_visited', 'num_obs', 'num_obs_events', 'num_obs_types', 'num_lab_batteries_ordered', 'has_consultation_False', 'has_consultation_True', 'visited_majors_False', 'visited_majors_True', 'visited_otf_False', 'visited_otf_True', 'visited_paeds_False', 'visited_paeds_True', 'visited_rat_False', 'visited_rat_True', 'visited_resus_False', 'visited_resus_True', 'visited_sdec_False', 'visited_sdec_True', 'visited_sdec_waiting_False', 'visited_sdec_waiting_True', 'visited_unknown_False', 'visited_unknown_True', 'visited_utc_False', 'visited_utc_True', 'visited_waiting_False', 'visited_waiting_True', 'num_obs_blood_pressure', 'num_obs_pulse', 'num_obs_air_or_oxygen', 'num_obs_glasgow_coma_scale_best_motor_response', 'num_obs_level_of_consciousness', 'num_obs_news_score_result', 'num_obs_manchester_triage_acuity', 'num_obs_objective_pain_score', 'num_obs_subjective_pain_score', 'num_obs_temperature', 'num_obs_oxygen_delivery_method', 'num_obs_pupil_reaction_right', 'num_obs_oxygen_flow_rate', 'num_obs_uclh_sskin_areas_observed', 'latest_obs_pulse', 'latest_obs_respirations', 'latest_obs_level_of_consciousness', 'latest_obs_news_score_result', 'latest_obs_manchester_triage_acuity', 'latest_obs_objective_pain_score', 'latest_obs_temperature', 'lab_orders_bc_False', 'lab_orders_bc_True', 'lab_orders_bon_False', 'lab_orders_bon_True', 'lab_orders_crp_False', 'lab_orders_crp_True', 'lab_orders_csnf_False', 'lab_orders_csnf_True', 'lab_orders_ddit_False', 'lab_orders_ddit_True', 'lab_orders_ncov_False', 'lab_orders_ncov_True', 'lab_orders_rflu_False', 'lab_orders_rflu_True', 'lab_orders_xcov_False', 'lab_orders_xcov_True', 'latest_lab_results_crea', 'latest_lab_results_hctu', 'latest_lab_results_k', 'latest_lab_results_lac', 'latest_lab_results_na', 'latest_lab_results_pco2', 'latest_lab_results_ph', 'latest_lab_results_wcc', 'latest_lab_results_alb', 'latest_lab_results_htrt'], 'importances': [0.008522354066371918, 0.008131534792482853, 0.005864954087883234, 0.01838778890669346, 0.008991928771138191, 0.05714333429932594, 0.0, 0.0, 0.0, 0.004175173118710518, 0.0029791926499456167, 0.005155919585376978, 0.00883033312857151, 0.0, 0.0, 0.0060632373206317425, 0.03788682445883751, 0.0308400709182024, 0.024043474346399307, 0.07728108763694763, 0.0, 0.0059556374326348305, 0.006368630100041628, 0.003310894127935171, 0.005870932247489691, 0.07313118875026703, 0.02008918859064579, 0.013537026010453701, 0.004102907609194517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003809602465480566, 0.0, 0.011287982575595379, 0.0027528777718544006, 0.0067667667753994465, 0.004753147251904011, 0.016880907118320465, 0.005798147991299629, 0.012107975780963898, 0.005574553273618221, 0.03745686635375023, 0.0, 0.010590678080916405, 0.0, 0.007848676294088364, 0.0034012245014309883, 0.007608588319271803, 0.003733427496626973, 0.0034718795213848352, 0.0022597843781113625, 0.06949556618928909, 0.0019915185403078794, 0.007285783067345619, 0.0035581011325120926, 0.0, 0.004668738227337599, 0.02521573193371296, 0.0, 0.005630819126963615, 0.0068956115283071995, 0.002723122015595436, 0.016431091353297234, 0.01504108402878046, 0.007177324965596199, 0.005190321709960699, 0.030588064342737198, 0.004908763337880373, 0.007758032996207476, 0.0, 0.025546306744217873, 0.0, 0.005771547090262175, 0.052093591541051865, 0.009660473093390465, 0.0086259376257658, 0.0, 0.0, 0.006003187503665686, 0.018534807488322258, 0.003805062035098672, 0.0, 0.005595318507403135, 0.006606217939406633, 0.006749584339559078, 0.005568852182477713, 0.006685983389616013, 0.0065507483668625355, 0.007634185254573822, 0.010091325268149376, 0.006409717258065939, 0.010740851052105427], 'has_importance_values': True}, 'dataset_info': {'train_valid_test_set_no': {'train_set_no': 14969, 'valid_set_no': 1691, 'test_set_no': 5519}, 'train_valid_test_class_balance': {'y_train_class_balance': {0: 0.8446121985436569, 1: 0.15538780145634312}, 'y_valid_class_balance': {0: 0.7983441750443524, 1: 0.20165582495564754}, 'y_test_class_balance': {0: 0.812103641964124, 1: 0.18789635803587607}}}}, calibration_info={}, test_results={'test_auc': np.float64(0.7931831257312546), 'test_logloss': 0.40706388956815903, 'test_auprc': np.float64(0.4912035432847283)}, balance_info={'is_balanced': False, 'original_size': 14969, 'balanced_size': 14969, 'original_positive_rate': np.float64(0.15538780145634312), 'balanced_positive_rate': np.float64(0.15538780145634312), 'majority_to_minority_ratio': 1.0})



To get a better view of the same output


```python
from dataclasses import fields
print("\nDataclass fields in TrainingResults:")
for field in fields(trained_models['admissions_1530'].training_results):
    print(field.name)
```

    
    Dataclass fields in TrainingResults:
    prediction_time
    training_info
    calibration_info
    test_results
    balance_info


The prediction time for this model has also been saved.


```python
# See the prediction time for this model
trained_models['admissions_1530'].training_results.prediction_time
```




    (15, 30)



Within the training_results, a training_info object contains information related to model training


```python
metrics_dict = trained_models['admissions_1530'].training_results.training_info
metrics_dict.keys()
```




    dict_keys(['best_hyperparameters', 'cv_trials', 'features', 'dataset_info'])




```python
for key, value in trained_models['admissions_1530'].training_results.training_info.items():
    print(key)
```

    best_hyperparameters
    cv_trials
    features
    dataset_info



```python

```


```python


prediction_time_model_name = 'admissions_1530'

# Get the results dictionary from cross validation trials
results = trained_models[prediction_time_model_name].training_results.training_info['cv_trials']

# Show the first trial results
results[0].cv_results
```




    {'train_auc': np.float64(0.9580444574749587),
     'train_logloss': np.float64(0.17128741291716607),
     'train_auprc': np.float64(0.8715806541831641),
     'valid_auc': np.float64(0.8185838772831474),
     'valid_logloss': np.float64(0.35776512326952187),
     'valid_auprc': np.float64(0.49339818684816883)}




```python

training_info = trained_models[prediction_time_model_name].training_results.training_info

def find_best_trial(trials_list):
    """Find the trial with the lowest validation logloss."""
    return min(trials_list, key=lambda trial: trial.cv_results['valid_logloss'])

best_trial = find_best_trial(training_info["cv_trials"])

# print the best parameters
best_parameters = best_trial.parameters
best_parameters
```




    {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}




#### Retreiving saved information about the number of training, validation and test set samples, and the class balance


```python
model_metadata = trained_models[prediction_time_model_name].training_results.training_info
print(f"Number in each set for model called {prediction_time_model_name}:\n{model_metadata['dataset_info']['train_valid_test_set_no']}")

def print_class_balance(d):
    for k in d:
        print(f"{k.split('_')[1]}: {d[k][0]:.1%} neg, {d[k][1]:.1%} pos")


print_class_balance(model_metadata['dataset_info']['train_valid_test_class_balance'])

```

    Number in each set for model called admissions_1530:
    {'train_set_no': 14969, 'valid_set_no': 1691, 'test_set_no': 5519}
    train: 84.5% neg, 15.5% pos
    valid: 79.8% neg, 20.2% pos
    test: 81.2% neg, 18.8% pos


Class balance information is also saved in this key, which will store information about the differences between the class balance when forcing the training set to be balanced


```python
trained_models[prediction_time_model_name].training_results.balance_info
```




    {'is_balanced': False,
     'original_size': 14969,
     'balanced_size': 14969,
     'original_positive_rate': np.float64(0.15538780145634312),
     'balanced_positive_rate': np.float64(0.15538780145634312),
     'majority_to_minority_ratio': 1.0}



#### Retreiving saved information about the best parameters for each model


```python
from patientflow.load import get_model_key



def find_best_trial(trials_list):
    """Find the trial with the lowest validation logloss."""
    return min(trials_list, key=lambda trial: trial.cv_results['valid_logloss'])

best_trial = find_best_trial(training_info["cv_trials"])


for prediction_time in train_visits.prediction_time.unique():
    prediction_time_model_name = get_model_key("admissions", prediction_time)
    training_info = trained_models[prediction_time_model_name].training_results.training_info
    best_trial = find_best_trial(training_info["cv_trials"])
    hour, minute = prediction_time
    print(f"Best hyperparameters for {hour:02d}:{minute:02d} model: {best_trial.parameters}")
```

    Best hyperparameters for 12:00 model: {'colsample_bytree': 0.8, 'n_estimators': 30, 'subsample': 0.9}
    Best hyperparameters for 15:30 model: {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}
    Best hyperparameters for 06:00 model: {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}
    Best hyperparameters for 09:30 model: {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}
    Best hyperparameters for 22:00 model: {'colsample_bytree': 0.7, 'n_estimators': 30, 'subsample': 0.9}


Plot the performance of each training run.


```python
from patientflow.viz.training_results import plot_trial_results
fig = plot_trial_results(trained_models[prediction_time_model_name].training_results.training_info['cv_trials'])

```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_55_0.png)
    



```python
# After sigmoid calibration
print(f"Results for each training run are saved using the hyperparameters as keys:")

for key, value in trained_models[prediction_time_model_name].training_results.test_results.items():
    print(f"{key}: {value:.2f}")

```

    Results for each training run are saved using the hyperparameters as keys:
    test_auc: 0.84
    test_logloss: 0.41
    test_auprc: 0.60


### Plot feature importances and shap plots for each of the five models

The following cells show Shap plots and feature importance plots for each of the five prediction times.

- **Feature Importance Plot**
A feature importance plot is a visual representation that shows the significance of each feature (or variable) in a machine learning model. It helps to understand which features contribute most to the model's predictions. The importance of a feature is typically determined by how much it improves the model's performance. The plot tells you which inputs, in overall terms, have the most influence on the output of the model. This plot is particularly useful for model interpretation, and gaining insights into the underlying data.

- **SHAP Plot**
A SHAP (SHapley Additive exPlanations) plot provides a detailed view of the impact each feature has on the prediction of a machine learning model on a particular dataset, which in the case is the test set. Unlike the feature importance plot, SHAP values explain the contribution of each feature for each individual hospital visit in the test set (with each observation being represented as a dot in the plot). SHAP plots combine game theory and local explanations to show how much each feature increases or decreases the prediction for a given visit. This helps to interpret not only the overall model behavior but also the specific decisions for individual visits, offering a more granular and transparent view of model predictions.

#### Feature plots

The cell below shows feature plots for the main model. To see the same for the minimal model, change the model name to 'admissions_minimal'


```python
from patientflow.viz.feature_plot import plot_features


plot_features(
    trained_models=list(trained_models.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    top_n=20  # optional, showing default value
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_59_0.png)
    


## Inspecting the base model


```python
# without balanced training
from patientflow.viz.distribution_plots import plot_prediction_distributions
plot_prediction_distributions(
    trained_models=list(trained_models.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)


```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_61_0.png)
    



```python
# without balanced training
from patientflow.viz.calibration_plot import plot_calibration

plot_calibration(
    trained_models=list(trained_models.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    strategy="uniform",  # optional
    suptitle="Base model with imbalanced training data"  # optional
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_62_0.png)
    



```python
## without balanced training
from patientflow.viz.madcap_plot import generate_madcap_plots
generate_madcap_plots(
    trained_models=list(trained_models.values()),
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_63_0.png)
    


## Trying with balanced samples


```python
# train admissions model
from patientflow.train.classifiers import train_multiple_classifiers

prediction_times = ed_visits.prediction_time.unique()

model_name = 'admissions_balanced'
trained_models_balanced = train_multiple_classifiers(
    train_visits=train_visits,
    valid_visits=valid_visits,
    test_visits=test_visits,
    grid=grid,
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings=ordinal_mappings,
    prediction_times=prediction_times,
    model_name=model_name,
    calibrate_probabilities=False,
    calibration_method='sigmoid',
    use_balanced_training=True,
    visit_col='visit_number' # visit_col is needed to ensure we get only one snapshot for each visit in the training set; snapshots are randomly sampled
)

# save models and metadata
from patientflow.train.utils import save_model

save_model(trained_models_balanced, model_name, model_file_path)

print(f"Models have been saved to {model_file_path}")
```

    
    Processing: (12, 0)
    
    Processing: (15, 30)
    
    Processing: (6, 0)
    
    Processing: (9, 30)
    
    Processing: (22, 0)
    Models have been saved to /Users/zellaking/Repos/patientflow/trained-models/public



```python
for key, value in trained_models_balanced["admissions_balanced_1530"].training_results.balance_info.items():
    print(f"{key}: {value}")
```

    is_balanced: True
    original_size: 14969
    balanced_size: 4652
    original_positive_rate: 0.15538780145634312
    balanced_positive_rate: 0.5
    majority_to_minority_ratio: 1.0



```python
# with balanced training, not calibrated
from patientflow.viz.distribution_plots import plot_prediction_distributions
plot_prediction_distributions(
    trained_models=list(trained_models_balanced.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_67_0.png)
    



```python
# with balanced training, not calibrated
from patientflow.viz.calibration_plot import plot_calibration

plot_calibration(
    trained_models=list(trained_models_balanced.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    strategy="uniform",  # optional
    suptitle="Balanced model prior to calibration"  # optional
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_68_0.png)
    



```python
# with balanced training, not calibrated
from patientflow.viz.madcap_plot import generate_madcap_plots
generate_madcap_plots(
    trained_models=list(trained_models_balanced.values()),
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    suptitle="Balanced model prior to calibration"  
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_69_0.png)
    


## Adding calibration


```python
# train admissions model
from patientflow.train.classifiers import train_multiple_classifiers

prediction_times = ed_visits.prediction_time.unique()

model_name = 'admissions_balanced_calibrated'
trained_models_balanced_calibrated = train_multiple_classifiers(
    train_visits=train_visits,
    valid_visits=valid_visits,
    test_visits=test_visits,
    grid=grid,
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings=ordinal_mappings,
    prediction_times=prediction_times,
    model_name=model_name,
    calibrate_probabilities=True,
    calibration_method='sigmoid', # Platt scaling
    use_balanced_training=True,
    visit_col='visit_number' # visit_col is needed to ensure we get only one snapshot for each visit in the training set; snapshots are randomly sampled
)

# save models and metadata
from patientflow.train.utils import save_model

save_model(trained_models_balanced_calibrated, model_name, model_file_path)

print(f"Models have been saved to {model_file_path}")
```

    
    Processing: (12, 0)
    
    Processing: (15, 30)
    
    Processing: (6, 0)
    
    Processing: (9, 30)
    
    Processing: (22, 0)
    Models have been saved to /Users/zellaking/Repos/patientflow/trained-models/public



```python
for key, value in trained_models_balanced_calibrated["admissions_balanced_calibrated_1530"].training_results.balance_info.items():
    print(f"{key}: {value}")
```

    is_balanced: True
    original_size: 14969
    balanced_size: 4652
    original_positive_rate: 0.15538780145634312
    balanced_positive_rate: 0.5
    majority_to_minority_ratio: 1.0



```python
from patientflow.viz.distribution_plots import plot_prediction_distributions
plot_prediction_distributions(
    trained_models=list(trained_models_balanced_calibrated.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_73_0.png)
    



```python
# with balanced training, calibrated
from patientflow.viz.calibration_plot import plot_calibration
plot_calibration(
    trained_models=list(trained_models_balanced_calibrated.values()),  # Convert dict values to list
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    strategy="uniform",  # optional
    suptitle="Balanced model after calibration"  # optional
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_74_0.png)
    



```python
# with balanced training, calibrated
from patientflow.viz.madcap_plot import generate_madcap_plots
generate_madcap_plots(
    trained_models=list(trained_models_balanced_calibrated.values()),
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    suptitle="Balanced model after calibration"  
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_75_0.png)
    



```python
from patientflow.viz.madcap_plot import generate_madcap_plots_by_group
generate_madcap_plots_by_group(
    trained_models=list(trained_models_balanced_calibrated.values()),
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data,
    grouping_var="age_group",
    grouping_var_name="Age Group",
    plot_difference=False
)


```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_76_0.png)
    



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_76_1.png)
    



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_76_2.png)
    



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_76_3.png)
    



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_76_4.png)
    



```python
from patientflow.viz.feature_plot import plot_features

plot_features(
    list(trained_models_balanced_calibrated.values()),
    media_file_path,
    suptitle="Feature Importance for balanced and calibrated model",
)
```


    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_77_0.png)
    


## Shap plots - main model

The cell below shows SHAP plots for the balanced and calibrated model at each prediction time. 


```python
from patientflow.viz.shap_plot import plot_shap
    
plot_shap(
    trained_models=list(trained_models_balanced_calibrated.values()),
    media_file_path=media_file_path,
    test_visits=test_visits,
    exclude_from_training_data=exclude_from_training_data
)
```

    Predicted classification (not admitted, admitted):  [1089  743]



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_79_1.png)
    


    Predicted classification (not admitted, admitted):  [1781  993]



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_79_3.png)
    


    Predicted classification (not admitted, admitted):  [3059 1678]



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_79_5.png)
    


    Predicted classification (not admitted, admitted):  [3501 2025]



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_79_7.png)
    


    Predicted classification (not admitted, admitted):  [2876 1749]



    
![png](4a_Predict_probability_of_admission_from_ED_files/4a_Predict_probability_of_admission_from_ED_79_9.png)
    



```python

```


```python

```
