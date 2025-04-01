# Make predictions using patient-level snapshots

## Things to consider when training predictive models using snapshots

**Random versus temporal splits**

When dividing your data into training, validation and test sets, a random allocation will make your models appear to perform better than they actually would in practice. This is because random splits ignore the temporal nature of healthcare data, where patterns may change over time. A more realistic approach is to use temporal splits, where you train on earlier data and validate/test on later data, mimicking how the model would be deployed in a real-world setting.

**Multiple snapshots per visit**

To use `patientflow` your data should be in snapshot form. I showed how to create this in the last notebook. I defined a series of prediction times, and then sampled finished visit to get snapshots that represent those visits while still in progress. When you follow this method, you may end up with multiple snapshots. Is this OK, for your analysis? You will need to decide whether you include all snapshots from a single visit into a predictive model. These snapshots from the same visit are inherently correlated, which may violate assumptions of many statistical and machine learning methods.

**Multiple visits per patient**

The patient identifier is also important, because if the same patient appears in training and test sets, there is the potential for data leakage. We took the decision to probabilistically allocate each patient to training, validation and test sets, where the probability of being allocated to each set is in proportion to the number of visits they made in any of those time periods.

`patientflow` includes functions that handle of all these considerations.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Create fake snapshots

See the previous notebook for more information about how this is done.

```python
from patientflow.generate import create_fake_snapshots
prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)]
snapshots_df=create_fake_snapshots(prediction_times=prediction_times, start_date='2023-01-01', end_date='2023-04-01')
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
      <th>patient_id</th>
      <th>visit_number</th>
      <th>is_admitted</th>
      <th>age</th>
      <th>latest_triage_score</th>
      <th>num_bmp_orders</th>
      <th>num_troponin_orders</th>
      <th>num_cbc_orders</th>
      <th>num_urinalysis_orders</th>
      <th>num_d-dimer_orders</th>
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
      <td>2690</td>
      <td>6</td>
      <td>0</td>
      <td>49</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>2471</td>
      <td>16</td>
      <td>0</td>
      <td>76</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>2987</td>
      <td>9</td>
      <td>0</td>
      <td>58</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>3472</td>
      <td>46</td>
      <td>0</td>
      <td>63</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01</td>
      <td>(9, 30)</td>
      <td>41</td>
      <td>35</td>
      <td>0</td>
      <td>83</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

## Train a model to predict the outcome of each snapshot

Let's train a model to predict admission for the 9:30 prediction time. We will specify that the triage scores are ordinal, to make use of sklearn's OrdinalEncoder to maintain the natural order of categories.

We exclude columns that are not relevant to the prediction of probability of admission, including `snapshot_date` and `prediction_time`. Note that here we are trying a different model for each prediction time. That was a design decision, which allows the model to pick up different signals of the outcome at different times of day. You'll see the results of this in later notebooks where I show shap plots for models at different times of day.

If the same patient appears in training, validation or test sets, there is the potential for data leakage. The `create_temporal_splits()` function below will randomly allocate each patient_id to training, validation and test sets, where the probability of being allocated to each is in proportion to the number of visits they made in any of those time periods.

```python
from datetime import date
from patientflow.prepare import create_temporal_splits

# set the temporal split
start_training_set = date(2023, 1, 1)
start_validation_set = date(2023, 2, 15) # 6 week training set
start_test_set = date(2023, 3, 1) # 2 week validation set
end_test_set = date(2023, 4, 1) # 1 month test set

# create the temporal splits
train_visits, valid_visits, test_visits = create_temporal_splits(
    snapshots_df,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date", # states which column contains the date to use when making the splits
    patient_id="patient_id", # states which column contains the patient id to use when making the splits
    visit_col="visit_number", # states which column contains the visit number to use when making the splits

)



```

    Patient Set Overlaps (before random assignment):
    Train-Valid: 0 of 2158
    Valid-Test: 29 of 1513
    Train-Test: 100 of 2500
    All Sets: 0 of 3021 total patients
    Split sizes: [1811, 629, 1242]

You will need to decide whether you include all snapshots from a single visit into a predictive model. If you do, there will be non-independence in the data.

Since we train a different model for each prediction time, then it is only visits spanning more than 24 hours that would have multiple rows. If your snapshots are drawn from visits to ED, this should hopefully not happen too often (though sadly it is becoming more common in the UK). If your snapshots are drawn from inpatient visits, then it is very likely that you will have multiple rows per patient.

We took the decision to select one visit at random, even for our ED visits. The function below gives you the option. If you specify `single_snapshot_per_visit` as True, the function will expect a `visit_col` parameter.

```python
from patientflow.train.classifiers import train_classifier

# exclude columns that are not needed for training
exclude_from_training_data=['patient_id', 'visit_number', 'snapshot_date', 'prediction_time']

# train the patient-level model
model = train_classifier(
    train_visits,
    valid_visits,
    test_visits,
    grid={"n_estimators": [20, 30, 40]},
    prediction_time=(9, 30),
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings={'latest_triage_score': [1, 2, 3, 4, 5]},
    single_snapshot_per_visit=True,
    visit_col='visit_number', # as we are using a single snapshot per visit, we need to specify which column contains the visit number
    use_balanced_training=True,
    calibrate_probabilities=True,
    calibration_method='sigmoid'
)

```

**About the parameters in `train_classifer()`**

There are a few parameters in this function to explain.

- `grid`: specifies the grid to use in hyperparameter tuning.
- `prediction_time`: is used to identify which patient snapshots to use for training.
- `single_snapshot_per_visit`: if this is True, the function will randomly pick one snapshot for any visit, using `visit_col` as the column name that identifies the visit identifier.
- `exclude_from_training_data`: certain columns in the data should not be used for training, including visit numbers and dates.
- `ordinal_mappings`: the function makes use of SKLearn's Ordinal Mapping encoder.
- `use_balanced_training`: in healthcare contexts, there are often fewer observations in the positive class. Set this to True for imbalanced samples (common for ED visits, when most patients are discharged, and for predicting inpatient discharge from hospital when most patients remain). It will downsample the negative class.
- `calibrate_probabilities`: when you downsample the negative class, it is a good idea to calibrate the probabilities to account for this class imbalance. Setting this to True will apply isotonic regression to calibrate the predicted probabilities, ensuring they better reflect the probabilities in the original data distribution.
- `calibration_method`: options are sigmoid or isotonic; I have found that sigmoid (the default) works better.

**About machine learning choices**

By default, the function will use an XGBoost classifier, initialised with the hyperparamter grid provided with log loss as the evaluation metric. Chronological ross-validation is used, with the best hyperparameters selected based on minimising log loss in the validation set. We chose XGBoost because it is quick to train, generally performs well, and handles missing values.

If you wish to use a different classifer, you can use another parameter, not shown here:

- `model_class` (not shown here): You can pass your own model in an optional model_class argument, which expects classifier class (like XGBClassifier or other scikit-learn compatible classifiers) that can be instantiated and initialised with the parameters provided

## Inspecting the object returned by `train_classifier()`

The function returns an object of type TrainedClassifer(). Meta data and metrics from the training process are returned with it.

```python
print(f'Object returned is of type: {type(model)}')

print(f'\nThe metadata from the training process are returned in the `training_results` attribute:')
model.training_results
```

    Object returned is of type: <class 'patientflow.model_artifacts.TrainedClassifier'>

    The metadata from the training process are returned in the `training_results` attribute:





    TrainingResults(prediction_time=(9, 30), training_info={'cv_trials': [HyperParameterTrial(parameters={'n_estimators': 20}, cv_results={'train_auc': np.float64(0.9891000722886609), 'train_logloss': np.float64(0.2401767879184195), 'train_auprc': np.float64(0.9915814710451907), 'valid_auc': np.float64(0.6902652943461767), 'valid_logloss': np.float64(0.7612948055228466), 'valid_auprc': np.float64(0.6423223640376016)}), HyperParameterTrial(parameters={'n_estimators': 30}, cv_results={'train_auc': np.float64(0.9964572270616507), 'train_logloss': np.float64(0.2037677992999191), 'train_auprc': np.float64(0.9968371157941393), 'valid_auc': np.float64(0.6828805304172951), 'valid_logloss': np.float64(0.7965992894598218), 'valid_auprc': np.float64(0.6440393638797277)}), HyperParameterTrial(parameters={'n_estimators': 40}, cv_results={'train_auc': np.float64(0.9986821301541798), 'train_logloss': np.float64(0.17958817128501964), 'train_auprc': np.float64(0.9986379247873801), 'valid_auc': np.float64(0.6922204225513049), 'valid_logloss': np.float64(0.8288787785614758), 'valid_auprc': np.float64(0.6543331451114808)})], 'features': {'names': ['age', 'latest_triage_score', 'num_bmp_orders_0', 'num_bmp_orders_1', 'num_troponin_orders_0', 'num_troponin_orders_1', 'num_cbc_orders_0', 'num_cbc_orders_1', 'num_urinalysis_orders_0', 'num_urinalysis_orders_1', 'num_d-dimer_orders_0', 'num_d-dimer_orders_1'], 'importances': [0.11312869191169739, 0.3690102994441986, 0.11062111705541611, 0.0, 0.10884614288806915, 0.0, 0.11563073843717575, 0.0, 0.0881805568933487, 0.0, 0.09458249062299728, 0.0], 'has_importance_values': True}, 'dataset_info': {'train_valid_test_set_no': {'train_set_no': 299, 'valid_set_no': 106, 'test_set_no': 213}, 'train_valid_test_class_balance': {'y_train_class_balance': {0: 0.7123745819397993, 1: 0.28762541806020064}, 'y_valid_class_balance': {0: 0.6698113207547169, 1: 0.330188679245283}, 'y_test_class_balance': {0: 0.6901408450704225, 1: 0.30985915492957744}}}}, calibration_info={'method': 'sigmoid'}, test_results={'test_auc': 0.7148010719439291, 'test_logloss': 0.5654510176351998, 'test_auprc': 0.5559247479672749}, balance_info={'is_balanced': True, 'original_size': 299, 'balanced_size': 172, 'original_positive_rate': np.float64(0.28762541806020064), 'balanced_positive_rate': np.float64(0.5), 'majority_to_minority_ratio': 1.0})

To get a better view of what is included within the results, here is a list of the fields returned:

```python
from dataclasses import fields
print("\nDataclass fields in TrainingResults:")
for field in fields(model.training_results):
    print(field.name)
```

    Dataclass fields in TrainingResults:
    prediction_time
    training_info
    calibration_info
    test_results
    balance_info

The prediction time has been saved.

```python
print(f'The prediction time is: {model.training_results.prediction_time}')
```

    The prediction time is: (9, 30)

An object called training_info contains information related to model training. To simplify the code, I'll assign it to a variable called results. It will tell us the size and class balance of each set

```python
results = model.training_results.training_info

print(f"The training_info object contains the following keys: {results.keys()}")

print(f"\nNumber in each set{results['dataset_info']['train_valid_test_set_no']}")

def print_class_balance(d):
    for k in d:
        print(f"{k.split('_')[1]}: {d[k][0]:.1%} neg, {d[k][1]:.1%} pos")


print_class_balance(results['dataset_info']['train_valid_test_class_balance'])
```

    The training_info object contains the following keys: dict_keys(['cv_trials', 'features', 'dataset_info'])

    Number in each set{'train_set_no': 299, 'valid_set_no': 106, 'test_set_no': 213}
    train: 71.2% neg, 28.8% pos
    valid: 67.0% neg, 33.0% pos
    test: 69.0% neg, 31.0% pos

Class balance information is also saved in the training_results, which will store information about the differences between the class balance when forcing the training set to be balanced

```python
model.training_results.balance_info
```

    {'is_balanced': True,
     'original_size': 299,
     'balanced_size': 172,
     'original_positive_rate': np.float64(0.28762541806020064),
     'balanced_positive_rate': np.float64(0.5),
     'majority_to_minority_ratio': 1.0}

And the type of calibration done on balanced samples is saved in training_results also

```python
model.training_results.calibration_info
```

    {'method': 'sigmoid'}

Results of hyperparameter tuning are saved in a HyperParameterTrial object

```python
# results are stored in a HyperParameterTrial object
results['cv_trials']
```

    [HyperParameterTrial(parameters={'n_estimators': 20}, cv_results={'train_auc': np.float64(0.9891000722886609), 'train_logloss': np.float64(0.2401767879184195), 'train_auprc': np.float64(0.9915814710451907), 'valid_auc': np.float64(0.6902652943461767), 'valid_logloss': np.float64(0.7612948055228466), 'valid_auprc': np.float64(0.6423223640376016)}),
     HyperParameterTrial(parameters={'n_estimators': 30}, cv_results={'train_auc': np.float64(0.9964572270616507), 'train_logloss': np.float64(0.2037677992999191), 'train_auprc': np.float64(0.9968371157941393), 'valid_auc': np.float64(0.6828805304172951), 'valid_logloss': np.float64(0.7965992894598218), 'valid_auprc': np.float64(0.6440393638797277)}),
     HyperParameterTrial(parameters={'n_estimators': 40}, cv_results={'train_auc': np.float64(0.9986821301541798), 'train_logloss': np.float64(0.17958817128501964), 'train_auprc': np.float64(0.9986379247873801), 'valid_auc': np.float64(0.6922204225513049), 'valid_logloss': np.float64(0.8288787785614758), 'valid_auprc': np.float64(0.6543331451114808)})]

```python

# Find the trial with the lowest validation logloss
best_trial = min(results["cv_trials"], key=lambda trial: trial.cv_results['valid_logloss'])

# print the best parameters
print(f'The best parameters are: {best_trial.parameters}')
```

    The best parameters are: {'n_estimators': 20}

```python
print(f'The results on the test set were:')
model.training_results.test_results

```

    The results on the test set were:





    {'test_auc': 0.7148010719439291,
     'test_logloss': 0.5654510176351998,
     'test_auprc': 0.5559247479672749}

Note that each record in the snapshots dataframe is indexed by a unique snapshot_id.

## Conclusion

Here I have shown how `patientflow` can help you

- handle multiple snapshots per visit and multiple visits per patient
- impose a temporal split on your training and test sets, allowing for the point above
- train a model to predict some later outcome using functions that handle class imbalance and calibration

In the next notice, I show how to evaluate models applied to patient snapshots.

A process like this creates a predicted probability of admission for each patient, based on what is known about them at the time of the snapshot. However, bed managers really want predictions for the whole cohort of patients in the ED at a point in time. This is where `patientflow` comes into its own. In the next notebook, I show how to do this.
