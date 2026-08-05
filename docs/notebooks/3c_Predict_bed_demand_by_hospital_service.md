# 3c. Predict bed demand by hospital service

Hospitals organise their inpatient beds by hospital service — for example, medical, surgical, paediatric, or haematology/oncology beds. When predicting emergency demand for beds, it is useful to disaggregate overall demand into predictions for each hospital service, so that bed managers can plan for pressures in specific areas. In the data used here, the hospital service is recorded as the `specialty` a patient is admitted under.

In this notebook I show how `patientflow` can be used to predict demand by hospital service. I also show an example of stratifying predictions by an observed patient characteristic (sex). In both cases, I focus on predictions from patients currently in the ED. Later notebooks will consider patients yet-to-arrive, who may need admission with a prediction window.

### Predicting which hospital service a patient will need

When patients are still in the ED, we don't yet know which hospital service they will be admitted to.

I demonstrate the use of a `SequenceToOutcomePredictor` class, which predicts each patient's probability of admission to each hospital service, based on sequences of consult requests made while patients are in the ED.

The `SequenceToOutcomePredictor` approach could be used with other sequence data, such as sequences of locations or procedures, if you deem these likely to be associated with a patient being admitted to a particular hospital service. The key assumption of the sequence design is that the order is meaningful; for example, a surgical consult following a medical consult, or vice versa, is meaningful for the patient's likelihood of being admitted under surgery.

If you don't have sequences for predicting hospital service, you might have simpler data, such as a single reason code for each visit, entered on triage (eg heart problem, broken bone), that suggests which service a patient will end up in. I demonstrate a `ValueToOutcomePredictor` which can be used with such data.

### Combining specialty prediction with admission prediction

I then combine the specialty prediction model with the admission probability model (shown in previous notebooks) to calculate the joint probability that a patient will both be admitted and require a specific specialty. Formally, this derives P(admitted AND specialty X) = P(admitted) × P(specialty X | admitted). This joint probability approach means we can generate specialty-specific bed count predictions. I demonstrate the joint probability for one group snapshot.

I deliberately excluded consult types from the admissions model to ensure the two models use independent signals, avoiding potential overfitting when combining their predictions.

### Stratifying by observed characteristics

Finally, I show a different type of subgroup analysis by stratifying patients by sex. Since sex is directly observed rather than predicted, I create separate bed count distributions for male and female patients.

## Load real patient data

Following the approach taken in the previous notebook, I'll first load some real patient data.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
from patientflow.load import set_file_paths, load_data, load_config_file

# set project root
from patientflow.load import set_project_root
project_root = set_project_root()

# set file paths
data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
        project_root,
        data_folder_name='data-public', # change this to data-synthetic if you don't have the public dataset
        verbose=False)

# load the data
ed_visits = load_data(data_file_path,
                    file_name='ed_visits.csv',
                    index_column = 'snapshot_id',
                    sort_columns = ["visit_number", "snapshot_date", "prediction_time"],
                    eval_columns = ["prediction_time", "consultation_sequence", "final_sequence"])
ed_visits.snapshot_date = pd.to_datetime(ed_visits.snapshot_date).dt.date

# load the config file to set the dates for the training, validation and test sets
params = load_config_file(config_path)
start_training_set, start_calibration_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_calibration_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]

# apply the temporal splits
from patientflow.prepare import create_temporal_splits

# create the temporal splits
train_visits, calibration_visits, valid_visits, test_visits = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date", # states which column contains the date to use when making the splits
    visit_col="visit_number", # states which column contains the visit number to use when making the splits
    start_calibration=start_calibration_set,
)

```

    Inferred project root: /Users/zellaking/Repos/patientflow


    Split sizes: [51170, 10901, 10415, 29134]

## Train a model to predict probability of admission to each specialty

### Predict specialty of admission using sequences of consults

In this example, the data used as input comprise sequences of consults issued while the patient was in the ED. The `consultation_sequence` column shows the ordered sequence of consultation requests up to the moment of the snapshot, and the `final_sequence` shows the ordered sequence at the end of the ED visit. The `specialty` column records which specialty the patient was admitted to.

```python
ed_visits[(ed_visits.is_admitted) & (ed_visits.prediction_time == (9,30))][['consultation_sequence', 'final_sequence', 'specialty']].head(10)

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
      <th>consultation_sequence</th>
      <th>final_sequence</th>
      <th>specialty</th>
    </tr>
    <tr>
      <th>snapshot_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>183349</th>
      <td>['acute', 'discharge']</td>
      <td>['acute', 'discharge']</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>132235</th>
      <td>['paeds']</td>
      <td>['paeds']</td>
      <td>paediatric</td>
    </tr>
    <tr>
      <th>114978</th>
      <td>[]</td>
      <td>['acute']</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>199212</th>
      <td>[]</td>
      <td>[]</td>
      <td>paediatric</td>
    </tr>
    <tr>
      <th>202378</th>
      <td>[]</td>
      <td>['surgical']</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>200273</th>
      <td>[]</td>
      <td>['acute']</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>171735</th>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>140899</th>
      <td>['haem_onc']</td>
      <td>['haem_onc']</td>
      <td>haem/onc</td>
    </tr>
    <tr>
      <th>122882</th>
      <td>['acute']</td>
      <td>['acute', 'medical', 'elderly']</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>159335</th>
      <td>[]</td>
      <td>['surgical']</td>
      <td>surgical</td>
    </tr>
  </tbody>
</table>
</div>

Below I demonstrate training the model. A rooted decision-tree is used to calculate:

- the probability of an ordered sequence of consultations observed at the snapshot (which could be none) resulting in each final sequence at the end of the ED visit
- the probability of each of those final sequences being associated with admission to each specialty (the outcome)

This sequence predictor could be applied to other types of data, such as sequences of ED locations, or sequences of clinical teams visited. Therefore, the `SequenceToOutcomePredictor` arguments have been given generic names:

- `input_var` - the interim node in the decision tree, observed at the snapshot
- `grouping_var` - the terminal node in the decision tree, observed in this example at the end of the ED visit
- `outcome_var` - the final outcome to be predicted

The `apply_special_category_filtering` argument provides for the handling of certain categories in a specific way. For example, patients under 18 might always be assumed to be visiting paediatric specialties. I demonstrate this in a later notebook.

```python
from patientflow.predictors.sequence_to_outcome_predictor import SequenceToOutcomePredictor

spec_model = SequenceToOutcomePredictor(
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=False,
)

_ = spec_model.fit(train_visits)
```

From the weights that are returned, we can view the probability of being admitted to each specialty for a patient who has no consultation sequence at the time of prediction

```python
print(
    f'Probability of being admitted to each specialty at the end of the visit if no consultation result has been made by the time of the snapshot:\n'
    f'{dict((k, round(v, 3)) for k, v in spec_model.weights[()].items())}'
)
```

    Probability of being admitted to each specialty at the end of the visit if no consultation result has been made by the time of the snapshot:
    {'surgical': 0.251, 'medical': 0.607, 'paediatric': 0.063, 'haem/onc': 0.079}

Similarly, we can view the probability of being admitted to each specialty after a consultation request to acute medicine

```python
print(
    f'\nProbability of being admitted to each specialty if one consultation request to acute medicine has taken place by the time of the snapshot:\n'
    f'{dict((k, round(v, 3)) for k, v in spec_model.weights[("acute",)].items())}'
)
```

    Probability of being admitted to each specialty if one consultation request to acute medicine has taken place by the time of the snapshot:
    {'surgical': 0.016, 'medical': 0.948, 'paediatric': 0.001, 'haem/onc': 0.035}

The intermediate mapping of consultation_sequence to final_sequence can be accessed from the trained model like this. The first row shows the probability of a null sequence (ie no consults yet) ending in any of the final_sequence options.

```python
spec_model.input_to_grouping_probs.iloc[:, :10]
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
      <th>final_sequence</th>
      <th>()</th>
      <th>(acute,)</th>
      <th>(acute, acute)</th>
      <th>(acute, acute, acute)</th>
      <th>(acute, acute, icu)</th>
      <th>(acute, acute, medical)</th>
      <th>(acute, acute, medical, surgical)</th>
      <th>(acute, acute, mental_health)</th>
      <th>(acute, acute, surgical)</th>
      <th>(acute, allied)</th>
    </tr>
    <tr>
      <th>consultation_sequence</th>
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
      <th>()</th>
      <td>0.011433</td>
      <td>0.438093</td>
      <td>0.013943</td>
      <td>0.000279</td>
      <td>0.000000</td>
      <td>0.000279</td>
      <td>0.000279</td>
      <td>0.000558</td>
      <td>0.000558</td>
      <td>0.005020</td>
    </tr>
    <tr>
      <th>(acute,)</th>
      <td>0.000000</td>
      <td>0.828191</td>
      <td>0.004909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.007714</td>
    </tr>
    <tr>
      <th>(acute, acute)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(acute, acute, medical)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(acute, ambulatory)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>(surgical, other)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(surgical, surgical)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(surgical, surgical, acute)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(surgical, surgical, icu)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(surgical, surgical, medical)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 10 columns</p>
</div>

#### Using the `SequenceToOutcomePredictor`

Below I apply the predict function to get each patient's probability of being admitted to the four specialties.

```python
test_visits['consultation_sequence'].head().apply(spec_model.predict)
```

    snapshot_id
    192732    {'surgical': 0.8227405247813412, 'medical': 0....
    209659    {'surgical': 0.8954703832752613, 'medical': 0....
    207377    {'surgical': 0.0, 'medical': 0.833333333333333...
    216864    {'surgical': 0.25145865945638257, 'medical': 0...
    207071    {'surgical': 0.25145865945638257, 'medical': 0...
    Name: consultation_sequence, dtype: object

A dictionary is returned for each patient, with probabilities summed to 1. To get each patient's probability of admission to one specialty indexed in the dictionary, we can select that key as shown below:

```python
print("Probability of admission to medical specialty for the first five patients:")
test_visits['consultation_sequence'].head().apply(spec_model.predict).apply(lambda x: x['medical']).values

```

    Probability of admission to medical specialty for the first five patients:





    array([0.13236152, 0.07665505, 0.83333333, 0.60722926, 0.60722926])

### Predicting specialty of admission using a simpler input

If your data for predicting specialty has a simpler structure, say in the form of a string variable containing reasons for presentation at ED, `patientflow` offers a simpler model.

To illustrate this, I create a temporary column by truncating the sequence data to the first item in the list only.

```python
ed_visits['temp_consultation_sequence'] = ed_visits['consultation_sequence'].apply(
    lambda x: x[0].strip("'") if isinstance(x, (list, tuple)) and len(x) > 0 else None
)

ed_visits['temp_final_sequence'] = ed_visits['final_sequence'].apply(
    lambda x: x[0].strip("'") if isinstance(x, (list, tuple)) and len(x) > 0 else None
)

ed_visits[(ed_visits.is_admitted) & (ed_visits.prediction_time == (9,30))][['temp_consultation_sequence', 'temp_final_sequence', 'specialty']].head(10)

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
      <th>temp_consultation_sequence</th>
      <th>temp_final_sequence</th>
      <th>specialty</th>
    </tr>
    <tr>
      <th>snapshot_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>183349</th>
      <td>acute</td>
      <td>acute</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>132235</th>
      <td>paeds</td>
      <td>paeds</td>
      <td>paediatric</td>
    </tr>
    <tr>
      <th>114978</th>
      <td>None</td>
      <td>acute</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>199212</th>
      <td>None</td>
      <td>None</td>
      <td>paediatric</td>
    </tr>
    <tr>
      <th>202378</th>
      <td>None</td>
      <td>surgical</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>200273</th>
      <td>None</td>
      <td>acute</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>171735</th>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>140899</th>
      <td>haem_onc</td>
      <td>haem_onc</td>
      <td>haem/onc</td>
    </tr>
    <tr>
      <th>122882</th>
      <td>acute</td>
      <td>acute</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>159335</th>
      <td>None</td>
      <td>surgical</td>
      <td>surgical</td>
    </tr>
  </tbody>
</table>
</div>

```python
# create the temporal splits
train_visits, calibration_visits, valid_visits, test_visits = create_temporal_splits(
    ed_visits,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date", # states which column contains the date to use when making the splits
    visit_col="visit_number", # states which column contains the visit number to use when making the splits
    start_calibration=start_calibration_set,
)

from patientflow.predictors.value_to_outcome_predictor import ValueToOutcomePredictor

spec_model_simple = ValueToOutcomePredictor(
    input_var="temp_consultation_sequence",
    grouping_var="temp_final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=False,
)

_ = spec_model_simple.fit(train_visits)
```

    Split sizes: [51170, 10901, 10415, 29134]

The weights, which map the input variable to specialty, and the intermediate mappings from input to grouping variables can be viewed in the same way as before. The weights are returned with a key of an empty string rather than a None value for probabilities with a Null value in the input variable.

```python
print(
    f'Probability of being admitted to each specialty at the end of the visit if the value of the input is "medical" at the time of the snapshot:\n'
    f'{dict((k, round(v, 3)) for k, v in spec_model_simple.weights['medical'].items())}'
)

print(
    f'\nProbability of being admitted to each specialty at the end of the visit if no input has been recorded by the time of the snapshot:\n'
    f'{dict((k, round(v, 3)) for k, v in spec_model_simple.weights[''].items())}'
)
```

    Probability of being admitted to each specialty at the end of the visit if the value of the input is "medical" at the time of the snapshot:
    {'haem/onc': 0.023, 'medical': 0.921, 'paediatric': 0.006, 'surgical': 0.051}

    Probability of being admitted to each specialty at the end of the visit if no input has been recorded by the time of the snapshot:
    {'haem/onc': 0.061, 'medical': 0.651, 'paediatric': 0.059, 'surgical': 0.229}

```python
spec_model_simple.input_to_grouping_probs
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
      <th>temp_final_sequence</th>
      <th></th>
      <th>acute</th>
      <th>allied</th>
      <th>ambulatory</th>
      <th>discharge</th>
      <th>elderly</th>
      <th>haem_onc</th>
      <th>icu</th>
      <th>medical</th>
      <th>mental_health</th>
      <th>neuro</th>
      <th>obs_gyn</th>
      <th>other</th>
      <th>paeds</th>
      <th>palliative</th>
      <th>surgical</th>
      <th>probability_of_input_value</th>
    </tr>
    <tr>
      <th>temp_consultation_sequence</th>
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
      <th></th>
      <td>0.011433</td>
      <td>0.547407</td>
      <td>0.000279</td>
      <td>0.007808</td>
      <td>0.003067</td>
      <td>0.001952</td>
      <td>0.040993</td>
      <td>0.007808</td>
      <td>0.027607</td>
      <td>0.006972</td>
      <td>0.038204</td>
      <td>0.030675</td>
      <td>0.000558</td>
      <td>0.045733</td>
      <td>0.000279</td>
      <td>0.229225</td>
      <td>0.510317</td>
    </tr>
    <tr>
      <th>acute</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.221574</td>
    </tr>
    <tr>
      <th>allied</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000142</td>
    </tr>
    <tr>
      <th>ambulatory</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018215</td>
    </tr>
    <tr>
      <th>discharge</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000996</td>
    </tr>
    <tr>
      <th>elderly</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001138</td>
    </tr>
    <tr>
      <th>haem_onc</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040558</td>
    </tr>
    <tr>
      <th>icu</th>
      <td>0.000000</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.967742</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004412</td>
    </tr>
    <tr>
      <th>medical</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011100</td>
    </tr>
    <tr>
      <th>mental_health</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.008965</td>
    </tr>
    <tr>
      <th>neuro</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002846</td>
    </tr>
    <tr>
      <th>obs_gyn</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025189</td>
    </tr>
    <tr>
      <th>other</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000427</td>
    </tr>
    <tr>
      <th>paeds</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026896</td>
    </tr>
    <tr>
      <th>palliative</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000142</td>
    </tr>
    <tr>
      <th>surgical</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.127081</td>
    </tr>
  </tbody>
</table>
</div>

## Combining specialty prediction with admission prediction

I now have a model I can use to predict a patient's probability of admission to each of the four specialties: medical, surgical, haematology/oncology or paediatric, if admitted. I'll use these probabilities, with each patient's probability of admission after ED, to generate predicted bed count distributions for each specialty.

For that I'll also need an admission prediction model, which is set up below.

```python
from patientflow.train.classifiers import train_classifier

prediction_times = [(6, 0), (9, 30), (12, 0), (15, 30), (22, 0)]
ordinal_mappings = {
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
        "Severe_Very Severe",
    ],
    "latest_obs_level_of_consciousness": [
        "A", #alert
        "C", #confused
        "V", #voice - responds to voice stimulus
        "P", #pain - responds to pain stimulus
        "U" #unconscious - no response to pain or voice stimulus
    ]    }
exclude_from_training_data = [ 'snapshot_date', 'prediction_time','visit_number', 'consultation_sequence', 'specialty', 'final_sequence', ]


admission_model = train_classifier(
    train_visits=train_visits,
    valid_visits=valid_visits,
    test_visits=test_visits,
    grid={"n_estimators": [20, 30, 40]},
    exclude_from_training_data=exclude_from_training_data,
    ordinal_mappings=ordinal_mappings,
    prediction_time=(9,30),
    visit_col="visit_number",
    calibrate_probabilities=True,
    calibration_method="isotonic",
    use_balanced_training=True,
    calibration_visits=calibration_visits,
)

```

### Prepare group snapshots

The preparation of group snapshots below is similar to previous notebooks.

```python
from patientflow.prepare import prepare_patient_snapshots, prepare_group_snapshot_dict

prob_dist_dict = {}
first_group_snapshot_key = test_visits.snapshot_date.min()

prediction_snapshots = test_visits[(test_visits.snapshot_date == first_group_snapshot_key) & (test_visits.prediction_time == (9,30))]

# format patient snapshots for input into the admissions model
X_test, y_test = prepare_patient_snapshots(
    df=prediction_snapshots,
    prediction_time=(9,30),
    single_snapshot_per_visit=False,
    exclude_columns=exclude_from_training_data,
    visit_col='visit_number'
)

# prepare group snapshots dict to indicate which patients comprise the group we want to predict for
group_snapshots_dict = prepare_group_snapshot_dict(
    prediction_snapshots
)

```

Below I demonstrate predictions for each specialty in turn.

```python
import matplotlib.pyplot as plt
from patientflow.viz.probability_distribution import plot_prob_dist
from patientflow.viz.pipeline_plots import create_colour_dict
from patientflow.aggregate import get_prob_dist
from patientflow.viz.utils import format_prediction_time

spec_colour_dict = create_colour_dict()
plot_order = ["medical", "surgical", "haem/onc", "paediatric"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, specialty in zip(axes.flat, plot_order):

    prob_admission_to_specialty = prediction_snapshots['consultation_sequence'].apply(spec_model.predict).apply(lambda x, s=specialty: x[s])

    prob_dist_dict = get_prob_dist(
            group_snapshots_dict, X_test, y_test, admission_model,
            weights=prob_admission_to_specialty
        )

    title = specialty.title()
    plot_prob_dist(prob_dist_dict[first_group_snapshot_key]['agg_predicted'], title,
        include_titles=True,
        truncate_at_beds=20,
        bar_colour=spec_colour_dict["single"].get(specialty, "#5B9BD5"),
        ax=ax,
        text_size=14,
    )

fig.suptitle(
    f'Probability distribution for number of beds needed by the '
    f'{len(prediction_snapshots)} patients\n'
    f'in the ED at {format_prediction_time((9,30))} '
    f'on {first_group_snapshot_key}',
    fontsize=16,
)
fig.tight_layout()
plt.show()
```

![png](3c_Predict_bed_demand_by_hospital_service_files/3c_Predict_bed_demand_by_hospital_service_30_0.png)

To compare these with the overall predictions (not by specialty), use the same function without weighting the probability for each specialty.

```python
# get probability distribution for this time of day
prob_dist_dict = get_prob_dist(
        group_snapshots_dict, X_test, y_test, admission_model
        # commenting out the weights argument
        # weights=prob_admission_to_specialty
    )

title = (
    f'Probability distribution for total number of beds needed by the '
    f'{len(prediction_snapshots)} patients\n'
    f'in the ED at {format_prediction_time((9,30))} '
    f'on {first_group_snapshot_key} '
)
plot_prob_dist(prob_dist_dict[first_group_snapshot_key]['agg_predicted'], title,
    include_titles=True, truncate_at_beds=20)
```

![png](3c_Predict_bed_demand_by_hospital_service_files/3c_Predict_bed_demand_by_hospital_service_32_0.png)

## Stratifying by observed characteristics

Disaggregation of predictions using unchanging attributes like sex is very straightforward. Here I show breakdowns by sex.

```python
sex_colour_dict = {"M": "#9467BD", "F": "#8C564B"}
sex_labels = {"M": "male", "F": "female"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, sex in zip(axes.flat, ['M', 'F']):

    prediction_snapshots = test_visits[(test_visits.snapshot_date == first_group_snapshot_key) &
                                       (test_visits.sex == sex) &
                                       (test_visits.prediction_time == (9,30))]

    group_snapshots_dict = prepare_group_snapshot_dict(
        prediction_snapshots
    )

    prob_dist_dict = get_prob_dist(
            group_snapshots_dict, X_test, y_test, admission_model
        )

    title = f'{len(prediction_snapshots)} {sex_labels[sex]} patients'
    plot_prob_dist(prob_dist_dict[first_group_snapshot_key]['agg_predicted'], title,
        include_titles=True,
        truncate_at_beds=20,
        bar_colour=sex_colour_dict[sex],
        ax=ax,
        text_size=14,
    )

fig.suptitle(
    f'Probability distribution for number of beds needed\n'
    f'in the ED at {format_prediction_time((9,30))} '
    f'on {first_group_snapshot_key}, by sex',
    fontsize=16,
)
fig.tight_layout()
plt.show()
```

![png](3c_Predict_bed_demand_by_hospital_service_files/3c_Predict_bed_demand_by_hospital_service_34_0.png)

## Summary

In this notebook I have shown how to predict bed demand by hospital service, using both sequence-based and simpler input data to predict which service a patient will be admitted to. I also demonstrated stratification by observed patient characteristics such as sex.

In the next notebook, I evaluate these predictions across the full test set and compare them against a baseline.
