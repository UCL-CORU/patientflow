# Modelling probability of admission to specialty, if admitted

This notebook demonstrates the second stage of prediction, to generate a probability of admission to a specialty for each patient in the ED if they are admitted.

Here consult sequences provide the input to prediction, and the model is trained only on visits by adult patients that ended in admission. Patients less than 18 at the time of arrival to the ED are assumed to be admitted to paediatric wards. This assumption could be relaxed by changing the training data to include children, and changing how the inference stage is done.

This approach assumes that, if admitted, a patient's probability of admission to any particular specialty is independent of their probability of admission to hospital.

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
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-public
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public/media

## Train the model

This is the function that trains the specialty model, loaded from a file. Below we will break it down step-by-step.

```python
from patientflow.train.sequence_predictor import train_sequence_predictor, get_default_visits
??train_sequence_predictor
```

    [0;31mSignature:[0m
    [0mtrain_sequence_predictor[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mtrain_visits[0m[0;34m:[0m [0mpandas[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframe[0m[0;34m.[0m[0mDataFrame[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmodel_name[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mvisit_col[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0minput_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrouping_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0moutcome_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0mpatientflow[0m[0;34m.[0m[0mpredictors[0m[0;34m.[0m[0msequence_predictor[0m[0;34m.[0m[0mSequencePredictor[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m
    [0;32mdef[0m [0mtrain_sequence_predictor[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mtrain_visits[0m[0;34m:[0m [0mDataFrame[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmodel_name[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mvisit_col[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0minput_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrouping_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0moutcome_var[0m[0;34m:[0m [0mstr[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0mSequencePredictor[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"""Train a specialty prediction model.[0m
    [0;34m[0m
    [0;34m    Args:[0m
    [0;34m        train_visits: Training data containing visit information[0m
    [0;34m        model_name: Name identifier for the model[0m
    [0;34m        visit_col: Column name containing visit identifiers[0m
    [0;34m        input_var: Column name for input sequence[0m
    [0;34m        grouping_var: Column name for grouping sequence[0m
    [0;34m        outcome_var: Column name for target variable[0m
    [0;34m[0m
    [0;34m    Returns:[0m
    [0;34m        Trained SequencePredictor model[0m
    [0;34m    """[0m[0;34m[0m
    [0;34m[0m    [0mvisits_single[0m [0;34m=[0m [0mselect_one_snapshot_per_visit[0m[0;34m([0m[0mtrain_visits[0m[0;34m,[0m [0mvisit_col[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0madmitted[0m [0;34m=[0m [0mvisits_single[0m[0;34m[[0m[0;34m[0m
    [0;34m[0m        [0;34m([0m[0mvisits_single[0m[0;34m.[0m[0mis_admitted[0m[0;34m)[0m [0;34m&[0m [0;34m~[0m[0;34m([0m[0mvisits_single[0m[0;34m.[0m[0mspecialty[0m[0;34m.[0m[0misnull[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;34m][0m[0;34m[0m
    [0;34m[0m    [0mfiltered_admitted[0m [0;34m=[0m [0mget_default_visits[0m[0;34m([0m[0madmitted[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0mfiltered_admitted[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0minput_var[0m[0;34m][0m [0;34m=[0m [0mfiltered_admitted[0m[0;34m[[0m[0minput_var[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;34m[0m
    [0;34m[0m        [0;32mlambda[0m [0mx[0m[0;34m:[0m [0mtuple[0m[0;34m([0m[0mx[0m[0;34m)[0m [0;32mif[0m [0mx[0m [0;32melse[0m [0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mfiltered_admitted[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mgrouping_var[0m[0;34m][0m [0;34m=[0m [0mfiltered_admitted[0m[0;34m[[0m[0mgrouping_var[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;34m[0m
    [0;34m[0m        [0;32mlambda[0m [0mx[0m[0;34m:[0m [0mtuple[0m[0;34m([0m[0mx[0m[0;34m)[0m [0;32mif[0m [0mx[0m [0;32melse[0m [0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0mspec_model[0m [0;34m=[0m [0mSequencePredictor[0m[0;34m([0m[0;34m[0m
    [0;34m[0m        [0minput_var[0m[0;34m=[0m[0minput_var[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mgrouping_var[0m[0;34m=[0m[0mgrouping_var[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0moutcome_var[0m[0;34m=[0m[0moutcome_var[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mspec_model[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mfiltered_admitted[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mreturn[0m [0mspec_model[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      ~/Repos/patientflow/src/patientflow/train/sequence_predictor.py
    [0;31mType:[0m      function

The first step in the function above is to handle the fact that there are multiple snapshots per visit and we only want one for each visit in the training set.

```python
from patientflow.prepare import select_one_snapshot_per_visit

visits_single = select_one_snapshot_per_visit(ed_visits, visit_col = 'visit_number')

print(ed_visits.shape)
print(visits_single.shape)
```

    (79814, 69)
    (64497, 68)

To train the specialty model, we only use a subset of the columns. Here we can see the relevant columns

```python
display(visits_single[['consultation_sequence', 'final_sequence', 'specialty', 'is_admitted', 'age_group']].head(10))

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
      <th>is_admitted</th>
      <th>age_group</th>
    </tr>
    <tr>
      <th>snapshot_id</th>
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
      <td>[]</td>
      <td>[]</td>
      <td>medical</td>
      <td>False</td>
      <td>55-64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[]</td>
      <td>[]</td>
      <td>surgical</td>
      <td>False</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[]</td>
      <td>[]</td>
      <td>medical</td>
      <td>False</td>
      <td>35-44</td>
    </tr>
    <tr>
      <th>5</th>
      <td>['haem_onc']</td>
      <td>['haem_onc']</td>
      <td>haem/onc</td>
      <td>False</td>
      <td>65-74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>['surgical']</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>False</td>
      <td>25-34</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[]</td>
      <td>['haem_onc']</td>
      <td>medical</td>
      <td>False</td>
      <td>65-74</td>
    </tr>
    <tr>
      <th>11</th>
      <td>['haem_onc']</td>
      <td>['haem_onc']</td>
      <td>medical</td>
      <td>False</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>12</th>
      <td>['haem_onc']</td>
      <td>['haem_onc']</td>
      <td>haem/onc</td>
      <td>False</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[]</td>
      <td>[]</td>
      <td>haem/onc</td>
      <td>False</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>15</th>
      <td>['ambulatory']</td>
      <td>['ambulatory']</td>
      <td>NaN</td>
      <td>False</td>
      <td>0-17</td>
    </tr>
  </tbody>
</table>
</div>

We filter down to only include admitted patients, and remove any with a null value for the specialty column, since this is the model aims to predict.

```python
admitted = visits_single[
    (visits_single.is_admitted) & ~(visits_single.specialty.isnull())
]
```

Note that some visits that ended in admission had no consult request at the time they were sampled, as we can see below, where visits have an empty tuple

```python
display(admitted[['consultation_sequence', 'final_sequence', 'specialty', 'is_admitted', 'age_group']].head(10))


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
      <th>is_admitted</th>
      <th>age_group</th>
    </tr>
    <tr>
      <th>snapshot_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>['surgical']</td>
      <td>['surgical', 'surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>45-54</td>
    </tr>
    <tr>
      <th>58</th>
      <td>['surgical']</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>35-44</td>
    </tr>
    <tr>
      <th>77</th>
      <td>[]</td>
      <td>['acute']</td>
      <td>medical</td>
      <td>True</td>
      <td>65-74</td>
    </tr>
    <tr>
      <th>117</th>
      <td>[]</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>35-44</td>
    </tr>
    <tr>
      <th>125</th>
      <td>['surgical']</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>25-34</td>
    </tr>
    <tr>
      <th>128</th>
      <td>['surgical']</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>141</th>
      <td>[]</td>
      <td>['surgical']</td>
      <td>surgical</td>
      <td>True</td>
      <td>65-74</td>
    </tr>
    <tr>
      <th>163</th>
      <td>['acute']</td>
      <td>['acute']</td>
      <td>medical</td>
      <td>True</td>
      <td>65-74</td>
    </tr>
    <tr>
      <th>176</th>
      <td>[]</td>
      <td>['surgical']</td>
      <td>medical</td>
      <td>True</td>
      <td>75-102</td>
    </tr>
    <tr>
      <th>227</th>
      <td>[]</td>
      <td>['paeds']</td>
      <td>paediatric</td>
      <td>True</td>
      <td>0-17</td>
    </tr>
  </tbody>
</table>
</div>

The UCLH data (not shared publicly) includes more detailed data on consult type, as shown in the `code` column in the dataset below. The public data has been simplified to a higher level (identified in the mapping below as `type`).

```python
from pathlib import Path
model_input_path = project_root / 'src' /  'patientflow'/ 'model-input'
name_mapping = pd.read_csv(str(model_input_path) + '/consults-mapping.csv')
name_mapping
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CON124</td>
      <td>Inpatient consult to Neuro Ophthalmology</td>
      <td>neuro</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>CON9</td>
      <td>Inpatient consult to Neurology</td>
      <td>neuro</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CON34</td>
      <td>Inpatient consult to Dietetics (N&amp;D) - Not TPN</td>
      <td>allied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CON134</td>
      <td>Inpatient consult to PERRT</td>
      <td>icu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>CON163</td>
      <td>IP Consult to MCC Complementary Therapy Team</td>
      <td>pain</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111</th>
      <td>112</td>
      <td>CON77</td>
      <td>Inpatient consult to Paediatric Allergy</td>
      <td>paeds</td>
    </tr>
    <tr>
      <th>112</th>
      <td>113</td>
      <td>CON168</td>
      <td>Inpatient consult to Acute Oncology Service</td>
      <td>haem_onc</td>
    </tr>
    <tr>
      <th>113</th>
      <td>114</td>
      <td>CON84</td>
      <td>Inpatient consult to Paediatric Hematology - C...</td>
      <td>haem_onc</td>
    </tr>
    <tr>
      <th>114</th>
      <td>115</td>
      <td>CON122</td>
      <td>Inpatient consult to Paediatric Epilepsy Service</td>
      <td>paeds</td>
    </tr>
    <tr>
      <th>115</th>
      <td>116</td>
      <td>CON74</td>
      <td>Inpatient consult to Smoking Cessation Program</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 4 columns</p>
</div>

For example, the code for a consult with Acute Medicine is convered to a more general category in the public dataset

```python
name_mapping[name_mapping.code == 'CON157']
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>CON157</td>
      <td>Inpatient consult to Acute Medicine</td>
      <td>acute</td>
    </tr>
  </tbody>
</table>
</div>

The medical group includes many of the more specific types

```python
name_mapping[name_mapping.type == 'medical']
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
      <th>id</th>
      <th>code</th>
      <th>name</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>CON165</td>
      <td>Inpatient consult to Nutrition Team (TPN)</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>CON54</td>
      <td>Inpatient consult to Respiratory Medicine</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>CON43</td>
      <td>Inpatient consult to Cardiology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>CON5</td>
      <td>Inpatient consult to Infectious Diseases</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>CON132</td>
      <td>Inpatient consult to Adult Diabetes CNS</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>CON68</td>
      <td>Inpatient consult to Gastroenterology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>CON60</td>
      <td>Inpatient consult to Endocrinology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>CON156</td>
      <td>Inpatient consult to Adult Endocrine &amp; Diabetes</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>62</th>
      <td>63</td>
      <td>CON44</td>
      <td>Inpatient consult to Rheumatology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>66</th>
      <td>67</td>
      <td>CON147</td>
      <td>Inpatient consult to Cardiac Rehabilitation</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>70</th>
      <td>71</td>
      <td>CON62</td>
      <td>Inpatient consult to Internal Medicine</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>71</th>
      <td>72</td>
      <td>CON127</td>
      <td>Inpatient consult to Hepato-Biliary Medicine</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>76</th>
      <td>77</td>
      <td>CON171</td>
      <td>Inpatient consult to Tropical Medicine</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>91</th>
      <td>92</td>
      <td>CON153</td>
      <td>Inpatient consult to Clinical Biochemistry</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>95</th>
      <td>96</td>
      <td>CON8</td>
      <td>Inpatient consult to Renal Medicine</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>98</th>
      <td>99</td>
      <td>CON81</td>
      <td>Inpatient consult to Paediatric Endocrinology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>101</th>
      <td>102</td>
      <td>CON151</td>
      <td>Inpatient consult to Virology</td>
      <td>medical</td>
    </tr>
    <tr>
      <th>106</th>
      <td>107</td>
      <td>CON28</td>
      <td>Inpatient consult to Integrative Medicine</td>
      <td>medical</td>
    </tr>
  </tbody>
</table>
</div>

## Separate into training, validation and test sets

As part of preparing the data, each visit has already been allocated into one of three sets - training, vaidation and test sets.

```python
from patientflow.prepare import create_temporal_splits

# note that we derive the training set from visits_single, as the SequencePredictor() does the preprocessing mentioned above
train_visits, _, _ = create_temporal_splits(
    visits_single,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="snapshot_date",
)
```

    Split sizes: [42852, 5405, 16240]

## Train the model

Here, we load the SequencePredictor(), a function that takes a sequence as input (in this case consultation_sequence), a grouping variable (in this case final_sequence) and a outcome variable (in this case specialty), and uses a grouping variable to create a rooted directed tree. Each new consult in the sequence is a branching node of the tree. The grouping variable, final sequence, serves as the terminal nodes of the tree. The function maps the probability of each part-complete sequence of consults ending (via each final_sequence) in each specialty of admission.

```python
from patientflow.predictors.sequence_predictor import SequencePredictor

spec_model = SequencePredictor(
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=False,
)

spec_model.fit(train_visits)


```

<style>#sk-container-id-24 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-24 {
  color: var(--sklearn-color-text);
}

#sk-container-id-24 pre {
  padding: 0;
}

#sk-container-id-24 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-24 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-24 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-24 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-24 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-24 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-24 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-24 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-24 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-24 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-24 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-24 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-24 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-24 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-24 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-24 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-24 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-24 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-24 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-24 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-24 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-24 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-24 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-24 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-24 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-24 div.sk-label label.sk-toggleable__label,
#sk-container-id-24 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-24 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-24 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-24 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-24 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-24 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-24 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-24 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-24 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-24 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-24 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-24 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-24 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-24" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SequencePredictor(

    input_var=&#x27;consultation_sequence&#x27;,
    grouping_var=&#x27;final_sequence&#x27;,
    outcome_var=&#x27;specialty&#x27;,
    apply_special_category_filtering=False,
    admit_col=&#x27;is_admitted&#x27;

)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" checked><label for="sk-estimator-id-21" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;SequencePredictor<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></label><div class="sk-toggleable__content "><pre>SequencePredictor(
input_var=&#x27;consultation_sequence&#x27;,
grouping_var=&#x27;final_sequence&#x27;,
outcome_var=&#x27;specialty&#x27;,
apply_special_category_filtering=False,
admit_col=&#x27;is_admitted&#x27;
)</pre></div> </div></div></div></div>

Meta data about the model can be viewed in the metrics object

```python
spec_model.metrics
```

    {'train_dttm': '2025-03-20 16:30',
     'train_set_no': 42852,
     'start_date': '3/1/2031',
     'end_date': '8/9/2031'}

Passing an empty tuple to the trained model shows the probability of ending in each specialty, if a visit has had no consults yet.

```python
print("For a visit which has no consult at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below")
print({k: round(v, 3) for k, v in spec_model.predict(tuple()) .items()})



```

    For a visit which has no consult at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below
    {'surgical': 0.258, 'medical': 0.574, 'paediatric': 0.078, 'haem/onc': 0.09}

The probabilities for each consult sequence ending in a given observed specialty have been saved in the model. These can be accessed as follows:

```python
spec_model.weights[()].keys()
```

    dict_keys(['surgical', 'medical', 'paediatric', 'haem/onc'])

```python
weights = spec_model.weights
print("For a visit which has one consult to acute medicine at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below")
print({k: round(v, 3) for k, v in weights[tuple(['acute'])].items()})

```

    For a visit which has one consult to acute medicine at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below
    {'surgical': 0.013, 'medical': 0.946, 'paediatric': 0.002, 'haem/onc': 0.039}

The intermediate mapping of consultation_sequence to final_sequence can be accessed from the trained model like this. The first row shows the probability of a null sequence (ie no consults yet) ending in any of the final_sequence options.

```python
spec_model.input_to_grouping_probs
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
      <th>('acute',)</th>
      <th>('acute', 'acute')</th>
      <th>('acute', 'acute', 'medical')</th>
      <th>('acute', 'acute', 'medical', 'surgical')</th>
      <th>('acute', 'acute', 'mental_health')</th>
      <th>('acute', 'acute', 'palliative')</th>
      <th>('acute', 'acute', 'surgical')</th>
      <th>('acute', 'allied')</th>
      <th>('acute', 'allied', 'acute')</th>
      <th>...</th>
      <th>('surgical', 'surgical')</th>
      <th>('surgical', 'surgical', 'acute')</th>
      <th>('surgical', 'surgical', 'acute', 'mental_health', 'discharge', 'discharge')</th>
      <th>('surgical', 'surgical', 'acute', 'surgical')</th>
      <th>('surgical', 'surgical', 'icu')</th>
      <th>('surgical', 'surgical', 'medical')</th>
      <th>('surgical', 'surgical', 'obs_gyn')</th>
      <th>('surgical', 'surgical', 'other')</th>
      <th>('surgical', 'surgical', 'surgical')</th>
      <th>probability_of_grouping_sequence</th>
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
      <th>()</th>
      <td>0.009819</td>
      <td>0.458837</td>
      <td>0.013218</td>
      <td>0.000755</td>
      <td>0.000378</td>
      <td>0.000755</td>
      <td>0.000000</td>
      <td>0.000378</td>
      <td>0.005665</td>
      <td>0.000378</td>
      <td>...</td>
      <td>0.007553</td>
      <td>0.000755</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000378</td>
      <td>0.534194</td>
    </tr>
    <tr>
      <th>('acute',)</th>
      <td>0.000000</td>
      <td>0.830409</td>
      <td>0.005848</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014620</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.206980</td>
    </tr>
    <tr>
      <th>('acute', 'acute')</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.909091</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004438</td>
    </tr>
    <tr>
      <th>('acute', 'allied')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000202</td>
    </tr>
    <tr>
      <th>('acute', 'ambulatory')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000403</td>
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
      <th>('surgical', 'medical')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>('surgical', 'obs_gyn')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>('surgical', 'surgical')</th>
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
      <td>...</td>
      <td>0.814815</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.005447</td>
    </tr>
    <tr>
      <th>('surgical', 'surgical', 'acute')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>('surgical', 'surgical', 'icu')</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000202</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 275 columns</p>
</div>

```python
# save models and metadata
from patientflow.train.utils import save_model

save_model(spec_model, "specialty_no_filtering", model_file_path)
print(f"Model has been saved to {model_file_path}")
```

    Model has been saved to /Users/zellaking/Repos/patientflow/trained-models/public

## Handle special categories

At UCLH, we assume that all under 18s will be admitted to a paediatric specialty. Their visits are therefore used to train the specialty predictor. An `apply_special_category_filtering` parameter can be set in the `SequencePredictor` to handle certain categories differently. When this is set, the `SequencePredictor` will retrieve the relevant logic that has been defined in a class called `SpecialCategoryParams`.

```python
train_visits.snapshot_date.min()
```

    '3/1/2031'

```python
spec_model= SequencePredictor(
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=True,
)

spec_model.fit(train_visits)

weights = spec_model.weights
print("For a visit which has no consult at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below")
print({k: round(v, 3) for k, v in spec_model.predict(tuple()) .items()})
print("For a visit which has one consult to acute medicine at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below")
print({k: round(v, 3) for k, v in weights[tuple(['acute'])].items()})

save_model(spec_model, "specialty", model_file_path)


```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[128], line 8
          1 spec_model= SequencePredictor(
          2     input_var="consultation_sequence",
          3     grouping_var="final_sequence",
          4     outcome_var="specialty",
          5     apply_special_category_filtering=True,
          6 )
    ----> 8 spec_model.fit(train_visits)
         10 weights = spec_model.weights
         11 print("For a visit which has no consult at the time of a snapsnot, the probabilities of ending up under a medical, surgical or haem/onc specialty are shown below")


    File ~/Repos/patientflow/src/patientflow/predictors/sequence_predictor.py:178, in SequencePredictor.fit(self, X)
        176     self.metrics["end_date"] = X["snapshot_date"].max()
        177     self.metrics["unique_outcomes"] = len(X[self.outcome_var].unique())
    --> 178     self.metrics["unique_input_sequences"] = len(X[self.input_var].unique())
        179     self.metrics["unique_grouping_sequences"] = len(X[self.grouping_var].unique())
        181 # Preprocess the data


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/pandas/core/series.py:2407, in Series.unique(self)
       2344 def unique(self) -> ArrayLike:  # pylint: disable=useless-parent-delegation
       2345     """
       2346     Return unique values of Series object.
       2347
       (...)
       2405     Categories (3, object): ['a' < 'b' < 'c']
       2406     """
    -> 2407     return super().unique()


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/pandas/core/base.py:1025, in IndexOpsMixin.unique(self)
       1023     result = values.unique()
       1024 else:
    -> 1025     result = algorithms.unique1d(values)
       1026 return result


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/pandas/core/algorithms.py:401, in unique(values)
        307 def unique(values):
        308     """
        309     Return unique values based on a hash table.
        310
       (...)
        399     array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
        400     """
    --> 401     return unique_with_mask(values)


    File ~/miniconda3/envs/patientflow/lib/python3.12/site-packages/pandas/core/algorithms.py:440, in unique_with_mask(values, mask)
        438 table = hashtable(len(values))
        439 if mask is None:
    --> 440     uniques = table.unique(values)
        441     uniques = _reconstruct_data(uniques, original.dtype, original)
        442     return uniques


    File pandas/_libs/hashtable_class_helper.pxi:7248, in pandas._libs.hashtable.PyObjectHashTable.unique()


    File pandas/_libs/hashtable_class_helper.pxi:7195, in pandas._libs.hashtable.PyObjectHashTable._unique()


    TypeError: unhashable type: 'list'

The handling of special categories is saved as an attribute of the trained model as shown below.

```python
spec_model.special_params
```

    {'special_category_func': <bound method SpecialCategoryParams.special_category_func of <patientflow.prepare.SpecialCategoryParams object at 0x28a867ef0>>,
     'special_category_dict': {'medical': 0.0,
      'surgical': 0.0,
      'haem/onc': 0.0,
      'paediatric': 1.0},
     'special_func_map': {'paediatric': <bound method SpecialCategoryParams.special_category_func of <patientflow.prepare.SpecialCategoryParams object at 0x28a867ef0>>,
      'default': <bound method SpecialCategoryParams.opposite_special_category_func of <patientflow.prepare.SpecialCategoryParams object at 0x28a867ef0>>}}

```python

```
