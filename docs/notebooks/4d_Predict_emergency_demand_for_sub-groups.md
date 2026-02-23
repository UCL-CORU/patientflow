# 4d. Predict emergency demand for sub-groups

In the previous notebook I demonstrated how we evaluate the models used at UCLH. As a final step, I now show the same implementation in code, but including extra functionality to handle certain sub-groups differently from others.

At UCLH, it is standard practice to admit paediatric patients (defined as patients under 18 on the day of arrival at the ED) to paediatric wards, and not to admit adult patients (18 or over) to paediatric wards.

The two models that enable prediction by sub-groups (specialty of admission, and yet-to-arrive by specialty) offer parameters that allow you to specify that certain groups are handled differently. In the UCLH example, this means disregarding any consult requests for patients in the ED when predicting which specialty they will be admitted to, and counting all yet-to-arrive patients under 18 as paediatric admissions.

Most of the code below is the same as in the previous notebook. I limit the narrative here to pointing out how the special sub-groups are handled.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

## Load data and train models

The data loading, configuration, and model training steps are identical to those demonstrated in detail in notebook 4b. Here we use `prepare_prediction_inputs` to perform all of these steps in a single call.

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you don't have the public data, change `data_folder_name` from `'data-public'` to `'data-synthetic'`.

```python
from patientflow.train.emergency_demand import prepare_prediction_inputs

data_folder_name = 'data-public'
prediction_inputs = prepare_prediction_inputs(data_folder_name)

# Unpack the results
admissions_models = prediction_inputs['admission_models']
spec_model = prediction_inputs['specialty_model']
yta_model_by_spec = prediction_inputs['yta_model']
ed_visits = prediction_inputs['ed_visits']
inpatient_arrivals = prediction_inputs['inpatient_arrivals']
specialties = prediction_inputs['specialties']
params = prediction_inputs['config']
```

    /Users/zellaking/Repos/patientflow/src/patientflow/train/emergency_demand.py:470: UserWarning: Parsing dates in %Y-%m-%d format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
      ed_visits["snapshot_date"] = pd.to_datetime(


    Split sizes: [62071, 10415, 29134]
    Split sizes: [7716, 1285, 3898]

    Processing: (6, 0)

    Processing: (9, 30)

    Processing: (12, 0)

    Processing: (15, 30)

    Processing: (22, 0)

```python
import pandas as pd
from datetime import timedelta
from patientflow.prepare import create_temporal_splits
from patientflow.load import get_model_key

model_name = 'admissions'

# Extract config parameters
start_training_set = params['start_training_set']
start_validation_set = params['start_validation_set']
start_test_set = params['start_test_set']
end_test_set = params['end_test_set']
prediction_window = timedelta(minutes=params['prediction_window'])
x1, y1, x2, y2 = params['x1'], params['y1'], params['x2'], params['y2']
yta_time_interval = timedelta(minutes=params['yta_time_interval'])

# Create temporal splits for the ED visits
train_visits_df, valid_visits_df, test_visits_df = create_temporal_splits(
    ed_visits, start_training_set, start_validation_set,
    start_test_set, end_test_set, col_name='snapshot_date',
)

# Create temporal splits for inpatient arrivals
inpatient_arrivals['arrival_datetime'] = pd.to_datetime(
    inpatient_arrivals['arrival_datetime'], utc=True
)
train_inpatient_arrivals_df, _, test_inpatient_arrivals_df = create_temporal_splits(
    inpatient_arrivals, start_training_set, start_validation_set,
    start_test_set, end_test_set, col_name='arrival_datetime',
)

# Define columns excluded from training data (needed by evaluation functions)
exclude_from_training_data = [
    'snapshot_date', 'prediction_time', 'visit_number',
    'consultation_sequence', 'specialty', 'final_sequence',
]
```

    Split sizes: [62071, 10415, 29134]
    Split sizes: [7716, 1285, 3898]

## Explore the cohort-aware specialty model

The `prepare_prediction_inputs` function trained a `MultiSubgroupPredictor` for specialty prediction. This model handles paediatric and adult patients as separate cohorts, each with their own `SequenceToOutcomePredictor`. Let's explore how it works.

The `MultiSubgroupPredictor` was configured with subgroup functions that identify paediatric patients (age group 0-17) and adult patients, then trains separate specialty predictors for each subgroup.

```python
from patientflow.predictors.sequence_to_outcome_predictor import SequenceToOutcomePredictor
from patientflow.predictors.subgroup_predictor import MultiSubgroupPredictor

def create_subgroup_functions_from_age_group():
    """Create subgroup functions that work with age_group categorical variable."""

    def is_paediatric(row):
        return row.get("age_group") == "0-17"

    def is_adult(row):
        # All non-paediatric patients are adults
        return row.get("age_group") != "0-17"

    return {
        "paediatric": is_paediatric,
        "adult": is_adult,
    }

subgroup_functions = create_subgroup_functions_from_age_group()

spec_model = MultiSubgroupPredictor(
    subgroup_functions=subgroup_functions,
    base_predictor_class=SequenceToOutcomePredictor,
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    min_samples=50,  # Minimum samples required per subgroup
)
spec_model = spec_model.fit(train_visits_df)
```

By training on the data, we have derived the following mapping. The intended containment of children to paediatric specialties only, and excluding adults form paediatric specialties did not work as intended. That is because `infer_specialty_to_subgroups` function includes any subgroup that appears **at least once** for a specialty in the training data. This means that a few edge cases (e.g., an adult patient incorrectly coded as being admitted to paediatric specialty, or vice versa, or a legitimate reason for breaking the usual policy), both subgroups will be included.

```python
spec_model.specialty_to_subgroups
```

    {'medical': ['paediatric', 'adult'],
     'surgical': ['paediatric', 'adult'],
     'paediatric': ['paediatric', 'adult'],
     'haem/onc': ['paediatric', 'adult']}

Looking at the actual subgroup mapping in the data, we see that some children do go to adult specialties and vice versa. From the data below, 3% of children were admitted to surgical specialties; these might be genuine decisions rather than coding errors.

```python

for specialty in spec_model.specialty_to_subgroups.keys():
    spec_mask = train_visits_df['specialty'] == specialty
    spec_data = train_visits_df[spec_mask]

    paediatric_count = spec_data.apply(subgroup_functions['paediatric'], axis=1).sum()
    adult_count = spec_data.apply(subgroup_functions['adult'], axis=1).sum()
    total = len(spec_data)

    print(f"\n{specialty} specialty:")
    print(f"  Total patients: {total}")
    print(f"  Paediatric (age_group == '0-17'): {paediatric_count} ({paediatric_count/total*100:.1f}%)")
    print(f"  Adult (age_group != '0-17'): {adult_count} ({adult_count/total*100:.1f}%)")

    # Check for missing age_group values
    missing_age = spec_data['age_group'].isna().sum()
    if missing_age > 0:
        print(f"  Missing age_group: {missing_age} ({missing_age/total*100:.1f}%)")
```

    medical specialty:
      Total patients: 5392
      Paediatric (age_group == '0-17'): 12 (0.2%)
      Adult (age_group != '0-17'): 5380 (99.8%)

    surgical specialty:
      Total patients: 2185
      Paediatric (age_group == '0-17'): 70 (3.2%)
      Adult (age_group != '0-17'): 2115 (96.8%)

    paediatric specialty:
      Total patients: 528
      Paediatric (age_group == '0-17'): 513 (97.2%)
      Adult (age_group != '0-17'): 15 (2.8%)

    haem/onc specialty:
      Total patients: 707
      Paediatric (age_group == '0-17'): 1 (0.1%)
      Adult (age_group != '0-17'): 706 (99.9%)

We can override this mapping if we wished to enforce stricter policy rules, as shown below.

```python
expected_mapping = {
    'paediatric': ['paediatric'],
    'medical': ['adult'],
    'surgical': ['adult'],
    'haem/onc': ['adult'],
}

# Override the inferred mapping
spec_model.specialty_to_subgroups = expected_mapping

print("Updated specialty_to_subgroups mapping:")
spec_model.specialty_to_subgroups
```

    Updated specialty_to_subgroups mapping:





    {'paediatric': ['paediatric'],
     'medical': ['adult'],
     'surgical': ['adult'],
     'haem/onc': ['adult']}

## Generate predicted distributions for each specialty and prediction time for patients in ED

Now that the models have been trained with the special parameters, we proceed with generating and evaluating predictions. The approach below uses a similar function to the `get_specialty_probability_distributions` function shown in the previous notebook, with some additional logic to identify sub-groups that need special processing.

The function

- retrieves the special parameters than were saved with the specialty predictor
- ensures that only eligible patient snapshots are included in the predictions for each specialty. A temporary version of the test set, called `test_df_eligible` is created for each iteration through the various specialties using only the eligible visits

Why is this necessary? Imagine an ED that currently has 75 adult patients and 25 children. Tje maximum number of beds that could be needed in the paediatric specialties is 25 and the maximum number of beds that could be needed in the adult specialties is 75. Without filtering a probabilty distribution for 100 beds would be produced. The logic below means that adult patients are excluded from the predicted distribution for the paediatric specialty, and children from the predicted distributions for the adult specialty.

```python
from patientflow.prepare import prepare_patient_snapshots, prepare_group_snapshot_dict
from patientflow.aggregate import get_prob_dist
from patientflow.predict.emergency_demand import get_specialty_probs
from patientflow.load import get_model_key


def get_specialty_probability_distributions_with_special_categories(
    test_visits_df,
    spec_model,
    admissions_models,
    model_name,
    exclude_from_training_data,
    specialties=['medical', 'surgical', 'haem/onc', 'paediatric'],
    baseline_prob_dict=None,
):
    """
    Calculate probability distributions for emergency department patients by specialty and prediction time.

    Args:
        test_visits_df: DataFrame containing test visit data
        spec_model: Model for specialty predictions (SequenceToOutcomePredictor)
        admissions_models: Dictionary of admission prediction models
        model_name: Name of the model to use
        specialties: List of specialties to consider
        exclude_from_training_data: List of columns to exclude from training data
        baseline_prob_dict: Optional dict of baseline probabilities to use instead of spec_model predictions

    Returns:
        Dictionary containing probability distributions for each specialty and prediction time
    """
    # Get specialty prediction parameters
    special_params = spec_model.special_params
    special_category_func = special_params["special_category_func"]
    special_category_dict = special_params["special_category_dict"]
    special_func_map = special_params["special_func_map"]

    # Get predictions of admission to specialty
    if baseline_prob_dict is not None:
        # Use baseline probabilities instead of model predictions
        # Create paediatric dictionary for age group 0-17
        paediatric_dict = {key: 0 for key in baseline_prob_dict.keys()}
        paediatric_dict['paediatric'] = 1

        # Apply different dictionaries based specialty category function
        test_visits_df.loc[:, "specialty_prob"] = test_visits_df.apply(
            lambda row: paediatric_dict if special_category_func(row) else baseline_prob_dict,
            axis=1
        )
    else:
        # Use spec_model to get predictions
        test_visits_df.loc[:, "specialty_prob"] = get_specialty_probs(
            specialties,
            spec_model,
            test_visits_df,
            special_category_func=special_category_func,
            special_category_dict=special_category_dict,
        )

    # Initialize dictionary to store probability distributions
    prob_dist_dict_all = {}

    # Process each time of day
    for _prediction_time in test_visits_df.prediction_time.unique():
        prob_dist_dict_for_pats_in_ED = {}
        print("\nProcessing :" + str(_prediction_time))
        model_key = get_model_key(model_name, _prediction_time)

        for specialty in specialties:
            print(f"Predicting bed counts for {specialty} specialty, for all snapshots in the test set")

            # Get indices of patients who are eligible for this specialty
            func = special_func_map.get(specialty, special_func_map["default"])
            non_zero_indices = test_visits_df[
                test_visits_df.apply(func, axis=1)
            ].index

            test_df_eligible = test_visits_df.copy()
            test_df_eligible = test_df_eligible.loc[non_zero_indices]

            # Get probability of admission to specialty for eligible patients
            prob_admission_to_specialty = test_df_eligible["specialty_prob"].apply(
                lambda x: x[specialty]
            )

            # Prepare patient snapshots
            X_test, y_test = prepare_patient_snapshots(
                df=test_df_eligible,
                prediction_time=_prediction_time,
                single_snapshot_per_visit=False,
                exclude_columns=exclude_from_training_data,
                visit_col='visit_number'
            )

            # Filter probabilities for eligible patients
            filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[
                non_zero_indices
            ]

            # Prepare group snapshots
            group_snapshots_dict = prepare_group_snapshot_dict(
                test_df_eligible[test_df_eligible.prediction_time == _prediction_time]
            )

            admitted_to_specialty = test_df_eligible['specialty'] == specialty

            # Get probability distribution for this time and specialty
            prob_dist_dict_for_pats_in_ED[specialty] = get_prob_dist(
                group_snapshots_dict, X_test, y_test, admissions_models[model_key],
                weights=filtered_prob_admission_to_specialty,
                category_filter=admitted_to_specialty,
                normal_approx_threshold=30
            )

        prob_dist_dict_all[f'{model_key}'] = prob_dist_dict_for_pats_in_ED

    return prob_dist_dict_all
```

```python
prob_dist_dict_all = get_specialty_probability_distributions_with_special_categories(
    test_visits_df=test_visits_df,
    spec_model=spec_model,
    admissions_models=admissions_models,
    model_name=model_name,
    exclude_from_training_data=exclude_from_training_data,
)
```

    Processing :(22, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(6, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(15, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(9, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(12, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

## Visualise the performance of emergency demand prediction models for patients in the ED

Below I generate EPUDD plots for each specialties, using both the baseline predictions, and the sequence predictor. The plots are very similar to the previous notebook. Using age as a fixed category for identifying patients destined for paediatric or adult wards yields similar results.

```python
baseline_probs = train_inpatient_arrivals_df[~(train_inpatient_arrivals_df.is_child) &
                                             (train_inpatient_arrivals_df.specialty.isin(['medical', 'surgical', 'haem/onc']))]['specialty'].value_counts(normalize=True).to_dict()
baseline_probs['paediatric'] = 0

prob_dist_dict_all_baseline = get_specialty_probability_distributions_with_special_categories(
    test_visits_df=test_visits_df,
    spec_model=spec_model,
    admissions_models=admissions_models,
    model_name=model_name,
    exclude_from_training_data=exclude_from_training_data,
    baseline_prob_dict=baseline_probs
)
```

    Processing :(22, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(6, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(15, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(9, 30)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

    Processing :(12, 0)
    Predicting bed counts for medical specialty, for all snapshots in the test set
    Predicting bed counts for surgical specialty, for all snapshots in the test set
    Predicting bed counts for haem/onc specialty, for all snapshots in the test set
    Predicting bed counts for paediatric specialty, for all snapshots in the test set

```python
from patientflow.viz.epudd import plot_epudd

for specialty in ['medical', 'surgical', 'haem/onc', 'paediatric']:

    print(f'\nEPUDD plots for {specialty} specialty: baseline vs sequence predictor')

    specialty_prob_dist_baseline = {time: dist_dict[specialty] for time, dist_dict in prob_dist_dict_all_baseline.items()}
    specialty_prob_dist = {time: dist_dict[specialty] for time, dist_dict in prob_dist_dict_all.items()}

    plot_epudd(ed_visits.prediction_time.unique(),
        specialty_prob_dist_baseline,
        model_name="admissions",
        suptitle=f"EPUDD plots for {specialty} specialty using baseline probability")

    plot_epudd(ed_visits.prediction_time.unique(),
        specialty_prob_dist,
        model_name="admissions",
        suptitle=f"EPUDD plots for {specialty} specialty using sequence predictor")
```

    EPUDD plots for medical specialty: baseline vs sequence predictor

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_1.png)

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_2.png)

    EPUDD plots for surgical specialty: baseline vs sequence predictor

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_4.png)

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_5.png)

    EPUDD plots for haem/onc specialty: baseline vs sequence predictor

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_7.png)

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_8.png)

    EPUDD plots for paediatric specialty: baseline vs sequence predictor

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_10.png)

![png](4d_Predict_emergency_demand_for_sub-groups_files/4d_Predict_emergency_demand_for_sub-groups_18_11.png)

## Summary

In this notebook I have shown how to specify that certain groups are handled differently. In the UCLH case, we assume that all patients under 18 will be admitted to a paediatric specialty. I have demonstrated how you can use the functions in patientflow to handle such special cases.

Note: the approach illustrated here had been deprecated. The code will still work, but this notebook will be replaced in due course.

## Appendix: Legacy approach for sub-group handling

The following sections show the legacy approach for handling sub-groups, maintained for backward compatibility. The modern approach uses `MultiSubgroupPredictor` as shown above.

## Train specialty model - legacy code

The legacy approach, prior to implementing the MultiSubgroupPredictor was, when training the model predicting specialty of admission, to set a `apply_special_category_filtering` parameter to True. The legacy approach has been included here for completeness.

## Changes required in your implementation

Listed below are the functions that relate to this special handling. However, note that the way this is handled has been deprecated, in favour of a more generalisable approach.

1. **SpecialCategoryParams Class** (`src/patientflow/prepare.py`):

- Specialty names and flags in `special_category_dict`
- Age detection logic in `special_category_func`
- Category mapping in `special_func_map`

2. **SequenceToOutcomePredictor Class** (`src/patientflow/predictors/sequence_to_outcome_predictor.py`):

- Uses `create_special_category_objects` in `_preprocess_data`
- Filters data based on special categories
- Handles specialty predictions differently for special categories

3. **ValueToOutcomePredictor Class** (`src/patientflow/predictors/value_to_outcome_predictor.py`):

- Similar to SequenceToOutcomePredictor, uses special category filtering
- Applies the same filtering logic in `_preprocess_data`

4. **create_yta_filters Function** (`src/patientflow/prepare.py`):

- Creates specialty filters based on special category parameters
- Generates filter configurations for each specialty

5. **get_specialty_probs Function** (`src/patientflow/predict/emergency_demand.py`):

- Uses special category functions to determine specialty probabilities
- Applies different probability distributions for special categories

6. **create_predictions Function** (`src/patientflow/predict/emergency_demand.py`):

- Validates that requested specialties match special category dictionary
- Uses special function map for filtering predictions
- Applies different prediction logic for special categories

7. **WeightedPoissonPredictor Class** (`src/patientflow/predictors/weighted_poisson_predictor.py`):

- Uses specialty filters for predictions
- Handles different prediction logic for special categories

8. **Tests** (`tests/test_create_predictions.py`):

- Test cases for special category handling
- Validation of special category predictions

To modify your implementation for different specialty names and rules, you would need to:

1. Create a new class that inherits from `SpecialCategoryParams` with your custom logic
2. Update all the specialty names and flags in the special category dictionary
3. Modify the detection functions for your special categories
4. Update the filter configurations in `create_yta_filters`
5. Ensure all test cases are updated to reflect your new specialty structure
6. Update any documentation or examples that reference the specialty names

```python
from patientflow.predictors.sequence_to_outcome_predictor import SequenceToOutcomePredictor

spec_model = SequenceToOutcomePredictor(
    input_var="consultation_sequence",
    grouping_var="final_sequence",
    outcome_var="specialty",
    apply_special_category_filtering=True,
)

spec_model = spec_model.fit(train_visits_df)

spec_model
```

<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
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

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
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

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
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

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
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

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
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
  margin-left: 0.5em;
  text-align: center;
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

#sk-container-id-1 a.estimator_doc_link {
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

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SequenceToOutcomePredictor(

    input_var=&#x27;consultation_sequence&#x27;,
    grouping_var=&#x27;final_sequence&#x27;,
    outcome_var=&#x27;specialty&#x27;,
    apply_special_category_filtering=True,
    admit_col=&#x27;is_admitted&#x27;

)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>SequenceToOutcomePredictor</div></div><div><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></div></label><div class="sk-toggleable__content "><pre>SequenceToOutcomePredictor(
input_var=&#x27;consultation_sequence&#x27;,
grouping_var=&#x27;final_sequence&#x27;,
outcome_var=&#x27;specialty&#x27;,
apply_special_category_filtering=True,
admit_col=&#x27;is_admitted&#x27;
)</pre></div> </div></div></div></div>

Under the hood, the `SequenceToOutcomePredictor` will call a `create_special_category_objects()` function that returns rules for how to handle each subgroup. The implementation here is primarily designed to handle paediatric patients (under 18) as a special category. A `SpecialCategoryParams` class generates a dictionary mapping specialties to flags (1.0 for paediatric, 0.0 for others) and functions to identify paediatric patients based on age data. It provides methods to handle both age formats (age_on_arrival or age_group).

The `SequenceToOutcomePredictor` applies these rules during both training and prediction, ensuring consistent handling of special categories across the entire prediction pipeline

The `SpecialCategoryParams` class is designed to be picklable, which is necessary for saving the specialty predictor model to disk.

The output from `create_special_category_objects` is shown below. Note that the output is specific to the UCLH implementation, and that its use has been deprecated in favour of a more generalisable approach.

See below for notes about how to change this for your implementation.

```python
from patientflow.prepare import create_special_category_objects
create_special_category_objects(train_visits_df.columns)
```

    {'special_category_func': <function patientflow.predictors.subgroup_definitions._is_paediatric(row)>,
     'special_category_dict': {'medical': 0.0,
      'surgical': 0.0,
      'haem/onc': 0.0,
      'paediatric': 1.0},
     'special_func_map': {'paediatric': <function patientflow.predictors.subgroup_definitions._is_paediatric(row)>,
      'default': <function patientflow.predictors.subgroup_definitions._is_adult(row)>}}

### Saving of special category information

The `ParametricIncomingAdmissionPredictor` class uses the special category objects during initialisation to create static filters that map specialties to their configurations (e.g., {'is_child': True} for paediatric cases), but does not need them in the predict method. The filters are saved with the instance.

```python
yta_model_by_spec.filters
```

    {'medical': {'specialty': 'medical', 'is_child': False},
     'surgical': {'specialty': 'surgical', 'is_child': False},
     'haem/onc': {'specialty': 'haem/onc', 'is_child': False},
     'paediatric': {'is_child': True}}

In contrast, the `SequenceToOutcomePredictor` save the special parameters as a function, which is used by the predict method to filter and categorise patients based on their characteristics.

```python
spec_model.special_params
```

    {'special_category_func': <function patientflow.predictors.subgroup_definitions._is_paediatric(row)>,
     'special_category_dict': {'medical': 0.0,
      'surgical': 0.0,
      'haem/onc': 0.0,
      'paediatric': 1.0},
     'special_func_map': {'paediatric': <function patientflow.predictors.subgroup_definitions._is_paediatric(row)>,
      'default': <function patientflow.predictors.subgroup_definitions._is_adult(row)>}}
