# 4b. Stratify predictions by patient subgroups

In the previous notebook I introduced the production data structures used to organise predictions. Before assembling the full prediction pipeline, I now show how to stratify predictions by observable patient characteristics using `MultiSubgroupPredictor`.

In practice, different patient subgroups may need to be handled differently. For example, at UCLH:

* Paediatric patients (under 18 on the day of arrival) are almost always admitted to paediatric wards, and adults are never admitted to paediatric wards
* Older adults versus younger adults may have different specialty distributions
* Men versus women may have different patterns of admission

`MultiSubgroupPredictor` handles this by training a separate `SequenceToOutcomePredictor` for each subgroup, using that subgroup's own consult sequences. During training, it also learns from the data which subgroups contribute to each specialty — for example, only paediatric patients are observed in paediatric admissions, so only they are included when predicting paediatric bed counts. At prediction time, each specialty's bed count distribution is generated using only patients from eligible subgroups, with their subgroup-specific specialty probabilities.

For yet-to-arrive patients, the incoming admission predictor similarly uses subgroup-specific arrival rates and specialty distributions.

This notebook introduces the subgroup handling as a building block; the next notebook (4c) shows how it is integrated into the full prediction pipeline.


```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Load data and train models

The data loading, configuration, and model training steps are identical to those demonstrated in detail in notebook 4c. Here we use `prepare_prediction_inputs` to perform all of these steps in a single call.

You can request the UCLH datasets on [Zenodo](https://zenodo.org/records/14866057). If you don't have the public data, change `data_folder_name` from `'data-public'` to `'data-synthetic'`.


```python
from patientflow.train.emergency_demand import prepare_prediction_inputs

data_folder_name = "data-public"
prediction_inputs = prepare_prediction_inputs(data_folder_name)

ed_visits = prediction_inputs["ed_visits"]
params = prediction_inputs["config"]

```

    Split sizes: [62071, 10415, 29134]
    Split sizes: [7716, 1285, 3898]
    
    Processing: (6, 0)


    
    Processing: (9, 30)


    
    Processing: (12, 0)


    
    Processing: (15, 30)


    
    Processing: (22, 0)



```python
from patientflow.prepare import create_temporal_splits

start_training_set = params["start_training_set"]
start_validation_set = params["start_validation_set"]
start_test_set = params["start_test_set"]
end_test_set = params["end_test_set"]

train_visits_df, _, _ = create_temporal_splits(
    ed_visits, start_training_set, start_validation_set,
    start_test_set, end_test_set, col_name="snapshot_date",
)

```

    Split sizes: [62071, 10415, 29134]


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



## Summary

In this notebook I introduced `MultiSubgroupPredictor`, which trains separate specialty predictors for different patient subgroups (here, paediatric versus adult) and controls which subgroups contribute to each specialty's bed count distribution. I showed how the model infers a specialty-to-subgroup mapping from the data, and how this mapping can be overridden to enforce policy rules.

In the next notebook (4c), I show how this subgroup-aware specialty model is integrated into the full prediction pipeline alongside the admission probability model and the yet-to-arrive model.
