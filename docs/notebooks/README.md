# About the notebooks

## Background

The notebooks in this folder demonstrate how you can use the PatientFlow repository. Notebooks combine commentary, code and the results produced by that code. Here's how different audiences can use the notebooks:

- **As a non-programmer seeking to understand the approach:** The narrative sections in each notebook introduce my approach to creating predictive models of emergency demand for a hospital.
- **As a data scientist interested in how to model emergency demand:** The code snippets, combined with the narrative, show how I trained, tested and applied my models in Python. The output of each notebook cell shows the results of running the code.
- **As a researcher interested in the patientflow package:** The repository contains a Python package that can be installed using an import statement in your code, so that you can use the functions I have developed. The notebooks demonstrate use of the functions in the PatientFlow package.

## Outline of the notebooks

The first notebook explains how to set up your environment to run the notebooks that follow. Instructions are also provided at the bottom of this page.

- **[0_Set_up_your_environment](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/0_Set_up_your_environment.md):** Shows how to set things up if you want to run these notebooks in a Jupyter environment

I then explain who are the users of predictive models of patient flow.

- **[1_Meet_the_users_of_our_predictions](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/1_Meet_the_users_of_our_predictions.md):** Talks about the users of patient flow predictions in acute hospitals.

A series of notebooks on patient snapshots.

- **[2a_Create_patient_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2a_Create_patient_snapshots.md):** Shows how to convert finished hospital visits into patient snapshots.
- **[2b_Predict_using_patient_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2b_Predict_using_patient_snapshots.md):** Shows how to make predictions using patient snapshots, handling multiple visits for a single patient, and multiple snapshots in a single visit.
- **[2c_Evaluate_patient_snapshot_models](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2c_Evaluate_patient_snapshot_models.md):** Demonstrates the use of convenient function to help you evaluate predictive models trained on patient snapshots.
- **[2d_Explore_the_datasets_provided](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2d_Explore_the_datasets_provided.md):** Provides exploratory plots of the two datasets that accompany this repository.

Two notebooks on group snapshots.

- **[3a_Prepare_group_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/3a_Create_group_snapshots.md):** Show how to create group snapshots from patient snapshots.

A set of notebooks follow, to show how we have used the functions in `patientflow` at UCLH to predict number of beds needed for emergency demand.

- **[4_Specify_emergency_demand_model](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4_Specify_emergency_demand_model.md):** Explains design choices that were made to develop a practical model, and shows an example of the output that is sent five times a day at UCLH.
- **[4a_Predict_probability_of_admission_from_ED](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4a_Predict_probability_of_admission_from_ED.md):** Shows how to train a machine learning model to predict a patient's probability of admission using patient data from the Emergency Department (ED). This includes dividing the data into training, validation, and testing sets, as well as into subsets based on the time of day the predictions are made, applying an XGBoost model for predictions, and saving the models for future use.
- **[4b_Predict_demand_from_patients_in_ED](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4b_Predict_demand_from_patients_in_ED.md)** Shows how to convert patient-level admission probabilities into a predictions of overall bed demand
- **[4c_Predict_probability_of_admission_to_specialty](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4c_Predict_probability_of_admission_to_specialty.md):** Shows how to train a model predicting specialty of admission; a sequence of consultation requests is mapped to a probability of being admitted to one of three specialties: medical, surgical, and haematology/oncology, with paediatric patients (under 18) handled differently
- **[4d_Predict_demand_from_patients_yet_to_arrive](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4d_Predict_demand_from_patients_yet_to_arrive.md):** Shows the use of a time-varying weighted Poisson distribution to predict a number of patients yet to arrive to the ED within a prediction window (say 8 hours). Demonstrates the use of a function that will assume ED performance targets are met when predicting the number admitted by the end of the prediction window
- **[4e_Predict_probabiity_of_admission_using_minimal_data](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4e_Predict_probabiity_of_admission_using_minimal_data.md):** Shows an example of doing live inference using the models trained in the previous steps
- **[4f_Bring_it_all_together](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4f_Bring_it_all_together.md):** Shows an example of doing live inference using the models trained in the previous steps

## Preparing your notebook environment

The `PATH_TO_PATIENTFLOW` environment variable needs to be set so notebooks know where the patientflow repository resides on your computer. You have various options:

- use a virtual environment and set PATH_TO_PATIENTFLOW up within that
- set PATH_TO_PATIENTFLOW globally on your computer
- let each notebook infer PATH_TO_PATIENTFLOW from the location of the notebook file, or specify it within the notebook

### To set the PATH_TO_PATIENTFLOW environment variable within your virtual environment

**Conda environments**

Add PATH_TO_PATIENTFLOW to the `environment.yml` file:

```yaml
variables:
  PATH_TO_PATIENTFLOW: /path/to/patientflow
```

**venv environment**

Add path_to_patientflow to the venv activation script:

```sh
echo 'export PATH_TO_PATIENTFLOW=/path/to/patientflow' >> venv/bin/activate  # Linux/Mac
echo 'set PATH_TO_PATIENTFLOW=/path/to/patientflow' >> venv/Scripts/activate.bat  # Windows
```

The environment variable will be set whenever you activate the virtual environment and unset when you deactivate it.
Replace /path/to/patientflow with your repository path.

### To set the project_root environment variable from within each notebook

A function called `set_project_root()` can be run in each notebook. If you include the name of a environment variable as shown below, the function will look in your global environment for a variable of this name.

Alternatively, if you call the function without any arguments, the function will try to infer the location of the patientflow repo from your currently active path.

```python
# to specify an environment variable that has been set elsewhere
project_root = set_project_root(env_var ="PATH_TO_PATIENTFLOW")

# to let the notebook infer the path
project_root = set_project_root()

```

You can also set an environment variable from within a notebook cell:

**Linux/Mac:**

```sh
%env PATH_TO_PATIENTFLOW=/path/to/patientflow
```

Windows:

```sh
%env PATH_TO_PATIENTFLOW=C:\path\to\patientflow
```

Replace /path/to/patientflow with the actual path to your cloned repository.

### To set project_root environment variable permanently on your system

**Linux/Mac:**

```sh
# Add to ~/.bashrc or ~/.zshrc:
export PATH_TO_PATIENTFLOW=/path/to/patientflow
```

**Windows:**

```sh
Open System Properties > Advanced > Environment Variables
Under User Variables, click New
Variable name: PATH_TO_PATIENTFLOW
Variable value: C:\path\to\patientflow
Click OK
```

Replace /path/to/patientflow with your repository path. Restart your terminal/IDE after setting.
