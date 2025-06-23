# About the notebooks

## Background

The notebooks in this folder demonstrate the core functionality of the `patientflow` package. They have been written by me, Dr Zella King, the primary author of this repository. My aim is to introduce, in a step-by-step approach, how to structure your data for use with the package, and how to use the functions. I conclude with a fully worked example of how we use these functions at University College London Hospital (UCLH) to predict emergency demand for beds.

## Outline of the notebooks

The first notebook explains how to set up your environment to run the notebooks that follow. Instructions are also provided at the bottom of this README.

- **[0_Set_up_your_environment](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/0_Set_up_your_environment.md):** Shows how to set things up if you want to run these notebooks in a Jupyter environment

I then explain who are the intended users of predictive models of patient flow.

- **[1_Meet_the_users_of_our_predictions](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/1_Meet_the_users_of_our_predictions.md):** Talks about the users of patient flow predictions in acute hospitals.

There is then a series of notebooks on preparing patient snapshots, training models on them, and evaluating the performance of those models. I also introduce the real data provided by UCLH in a summary notebook.

- **[2a_Create_patient_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2a_Create_patient_snapshots.md):** Shows how to convert finished hospital visits into patient snapshots.
- **[2b_Predict_using_patient_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2b_Predict_using_patient_snapshots.md):** Shows how to make predictions using patient snapshots, handling multiple visits for a single patient, and multiple snapshots in a single visit.
- **[2c_Evaluate_patient_snapshot_models](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2c_Evaluate_patient_snapshot_models.md):** Demonstrates the use of convenient function to help you evaluate predictive models trained on patient snapshots.
- **[2d_Explore_the_datasets_provided](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/2d_Explore_the_datasets_provided.md):** Provides exploratory plots of the two datasets that accompany this repository.

Next is a series of notebooks on preparing group snapshots, generating predictions for group snapshots, and evaluating the predictions.

- **[3a_Prepare_group_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/3a_Create_group_snapshots.md):** Show how to create group snapshots from patient snapshots.
- **[3b_Evaluate_group_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/3b_Evaluate_group_snapshots.md):** Show how to evaluate predicted bed count distribution generated form group snapshots.
- **[3c_Predict_bed_counts_without_using_patient_snapshots](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/3c_Predict_bed_counts_without_using_patient_snapshots.md):** Show how to predict demand, using historical data, when patient snapshots are not appropriate
- **[3d_Predict_bed_counts_for_subgroups](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/3d_Predict_bed_counts_for_subgroups.md):** Show how to disaggregate bed count distributions by subgroups such as age or specialty of admission

A set of notebooks follow, that show how we have used the functions in `patientflow` at UCLH to predict number of beds needed for emergency demand.

- **[4a_Specify_emergency_demand_model](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4a_Specify_emergency_demand_model.md):** Explains design choices that were made to develop a practical model, and shows an example of the output that is sent five times a day at UCLH.
- **[4b_Predict_emergency_demand](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4b_Predict_emergency_demand.md):** Shows a full implementation of the functions covered up to this point, to predict emergency demand at UCLH.
- **[4c_Evaluate_emergency_demand_predictions](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4b_Predict_demand_from_patients_in_ED.md)** Shows an evaluation of the emergency demand predictions generated using the public data provided with this repo.
- **[4d_Predict_emergency_demand_with_special_categories](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/4d_Predict_emergency_demand_with_special_categories.md)** Develops the logic shown in the previous notebook to include the handling of particular sub-groups of patients differently.

## Preparing your notebook environment

### Installation

You can install the `patientflow` package directly from PyPI:

```bash
pip install patientflow
```

For development purposes or to run these notebooks with the latest code, you may still want to use the Github repository directly. In that case, the `PATH_TO_PATIENTFLOW` environment variable needs to be set so notebooks know where the patientflow repository resides on your computer. You have various options:

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
