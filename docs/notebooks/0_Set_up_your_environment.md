# 0. Set up your environment

In this notebook I will

- Suggest how to set up your environment. You might find the checks below useful to confirm that your environment has been set up correctly for the following notebooks to run.
- Explain where the code expects to find data and where it saves media files by default.

Model files are not saved by these notebooks. Models are re-run for each notebook, so the notebooks will work if run in any order.

See also the [Notebooks README](README.md) in this folder for information about how to set the `project_root` variable.

## Set notebook to reload functions every time a cell is run

This is useful if you make any changes to any underlying code.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Check that the patientflow package has been installed

```python
try:
   import patientflow
   print(f"✓ patientflow {patientflow.__version__} imported successfully")
except ImportError:
   print("❌ patientflow not found - please install using one of the following methods:")
   print("   From PyPI: pip install patientflow")
   print("   For development: pip install -e '.[test]'")
except Exception as e:
   print(f"❌ Error: {e}")
```

    ✓ patientflow 0.2.0 imported successfully

## Set `project_root` variable

The variable called `project_root` tells the notebooks where the patientflow repository resides on your computer. All paths in the notebooks are set relative to `project_root`. There are various ways to set it, which are described in the notebooks [README](README.md).

```python
from patientflow.load import set_project_root
project_root = set_project_root()
```

    Inferred project root: /Users/zellaking/Repos/patientflow

## Set file paths

Now that you have set the project root, you can specify where the data will be loaded from, where images and models are saved, and where to load the config file from. By default, a function called `set_file_paths()` sets these as shown here.

```python
# Basic checks
print(f"patientflow version: {patientflow.__version__}")
print(f"Repository root: {project_root}")

# Verify data access
data_folder_name = 'data-synthetic'
data_file_path = project_root / data_folder_name
if data_file_path.exists():
    print("✓ Synthetic data found")
else:
    print("Synthetic data not found - check repository structure")
```

    patientflow version: 0.2.0
    Repository root: /Users/zellaking/Repos/patientflow
    ✓ Synthetic data found

The`set_file_paths` function will set file paths to default values within the `patientflow` folder, as shown below. File paths for saving media and models are derived from the name of the data folder.

In the notebooks that follow, no trained models are saved by default. All notebooks load data from `data_file_path` and train models from scratch. However, you may want to make use of `model_file_path` to save a model locally, especially they are time-consuming to run in your environment.

The config.yaml file will be loaded from the root directory. It specifies training, validation and test set dates, and some other parameters that will be discussed later.

```python
from patientflow.load import set_file_paths
data_file_path, media_file_path, model_file_path, config_path = set_file_paths(project_root,
               data_folder_name=data_folder_name)
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-synthetic
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/synthetic
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/synthetic/media

## Summary

In this notebook I have shown:

- How to configure your environment to run these notebooks
- Where the notebooks expect to find data, and where they will save media file, by default
