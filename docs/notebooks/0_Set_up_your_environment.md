# Set up your environment

Skip this notebook if you are just browsing. 

In this notebook I will explain where the code looks for data and saves models and media by default, and suggest how to set up your environment. There are two README files that may be useful:

* [Repository README](https://github.com/UCL-CORU/patientflow#) in the root of the repository
* [Notebooks README](README.md) in this folder

## Set notebook to reload functions every time a cell is fun

This is useful if you make any changes to any underlying code


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
   print("❌ patientflow not found - please check installation instructions in README")
   print("   pip install -e '.[test]'")
except Exception as e:
   print(f"❌ Error: {e}")
```

    ✓ patientflow 0.1.0 imported successfully


## Set project_root variable 

The variable called project_root tells the notebooks where the patientflow repository resides on your computer. All paths in the notebooks are set relative to project_root. There are various ways to set it, which are described in the notebooks [README](README.md). 


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

    patientflow version: 0.1.0
    Repository root: /Users/zellaking/Repos/patientflow
    ✓ Synthetic data found


The function will set file paths to default values, as shown here. You can override these as required. 


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

In this notebook you have seen 

* how to configure your environment to run these notebooks
* where the notebooks expect to find data, and where they will save models and media, by default

Now you are ready to explore the data that has been provided with this repository


