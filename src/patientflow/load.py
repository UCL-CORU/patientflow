"""
This module provides functionality for loading configuration files, data from CSV files, and trained machine learning models.

It includes the following features:

- **Loading Configurations**: Parse YAML configuration files and extract necessary parameters for data processing and modeling.
- **Data Handling**: Load and preprocess data from CSV files, including optional operations like setting an index, sorting, and applying literal evaluation on columns.
- **Model Management**: Load saved machine learning models, customize model filenames based on time, and categorize DataFrame columns into predefined groups for analysis.

The module handles common file and parsing errors, returning appropriate error messages or exceptions.

Functions
---------
parse_args:
    Parses command-line arguments for training models.
set_project_root:
    Validates project root path from specified environment variable.
load_config_file:
    Load a YAML configuration file and extract key parameters.
set_file_paths:
    Sets up the file paths based on UCLH-specific or default parameters.
set_data_file_names:
    Set file locations based on UCLH-specific or default data sources.
safe_literal_eval:
    Safely evaluate string literals into Python objects when loading from csv.
load_data:
    Load and preprocess data from a CSV or pickle file.
get_model_key:
    Generate a model name based on the time of day.
load_saved_model:
    Load a machine learning model saved in a joblib file.
get_dict_cols:
    Categorize columns from a DataFrame into predefined groups for analysis.
"""

import ast  # to convert tuples to strings
import os
from pathlib import Path
import sys

import pandas as pd

import yaml
from typing import Any, Dict, Tuple, Union, Optional
import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: The parsed arguments containing 'data_folder_name' and 'uclh' keys.
    """
    parser = argparse.ArgumentParser(description="Train emergency demand models")
    parser.add_argument(
        "--data_folder_name",
        type=str,
        default="data-synthetic",
        help="Location of data for training",
    )
    parser.add_argument(
        "--uclh",
        type=lambda x: x.lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Train using UCLH data (True) or Public data (False)",
    )
    args = parser.parse_args()
    return args


def set_project_root(env_var: Optional[str] = None) -> Path:
    """
    Sets project root path from environment variable or infers it from current path.

    First checks specified environment variable for project root path.
    If not found, searches current path hierarchy for highest-level 'patientflow' directory.

    Args:
        env_var (Optional[str]): Name of environment variable containing project root path

    Returns:
        Path: Validated project root path

    Raises:
        ValueError: If environment variable not set and 'patientflow' not found in path
        NotADirectoryError: If path doesn't exist
        TypeError: If env_var is not None and not a string
    """
    # Only try to get env path if env_var is provided
    env_path: Optional[str] = os.getenv(env_var) if env_var is not None else None
    project_root: Optional[Path] = None

    # Try getting from environment variable first
    if env_path is not None:
        try:
            project_root = Path(env_path)
            if not project_root.is_dir():
                raise NotADirectoryError(f"Path does not exist: {project_root}")
            print(f"Project root from environment: {project_root}")
            return project_root
        except (TypeError, ValueError) as e:
            print(f"Error converting {env_path} to Path: {e}")
            raise
    else:
        # If not in env var, try to infer from current path
        current: Path = Path().absolute()

        # Search through parents to find highest-level 'patientflow' directory
        for parent in [current, *current.parents]:
            if parent.name == "patientflow" and parent.is_dir():
                project_root = parent
                # Continue searching to find highest level

        if project_root:
            print(f"Inferred project root: {project_root}")
            return project_root

        print(
            f"Could not find project root - {env_var} not set and 'patientflow' not found in path"
        )
        print(f"\nCurrent directory: {Path().absolute()}")
        if env_var:
            print(f"\nRun one of these commands in a new cell to set {env_var}:")
            print("# Linux/Mac:")
            print(f"%env {env_var}=/path/to/project")
            print("\n# Windows:")
            print(f"%env {env_var}=C:\\path\\to\\project")
        raise ValueError("Project root not found")


def load_config_file(
    config_file_path: str, return_start_end_dates: bool = False
) -> Optional[Union[Dict[str, Any], Tuple[str, str]]]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file_path : str
        The path to the configuration file.
    return_start_end_dates : bool, optional
        If True, return only the start and end dates from the file (default is False).

    Returns
    -------
    dict or tuple or None
        If `return_start_end_dates` is True, returns a tuple of start and end dates (str).
        Otherwise, returns a dictionary containing the configuration parameters.
        Returns None if an error occurs during file reading or parsing.
    """
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

    try:
        if return_start_end_dates:
            # load the dates used in saved data for uclh versions
            if "file_dates" in config and config["file_dates"]:
                start_date, end_date = [str(item) for item in config["file_dates"]]
                return (start_date, end_date)
            else:
                print(
                    "Error: 'file_dates' key not found or empty in the configuration file."
                )
                return None

        params: Dict[str, Any] = {}

        if "prediction_times" in config:
            params["prediction_times"] = [
                tuple(item) for item in config["prediction_times"]
            ]
        else:
            print("Error: 'prediction_times' key not found in the configuration file.")
            sys.exit(1)

        if "modelling_dates" in config and len(config["modelling_dates"]) == 4:
            (
                params["start_training_set"],
                params["start_validation_set"],
                params["start_test_set"],
                params["end_test_set"],
            ) = [item for item in config["modelling_dates"]]
        else:
            print(
                f"Error: expecting 4 modelling dates and only got {len(config.get('modelling_dates', []))}"
            )
            return None

        params["x1"] = float(config.get("x1", 4))
        params["y1"] = float(config.get("y1", 0.76))
        params["x2"] = float(config.get("x2", 12))
        params["y2"] = float(config.get("y2", 0.99))
        params["prediction_window"] = config.get("prediction_window", 480)
        params["epsilon"] = config.get("epsilon", 10**-7)
        params["yta_time_interval"] = config.get("yta_time_interval", 15)

        return params

    except KeyError as e:
        print(f"Error: Missing key in the configuration file: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid value found in the configuration file: {e}")
        return None


def set_file_paths(
    project_root: Path,
    data_folder_name: str,
    train_dttm: Optional[str] = None,
    inference_time: bool = False,
    config_file: str = "config.yaml",
    prefix: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    """
    Sets up the file paths

    Args:
        project_root (Path): Root path of the project
        data_folder_name (str): Name of the folder where data files are located
        train_dttm (Optional[str], optional): A string representation of the datetime at which training commenced. Defaults to None
        inference_time (bool, optional): A flag indicating whether it is inference time or not. Defaults to False
        config_file (str, optional): Name of config file. Defaults to "config.yaml"
        prefix (Optional[str], optional): String to prefix model folder names. Defaults to None
        verbose (bool, optional): Whether to print path information. Defaults to True

    Returns:
        tuple: Contains (data_file_path, media_file_path, model_file_path, config_path)
    """

    config_path = Path(project_root) / config_file
    if verbose:
        print(f"Configuration will be loaded from: {config_path}")

    data_file_path = Path(project_root) / data_folder_name
    if verbose:
        print(f"Data files will be loaded from: {data_file_path}")

    model_id = data_folder_name.lstrip("data-")
    if prefix:
        model_id = f"{prefix}_{model_id}"
    if train_dttm:
        model_id = f"{model_id}_{train_dttm}"

    model_file_path = Path(project_root) / "trained-models" / model_id
    media_file_path = model_file_path / "media"

    if not inference_time:
        if verbose:
            print(f"Trained models will be saved to: {model_file_path}")
        model_file_path.mkdir(parents=True, exist_ok=True)
        (model_file_path / "model-output").mkdir(parents=False, exist_ok=True)
        media_file_path.mkdir(parents=False, exist_ok=True)
        if verbose:
            print(f"Images will be saved to: {media_file_path}")

    return data_file_path, media_file_path, model_file_path, config_path


def set_data_file_names(uclh, data_file_path, config_file_path=None):
    """
    Set file locations based on UCLH or default data source.

    Parameters
    ----------
    uclh : bool
        If True, use UCLH-specific file locations. If False, use default file locations.
    data_file_path : Path
        The base path to the data directory.
    config_file_path : str, optional
        The path to the configuration file, required if `uclh` is True.

    Returns
    -------
    tuple
        Paths to the required files (visits, arrivals) based on the configuration.
    """
    if not isinstance(data_file_path, Path):
        data_file_path = Path(data_file_path)

    if not uclh:
        csv_filename = "ed_visits.csv"
        yta_csv_filename = "inpatient_arrivals.csv"

        visits_csv_path = data_file_path / csv_filename
        yta_csv_path = data_file_path / yta_csv_filename

        return visits_csv_path, yta_csv_path

    else:
        start_date, end_date = load_config_file(
            config_file_path, return_start_end_dates=True
        )
        data_filename = (
            "uclh_visits_exc_beds_inc_minority_"
            + str(start_date)
            + "_"
            + str(end_date)
            + ".pickle"
        )
        csv_filename = "uclh_ed_visits.csv"
        yta_filename = (
            "uclh_yet_to_arrive_" + str(start_date) + "_" + str(end_date) + ".pickle"
        )
        yta_csv_filename = "uclh_inpatient_arrivals.csv"

        visits_path = data_file_path / data_filename
        yta_path = data_file_path / yta_filename

        visits_csv_path = data_file_path / csv_filename
        yta_csv_path = data_file_path / yta_csv_filename

    return visits_path, visits_csv_path, yta_path, yta_csv_path


def safe_literal_eval(s):
    """
    Safely evaluate a string literal into a Python object.
    Handles list-like strings by converting them to lists.

    Parameters
    ----------
    s : str
        The string to evaluate.

    Returns
    -------
    Any, list, or None
        The evaluated Python object if successful, a list if the input is list-like,
        or None for empty/null values.
    """
    if pd.isna(s) or str(s).strip().lower() in ["nan", "none", ""]:
        return None

    if isinstance(s, str):
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                # Remove square brackets and split by comma
                items = s[1:-1].split(",")
                # Strip whitespace from each item and remove empty strings
                return [item.strip() for item in items if item.strip()]
            except Exception:
                # If the above fails, fall back to ast.literal_eval
                pass

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If ast.literal_eval fails, return the original string
        return s


def data_from_csv(csv_path, index_column=None, sort_columns=None, eval_columns=None):
    """
    Loads data from a CSV file, with optional transformations. LEGACY!

    This function loads a CSV file into a pandas DataFrame and provides the following optional features:
    - Setting a specified column as the index.
    - Sorting the DataFrame by one or more specified columns.
    - Applying safe literal evaluation to specified columns to handle string representations of Python objects.

    Parameters
    ----------
    csv_path : str
        The relative or absolute path to the CSV file.
    index_column : str, optional
        The column to set as the index of the DataFrame. If not provided, no index column is set.
    sort_columns : list of str, optional
        A list of columns by which to sort the DataFrame. If not provided, the DataFrame is not sorted.
    eval_columns : list of str, optional
        A list of columns to which `safe_literal_eval` should be applied. This is useful for columns containing
        string representations of Python data structures (e.g., lists, dictionaries).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded data with any specified transformations applied.

    Raises
    ------
    SystemExit
        If the file cannot be found or another error occurs during loading or processing.

    Notes
    -----
    The function will terminate the program with a message if the file is not found or if any errors
    occur while loading the data. If sorting columns or applying `safe_literal_eval` fails,
    a warning message is printed, but execution continues.

    """
    path = os.path.join(Path().home(), csv_path)

    if not os.path.exists(path):
        print(f"Data file not found at path: {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path, parse_dates=True)
    except FileNotFoundError:
        print(f"Data file not found at path: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if index_column:
        try:
            if df.index.name != index_column:
                df = df.set_index(index_column)
        except KeyError:
            print(f"Index column '{index_column}' not found in dataframe")

    if sort_columns:
        try:
            df.sort_values(sort_columns, inplace=True)
        except KeyError:
            print("One or more sort columns not found in dataframe")

    if eval_columns:
        for column in eval_columns:
            if column in df.columns:
                try:
                    df[column] = df[column].apply(safe_literal_eval)
                except Exception as e:
                    print(f"Error applying safe_literal_eval to column '{column}': {e}")

    return df


def load_data(
    data_file_path,
    file_name,
    index_column=None,
    sort_columns=None,
    eval_columns=None,
    home_path=None,
    encoding=None,
):
    """
    Loads data from CSV or pickle file with optional transformations.

    Parameters
    ----------
    data_file_path : str
        Directory path containing the data file
    file_name : str
        Name of the CSV or pickle file to load
    index_column : str, optional
        Column to set as DataFrame index
    sort_columns : list of str, optional
        Columns to sort DataFrame by
    eval_columns : list of str, optional
        Columns to apply safe_literal_eval to
    home_path : str or Path, optional
        Base path to use instead of user's home directory
    encoding : str, optional
        The encoding to use when reading CSV files (e.g., 'utf-8', 'latin1')

    Returns
    -------
    pd.DataFrame
        Loaded and transformed DataFrame

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the file format is not supported or other processing errors occur
    """
    from pathlib import Path

    # Use provided home_path if available, otherwise default to user's home directory
    base_path = Path(home_path) if home_path else Path.home()
    path = base_path / data_file_path / file_name

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at path: {path}")

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, parse_dates=True, encoding=encoding)
        elif path.suffix.lower() == ".pkl":
            df = pd.read_pickle(path)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. Must be .csv or .pkl"
            )
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    if index_column and df.index.name != index_column:
        try:
            df = df.set_index(index_column)
        except KeyError:
            print(f"Warning: Index column '{index_column}' not found in dataframe")

    if sort_columns:
        try:
            df.sort_values(sort_columns, inplace=True)
        except KeyError:
            print("Warning: One or more sort columns not found in dataframe")

    if eval_columns:
        for column in eval_columns:
            if column in df.columns:
                try:
                    df[column] = df[column].apply(safe_literal_eval)
                except Exception as e:
                    print(
                        f"Warning: Error applying safe_literal_eval to column '{column}': {str(e)}"
                    )

    return df


def get_model_key(model_name, prediction_time):
    """
    Create a model name based on the time of day.

    Parameters
    ----------
    model_name : str
        The base name of the model.
    prediction_time : tuple of int
        A tuple representing the time of day (hour, minute).

    Returns
    -------
    str
        A string representing the model name based on the time of day.
    """

    hour_, min_ = prediction_time
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + "_" + f"{hour_:02}" + min_
    return model_name


def get_dict_cols(df):
    """
    Categorize DataFrame columns into predefined groups.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to categorize.

    Returns
    -------
    dict
        A dictionary where keys are column group names and values are lists of column names in each group.
    """
    not_used_in_training_vars = [
        "snapshot_id",
        "snapshot_date",
        "prediction_time",
        "visit_number",
        "training_validation_test",
        "random_number",
    ]
    arrival_and_demographic_vars = [
        "elapsed_los",
        "sex",
        "age_group",
        "age_on_arrival",
        "arrival_method",
    ]
    summary_vars = [
        "num_obs",
        "num_obs_events",
        "num_obs_types",
        "num_lab_batteries_ordered",
    ]

    location_vars = []
    observations_vars = []
    labs_vars = []
    consults_vars = [
        "has_consultation",
        "consultation_sequence",
        "final_sequence",
        "specialty",
    ]
    outcome_vars = ["is_admitted"]

    for col in df.columns:
        if (
            col in not_used_in_training_vars
            or col in arrival_and_demographic_vars
            or col in summary_vars
        ):
            continue
        elif "visited" in col or "location" in col:
            location_vars.append(col)
        elif "num_obs" in col or "latest_obs" in col:
            observations_vars.append(col)
        elif "lab_orders" in col or "latest_lab_results" in col:
            labs_vars.append(col)
        elif col in consults_vars or col in outcome_vars:
            continue  # Already categorized
        else:
            print(f"Column '{col}' did not match any predefined group")

    # Create a list of column groups
    col_group_names = [
        "not used in training",
        "arrival and demographic",
        "summary",
        "location",
        "observations",
        "lab orders and results",
        "consults",
        "outcome",
    ]

    # Create a list of the column names within those groups
    col_groups = [
        not_used_in_training_vars,
        arrival_and_demographic_vars,
        summary_vars,
        location_vars,
        observations_vars,
        labs_vars,
        consults_vars,
        outcome_vars,
    ]

    # Use dictionary to combine them
    dict_col_groups = {
        category: var_list for category, var_list in zip(col_group_names, col_groups)
    }

    return dict_col_groups
