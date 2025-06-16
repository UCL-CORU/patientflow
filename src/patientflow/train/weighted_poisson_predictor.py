from typing import List
import pandas as pd
from pandas import DataFrame
from datetime import timedelta

from patientflow.prepare import create_special_category_objects
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor


def create_yta_filters(df):
    """
    Create specialty filters for categorizing patients by specialty and age group.

    This function generates a dictionary of filters based on specialty categories,
    with special handling for pediatric patients. It uses the SpecialCategoryParams
    class to determine which specialties correspond to pediatric care.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing patient data with columns that include either
        'age_on_arrival' or 'age_group' for pediatric classification

    Returns:
    --------
    dict
        A dictionary mapping specialty names to filter configurations.
        Each configuration contains:
        - For pediatric specialty: {"is_child": True}
        - For other specialties: {"specialty": specialty_name, "is_child": False}

    Examples:
    ---------
    >>> df = pd.DataFrame({'patient_id': [1, 2], 'age_on_arrival': [10, 40]})
    >>> filters = create_yta_filters(df)
    >>> print(filters['paediatric'])
    {'is_child': True}
    >>> print(filters['medical'])
    {'specialty': 'medical', 'is_child': False}
    """
    # Get the special category parameters using the picklable implementation
    special_params = create_special_category_objects(df.columns)

    # Extract necessary data from the special_params
    special_category_dict = special_params["special_category_dict"]

    # Create the specialty_filters dictionary
    specialty_filters = {}

    for specialty, is_paediatric_flag in special_category_dict.items():
        if is_paediatric_flag == 1.0:
            # For the paediatric specialty, set `is_child` to True
            specialty_filters[specialty] = {"is_child": True}
        else:
            # For other specialties, set `is_child` to False
            specialty_filters[specialty] = {"specialty": specialty, "is_child": False}

    return specialty_filters


def train_weighted_poisson_predictor(
    train_visits: DataFrame,
    train_yta: DataFrame,
    prediction_window: timedelta,
    yta_time_interval: timedelta,
    prediction_times: List[float],
    num_days: int,
    epsilon: float = 10e-7,
) -> WeightedPoissonPredictor:
    """Train a yet-to-arrive prediction model.

    Args:
        train_visits: Visits dataset (used for identifying special categories)
        train_yta: Training data for yet-to-arrive predictions
        prediction_window: Time window for predictions as a timedelta
        yta_time_interval: Time interval for predictions as a timedelta
        prediction_times: List of prediction times
        epsilon: Epsilon parameter for model
        num_days: Number of days to consider

    Returns:
        Trained WeightedPoissonPredictor model

    Raises:
        TypeError: If prediction_window or yta_time_interval are not timedelta objects
    """
    if not isinstance(prediction_window, timedelta):
        raise TypeError("prediction_window must be a timedelta object")
    if not isinstance(yta_time_interval, timedelta):
        raise TypeError("yta_time_interval must be a timedelta object")

    if train_yta.index.name is None:
        if "arrival_datetime" in train_yta.columns:
            train_yta.loc[:, "arrival_datetime"] = pd.to_datetime(
                train_yta["arrival_datetime"], utc=True
            )
            train_yta.set_index("arrival_datetime", inplace=True)

    elif train_yta.index.name != "arrival_datetime":
        print("Dataset needs arrival_datetime column")

    specialty_filters = create_yta_filters(train_visits)

    yta_model = WeightedPoissonPredictor(filters=specialty_filters)
    yta_model.fit(
        train_df=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        num_days=num_days,
    )

    return yta_model
