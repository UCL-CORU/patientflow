"""
Generate fake Emergency Department visit data.

This module provides functions to generate fake datasets for patient visits to an emergency department (ED).
It generates arrival and departure times, triage scores, lab orders, and patient admissions.
The functions are used for illustrative purposes in some of the notebooks.

Functions
---------
create_fake_finished_visits(start_date, end_date, mean_patients_per_day)
    Generate synthetic patient visits, triage observations, and lab orders.

create_fake_snapshots(prediction_times, start_date, end_date, df, observations_df, lab_orders_df, mean_patients_per_day)
    Create patient-level snapshots at specific times with visit, triage, and lab features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time


def create_fake_finished_visits(
    start_date, end_date, mean_patients_per_day, admitted_only=False
):
    """
    Generate synthetic patient visit data for an emergency department.

    This function simulates a realistic distribution of patient arrivals, triage scores, lengths of stay,
    admissions, and lab orders over a specified date range. Some patients may have multiple visits.

    Parameters
    ----------
    start_date : str or datetime
        The starting date for the simulation (inclusive). Can be a datetime object or a string in 'YYYY-MM-DD' format.
    end_date : str or datetime
        The ending date for the simulation (exclusive). Can be a datetime object or a string in 'YYYY-MM-DD' format.
    mean_patients_per_day : float
        The average number of patient visits to generate per day.
    admitted_only : bool, optional
        If True, only return admitted patients. The mean_patients_per_day will be adjusted to maintain
        the same total number of admitted patients as would be expected in the full dataset.

    Returns
    -------
    visits_df : pandas.DataFrame
        DataFrame containing visit records with the following columns:
        - 'visit_number'
        - 'patient_id'
        - 'arrival_datetime'
        - 'departure_datetime'
        - 'is_admitted'
        - 'specialty'
        - 'age'
    observations_df : pandas.DataFrame
        DataFrame containing triage score observations with columns:
        - 'visit_number'
        - 'observation_datetime'
        - 'triage_score'
    lab_orders_df : pandas.DataFrame
        DataFrame containing lab test orders with columns:
        - 'visit_number'
        - 'order_datetime'
        - 'lab_name'

    Notes
    -----
    - Patients are more likely to arrive during daytime hours.
    - 20% of patients will have more than one visit during the simulation period.
    - Lab test ordering likelihood depends on the severity of the triage score.
    - When admitted_only=True, the mean_patients_per_day is adjusted to maintain the same number
      of admitted patients as would be expected in the full dataset.
    """

    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Set random seed for reproducibility
    np.random.seed(42)  # You can change this seed value as needed

    # Define admission probabilities based on triage score
    # Triage 1: 80% admission, Triage 2: 60%, Triage 3: 30%, Triage 4: 10%, Triage 5: 2%
    admission_probabilities = {
        1: 0.80,  # Highest severity - highest admission probability
        2: 0.60,
        3: 0.30,
        4: 0.10,
        5: 0.02,  # Lowest severity - lowest admission probability
    }

    # Define triage score distribution
    # Most common is 3-4, less common are 2 and 5, least common is 1 (most severe)
    triage_probabilities = [0.05, 0.15, 0.35, 0.35, 0.10]  # For scores 1-5

    # Calculate total days in range (changed to exclusive end date)
    days_range = (end_date - start_date).days

    # If admitted_only is True, adjust mean_patients_per_day to maintain the same number of admitted patients
    if admitted_only:
        # Calculate expected admission rate based on triage probabilities and admission probabilities
        expected_admission_rate = sum(
            triage_prob * admission_prob
            for triage_prob, admission_prob in zip(
                triage_probabilities, admission_probabilities.values()
            )
        )
        # Adjust mean_patients_per_day to maintain the same number of admitted patients
        mean_patients_per_day = mean_patients_per_day / expected_admission_rate

    # Generate random number of patients for each day using Poisson distribution
    daily_patients = np.random.poisson(mean_patients_per_day, days_range)

    # Calculate the total number of visits
    total_visits = sum(daily_patients)

    # Calculate approximately how many unique patients we need
    # If 20% of patients have more than one visit (let's assume they have exactly 2),
    # then for N total visits, we need approximately N * 0.8 + (N * 0.2) / 2 unique patients
    # Simplifying: N * (0.8 + 0.1) = N * 0.9 unique patients
    num_unique_patients = int(total_visits * 0.9)

    # Create patient ids
    patient_ids = list(range(1, num_unique_patients + 1))

    # Define common ED lab tests and their ordering probabilities based on triage score
    lab_tests = ["CBC", "BMP", "Troponin", "D-dimer", "Urinalysis"]
    lab_probabilities = {
        # Higher severity -> more likely to get labs
        1: {
            "CBC": 0.95,
            "BMP": 0.95,
            "Troponin": 0.90,
            "D-dimer": 0.70,
            "Urinalysis": 0.60,
        },
        2: {
            "CBC": 0.90,
            "BMP": 0.90,
            "Troponin": 0.80,
            "D-dimer": 0.60,
            "Urinalysis": 0.50,
        },
        3: {
            "CBC": 0.80,
            "BMP": 0.80,
            "Troponin": 0.60,
            "D-dimer": 0.40,
            "Urinalysis": 0.40,
        },
        4: {
            "CBC": 0.60,
            "BMP": 0.60,
            "Troponin": 0.30,
            "D-dimer": 0.20,
            "Urinalysis": 0.30,
        },
        5: {
            "CBC": 0.40,
            "BMP": 0.40,
            "Troponin": 0.15,
            "D-dimer": 0.10,
            "Urinalysis": 0.20,
        },
    }

    visits = []
    observations = []
    lab_orders = []
    visit_number = 1

    # Create a dictionary to track number of visits per patient
    patient_visit_count = {patient_id: 0 for patient_id in patient_ids}

    # Create a pool of patients who will have multiple visits (20% of patients)
    multi_visit_patients = set(
        np.random.choice(
            patient_ids, size=int(num_unique_patients * 0.2), replace=False
        )
    )

    for day_idx, num_patients in enumerate(daily_patients):
        current_date = start_date + timedelta(days=day_idx)

        # Generate patients for this day
        for _ in range(num_patients):
            # Select a patient ID based on our requirements
            # If we haven't assigned all patients yet, use a new one
            # Otherwise, pick from multi-visit patients
            available_new_patients = [
                pid for pid in patient_ids if patient_visit_count[pid] == 0
            ]

            if available_new_patients:
                # Use a new patient
                patient_id = np.random.choice(available_new_patients)
            else:
                # All patients have at least one visit, now use multi-visit patients
                patient_id = np.random.choice(list(multi_visit_patients))

            # Increment the visit count for this patient
            patient_visit_count[patient_id] += 1

            # Random hour for arrival (more likely during daytime)
            arrival_hour = np.random.normal(13, 4)  # Mean at 1 PM, std dev of 4 hours
            arrival_hour = max(0, min(23, int(arrival_hour)))  # Clamp between 0-23

            # Random minutes
            arrival_minute = np.random.randint(0, 60)

            # Create arrival datetime
            arrival_datetime = current_date.replace(
                hour=arrival_hour,
                minute=arrival_minute,
                second=np.random.randint(0, 60),
            )

            # Generate triage score (1-5)
            triage_score = np.random.choice([1, 2, 3, 4, 5], p=triage_probabilities)

            # Generate admission status based on triage score
            admission_prob = admission_probabilities[triage_score]
            is_admitted = np.random.choice(
                [0, 1], p=[1 - admission_prob, admission_prob]
            )

            # Generate specialty for admitted patients
            if is_admitted:
                specialty = np.random.choice(
                    ["medical", "surgical", "haem/onc", "paediatric"],
                    p=[0.65, 0.25, 0.05, 0.05],
                )
            else:
                specialty = None

            # Skip this visit if admitted_only is True and patient is not admitted
            if admitted_only and not is_admitted:
                continue

            # Generate length of stay (in minutes) - log-normal distribution
            # Most visits are 4 to 12 hours, but some can be shorter or longer
            length_of_stay = np.random.lognormal(mean=5.8, sigma=0.5)
            length_of_stay = max(
                60, min(2880, length_of_stay)
            )  # Between 1 hour and 48 hours

            # Make higher triage scores (more severe) stay longer on average
            if triage_score <= 2:
                length_of_stay *= 1.8  # 80% longer stays for more severe cases

            # Calculate departure time
            departure_datetime = arrival_datetime + timedelta(
                minutes=int(length_of_stay)
            )

            # For returning patients, use the same age as their first visit
            if patient_id in [v["patient_id"] for v in visits]:
                # Find the age from a previous visit
                age = next(v["age"] for v in visits if v["patient_id"] == patient_id)
            else:
                # Generate age with a distribution skewed towards older adults
                age = int(
                    np.random.lognormal(mean=3.8, sigma=0.5)
                )  # Centers around 45 years
                age = max(0, min(100, age))  # Clamp between 0-100 years

            # Add visit record (without triage score, but with patient_id)
            visits.append(
                {
                    "patient_id": patient_id,
                    "visit_number": visit_number,
                    "arrival_datetime": arrival_datetime,
                    "departure_datetime": departure_datetime,
                    "age": age,
                    "is_admitted": is_admitted,
                    "specialty": specialty,
                }
            )

            # Generate triage observation within first 10 minutes
            minutes_after_arrival = np.random.uniform(0, 10)
            observation_datetime = arrival_datetime + timedelta(
                minutes=minutes_after_arrival
            )

            observations.append(
                {
                    "visit_number": visit_number,
                    "observation_datetime": observation_datetime,
                    "triage_score": triage_score,
                }
            )

            # Generate lab orders if visit is longer than 2 hours
            if length_of_stay > 120:
                # For each lab test, decide if it should be ordered based on triage score
                for lab_test in lab_tests:
                    if np.random.random() < lab_probabilities[triage_score][lab_test]:
                        # Order time is after triage but within first 90 minutes
                        minutes_after_triage = np.random.uniform(
                            0, 90 - minutes_after_arrival
                        )
                        order_datetime = observation_datetime + timedelta(
                            minutes=minutes_after_triage
                        )

                        lab_orders.append(
                            {
                                "visit_number": visit_number,
                                "order_datetime": order_datetime,
                                "lab_name": lab_test,
                            }
                        )

            visit_number += 1

    # Create DataFrames and sort by time
    visits_df = pd.DataFrame(visits)
    visits_df = visits_df.sort_values("arrival_datetime").reset_index(drop=True)

    observations_df = pd.DataFrame(observations)
    observations_df = observations_df.sort_values("observation_datetime").reset_index(
        drop=True
    )

    lab_orders_df = pd.DataFrame(lab_orders)
    if not lab_orders_df.empty:
        lab_orders_df = lab_orders_df.sort_values("order_datetime").reset_index(
            drop=True
        )

    return visits_df, observations_df, lab_orders_df


def create_fake_snapshots(
    prediction_times,
    start_date,
    end_date,
    df=None,
    observations_df=None,
    lab_orders_df=None,
    mean_patients_per_day=50,
):
    """
    Generate patient-level snapshots at specific times for prediction modeling.

    For each specified time on each date in the range, this function returns a snapshot of patients
    who are currently in the emergency department, along with their visit features, latest triage score,
    and number of lab tests ordered prior to that time.

    Parameters
    ----------
    prediction_times : list of tuple of int
        A list of (hour, minute) tuples indicating times of day to create snapshots.
    start_date : str or datetime
        The starting date for generating snapshots (inclusive).
    end_date : str or datetime
        The ending date for generating snapshots (exclusive).
    df : pandas.DataFrame, optional
        Patient visit data from `create_fake_finished_visits`. If None, synthetic data is generated.
    observations_df : pandas.DataFrame, optional
        Triage score data from `create_fake_finished_visits`. If None, synthetic data is generated.
    lab_orders_df : pandas.DataFrame, optional
        Lab order data from `create_fake_finished_visits`. If None, synthetic data is generated.
    mean_patients_per_day : float, optional
        Average number of patients per day (used only if synthetic data is generated).

    Returns
    -------
    final_df : pandas.DataFrame
        A DataFrame with one row per patient visit present at the snapshot time. Columns include:
        - 'snapshot_date'
        - 'prediction_time'
        - 'patient_id'
        - 'visit_number'
        - 'is_admitted'
        - 'age'
        - 'latest_triage_score'
        - One column per lab test: 'num_<lab_name>_orders'

    Notes
    -----
    - Only patients present in the ED at the snapshot time are included.
    - Lab order columns reflect counts of tests ordered before the snapshot time.
    - If no patients are present at a snapshot time, that snapshot is omitted.
    """

    # Generate fake data if not provided
    if df is None or observations_df is None or lab_orders_df is None:
        df, observations_df, lab_orders_df = create_fake_finished_visits(
            start_date, end_date, mean_patients_per_day
        )

    # Add date conversion at the start
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    # Create date range (changed to exclusive end date)
    snapshot_dates = []
    current_date = start_date
    while current_date < end_date:  # Changed from <= to <
        snapshot_dates.append(current_date)
        current_date += timedelta(days=1)

    # Get unique lab test names
    lab_tests = lab_orders_df["lab_name"].unique() if not lab_orders_df.empty else []

    # Create empty list to store all results
    all_results = []

    # For each combination of date and time
    for date in snapshot_dates:
        for hour, minute in prediction_times:
            snapshot_datetime = datetime.combine(date, time(hour=hour, minute=minute))

            # Filter dataframe for this snapshot
            mask = (df["arrival_datetime"] <= snapshot_datetime) & (
                df["departure_datetime"] > snapshot_datetime
            )
            snapshot_df = df[mask].copy()  # Create copy to avoid SettingWithCopyWarning

            # Skip if no patients at this time
            if len(snapshot_df) == 0:
                continue

            # Get triage scores recorded before the snapshot time
            valid_observations = observations_df[
                (observations_df["visit_number"].isin(snapshot_df["visit_number"]))
                & (observations_df["observation_datetime"] <= snapshot_datetime)
            ].copy()

            # Keep only the most recent triage score for each visit
            if not valid_observations.empty:
                valid_observations = valid_observations.sort_values(
                    "observation_datetime"
                )
                valid_observations = (
                    valid_observations.groupby("visit_number").last().reset_index()
                )
                valid_observations = valid_observations.rename(
                    columns={"triage_score": "latest_triage_score"}
                )

            # Get lab orders placed before the snapshot time
            valid_orders = lab_orders_df[
                (lab_orders_df["visit_number"].isin(snapshot_df["visit_number"]))
                & (lab_orders_df["order_datetime"] <= snapshot_datetime)
            ].copy()

            # Initialize lab_counts with zeros for all visits in snapshot_df
            lab_counts = pd.DataFrame(
                0,
                index=pd.Index(
                    snapshot_df["visit_number"].unique(), name="visit_number"
                ),
                columns=[f"num_{test.lower()}_orders" for test in lab_tests],
            )

            # Update counts if there are any valid orders
            if not valid_orders.empty:
                order_counts = (
                    valid_orders.groupby(["visit_number", "lab_name"])
                    .size()
                    .unstack(fill_value=0)
                )
                order_counts.columns = [
                    f"num_{test.lower()}_orders" for test in order_counts.columns
                ]
                # Update the counts in lab_counts where we have orders
                lab_counts.update(order_counts)

            lab_counts = lab_counts.reset_index()

            # Add snapshot information columns
            snapshot_df["snapshot_date"] = date
            snapshot_df["prediction_time"] = [(hour, minute)] * len(snapshot_df)

            # Merge with valid observations to get triage scores, handling empty case
            if not valid_observations.empty:
                snapshot_df = pd.merge(
                    snapshot_df,
                    valid_observations[["visit_number", "latest_triage_score"]],
                    on="visit_number",
                    how="left",
                )
            else:
                snapshot_df["latest_triage_score"] = pd.Series(
                    [np.nan] * len(snapshot_df),
                    dtype="float64",
                    index=snapshot_df.index,
                )
            # Merge with lab counts
            snapshot_df = pd.merge(
                snapshot_df, lab_counts, on="visit_number", how="left"
            )

            # Fill NA values in lab count columns with 0
            for col in snapshot_df.columns:
                if col.endswith("_orders"):
                    snapshot_df[col] = snapshot_df[col].fillna(0)
            if not snapshot_df.empty:
                # Optionally check for all-NA in key columns
                snapshot_cols = [
                    "snapshot_date",
                    "prediction_time",
                    "snapshot_datetime",
                ]
                # Only check columns that exist in the DataFrame
                check_cols = [
                    col for col in snapshot_cols if col in snapshot_df.columns
                ]

                if not check_cols or not snapshot_df[check_cols].isna().all().any():
                    all_results.append(snapshot_df)
                else:
                    print(
                        f"Skipping DataFrame with all-NA values in key columns: {check_cols}"
                    )
            else:
                print("Skipping empty DataFrame")

    # Combine all results into single dataframe
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Define column order
        snapshot_cols = ["snapshot_date", "prediction_time"]
        visit_cols = [
            "patient_id",
            "visit_number",
            "is_admitted",
            "age",
            "latest_triage_score",
        ]
        lab_cols = [col for col in final_df.columns if col.endswith("_orders")]

        # Ensure all required columns exist
        for col in visit_cols:
            if col not in final_df.columns:
                if col == "latest_triage_score":
                    final_df[col] = pd.NA
                else:
                    final_df[col] = None

        # Reorder columns
        final_df = final_df[snapshot_cols + visit_cols + lab_cols]
    else:
        # Create empty dataframe with correct columns if no results found
        lab_cols = [f"num_{test.lower()}_orders" for test in lab_tests]
        columns = [
            "snapshot_date",
            "prediction_time",
            "visit_number",
            "is_admitted",
            "age",
            "latest_triage_score",
        ] + lab_cols
        final_df = pd.DataFrame(columns=columns)

    # Name the index snapshot_id before returning
    final_df.index.name = "snapshot_id"
    return final_df
