#!/usr/bin/env python3
"""
Data Processing and Anonymization for Hospital Visit Records.

This script provides functions to preprocess hospital visit data, including:
- Calculating patient age on arrival and grouping by age ranges.
- Shifting dates forward to anonymize visit data.
- Mapping consultation codes to predefined consultation types.
- Resampling arrival hours in a target dataset based on source data.

The script can be run as a standalone program to adjust arrival hours in a
target dataset using a source dataset's time distribution.

Functions
---------
- prepare_age_and_dates(df) : Calculates patient age on arrival and categorizes into age groups.
- shift_dates_into_future(df, yta, seed_path) : Shifts all date-related fields into the future for anonymization.
- map_consultations_to_types(df, name_mapping) : Maps consultation codes to their respective consultation types.
- resample_hours(df_source, df_target) : Resamples arrival hours in the target dataset based on the source dataset.
- main() : Command-line interface for resampling hours using input CSV files.

Usage
-----
To run the script from the command line:
    python3 convert.py --source <source_csv> --target <target_csv> --output <output_csv>
"""

import pandas as pd
import numpy as np
import argparse


def prepare_age_and_dates(df):
    """
    Prepare age and date-related features in the dataset.

    This function calculates the age of individuals on arrival based on their
    date of birth and arrival datetime. It also categorizes them into age groups.
    If `snapshot_datetime` exists in the dataframe, it computes the prediction
    time, snapshot date, and elapsed length of stay (LOS).

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing at least the columns 'date_of_birth' and
        'arrival_datetime'. If 'snapshot_datetime' exists, additional
        computations are performed.

    Returns
    -------
    pandas.DataFrame
        The modified dataframe with additional columns:
        - 'age_on_arrival': Numeric representation of age at arrival.
        - 'age_group': Categorical age group.
        - 'prediction_time': Tuple representing the hour and minute of the snapshot.
        - 'snapshot_date': Date extracted from 'snapshot_datetime'.
        - 'elapsed_los': Time in seconds since arrival (if snapshot is available).
    """
    df["age_on_arrival"] = (
        pd.to_timedelta(
            (
                pd.to_datetime(df["arrival_datetime"]).dt.date
                - pd.to_datetime(df["date_of_birth"]).dt.date
            )
        ).dt.days
        / 365.2425
    ).apply(lambda x: np.floor(x) if pd.notna(x) else x)

    bins = [-1, 18, 25, 35, 45, 55, 65, 75, 102]
    labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-115"]
    df["age_group"] = pd.cut(df["age_on_arrival"], bins=bins, labels=labels, right=True)

    if "snapshot_datetime" in df.columns:
        df["prediction_time"] = (
            df["snapshot_datetime"]
            .dt.strftime("%H,%M")
            .apply(lambda x: tuple(map(int, x.split(","))))
        )
        df["snapshot_date"] = pd.to_datetime(df["snapshot_datetime"]).dt.date
        df["elapsed_los"] = (
            df["snapshot_datetime"] - df["arrival_datetime"]
        ).dt.total_seconds()

    return df


def shift_dates_into_future(df, yta, seed_path):
    """
    Shift all date-related columns into the future to anonymize visit data.

    This function reads a random seed from a file, generates a random number
    of weeks to shift all date-related columns, and applies this shift.

    Parameters
    ----------
    df : pandas.DataFrame
        The main dataset containing visit records with date columns.
    yta : pandas.DataFrame
        Additional dataset with arrival and departure datetime fields to be shifted.
    seed_path : str
        Path to the file containing the seed value.

    Returns
    -------
    tuple
        A tuple containing the modified `df` and `yta` dataframes.
    """
    print("\nConverting dates to anonymise visits. Current min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())

    with open(seed_path, "r") as file:
        seed = int(file.read().strip())

    np.random.seed(seed)
    n = np.random.randint(1, 10 * 52)  # Random shift in weeks

    df.loc[:, "snapshot_date"] += pd.Timedelta(days=n * 7)
    df.loc[:, "snapshot_datetime"] += pd.Timedelta(days=n * 7)
    df.loc[:, "arrival_datetime"] += pd.Timedelta(days=n * 7)
    df.loc[:, "departure_datetime"] += pd.Timedelta(days=n * 7)

    print("New min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())

    yta["arrival_datetime"] += pd.Timedelta(days=n * 7)
    yta["departure_datetime"] += pd.Timedelta(days=n * 7)

    return df, yta


def map_consultations_to_types(df, name_mapping):
    """
    Map consultation codes to their respective types using a predefined mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing columns 'consultation_sequence' and 'final_sequence',
        which store lists of consultation codes.
    name_mapping : pandas.DataFrame
        A dataframe with 'code' and 'type' columns mapping consultation codes
        to their respective types.

    Returns
    -------
    pandas.DataFrame
        The modified dataframe with mapped consultation types.
    """
    code_to_type = dict(zip(name_mapping["code"], name_mapping["type"]))

    def map_codes_to_types(codes):
        return [code_to_type.get(code, "unknown") for code in codes]

    df["consultation_sequence"] = df["consultation_sequence"].apply(map_codes_to_types)
    df["final_sequence"] = df["final_sequence"].apply(map_codes_to_types)

    return df


def resample_hours(df_source, df_target):
    """
    Resample arrival hours in `df_target` based on the probability distribution
    of arrival hours from `df_source`.

    Parameters
    ----------
    df_source : pandas.DataFrame
        Source dataframe with 'arrival_datetime' column to derive the hour distribution.
    df_target : pandas.DataFrame
        Target dataframe where arrival hours will be modified.

    Returns
    -------
    pandas.DataFrame
        Modified target dataframe with new arrival hours.
    """
    arrival_hours = df_source["arrival_datetime"].dt.hour
    hour_counts = arrival_hours.value_counts()
    total_arrivals = len(arrival_hours)
    hour_probabilities = hour_counts / total_arrivals

    hours = np.array(hour_probabilities.index)
    probabilities = np.array(hour_probabilities.values)

    new_hours = np.random.choice(hours, size=len(df_target), p=probabilities)
    new_datetimes = pd.Series(
        [x.replace(hour=h) for x, h in zip(df_target["arrival_datetime"], new_hours)]
    )
    df_target["arrival_datetime"] = new_datetimes

    return df_target


def main():
    """
    Main function to resample hours from a source dataset to a target dataset.

    This function reads input CSV files, resamples arrival times in the target
    dataset based on the probability distribution from the source dataset, and
    saves the modified target dataset to an output file.
    """
    parser = argparse.ArgumentParser(
        description="Resample hours from source arrivals data to target data"
    )
    parser.add_argument(
        "--source",
        default="data-public/inpatient_arrivals.csv",
        help="Source CSV file with original hour distribution",
    )
    parser.add_argument(
        "--target",
        default="data-synthetic/inpatient_arrivals.csv",
        help="Target CSV file to modify",
    )
    parser.add_argument(
        "--output",
        default="data-synthetic/inpatient_arrivals_modified.csv",
        help="Output file path",
    )

    args = parser.parse_args()

    df_source = pd.read_csv(args.source, parse_dates=["arrival_datetime"])
    df_target = pd.read_csv(args.target, parse_dates=["arrival_datetime"])

    df_new = resample_hours(df_source, df_target)

    df_new.to_csv(args.output, index=False)
    print(f"Modified {len(df_new)} arrival times and saved to {args.output}")


if __name__ == "__main__":
    main()
