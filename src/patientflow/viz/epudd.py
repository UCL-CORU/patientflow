"""
Generate plots comparing observed values with model predictions for discrete distributions.

An Evaluating Predictions for Unique, Discrete, Distributions (EPUDD) plot displays the
model's predicted CDF values alongside the actual observed values'
positions within their predicted CDF intervals. For discrete distributions, each predicted
value has an associated probability, and the CDF is calculated by sorting the values and
computing cumulative probabilities.

The plot can show three possible positions for each observation within its predicted interval:

    * lower bound of the interval
    * midpoint of the interval
    * upper bound of the interval

By default, the plot only shows the midpoint of the interval.

For a well-calibrated model, the observed values should fall within their predicted
intervals, with the distribution of positions showing appropriate uncertainty.

The visualisation helps assess model calibration by comparing:
1. The predicted cumulative distribution function (CDF) values
2. The actual positions of observations within their predicted intervals
3. The spread and distribution of these positions

Functions
------------
plot_epudd : function
    Generates and plots the comparison of model predictions with observed values.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from patientflow.load import get_model_key


def _calculate_cdf_values(
    agg_predicted: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate CDF values for discrete distribution."""
    upper_cdf: np.ndarray = agg_predicted.cumsum()
    lower_cdf: np.ndarray = np.hstack((0, upper_cdf[:-1]))
    mid_cdf: np.ndarray = (upper_cdf + lower_cdf) / 2
    return lower_cdf, mid_cdf, upper_cdf


def _create_distribution_records(
    prob_dist_dict: Dict, cdf_types: List[str]
) -> List[Dict]:
    """Create distribution records with CDF values for all time points."""
    all_distributions: List[Dict] = []

    for dt, data in prob_dist_dict.items():
        agg_predicted: np.ndarray = np.array(data["agg_predicted"]["agg_proba"])
        lower_cdf, mid_cdf, upper_cdf = _calculate_cdf_values(agg_predicted)

        cdf_values = {"lower": lower_cdf, "mid": mid_cdf, "upper": upper_cdf}

        for j, prob in enumerate(agg_predicted):
            record = {
                "num_adm_pred": j,
                "prob": prob,
                "dt": dt,
            }

            for cdf_type in cdf_types:
                record[f"{cdf_type}_predicted_cdf"] = cdf_values[cdf_type][j]

            all_distributions.append(record)

    return all_distributions


def _create_observation_records(prob_dist_dict: Dict) -> List[Dict]:
    """Create observation records for all time points."""
    return [
        {"date": dt, "num_adm": data["agg_observed"], "dt": dt}
        for dt, data in prob_dist_dict.items()
    ]


def _plot_predictions(
    ax, distr_coll: pd.DataFrame, num_time_points: int, pred_types: List[str]
) -> None:
    """Plot model predictions for specified types."""
    type_labels = {"lower": "Lower Bound", "mid": "Midpoint", "upper": "Upper Bound"}

    for pred_type in pred_types:
        col_name: str = f"{pred_type}_predicted_cdf"
        df_temp: pd.DataFrame = distr_coll[[col_name, "prob"]].copy()
        df_temp = df_temp.sort_values(by=col_name)
        df_temp["cum_weight_normed"] = df_temp["prob"].cumsum() / num_time_points

        ax.scatter(
            df_temp[col_name],
            df_temp["cum_weight_normed"],
            color="grey",
            label=f"Predicted {type_labels[pred_type]}",
            marker="o",
            s=5,
        )


def _plot_observations(
    ax,
    merged_df: pd.DataFrame,
    obs_types: List[str],
    colors: Dict[str, str],
    is_first_subplot: bool,
) -> None:
    """Plot actual observations for specified types."""
    type_labels = {"lower": "Lower Bound", "mid": "Midpoint", "upper": "Upper Bound"}

    for obs_type in obs_types:
        col_name_obs: str = f"{obs_type}_observed_cdf"
        values: np.ndarray = merged_df[col_name_obs].values
        unique_values, counts = np.unique(np.sort(values), return_counts=True)
        cum_weights: np.ndarray = np.cumsum(counts) / len(values)

        ax.scatter(
            unique_values,
            cum_weights,
            color=colors[obs_type],
            label=f"Observed {type_labels[obs_type]}" if is_first_subplot else None,
            marker="o",
            s=20,
        )


def _setup_subplot(
    ax, prediction_time: Tuple[int, int], is_first_subplot: bool
) -> None:
    """Configure subplot appearance and labels."""
    hour, minutes = prediction_time
    ax.set_xlabel("CDF value (probability threshold)")
    ax.set_ylabel("Proportion of observations ≤ threshold")
    ax.set_title(f"EPUDD plot for {hour}:{minutes:02}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if is_first_subplot:
        ax.legend()


def plot_epudd(
    prediction_times: List[Tuple[int, int]],
    prob_dist_dict_all: Dict[str, Dict],
    model_name: str = "admissions",
    return_figure: bool = False,
    return_dataframe: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    media_file_path: Optional[Path] = None,
    file_name=None,
    plot_all_bounds: bool = False,
) -> Union[
    Figure, Dict[str, pd.DataFrame], Tuple[Figure, Dict[str, pd.DataFrame]], None
]:
    """
    Generates plots comparing model predictions with observed values for discrete distributions.

    For discrete distributions, each predicted value has an associated probability. The CDF
    is calculated by sorting the values and computing cumulative probabilities, normalized
    by the number of time points.

    Parameters
    ----------
    prediction_times : list of tuple
        List of (hour, minute) tuples representing times for which predictions were made.
    prob_dist_dict_all : dict
        Dictionary of probability distributions keyed by model_key. Each entry contains
        information about predicted distributions and observed values for different
        snapshot dates. The predicted distributions should be discrete probability mass
        functions, with each value having an associated probability.
    model_name : str, optional
        Base name of the model to construct model keys, by default "admissions".
    return_figure : bool, optional
        If True, returns the figure object instead of displaying it, by default False.
    return_dataframe : bool, optional
        If True, returns a dictionary of observation dataframes by model_key, by default False.
        The dataframes contain the merged observation and prediction data for analysis.
    figsize : tuple of (float, float), optional
        Size of the figure in inches as (width, height). If None, calculated automatically
        based on number of plots, by default None.
    suptitle : str, optional
        Super title for the entire figure, displayed above all subplots, by default None.
    media_file_path : Path, optional
        Path to save the plot, by default None. If provided, saves the plot as a PNG file.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "plot_epudd.png".
    plot_all_bounds : bool, optional
        If True, plots all bounds (lower, mid, upper). If False, only plots mid bounds.
        By default False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots, if return_figure is True.
    dict
        Dictionary of observation dataframes by model_key, if return_dataframe is True.
    tuple
        Tuple of (figure, dataframes_dict) if both return_figure and return_dataframe are True.
    None
        If neither return_figure nor return_dataframe is True, displays the plots and returns None.

    Notes
    -----
    For discrete distributions, the CDF is calculated by:

        1. Sorting the predicted values
        2. Computing cumulative probabilities for each value
        3. Normalizing by the number of time points

    The plot shows three possible positions for each observation:

        * lower_cdf (pink): Uses the lower bound of the CDF interval
        * mid_cdf (green): Uses the midpoint of the CDF interval
        * upper_cdf (light blue): Uses the upper bound of the CDF interval

    The black points represent the model's predicted CDF values, calculated from the sorted
    values and their associated probabilities, while the colored points show where the actual
    observations fall within their predicted intervals. For a well-calibrated model, the
    observed values should fall within their predicted intervals, with the distribution of
    positions showing appropriate uncertainty.

    """
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted: List[Tuple[int, int]] = sorted(
        prediction_times,
        key=lambda x: x[0] * 60 + x[1],
    )

    # Calculate figure parameters
    num_plots: int = len(prediction_times_sorted)
    figsize = figsize or (num_plots * 5, 4)

    # Create subplot layout
    fig: Figure
    axs: np.ndarray
    fig, axs = plt.subplots(1, num_plots, figsize=figsize)
    axs = [axs] if num_plots == 1 else axs

    # Define plotting types and colors
    all_types = ["lower", "mid", "upper"]
    plot_types = all_types if plot_all_bounds else ["mid"]
    colors: Dict[str, str] = {
        "lower": "#FF1493",  # deeppink
        "mid": "#228B22",  # chartreuse4/forest green
        "upper": "#ADD8E6",  # lightblue
    }

    all_obs_dfs: Dict[str, pd.DataFrame] = {}

    # Process each subplot
    for i, prediction_time in enumerate(prediction_times_sorted):
        model_key: str = get_model_key(model_name, prediction_time)
        prob_dist_dict: Dict = prob_dist_dict_all[model_key]

        if not prob_dist_dict:
            continue

        # Create distribution and observation dataframes
        all_distributions = _create_distribution_records(prob_dist_dict, all_types)
        distr_coll: pd.DataFrame = pd.DataFrame(all_distributions)

        all_observations = _create_observation_records(prob_dist_dict)
        adm_coll: pd.DataFrame = pd.DataFrame(all_observations)

        # For each actual observation, find its position in the predicted CDF
        # by matching datetime and admission count to get lower/mid/upper bounds
        merged_df: pd.DataFrame = pd.merge(
            adm_coll,
            distr_coll.rename(
                columns={
                    "num_adm_pred": "num_adm",
                    **{f"{t}_predicted_cdf": f"{t}_observed_cdf" for t in all_types},
                }
            ),
            on=["dt", "num_adm"],
            how="inner",
        )

        if merged_df.empty:
            continue

        all_obs_dfs[model_key] = merged_df
        ax = axs[i]
        num_time_points: int = len(prob_dist_dict)

        # Plot predictions and observations
        _plot_predictions(ax, distr_coll, num_time_points, plot_types)
        _plot_observations(ax, merged_df, plot_types, colors, i == 0)
        _setup_subplot(ax, prediction_time, i == 0)

    # Final plot configuration
    plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)
    if media_file_path:
        if file_name:
            filename = file_name
        else:
            filename = "plot_epudd.png"
        plt.savefig(media_file_path / filename, dpi=300)

    # Return based on flags
    if return_figure and return_dataframe:
        return fig, all_obs_dfs
    elif return_figure:
        return fig
    elif return_dataframe:
        plt.show()
        plt.close()
        return all_obs_dfs
    else:
        plt.show()
        plt.close()
        return None
