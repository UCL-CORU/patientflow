"""
Generate plots comparing observed values with model predictions for discrete distributions.

These plots display the model's predicted CDF values alongside the actual observed values'
positions within their predicted CDF intervals. For discrete distributions, each predicted
value has an associated probability, and the CDF is calculated by sorting the values and
computing cumulative probabilities.

The plot shows three possible positions for each observation within its predicted interval:
- lower bound of the interval
- midpoint of the interval
- upper bound of the interval

For a well-calibrated model, the observed values should fall within their predicted
intervals, with the distribution of positions showing appropriate uncertainty.

Key Functions:
- adjusted_qq_plot: Generates and plots the comparison of model predictions with observed values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patientflow.load import get_model_key


def adjusted_qq_plot(
    prediction_times,
    prob_dist_dict_all,
    model_name="admissions",
    return_figure=False,
    return_dataframe=False,
    figsize=None,
    suptitle=None,
    media_file_path=None,
):
    """
    Generate plots comparing model predictions with observed values for discrete distributions.

    For discrete distributions, each predicted value has an associated probability. The CDF
    is calculated by sorting the values and computing cumulative probabilities, normalized
    by the number of time points. The plot shows three possible positions for each observation:
    - lower bound of the interval (pink points)
    - midpoint of the interval (green points)
    - upper bound of the interval (light blue points)

    The black points represent the model's predicted CDF values, calculated from the sorted
    values and their associated probabilities, while the colored points show where the actual
    observations fall within their predicted intervals.

    Parameters
    ----------
    prediction_times : list of tuple
        List of (hour, minute) tuples representing times for which predictions were made.
    prob_dist_dict_all : dict
        Dictionary of probability distributions keyed by model_key. Each entry contains
        information about predicted distributions and observed values for different
        horizon dates. The predicted distributions should be discrete probability mass
        functions, with each value having an associated probability.
    model_name : str, optional
        Base name of the model to construct model keys, by default "admissions".
    return_figure : bool, optional
        If True, returns the figure object instead of displaying it, by default False.
    return_dataframe : bool, optional
        If True, returns a dictionary of observation dataframes by model_key, by default False.
    figsize : tuple of (float, float), optional
        Size of the figure in inches as (width, height). If None, calculated automatically
        based on number of plots, by default None.
    suptitle : str, optional
        Super title for the entire figure, displayed above all subplots, by default None.
    media_file_path : Path, optional
        Path to save the plot, by default None.

    Returns
    -------
    matplotlib.figure.Figure or dict or tuple or None
        If return_figure is True, returns the figure object containing the plots.
        If return_dataframe is True, returns a dictionary of observation dataframes by model_key.
        If both are True, returns a tuple (figure, dataframes_dict).
        Otherwise displays the plots and returns None.

    Notes
    -----
    For discrete distributions, the CDF is calculated by:
    1. Sorting the predicted values
    2. Computing cumulative probabilities for each value
    3. Normalizing by the number of time points

    The plot shows three possible positions for each observation:
    - lower_cdf (pink): Uses the lower bound of the CDF interval
    - mid_cdf (green): Uses the midpoint of the CDF interval
    - upper_cdf (light blue): Uses the upper bound of the CDF interval

    The black points represent the model's predicted CDF values, calculated from the sorted
    values and their associated probabilities, while the colored points show where the actual
    observations fall within their predicted intervals. For a well-calibrated model, the
    observed values should fall within their predicted intervals, with the distribution of
    positions showing appropriate uncertainty.

    Examples
    --------
    >>> prediction_times = [(8, 0), (12, 0), (16, 0)]
    >>> adjusted_qq_plot(prediction_times, prob_dist_dict, model_name="bed_demand",
    ...           figsize=(15, 5), suptitle="Bed Demand Model Predictions vs Observations")
    """
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60
        + x[1],  # Convert (hour, minute) to minutes since midnight
    )

    num_plots = len(prediction_times_sorted)
    if figsize is None:
        figsize = (num_plots * 5, 4)

    # Create subplot layout
    fig, axs = plt.subplots(1, num_plots, figsize=figsize)

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    # Dictionary to store observation dataframes by model_key
    all_obs_dfs = {}

    # Loop through each subplot
    for i, prediction_time in enumerate(prediction_times_sorted):
        # Get model key and corresponding prob_dist_dict
        model_key = get_model_key(model_name, prediction_time)
        prob_dist_dict = prob_dist_dict_all[model_key]

        if not prob_dist_dict:
            continue

        # ----- COLLECT MODEL PREDICTIONS -----
        all_distributions = []
        for dt in prob_dist_dict:
            agg_predicted = np.array(prob_dist_dict[dt]["agg_predicted"]["agg_proba"])

            # Calculate CDF values
            upper_cdf = agg_predicted.cumsum()
            lower_cdf = np.hstack((0, upper_cdf[:-1]))
            mid_cdf = (upper_cdf + lower_cdf) / 2

            # Store all predicted distributions for each time point
            for j, prob in enumerate(agg_predicted):
                all_distributions.append(
                    {
                        "num_adm_pred": j,
                        "prob": prob,
                        "sample_time": dt,  # Using the same name as in R code
                        "upper_M_discrete_value": upper_cdf[j],
                        "lower_M_discrete_value": lower_cdf[j],
                        "mid_M_discrete_value": mid_cdf[j],
                    }
                )

        # Create DataFrame with all distributions
        distr_coll = pd.DataFrame(all_distributions)

        # ----- COLLECT OBSERVATIONS -----
        all_observations = []
        time_pts = []
        num_time_points = len(prob_dist_dict.keys())
        for dt in prob_dist_dict:
            agg_observed = prob_dist_dict[dt]["agg_observed"]
            time_pts.append(dt)

            # Store observation data
            all_observations.append(
                {"date": dt, "num_adm": agg_observed, "sample_time": dt}
            )

        # Create DataFrame with all observations
        adm_coll = pd.DataFrame(all_observations)

        # ----- MERGE OBSERVATIONS WITH PREDICTIONS (equivalent to R merge) -----
        # This is the equivalent of the R code line:
        # adm_coll = merge(adm_coll, distr_coll[, .(sample_time, num_adm = num_adm_pred,
        #                 lower_E = lower_M_discrete_value, upper_E = upper_M_discrete_value)],
        #                 by = c("sample_time", "num_adm"))

        merged_df = pd.merge(
            adm_coll,
            distr_coll.rename(
                columns={
                    "num_adm_pred": "num_adm",
                    "lower_M_discrete_value": "lower_E",
                    "mid_M_discrete_value": "mid_E",
                    "upper_M_discrete_value": "upper_E",
                }
            ),
            on=["sample_time", "num_adm"],
            how="inner",
        )

        if merged_df.empty:
            continue

        # Store the observation dataframe
        all_obs_dfs[model_key] = merged_df

        # Set up the plot
        ax = axs[i]

        # For lower, mid, and upper model predictions
        for pred_type in ["lower", "mid", "upper"]:
            # Get the column name from distr_coll
            col_name = f"{pred_type}_M_discrete_value"

            # Extract values and probabilities as a temporary dataframe
            df_temp = distr_coll[[col_name, "prob"]].copy()

            # Sort by values - this is the key correction
            df_temp = df_temp.sort_values(by=col_name)

            # Calculate cumulative weights after sorting
            df_temp["cum_weight"] = df_temp["prob"].cumsum()
            df_temp["cum_weight_normed"] = df_temp["prob"].cumsum() / num_time_points

            # Define colors for each prediction type (similar to the R code)
            colors = {"lower": "deeppink", "mid": "chartreuse4", "upper": "lightblue"}

            # Plot model predictions with appropriate colors
            ax.scatter(
                df_temp[col_name],  # Use the sorted values
                df_temp[
                    "cum_weight_normed"
                ],  # Use the correctly calculated normed weights
                color="grey",
                label=f"Model {pred_type}",
                marker="o",
                s=5,
            )

        # ----- PLOT ACTUAL OBSERVATIONS (COLORED POINTS) -----

        # Define colors for the observations
        colors = {
            "lower": "#FF1493",  # deeppink
            "mid": "#228B22",  # chartreuse4/forest green
            "upper": "#ADD8E6",  # lightblue
        }

        # For lower, mid, and upper actual observations
        for obs_type in ["lower", "mid", "upper"]:
            # Get the column name from merged_df
            col_name = f"{obs_type}_E"

            # Extract values
            values = merged_df[col_name].values

            # Calculate distribution similar to R approach
            sorted_values = np.sort(values)
            n = len(sorted_values)

            # Calculate empirical CDF
            unique_values, counts = np.unique(sorted_values, return_counts=True)
            cum_weights = np.cumsum(counts) / n

            # Plot actual observations as colored points
            ax.scatter(
                unique_values,
                cum_weights,
                color=colors[obs_type],
                label=f"Actual {obs_type}"
                if i == 0
                else None,  # Only add label in first subplot
                marker="o",
                s=20,
            )

        # Set labels and title
        hour, minutes = prediction_time
        ax.set_xlabel("CDF value (probability threshold)")
        ax.set_ylabel("Proportion of observations ≤ threshold")
        ax.set_title(f"Adjusted QQ plot for {hour}:{minutes:02}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Add legend to first plot only
        if i == 0:
            ax.legend()

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)

    if media_file_path:
        plt.savefig(media_file_path / "adjusted_qq_plot.png", dpi=300)

    # Determine what to return
    if return_figure:
        if return_dataframe:
            return fig, all_obs_dfs
        return fig
    elif return_dataframe:
        plt.show()
        plt.close()
        return all_obs_dfs
    else:
        plt.show()
        plt.close()
