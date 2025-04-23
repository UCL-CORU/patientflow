"""
Generate CORU plots comparing observed values with model predictions.

These plots display the proportion of observed values that fall below each probability 
threshold (CDF value) from the model's predictions. For a well-calibrated model, this
proportion should match the probability threshold itself, resulting in points lying along
the diagonal line y=x.

Key Functions:
- coru_plot: Generates and plots the CORU plot based on the provided observed and predicted data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patientflow.load import get_model_key


def coru_plot(
    prediction_times,
    prob_dist_dict_all,
    model_name="admissions",
    return_figure=False,
    return_dataframe=False,
    figsize=None,
    suptitle=None,
):
    """
    Generate Calibration of Ranks Using probability (CORU) plots comparing observed values 
    with model predictions.
    
    CORU plots display the proportion of observed values that fall below each probability 
    threshold (CDF value) from the model's predictions. For a well-calibrated model, this
    proportion should match the probability threshold itself, resulting in points lying along
    the diagonal line y=x.
    
    Parameters
    ----------
    prediction_times : list of tuple
        List of (hour, minute) tuples representing times for which predictions were made.
    prob_dist_dict_all : dict
        Dictionary of probability distributions keyed by model_key. Each entry contains
        information about predicted distributions and observed values for different 
        horizon dates.
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
    
    Returns
    -------
    matplotlib.figure.Figure or dict or tuple or None
        If return_figure is True, returns the figure object containing the CORU plots.
        If return_dataframe is True, returns a dictionary of observation dataframes by model_key.
        If both are True, returns a tuple (figure, dataframes_dict).
        Otherwise displays the plots and returns None.
    
    Notes
    -----
    The CORU plot shows three curves for each prediction:
    - lower_cdf (pink): Uses the lower bound of the CDF interval
    - mid_cdf (green): Uses the midpoint of the CDF interval
    - upper_cdf (light blue): Uses the upper bound of the CDF interval
    
    For a well-calibrated model, all three curves should closely follow the diagonal line.
    Systematic deviations indicate over- or under-confidence in the model's predictions.
    
    Examples
    --------
    >>> prediction_times = [(8, 0), (12, 0), (16, 0)]
    >>> coru_plot(prediction_times, prob_dist_dict, model_name="bed_demand", 
    ...           figsize=(15, 5), suptitle="Bed Demand Model Calibration")
    """
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60 + x[1],  # Convert (hour, minute) to minutes since midnight
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

        # Initialize lists to store data for observed values and their associated CDF positions
        observations = []
        
        # Process data for current subplot
        for dt in prob_dist_dict:
            agg_predicted = np.array(prob_dist_dict[dt]["agg_predicted"])
            agg_observed = prob_dist_dict[dt]["agg_observed"]

            # if agg_observed == 0:
            #     print(f'{dt}: {agg_predicted}')
            
            # Calculate CDF values
            upper_cdf = agg_predicted.cumsum()
            lower_cdf = np.hstack((0, upper_cdf[:-1]))
            mid_cdf = (upper_cdf + lower_cdf) / 2
            
            # # Round observed value to nearest integer for indexing
            # agg_observed_int = int(round(agg_observed))
            
            # Record the CDF values at the observed point
            try:
                observations.append({
                    'date': dt,
                    'lower_cdf': lower_cdf[agg_observed],
                    'mid_cdf': mid_cdf[agg_observed],
                    'upper_cdf': upper_cdf[agg_observed],
                    'observed_value': agg_observed
                })
            except IndexError:
                # Handle case where observed value is out of range of predicted
                print(f"Warning: Observed value {agg_observed} out of range for date {dt}")
                continue
        
        if not observations:
            continue
        
        # Convert to DataFrame
        obs_df = pd.DataFrame(observations)
        
        # Store the observation dataframe
        all_obs_dfs[model_key] = obs_df
        
        # Create plot data for lower, mid, and upper CDF values
        plot_data = []
        
        for cdf_type in ['lower_cdf', 'mid_cdf', 'upper_cdf']:
            # Sort values
            sorted_values = sorted(obs_df[cdf_type])
            
            # Calculate empirical CDF
            n = len(sorted_values)
            cdf_data = []
            
            # Group by unique CDF values and count occurrences
            unique_values = pd.Series(sorted_values).value_counts().sort_index()
            cumulative_prop = 0
            
            for value, count in unique_values.items():
                proportion = count / n
                cumulative_prop += proportion
                cdf_data.append({
                    'value': value,
                    'cum_weight_normed': cumulative_prop,
                    'dist': cdf_type
                })
            
            plot_data.extend(cdf_data)
        
        # Convert to DataFrame for plotting
        plot_df = pd.DataFrame(plot_data)
        
        # Plot on current subplot
        ax = axs[i]
        
        # Reference line y=x
        ax.plot([0, 1], [0, 1], linestyle='--', color='black')
        
        # Plot points for different CDF types with smaller dots
        colors = {
            'lower_cdf': '#FF1493',  # approximate deeppink
            'mid_cdf': '#228B22',    # approximate chartreuse4/forest green
            'upper_cdf': '#ADD8E6'   # lightblue
        }
        
        for cdf_type in colors:
            df_subset = plot_df[plot_df['dist'] == cdf_type]
            if not df_subset.empty:
                ax.scatter(
                    df_subset['value'].values,
                    df_subset['cum_weight_normed'].values,
                    color=colors[cdf_type],
                    label=cdf_type,
                    marker='o',
                    s=20  # Set size of dots (default is 36)
                )
        
        # Set labels and title
        hour, minutes = prediction_time
        ax.set_xlabel("CDF value (probability threshold)")
        ax.set_ylabel("Proportion of observations ≤ threshold")
        ax.set_title(f"CORU Plot for {hour}:{minutes:02}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add legend to first plot only
        if i == 0:
            ax.legend()

    plt.tight_layout()
    
    # Add suptitle if provided
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)
    
    # Determine what to return
    if return_figure and return_dataframe:
        return fig, all_obs_dfs
    elif return_figure:
        return fig
    elif return_dataframe:
        return all_obs_dfs
    else:
        plt.show()