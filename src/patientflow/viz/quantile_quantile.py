"""
Generate Quantile-Quantile (QQ) plots to compare observed values with model predictions.

This module creates QQ plots for healthcare bed demand predictions, comparing observed
values with model predictions. A QQ plot is a graphical technique for determining if two
data sets come from populations with a common distribution. If the points form a line
approximately along the reference line y=x, this suggests the distributions are similar.

Functions
---------
qq_plot : function
    Generate multiple QQ plots comparing observed values with model predictions

Notes
-----
To prepare the predicted distribution:
* Treat the predicted distributions (saved as cdfs) for all time points of interest as if they were one distribution
* Within this predicted distribution, because each probability is over a discrete rather than continuous number of input values, the upper and lower of values of the probability range are saved at each value
* The mid point between upper and lower is calculated and saved
* The distribution of cdf mid points (one for each horizon date) is sorted by value of the mid point and a cdf of this is calculated (this is a cdf of cdfs, in effect)
* These are weighted by the probability of each value occurring

To prepare the observed distribution:
* Take observed number each horizon date and save the cdf of that value from its predicted distribution
* The distribution of cdf values (one per horizon date) is sorted
* These are weighted by the probability of each value occurring, which is a uniform probability (1 / over the number of horizon dates)
"""

# Import necessary libraries for data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from patientflow.load import get_model_key


def qq_plot(
    prediction_times,
    prob_dist_dict_all,
    model_name="admissions",
    return_figure=False,
    figsize=None,
    suptitle=None,
    media_file_path=None,
    file_name=None,
):
    """Generate multiple QQ plots comparing observed values with model predictions.

    Parameters
    ----------
    prediction_times : list of tuple
        List of (hour, minute) tuples for prediction times.
    prob_dist_dict_all : dict
        Dictionary of probability distributions keyed by model_key.
    model_name : str, default="admissions"
        Base name of the model to construct model keys.
    return_figure : bool, default=False
        If True, returns the figure object instead of displaying it.
    figsize : tuple of float, optional
        Size of the figure in inches as (width, height). If None, calculated automatically
        based on number of plots.
    suptitle : str, optional
        Super title for the entire figure, displayed above all subplots.
    media_file_path : Path, optional
        Path to save the plot.
    file_name : str, optional
        Custom filename to use when saving the plot. If not provided, defaults to "qq_plot.png".

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure if return_figure is True, otherwise displays the plot and returns None.

    Notes
    -----
    The function creates a QQ plot for each prediction time, comparing the observed
    distribution with the predicted distribution. Each subplot shows how well the
    model's predictions match the actual observations.
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

    # Loop through each subplot
    for i, prediction_time in enumerate(prediction_times_sorted):
        # Initialize lists to store CDF and observed data
        cdf_data = []
        observed_data = []

        # Get model key and corresponding prob_dist_dict
        model_key = get_model_key(model_name, prediction_time)
        prob_dist_dict = prob_dist_dict_all[model_key]

        # Process data for current subplot
        for dt in prob_dist_dict:
            agg_predicted = np.array(prob_dist_dict[dt]["agg_predicted"])
            agg_observed = prob_dist_dict[dt]["agg_observed"]

            upper = agg_predicted.cumsum()
            lower = np.hstack((0, upper[:-1]))
            mid = (upper + lower) / 2

            cdf_data.append(np.column_stack((upper, lower, mid, agg_predicted)))
            # Round the observed data to nearest integer before using as index
            agg_observed_int = int(round(agg_observed))
            observed_data.append(mid[agg_observed_int])

        if not cdf_data:
            continue

        # Prepare data for plotting
        cdf_data = np.vstack(cdf_data)
        qq_model = pd.DataFrame(
            cdf_data, columns=["cdf_upper", "cdf_mid", "cdf_lower", "weights"]
        )
        qq_model = qq_model.sort_values("cdf_mid")
        qq_model["cum_weight"] = qq_model["weights"].cumsum()
        qq_model["cum_weight_normed"] = (
            qq_model["cum_weight"] / qq_model["weights"].sum()
        )

        qq_observed = pd.DataFrame(observed_data, columns=["cdf_observed"])
        qq_observed = qq_observed.sort_values("cdf_observed")
        qq_observed["weights"] = 1 / len(observed_data)
        qq_observed["cum_weight_normed"] = qq_observed["weights"].cumsum()

        qq_observed["max_model_cdf_at_this_value"] = qq_observed["cdf_observed"].apply(
            lambda x: qq_model[qq_model["cdf_mid"] <= x]["cum_weight_normed"].max()
        )

        # Plot on current subplot
        ax = axs[i]
        ax.set_aspect("equal")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Reference line y=x
        ax.plot([0, 1], [0, 1], linestyle="--")

        # Plot QQ data points
        ax.plot(
            qq_observed["max_model_cdf_at_this_value"],
            qq_observed["cum_weight_normed"],
            marker=".",
            linewidth=0,
        )

        # Set labels and title for subplot with hour:minute format
        hour, minutes = prediction_time
        ax.set_xlabel("Cdf of model distribution")
        ax.set_ylabel("Cdf of observed distribution")
        ax.set_title(f"QQ Plot for {hour}:{minutes:02}")

    plt.tight_layout()

    # Add suptitle if provided
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, y=1.05)

    if media_file_path:
        plt.savefig(media_file_path / (file_name or "qq_plot.png"), dpi=300)

    if return_figure:
        return fig
    else:
        plt.show()
        plt.close(fig)
