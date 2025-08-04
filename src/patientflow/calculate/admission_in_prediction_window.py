"""
This module provides functions to model and analyze a curve consisting of an exponential growth segment followed by an exponential decay segment. It includes functions to create the curve, calculate specific points on it, and evaluate probabilities based on its shape.

Its intended use is to derive the probability of a patient being admitted to a hospital within a certain elapsed time after their arrival in the Emergency Department (ED), given the hospital's aspirations for the time it takes patients to be admitted.
For this purpose, two points on the curve are required as parameters:

    * (x1,y1) : The target proportion of patients y1 (eg 76%) who have been admitted or discharged by time x1 (eg 4 hours).
    * (x2, y2) : The time x2 by which all but a small proportion y2 of patients have been admitted.

 It is assumed that values of y where x < x1 is a growth curve grow exponentially towards x1 and that (x1,y1) the curve switches to a decay curve.

Functions
---------
growth_curve : function
    Calculate exponential growth at a point where x < x1.
decay_curve : function
    Calculate exponential decay at a point where x >= x1.
create_curve : function
    Generate a full curve with both growth and decay segments.
get_y_from_aspirational_curve : function
    Read from the curve a value for y, the probability of being admitted, for a given moment x hours after arrival
calculate_probability : function
    Compute the probability of a patient being admitted by the end of a prediction window, given how much time has elapsed since their arrival.
get_survival_probability : function
    Calculate the probability of a patient still being in the ED after a certain time using survival curve data.

"""

import numpy as np
from datetime import timedelta
import warnings


def growth_curve(x, a, gamma):
    """
    Calculate the exponential growth value at a given x using specified parameters.
    The function supports both scalar and array inputs for x.

    Parameters
    ----------
    x : float or np.ndarray
        The x-value(s) at which to evaluate the curve.
    a : float
        The coefficient that defines the starting point of the growth curve when x is 0.
    gamma : float
        The growth rate coefficient of the curve.

    Returns
    -------
    float or np.ndarray
        The y-value(s) of the growth curve at x.

    """
    return a * np.exp(x * gamma)


def decay_curve(x, x1, y1, lamda):
    """
    Calculate the exponential decay value at a given x using specified parameters.
    The function supports both scalar and array inputs for x.

    Parameters
    ----------
    x : float or np.ndarray
        The x-value(s) at which to evaluate the curve.
    x1 : float
        The x-value where the growth curve transitions to the decay curve.
    y1 : float
        The y-value at the transition point, where the decay curve starts.
    lamda : float
        The decay rate coefficient.

    Returns
    -------
    float or np.ndarray
        The y-value(s) of the decay curve at x.

    """
    return y1 + (1 - y1) * (1 - np.exp(-lamda * (x - x1)))


def create_curve(x1, y1, x2, y2, a=0.01, generate_values=False):
    """
    Generates parameters for an exponential growth and decay curve.
    Optionally generates x-values and corresponding y-values across a default or specified range.

    Parameters
    ----------
    x1 : float
        The x-value where the curve transitions from growth to decay.
    y1 : float
        The y-value at the transition point x1.
    x2 : float
        The x-value defining the end of the decay curve for calculation purposes.
    y2 : float
        The y-value at x2, intended to fine-tune the decay rate.
    a : float, optional
        The initial value coefficient for the growth curve, defaults to 0.01.
    generate_values : bool, optional
        Flag to determine whether to generate x-values and y-values for visualization purposes.

    Returns
    -------
    tuple
        If generate_values is False, returns (gamma, lamda, a).
        If generate_values is True, returns (gamma, lamda, a, x_values, y_values).

    """
    # Validate inputs
    if not (x1 < x2):
        raise ValueError("x1 must be less than x2")
    if not (0 < y1 < y2 < 1):
        raise ValueError("y1 must be less than y2, and both must be between 0 and 1")

    # Constants for growth and decay
    gamma = np.log(y1 / a) / x1
    lamda = np.log((1 - y1) / (1 - y2)) / (x2 - x1)

    if generate_values:
        x_values = np.linspace(0, 20, 200)
        y_values = [
            (growth_curve(x, a, gamma) if x <= x1 else decay_curve(x, x1, y1, lamda))
            for x in x_values
        ]
        return gamma, lamda, a, x_values, y_values

    return gamma, lamda, a


def get_y_from_aspirational_curve(x, x1, y1, x2, y2):
    """
    Calculate the probability y that a patient will have been admitted by a specified x after their arrival, by reading from the aspirational curve that has been constrained to pass through points (x1, y1) and (x2, y2) with an exponential growth curve where x < x1 and an exponential decay where x < x2

    The function handles scalar or array inputs for x and determines y using either an exponential growth curve (for x < x1)
    or an exponential decay curve (for x >= x1). The curve parameters are derived to ensure the curve passes through
    specified points (x1, y1) and (x2, y2).

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate(s) at which to calculate the y-value on the curve. Can be a single value or an array of values.
    x1 : float
        The x-coordinate of the first key point on the curve, where the growth phase ends and the decay phase begins.
    y1 : float
        The y-coordinate of the first key point (x1), representing the target proportion of patients admitted by time x1.
    x2 : float
        The x-coordinate of the second key point on the curve, beyond which all but a few patients are expected to be admitted.
    y2 : float
        The y-coordinate of the second key point (x2), representing the target proportion of patients admitted by time x2.

    Returns
    -------
    float or np.ndarray
        The calculated y-value(s) (probability of admission) at the given x. The type of the return matches the input type
        for x (either scalar or array).

    """
    gamma, lamda, a = create_curve(x1, y1, x2, y2)
    y = np.where(x < x1, growth_curve(x, a, gamma), decay_curve(x, x1, y1, lamda))
    return y


def calculate_probability(
    elapsed_los: timedelta,
    prediction_window: timedelta,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Calculates the probability of an admission occurring within a specified prediction window after the moment of prediction,
    based on the patient's elapsed time in the ED prior to the moment of prediction and the length of the window

    Parameters
    ----------
    elapsed_los : timedelta
        The elapsed time since the patient arrived at the ED.
    prediction_window : timedelta
        The duration of the prediction window after the point of prediction, for which the probability is calculated.
    x1 : float
        The time target for the first key point on the curve.
    y1 : float
        The proportion target for the first key point (e.g., 76% of patients admitted by time x1).
    x2 : float
        The time target for the second key point on the curve.
    y2 : float
        The proportion target for the second key point (e.g., 99% of patients admitted by time x2).

    Returns
    -------
    float
        The probability of the event occurring within the given prediction window.

    Edge Case Handling
    ------------------
    When elapsed_los is extremely high, such as values significantly greater than x2, the admission probability prior to the current time (`prob_admission_prior_to_now`) can reach 1.0 despite the curve being asymptotic. This scenario can cause computational errors when calculating the conditional probability, as it involves a division by zero. In such cases, this function directly returns a probability of 1.0, reflecting certainty of admission.

    Example
    -------
    Calculate the probability that a patient, who has already been in the ED for 3 hours, will be admitted in the next 2 hours. The ED targets that 76% of patients are admitted or discharged within 4 hours, and 99% within 12 hours.

    >>> from datetime import timedelta
    >>> calculate_probability(timedelta(hours=3), timedelta(hours=2), 4, 0.76, 12, 0.99)

    """
    # Validate inputs
    if not isinstance(elapsed_los, timedelta):
        raise TypeError("elapsed_los must be a timedelta object")
    if not isinstance(prediction_window, timedelta):
        raise TypeError("prediction_window must be a timedelta object")

    # Convert timedelta to hours
    elapsed_hours = elapsed_los.total_seconds() / 3600
    prediction_window_hours = prediction_window.total_seconds() / 3600

    # Validate elapsed time to ensure it represents a reasonable time value in hours
    if elapsed_hours < 0:
        raise ValueError(
            "elapsed_los must be non-negative (cannot have negative elapsed time)"
        )

    if elapsed_hours > 168:  # 168 hours = 1 week
        warnings.warn(
            "elapsed_los appears to be longer than 168 hours (1 week). "
            "Check that the units of elapsed_los are correct"
        )

    if not np.isfinite(elapsed_hours):
        raise ValueError("elapsed_los must be a finite time duration")

    # Validate prediction window to ensure it represents a reasonable time value in hours
    if prediction_window_hours < 0:
        raise ValueError(
            "prediction_window must be non-negative (cannot have negative prediction window)"
        )

    if prediction_window_hours > 72:  # 72 hours = 3 days
        warnings.warn(
            "prediction_window appears to be longer than 72 hours (3 days). "
            "Check that the units of prediction_window are correct"
        )

    if not np.isfinite(prediction_window_hours):
        raise ValueError("prediction_window must be a finite time duration")

    # probability of still being in the ED now (a function of elapsed time since arrival)
    prob_admission_prior_to_now = get_y_from_aspirational_curve(
        elapsed_hours, x1, y1, x2, y2
    )

    # prob admission when adding the prediction window added to elapsed time since arrival
    prob_admission_by_end_of_window = get_y_from_aspirational_curve(
        elapsed_hours + prediction_window_hours, x1, y1, x2, y2
    )

    # Direct return for edge cases where `prob_admission_prior_to_now` reaches 1.0
    if prob_admission_prior_to_now == 1:
        return 1.0

    # Calculate the conditional probability of admission within the prediction window
    # given that the patient hasn't been admitted yet
    conditional_prob = (
        prob_admission_by_end_of_window - prob_admission_prior_to_now
    ) / (1 - prob_admission_prior_to_now)

    return conditional_prob


def get_survival_probability(survival_df, time_hours):
    """
    Calculate the probability of a patient still being in the ED after a specified time
    using survival curve data.

    Parameters
    ----------
    survival_df : pandas.DataFrame
        DataFrame containing survival curve data with columns:
        - time_hours: Time points in hours
        - survival_probability: Probability of still being in ED at each time point
    time_hours : float
        The time point (in hours) at which to calculate the survival probability

    Returns
    -------
    float
        The probability of still being in the ED at the specified time

    Notes
    -----
    - If the exact time_hours is not in the survival curve data, the function will
      interpolate between the nearest time points
    - If time_hours is less than the minimum time in the data, returns 1.0
    - If time_hours is greater than the maximum time in the data, uses exponential
      decay extrapolation based on the last few data points

    Examples
    --------
    >>> survival_df = pd.DataFrame({
    ...     'time_hours': [0, 2, 4, 6],
    ...     'survival_probability': [1.0, 0.8, 0.5, 0.2]
    ... })
    >>> get_survival_probability(survival_df, 3.5)
    0.65  # interpolated between 0.8 and 0.5
    """
    if time_hours < survival_df["time_hours"].min():
        return 1.0

    if time_hours > survival_df["time_hours"].max():
        # Use exponential decay extrapolation for times beyond the data
        max_time = survival_df["time_hours"].max()
        max_prob = survival_df["survival_probability"].iloc[-1]

        # Track whether we attempted exponential fitting and whether it failed
        attempted_exponential_fitting = False
        exponential_fitting_failed = False

        # Use the last 3 points to fit exponential decay if available
        if len(survival_df) >= 3:
            attempted_exponential_fitting = True

            # Get the last 3 points for fitting
            last_times = survival_df["time_hours"].iloc[-3:].values
            last_probs = survival_df["survival_probability"].iloc[-3:].values

            # Fit exponential decay: P(t) = P0 * exp(-lambda * (t - t0))
            # where P0 is the probability at t0, and lambda is the decay rate
            t0 = last_times[0]
            P0 = last_probs[0]

            # Calculate decay rate lambda using the last two points
            t1, t2 = last_times[-2], last_times[-1]
            P1, P2 = last_probs[-2], last_probs[-1]

            # For exponential fitting, replace P2 = 0 with a small positive value
            # to ensure the decay can be calculated while maintaining asymptotic behavior
            if P2 == 0:
                P2_fit = 1e-6  # Small positive value for fitting
            else:
                P2_fit = P2

            # lambda = -ln(P2/P1) / (t2 - t1)
            if P2_fit > 0 and P1 > 0 and t2 > t1:
                lambda_decay = -np.log(P2_fit / P1) / (t2 - t1)

                # Extrapolate using exponential decay
                extrapolated_prob = P0 * np.exp(-lambda_decay * (time_hours - t0))

                # Ensure the probability doesn't go below 0
                return max(0.0, extrapolated_prob)
            else:
                exponential_fitting_failed = True

        # Fallback: use the last known value if exponential fitting fails
        # Only warn if we actually attempted exponential fitting but it failed
        if attempted_exponential_fitting and exponential_fitting_failed:
            warnings.warn(
                f"Time {time_hours} hours exceeds maximum time in survival data "
                f"({max_time} hours). Exponential fitting failed (invalid data), using last known survival probability "
                f"({max_prob:.3f}) as fallback."
            )
        return max_prob

    # Find the closest time points for interpolation
    lower_idx = survival_df["time_hours"].searchsorted(time_hours, side="right") - 1
    upper_idx = lower_idx + 1

    if lower_idx < 0:
        return 1.0

    if upper_idx >= len(survival_df):
        return survival_df["survival_probability"].iloc[-1]

    # Get the surrounding points
    t1 = survival_df["time_hours"].iloc[lower_idx]
    t2 = survival_df["time_hours"].iloc[upper_idx]
    p1 = survival_df["survival_probability"].iloc[lower_idx]
    p2 = survival_df["survival_probability"].iloc[upper_idx]

    # Linear interpolation
    return p1 + (p2 - p1) * (time_hours - t1) / (t2 - t1)


def calculate_admission_probability_from_survival_curve(
    elapsed_los: timedelta,
    prediction_window: timedelta,
    survival_df,
):
    """
    Calculates the probability of an admission occurring within a specified prediction window after the moment of prediction,
    based on the patient's elapsed time in the ED prior to the moment of prediction and the length of the window.
    Uses empirical survival curve data instead of an aspirational curve.

    Parameters
    ----------
    elapsed_los : timedelta
        The elapsed time since the patient arrived at the ED.
    prediction_window : timedelta
        The duration of the prediction window after the point of prediction, for which the probability is calculated.
    survival_df : pandas.DataFrame
        DataFrame containing survival curve data with columns:
        - time_hours: Time points in hours
        - survival_probability: Probability of still being in ED at each time point

    Returns
    -------
    float
        The probability of admission occurring within the given prediction window.

    Notes
    -----
    This function calculates the conditional probability of admission within the prediction window,
    given that the patient hasn't been admitted yet. It uses the formula:

    P(admission in window | not admitted yet) =
        (P(not admitted at elapsed) - P(not admitted at elapsed + window)) / P(not admitted at elapsed)

    Since survival probability = P(not admitted), this becomes:

    P(admission in window | not admitted yet) =
        (survival_prob(elapsed) - survival_prob(elapsed + window)) / survival_prob(elapsed)

    Edge Case Handling
    ------------------
    When elapsed_los is extremely high, the survival probability can reach 0, which would cause
    division by zero. In such cases, this function returns a probability of 1.0, reflecting
    certainty of admission.

    When elapsed_hours + prediction_window_hours exceeds the maximum time in the survival data,
    the get_survival_probability function will return the last known survival probability and
    issue a warning. This may affect the accuracy of the admission probability calculation
    if the survival curve hasn't reached its asymptotic value.

    Example
    -------
    Calculate the probability that a patient, who has already been in the ED for 3 hours,
    will be admitted in the next 2 hours using empirical survival data.

    >>> from datetime import timedelta
    >>> survival_df = pd.DataFrame({
    ...     'time_hours': [0, 2, 4, 6, 8],
    ...     'survival_probability': [1.0, 0.8, 0.5, 0.2, 0.1]
    ... })
    >>> calculate_admission_probability_from_survival_curve(timedelta(hours=3), timedelta(hours=2), survival_df)

    """

    # Convert timedelta to hours
    elapsed_hours = elapsed_los.total_seconds() / 3600
    prediction_window_hours = prediction_window.total_seconds() / 3600

    # Validate elapsed time to ensure it represents a reasonable time value in hours
    if elapsed_hours < 0:
        raise ValueError(
            "elapsed_los must be non-negative (cannot have negative elapsed time)"
        )

    if elapsed_hours > 168:  # 168 hours = 1 week
        warnings.warn(
            "elapsed_los appears to be longer than 168 hours (1 week). "
            "Check that the units of elapsed_los are correct"
        )

    if not np.isfinite(elapsed_hours):
        raise ValueError("elapsed_los must be a finite time duration")

    # Validate prediction window to ensure it represents a reasonable time value in hours
    if prediction_window_hours < 0:
        raise ValueError(
            "prediction_window must be non-negative (cannot have negative prediction window)"
        )

    if prediction_window_hours > 72:  # 72 hours = 3 days
        warnings.warn(
            "prediction_window appears to be longer than 72 hours (3 days). "
            "Check that the units of prediction_window are correct"
        )

    if not np.isfinite(prediction_window_hours):
        raise ValueError("prediction_window must be a finite time duration")

    # Get survival probabilities using the existing function
    survival_prob_at_elapsed = get_survival_probability(survival_df, elapsed_hours)
    survival_prob_at_end_of_window = get_survival_probability(
        survival_df, elapsed_hours + prediction_window_hours
    )

    # Direct return for edge cases where survival probability is 0 (patient already admitted)
    if survival_prob_at_elapsed == 0:
        return 1.0

    # Calculate the conditional probability of admission within the prediction window
    # given that the patient hasn't been admitted yet
    # P(admission in window | not admitted yet) =
    # (P(not admitted at elapsed) - P(not admitted at elapsed + window)) / P(not admitted at elapsed)
    conditional_prob = (
        survival_prob_at_elapsed - survival_prob_at_end_of_window
    ) / survival_prob_at_elapsed

    return conditional_prob
