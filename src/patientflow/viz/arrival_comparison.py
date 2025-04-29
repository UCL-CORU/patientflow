from datetime import timedelta, datetime, time
from patientflow.calculate.arrival_rates import time_varying_arrival_rates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Create date range
def plot_arrival_comparison(df, prediction_time, snapshot_date, prediction_window, show_delta=True, show_only_delta=False):
    """
    Plot comparison between observed arrivals and expected arrival rates.
    
    Args:
        df (pd.DataFrame): DataFrame containing arrival data
        prediction_time (tuple): (hour, minute) of prediction time
        snapshot_date (datetime.date): Date to analyze
        prediction_window (int): Prediction window in minutes
        show_delta (bool): If True, plot the difference between actual and expected arrivals
        show_only_delta (bool): If True, only plot the delta between actual and expected arrivals
    """
    # Convert prediction time to datetime objects
    prediction_time_obj = time(hour=prediction_time[0], minute=prediction_time[1])
    snapshot_datetime = pd.Timestamp(datetime.combine(snapshot_date, prediction_time_obj), tz='UTC')

    df_copy = df.copy()

    if 'arrival_datetime' in df_copy.columns:
        df_copy.set_index('arrival_datetime', inplace=True)
    
    # Get arrivals within the prediction window
    arrivals = df_copy[
        (df_copy.index > snapshot_datetime) & 
        (df_copy.index <= snapshot_datetime + pd.Timedelta(minutes=prediction_window))
    ]
    
    # Sort arrivals by time
    arrivals = arrivals.sort_values('arrival_datetime')
    
    # Create cumulative count
    arrivals['cumulative_count'] = range(1, len(arrivals) + 1)
    
    # Calculate and plot expected arrivals based on arrival rates
    arrival_rates = time_varying_arrival_rates(df_copy, yta_time_interval=15)
    end_time = (datetime.combine(datetime.min, prediction_time_obj) + 
                timedelta(minutes=prediction_window)).time()
    
    mean_arrival_rates = {k: v for k, v in arrival_rates.items() 
                         if (k >= prediction_time_obj and k < end_time) or
                            (end_time < prediction_time_obj and 
                             (k >= prediction_time_obj or k < end_time))}
    
    # Convert arrival rate times to datetime objects
    arrival_times_piecewise = []
    for t in mean_arrival_rates.keys():
        if t < prediction_time_obj:
            dt = datetime.combine(snapshot_date + timedelta(days=1), t)
        else:
            dt = datetime.combine(snapshot_date, t)
        # Ensure timezone awareness
        if dt.tzinfo is None:
            dt = pd.Timestamp(dt, tz='UTC')
        arrival_times_piecewise.append(dt)
    
    arrival_times_piecewise.sort()
    
    # Calculate expected cumulative arrivals
    cumulative_rates = []
    current_sum = 0
    for t in arrival_times_piecewise:
        rate = mean_arrival_rates[t.time()]
        current_sum += rate
        cumulative_rates.append(current_sum)
    
    # Create figure with subplots if showing delta
    if show_delta and not show_only_delta:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax = ax1
    else:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    # Ensure arrivals index is timezone-aware
    if arrivals.index.tz is None:
        arrivals.index = arrivals.index.tz_localize('UTC')
    
    # Calculate the delta
    # Create a combined timeline of all points
    all_times = sorted(set([snapshot_datetime] + 
                         list(arrivals.index) + 
                         [snapshot_datetime + pd.Timedelta(minutes=prediction_window)] +
                         arrival_times_piecewise))
    
    # Interpolate both actual and expected to the combined timeline
    actual_counts = np.interp([t.timestamp() for t in all_times],
                            [t.timestamp() for t in [snapshot_datetime] + list(arrivals.index) + [snapshot_datetime + pd.Timedelta(minutes=prediction_window)]],
                            [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1] if len(arrivals) > 0 else 0])
    
    expected_counts = np.interp([t.timestamp() for t in all_times],
                              [t.timestamp() for t in arrival_times_piecewise],
                              cumulative_rates)
    
    # Calculate delta
    delta = actual_counts - expected_counts

    if not show_only_delta:
        # Plot actual and expected arrivals
        ax.step([snapshot_datetime] + list(arrivals.index) + [snapshot_datetime + pd.Timedelta(minutes=prediction_window)], 
             [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1] if len(arrivals) > 0 else 0], 
             where='post', label='Actual Arrivals')
        ax.step(arrival_times_piecewise, cumulative_rates, where='post', label='Expected Arrivals')    
        
        ax.set_xlabel('Arrival Time')
        ax.set_title(f'Cumulative Arrivals in the {int(prediction_window/60)} hours after {prediction_time} on {snapshot_date}')
        ax.legend()
    
    if show_delta or show_only_delta:
        if show_only_delta:
            ax.step(all_times, delta, where='post', label='Actual - Expected', color='red')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Arrival Time')
            ax.set_ylabel('Difference (Actual - Expected)')
            ax.set_title('Difference Between Actual and Expected Arrivals')
            ax.legend()
        else:
            ax2.step(all_times, delta, where='post', label='Actual - Expected', color='red')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Arrival Time')
            ax2.set_ylabel('Difference (Actual - Expected)')
            ax2.set_title('Difference Between Actual and Expected Arrivals')
            ax2.legend()
        
        plt.tight_layout()
    
    plt.show()
