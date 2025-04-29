from datetime import timedelta, datetime, time
from patientflow.calculate.arrival_rates import time_varying_arrival_rates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from patientflow.viz.utils import format_prediction_time


# Create date range
def plot_arrival_comparison(df, prediction_time, snapshot_date, prediction_window, show_delta=True, show_only_delta=False, yta_time_interval=15):
    """
    Plot comparison between observed arrivals and expected arrival rates.
    
    Args:
        df (pd.DataFrame): DataFrame containing arrival data
        prediction_time (tuple): (hour, minute) of prediction time
        snapshot_date (datetime.date): Date to analyze
        prediction_window (int): Prediction window in minutes
        show_delta (bool): If True, plot the difference between actual and expected arrivals
        show_only_delta (bool): If True, only plot the delta between actual and expected arrivals
        yta_time_interval (int): Time interval in minutes for calculating arrival rates
    """
    # Convert prediction time to datetime objects
    prediction_time_obj = time(hour=prediction_time[0], minute=prediction_time[1])
    snapshot_datetime = pd.Timestamp(datetime.combine(snapshot_date, prediction_time_obj), tz='UTC')
    
    # Use a consistent default date for plotting (January 1, 2024)
    default_date = datetime(2024, 1, 1)
    default_datetime = pd.Timestamp(datetime.combine(default_date, prediction_time_obj), tz='UTC')

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
    arrival_rates = time_varying_arrival_rates(df_copy, yta_time_interval=yta_time_interval)
    end_time = (datetime.combine(datetime.min, prediction_time_obj) + 
                timedelta(minutes=prediction_window)).time()
    
    mean_arrival_rates = {k: v for k, v in arrival_rates.items() 
                         if (k >= prediction_time_obj and k < end_time) or
                            (end_time < prediction_time_obj and 
                             (k >= prediction_time_obj or k < end_time))}
    
    # Convert arrival rate times to datetime objects using default date
    arrival_times_piecewise = []
    for t in mean_arrival_rates.keys():
        if t < prediction_time_obj:
            dt = datetime.combine(default_date + timedelta(days=1), t)
        else:
            dt = datetime.combine(default_date, t)
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
    
    # Convert arrival times to use default date for plotting
    arrival_times_plot = [default_datetime + (t - snapshot_datetime) for t in arrivals.index]
    
    # Calculate the delta
    # Create a combined timeline of all points using default date
    all_times = sorted(set([default_datetime] + 
                         arrival_times_plot + 
                         [default_datetime + pd.Timedelta(minutes=prediction_window)] +
                         arrival_times_piecewise))
    
    # Interpolate both actual and expected to the combined timeline
    actual_counts = np.interp([t.timestamp() for t in all_times],
                            [t.timestamp() for t in [default_datetime] + arrival_times_plot + [default_datetime + pd.Timedelta(minutes=prediction_window)]],
                            [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1] if len(arrivals) > 0 else 0])
    
    expected_counts = np.interp([t.timestamp() for t in all_times],
                              [t.timestamp() for t in arrival_times_piecewise],
                              cumulative_rates)
    
    # Calculate delta
    delta = actual_counts - expected_counts

    if not show_only_delta:
        # Plot actual and expected arrivals
        ax.step([default_datetime] + arrival_times_plot + [default_datetime + pd.Timedelta(minutes=prediction_window)], 
             [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1] if len(arrivals) > 0 else 0], 
             where='post', label='Actual Arrivals')
        ax.step(arrival_times_piecewise, cumulative_rates, where='post', label='Expected Arrivals')    
        
        ax.set_xlabel('Time')
        ax.set_title(f'Cumulative Arrivals in the {int(prediction_window/60)} hours after {format_prediction_time(prediction_time)} on {snapshot_date}')
        ax.legend()
    
    if show_delta or show_only_delta:
        if show_only_delta:
            ax.step(all_times, delta, where='post', label='Actual - Expected', color='red')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Difference (Actual - Expected)')
            ax.set_title(f'Difference Between Actual and Expected Arrivals in the {int(prediction_window/60)} hours after {format_prediction_time(prediction_time)} on {snapshot_date}')
            ax.legend()
        else:
            ax2.step(all_times, delta, where='post', label='Actual - Expected', color='red')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Difference (Actual - Expected)')
            ax2.set_title(f'Difference Between Actual and Expected Arrivals in the {int(prediction_window/60)} hours after {format_prediction_time(prediction_time)} on {snapshot_date}')
            ax2.legend()
        
        plt.tight_layout()
    
    # Format x-axis to show only hours and minutes
    for ax in plt.gcf().get_axes():
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    plt.show()

def plot_multiple_deltas(df, prediction_time, snapshot_dates, prediction_window, yta_time_interval=15):
    """
    Plot delta charts for multiple snapshot dates on the same figure.
    
    Args:
        df (pd.DataFrame): DataFrame containing arrival data
        prediction_time (tuple): (hour, minute) of prediction time
        snapshot_dates (list): List of datetime.date objects to analyze
        prediction_window (int): Prediction window in minutes
        yta_time_interval (int): Time interval in minutes for calculating arrival rates
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Convert prediction time to datetime object
    prediction_time_obj = time(hour=prediction_time[0], minute=prediction_time[1])
    
    # Use a consistent default date for plotting (January 1, 2024)
    default_date = datetime(2024, 1, 1)
    default_datetime = pd.Timestamp(datetime.combine(default_date, prediction_time_obj), tz='UTC')
    
    df_copy = df.copy()
    if 'arrival_datetime' in df_copy.columns:
        df_copy.set_index('arrival_datetime', inplace=True)
    
    # Calculate arrival rates once for all dates
    arrival_rates = time_varying_arrival_rates(df_copy, yta_time_interval=yta_time_interval)
    
    # Store all deltas for averaging
    all_deltas = []
    all_times_list = []
    
    for snapshot_date in snapshot_dates:
        snapshot_datetime = pd.Timestamp(datetime.combine(snapshot_date, prediction_time_obj), tz='UTC')
        
        # Get arrivals within the prediction window
        arrivals = df_copy[
            (df_copy.index > snapshot_datetime) & 
            (df_copy.index <= snapshot_datetime + pd.Timedelta(minutes=prediction_window))
        ]
        
        if len(arrivals) == 0:
            continue
            
        # Sort arrivals by time
        arrivals = arrivals.sort_values('arrival_datetime')
        arrivals['cumulative_count'] = range(1, len(arrivals) + 1)
        
        # Calculate expected arrivals
        end_time = (datetime.combine(datetime.min, prediction_time_obj) + 
                   timedelta(minutes=prediction_window)).time()
        
        mean_arrival_rates = {k: v for k, v in arrival_rates.items() 
                            if (k >= prediction_time_obj and k < end_time) or
                               (end_time < prediction_time_obj and 
                                (k >= prediction_time_obj or k < end_time))}
        
        # Convert arrival rate times to datetime objects using default date
        arrival_times_piecewise = []
        for t in mean_arrival_rates.keys():
            if t < prediction_time_obj:
                dt = datetime.combine(default_date + timedelta(days=1), t)
            else:
                dt = datetime.combine(default_date, t)
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
        
        # Convert arrival times to use default date for plotting
        arrival_times_plot = [default_datetime + (t - snapshot_datetime) for t in arrivals.index]
        
        # Create a combined timeline of all points using default date
        all_times = sorted(set([default_datetime] + 
                             arrival_times_plot + 
                             [default_datetime + pd.Timedelta(minutes=prediction_window)] +
                             arrival_times_piecewise))
        
        # Interpolate both actual and expected to the combined timeline
        actual_counts = np.interp([t.timestamp() for t in all_times],
                                [t.timestamp() for t in [default_datetime] + arrival_times_plot + [default_datetime + pd.Timedelta(minutes=prediction_window)]],
                                [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1]])
        
        expected_counts = np.interp([t.timestamp() for t in all_times],
                                  [t.timestamp() for t in arrival_times_piecewise],
                                  cumulative_rates)
        
        # Calculate delta
        delta = actual_counts - expected_counts
        
        # Store for averaging
        all_deltas.append(delta)
        all_times_list.append(all_times)
        
        # Plot delta for this snapshot date
        ax.step(all_times, delta, where='post', color='grey', alpha=0.5)
    
    # Calculate and plot average delta
    if all_deltas:
        # Find the common time points across all dates
        common_times = sorted(set().union(*[set(times) for times in all_times_list]))
        
        # Interpolate all deltas to common time points
        interpolated_deltas = []
        for times, delta in zip(all_times_list, all_deltas):
            # Only interpolate within the actual time range for each date
            min_time = min(times)
            max_time = max(times)
            valid_times = [t for t in common_times if min_time <= t <= max_time]
            
            if valid_times:
                interpolated = np.interp([t.timestamp() for t in valid_times],
                                       [t.timestamp() for t in times],
                                       delta)
                # Pad with NaN for times outside the valid range
                padded = np.full(len(common_times), np.nan)
                valid_indices = [i for i, t in enumerate(common_times) if t in valid_times]
                padded[valid_indices] = interpolated
                interpolated_deltas.append(padded)
        
        # Calculate average delta, ignoring NaN values
        avg_delta = np.nanmean(interpolated_deltas, axis=0)
        
        # Plot average delta as a solid line
        # Only plot where we have valid data (not NaN)
        valid_mask = ~np.isnan(avg_delta)
        if np.any(valid_mask):
            ax.step([t for t, m in zip(common_times, valid_mask) if m],
                   avg_delta[valid_mask],
                   where='post', color='black', linewidth=2)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Format the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Difference (Actual - Expected)')
    ax.set_title(f'Difference Between Actual and Expected Arrivals in the {int(prediction_window/60)} hours after {format_prediction_time(prediction_time)} on all dates')
    
    # Format x-axis to show only hours and minutes
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    plt.show()
