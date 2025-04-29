from datetime import timedelta, datetime, time
from patientflow.calculate.arrival_rates import time_varying_arrival_rates
import matplotlib.pyplot as plt
import pandas as pd


# Create date range
def plot_arrival_comparison(df, prediction_time, snapshot_date, prediction_window):
    """
    Plot comparison between observed arrivals and expected arrival rates.
    
    Args:
        df (pd.DataFrame): DataFrame containing arrival data
        prediction_time (tuple): (hour, minute) of prediction time
        snapshot_date (datetime.date): Date to analyze
        prediction_window (int): Prediction window in minutes
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
            arrival_times_piecewise.append(datetime.combine(snapshot_date + timedelta(days=1), t))
        else:
            arrival_times_piecewise.append(datetime.combine(snapshot_date, t))
    
    arrival_times_piecewise.sort()
    
    # Calculate expected cumulative arrivals

    cumulative_rates = []
    current_sum = 0
    for t in arrival_times_piecewise:
        rate = mean_arrival_rates[t.time()]
        current_sum += rate #* (15/60)  # Convert 15-minute rate to hourly equivalent
        cumulative_rates.append(current_sum)
    
        # Plot observed vs expected arrivals
    plt.figure(figsize=(10, 6))
    plt.step([snapshot_datetime] + list(arrivals.index) + [snapshot_datetime + pd.Timedelta(minutes=prediction_window)], 
         [0] + list(arrivals['cumulative_count']) + [arrivals['cumulative_count'].iloc[-1] if len(arrivals) > 0 else 0], 
         where='post', label='Actual Arrivals')
    plt.step(arrival_times_piecewise, cumulative_rates, where='post', label='Expected Arrivals')    
    
    plt.xlabel('')
    plt.xlabel('Arrival Time')
    plt.title(f'Cumulative Arrivals in the {int(prediction_window/60)} hours after {prediction_time} on {snapshot_date}')
    plt.legend()
    plt.show()
