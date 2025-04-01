# Visualise un-delayed demand for beds for admitted patients

One reason why Emergency Departments (EDs) fail to meet admissions targets is because beds are not available at the time patients arrive. Most discharges happen in the afternoon, so patients arriving at night into a full hospital often have to wait until late the following day to be admitted. UCLH wanted to understand their un-delayed demand over the course of a day - that is, when beds would be needed if patients were processed within the 4-hour target time. A formal term for this (used in queueing theory) is "offered load". Here we use "un-delayed demand" as it better conveys the concept to healthcare staff. UCLH wanted to understand their undelayed demand patterns to inform an improvement project focus on their emergency patient pathway.

This notebook generates charts that aim to answer the following questions: 

* What is the typical pattern of arrivals of admitted patients over the day?
* If ED was meeting 4-hour targets for admitted patients, when would beds for these patients need to be ready?**
* Are there differences in the above between weekends and weekdays? 
* The discharge window, when decision-makers are on wards, is between 8.30 am and 5 pm. If a hospital wants to meet its ED targets on a consistent basis, then there should be enough empty beds to cover demand over night by the end of the discharge window when decision-makers leave. How many beds would need to be available at the end of discharge window to cover all demand over night? 

To answer these questions we will prepare a model using only 

* historical arrival rates of patients later admitted
* 4-hour targets for ED. We assume that the hospital might want to choose a certain percentage of admitted patients to be processed into a bed within 4 hours, accordingly to current targets. 

Here we refer to both Emergency Department (EDs) and Same Day Emergency Care (SDECs) since patients can be admitted to the hospital from both. 




## Set up the notebook environment


```python
# Reload functions every time
%load_ext autoreload 
%autoreload 2
```


```python
from patientflow.load import set_project_root
project_root = set_project_root()
```

    Inferred project root: /Users/zellaking/Repos/patientflow


## Set file paths


```python
from patientflow.load import set_file_paths
from patientflow.load import load_config_file

# set file locations
data_folder_name = 'data-public' 

# set file paths
data_file_path, media_file_path, model_file_path, config_path = set_file_paths(project_root, 
               data_folder_name=data_folder_name, config_file='config-uclh.yaml')

# create subfolders for weekdays and weekends
media_file_path_weekdays = media_file_path / 'undelayed-demand-uclh' / 'weekdays'
media_file_path_weekends = media_file_path / 'undelayed-demand-uclh' / 'weekends'
media_file_path_all_days = media_file_path / 'undelayed-demand-uclh' / 'all_days'

    
media_file_path_weekdays.mkdir(parents=True, exist_ok=True)
media_file_path_weekends.mkdir(parents=True, exist_ok=True)
media_file_path_all_days.mkdir(parents=True, exist_ok=True)
```

    Configuration will be loaded from: /Users/zellaking/Repos/patientflow/config-uclh.yaml
    Data files will be loaded from: /Users/zellaking/Repos/patientflow/data-public
    Trained models will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public
    Images will be saved to: /Users/zellaking/Repos/patientflow/trained-models/public/media


## Load parameters

These are set in config.json. 


```python
# load params from config file
params = load_config_file(config_path)

# prediction_times = params["prediction_times"]
# start_training_set, start_validation_set, start_test_set, end_test_set = params["start_training_set"], params["start_validation_set"], params["start_test_set"], params["end_test_set"]
x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
# prediction_window = params["prediction_window"]
# epsilon = float(params["epsilon"])
yta_time_interval = params["yta_time_interval"]

print(f'The aspiration is for {y1*100}% of patients to be admitted within {x1} hours, and {y2*100}% of patients to be admitted within {x2} hours')
```

    The aspiration is for 76.0% of patients to be admitted within 4.0 hours, and 99.0% of patients to be admitted within 12.0 hours


## Load data

Here we load the data. NOTE - the public data is a subset of UCLH arrivals, so the charts here will underestimate the demand at UCLH. 


```python
import pandas as pd
import numpy as np
from patientflow.load import load_data

# load data
inpatient_arrivals = load_data(data_file_path, 
                    file_name='inpatient_arrivals.csv')
inpatient_arrivals['arrival_datetime'] = pd.to_datetime(inpatient_arrivals['arrival_datetime'], utc = True)
inpatient_arrivals.set_index('arrival_datetime', inplace=True)

# select only data from 2023 onwards
inpatient_arrivals[inpatient_arrivals.index > '2023-01-01']


weekdays = inpatient_arrivals[(inpatient_arrivals.index.weekday < 5) & (inpatient_arrivals.index > '2023-01-01')]
weekends = inpatient_arrivals[(inpatient_arrivals.index.weekday >= 5) & (inpatient_arrivals.index > '2023-01-01')]


```


```python
# summarise
print(f"Weekday dates span {weekdays.index.date.min()} to {weekdays.index.date.max()}")
print(f"Weekend dates span {weekends.index.date.min()} to {weekends.index.date.max()}")


```

    Weekday dates span 2031-03-03 to 2031-12-31
    Weekend dates span 2031-03-01 to 2031-12-28


Note how simple the dataset is that is used in the rest of this notebook. The only columns used are the arrival_datetime index, and (at the very end of the notebook) the specialty of admission. Similar breakdown as for specialty could be done for by sex or for adults/children, if useful.


```python
inpatient_arrivals.dtypes
```




    training_validation_test    object
    sex                         object
    specialty                   object
    is_child                      bool
    dtype: object



## Get arrival rates by hour


### Calculate the arrival rates for each hour of the day, at weekdays and weekends, using the historical data


```python
print(f"On average, there were {inpatient_arrivals[(inpatient_arrivals.index > '2023-01-01')].resample('D').size().mean():.3f} inpatients admitted per day")
```

    On average, there were 41.422 inpatients admitted per day



```python
from patientflow.calculate.arrival_rates import time_varying_arrival_rates

arrival_rates_by_time_interval = time_varying_arrival_rates(
    df=inpatient_arrivals[(inpatient_arrivals.index > '2023-01-01')],
    yta_time_interval=yta_time_interval,
    verbose=True,
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 4
          1 from patientflow.calculate.arrival_rates import time_varying_arrival_rates
          3 arrival_rates_by_time_interval = time_varying_arrival_rates(
    ----> 4     df=inpatient_arrivals[(inpatient_arrivals.index > '2023-01-01')],
          5     yta_time_interval=yta_time_interval,
          6     verbose=True,
          7 )


    NameError: name 'inpatient_arrivals' is not defined


The dictionary returned by this function breaks out the arrival rates into intervals defined by yta_time_interval


```python
arrival_rates_by_time_interval
```




    OrderedDict([(datetime.time(0, 0), 0.20915032679738563),
                 (datetime.time(0, 15), 0.19934640522875818),
                 (datetime.time(0, 30), 0.1895424836601307),
                 (datetime.time(0, 45), 0.1568627450980392),
                 (datetime.time(1, 0), 0.20915032679738563),
                 (datetime.time(1, 15), 0.18627450980392157),
                 (datetime.time(1, 30), 0.1830065359477124),
                 (datetime.time(1, 45), 0.18627450980392157),
                 (datetime.time(2, 0), 0.15359477124183007),
                 (datetime.time(2, 15), 0.1437908496732026),
                 (datetime.time(2, 30), 0.16666666666666666),
                 (datetime.time(2, 45), 0.1895424836601307),
                 (datetime.time(3, 0), 0.16339869281045752),
                 (datetime.time(3, 15), 0.13071895424836602),
                 (datetime.time(3, 30), 0.1568627450980392),
                 (datetime.time(3, 45), 0.15359477124183007),
                 (datetime.time(4, 0), 0.16993464052287582),
                 (datetime.time(4, 15), 0.17973856209150327),
                 (datetime.time(4, 30), 0.1830065359477124),
                 (datetime.time(4, 45), 0.19934640522875818),
                 (datetime.time(5, 0), 0.12745098039215685),
                 (datetime.time(5, 15), 0.14052287581699346),
                 (datetime.time(5, 30), 0.1111111111111111),
                 (datetime.time(5, 45), 0.10130718954248366),
                 (datetime.time(6, 0), 0.16339869281045752),
                 (datetime.time(6, 15), 0.12418300653594772),
                 (datetime.time(6, 30), 0.14705882352941177),
                 (datetime.time(6, 45), 0.17973856209150327),
                 (datetime.time(7, 0), 0.20915032679738563),
                 (datetime.time(7, 15), 0.2908496732026144),
                 (datetime.time(7, 30), 0.3431372549019608),
                 (datetime.time(7, 45), 0.32679738562091504),
                 (datetime.time(8, 0), 0.34967320261437906),
                 (datetime.time(8, 15), 0.3366013071895425),
                 (datetime.time(8, 30), 0.3202614379084967),
                 (datetime.time(8, 45), 0.4117647058823529),
                 (datetime.time(9, 0), 0.5130718954248366),
                 (datetime.time(9, 15), 0.5261437908496732),
                 (datetime.time(9, 30), 0.5522875816993464),
                 (datetime.time(9, 45), 0.5947712418300654),
                 (datetime.time(10, 0), 0.6045751633986928),
                 (datetime.time(10, 15), 0.6797385620915033),
                 (datetime.time(10, 30), 0.6568627450980392),
                 (datetime.time(10, 45), 0.8006535947712419),
                 (datetime.time(11, 0), 0.6764705882352942),
                 (datetime.time(11, 15), 0.7712418300653595),
                 (datetime.time(11, 30), 0.6535947712418301),
                 (datetime.time(11, 45), 0.7941176470588235),
                 (datetime.time(12, 0), 0.7026143790849673),
                 (datetime.time(12, 15), 0.7875816993464052),
                 (datetime.time(12, 30), 0.7222222222222222),
                 (datetime.time(12, 45), 0.8333333333333334),
                 (datetime.time(13, 0), 0.8333333333333334),
                 (datetime.time(13, 15), 0.738562091503268),
                 (datetime.time(13, 30), 0.7483660130718954),
                 (datetime.time(13, 45), 0.7973856209150327),
                 (datetime.time(14, 0), 0.7352941176470589),
                 (datetime.time(14, 15), 0.8104575163398693),
                 (datetime.time(14, 30), 0.6176470588235294),
                 (datetime.time(14, 45), 0.6666666666666666),
                 (datetime.time(15, 0), 0.6764705882352942),
                 (datetime.time(15, 15), 0.6895424836601307),
                 (datetime.time(15, 30), 0.5065359477124183),
                 (datetime.time(15, 45), 0.5816993464052288),
                 (datetime.time(16, 0), 0.5032679738562091),
                 (datetime.time(16, 15), 0.5751633986928104),
                 (datetime.time(16, 30), 0.6470588235294118),
                 (datetime.time(16, 45), 0.6764705882352942),
                 (datetime.time(17, 0), 0.6470588235294118),
                 (datetime.time(17, 15), 0.6633986928104575),
                 (datetime.time(17, 30), 0.565359477124183),
                 (datetime.time(17, 45), 0.6209150326797386),
                 (datetime.time(18, 0), 0.5915032679738562),
                 (datetime.time(18, 15), 0.5751633986928104),
                 (datetime.time(18, 30), 0.5620915032679739),
                 (datetime.time(18, 45), 0.6045751633986928),
                 (datetime.time(19, 0), 0.5),
                 (datetime.time(19, 15), 0.5947712418300654),
                 (datetime.time(19, 30), 0.5784313725490197),
                 (datetime.time(19, 45), 0.6274509803921569),
                 (datetime.time(20, 0), 0.5882352941176471),
                 (datetime.time(20, 15), 0.545751633986928),
                 (datetime.time(20, 30), 0.545751633986928),
                 (datetime.time(20, 45), 0.5424836601307189),
                 (datetime.time(21, 0), 0.369281045751634),
                 (datetime.time(21, 15), 0.3202614379084967),
                 (datetime.time(21, 30), 0.3104575163398693),
                 (datetime.time(21, 45), 0.3300653594771242),
                 (datetime.time(22, 0), 0.2777777777777778),
                 (datetime.time(22, 15), 0.238562091503268),
                 (datetime.time(22, 30), 0.23202614379084968),
                 (datetime.time(22, 45), 0.2777777777777778),
                 (datetime.time(23, 0), 0.21241830065359477),
                 (datetime.time(23, 15), 0.22875816993464052),
                 (datetime.time(23, 30), 0.20588235294117646),
                 (datetime.time(23, 45), 0.19934640522875818)])



Below, confirming that the sum of these intervals equates to the same value


```python
sum(arrival_rates_by_time_interval.values())
```




    41.42156862745098



To get the correct demominator when calculating average arrival rates per weekday and weekend, we need to calculate the number of weekdays or weekends, and pass this into the function. Assuming that there was at least one inpatient per day, we can do this by counting the number of days in each dataset


```python
num_weekdays = len(np.unique(weekdays.index.date))
num_weekends = len(np.unique(weekends.index.date))

print(f"On average, there were {len(weekdays)/num_weekdays:.1f} inpatients admitted per day over the {num_weekdays} weekdays included")
print(f"On average, there were {len(weekends)/num_weekends:.1f} inpatients admitted per day over the {num_weekends} weekend days included")
```

    On average, there were 44.3 inpatients admitted per day over the 218 weekdays included
    On average, there were 34.2 inpatients admitted per day over the 88 weekend days included


### Plot the arrival rates for each hour of the day, at weekdays and weekends, using the historical data


```python
from patientflow.viz.arrival_rates import plot_arrival_rates
from datetime import timedelta

# Set the plot to start at the 8th hour of the day (if not set the function will default to starting at midnight
start_plot_index = 8

# plot for weekdays
title = f'Hourly arrival rates of admitted patients starting at {start_plot_index} am - weekdays from {weekdays.index.date.min()} to {weekdays.index.date.max()}'
plot_arrival_rates(weekdays, 
                   title, 
                   time_interval=60, 
                   start_plot_index=start_plot_index, 
                   file_prefix = '1_',
                   media_file_path=media_file_path_weekdays,
                  num_days=num_weekdays)

# plot for weekends
title = f'Hourly arrival rates of admitted patients starting at {start_plot_index} am - weekends from {weekends.index.date.min()} to {weekends.index.date.max()}'
plot_arrival_rates(weekends, 
                   title, 
                   time_interval=60, 
                   start_plot_index=start_plot_index, 
                   file_prefix = '1_',
                   media_file_path=media_file_path_weekends,
                  num_days=num_weekends
                   )


# plot for both
title = f'Hourly arrival rates of admitted patients starting at {start_plot_index} am - all days from {inpatient_arrivals.index.date.min()} to {inpatient_arrivals.index.date.max()}'
plot_arrival_rates(inpatient_arrivals=weekdays,
                        inpatient_arrivals_2=weekends, 
                    labels=('Weekdays', 'Weekends'),
                   title=title, 
                   time_interval=60, 
                   start_plot_index=start_plot_index, 
                   file_prefix = '1_',
                   media_file_path=media_file_path_all_days,
                   num_days=num_weekdays,
                   num_days_2=num_weekends
                   )
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_24_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_24_1.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_24_2.png)
    


## Plot the time beds would be needed if each patient were admitted exactly 4 hours after arrival

We now make the assumption that the bed is needed exactly 4 hours after arrival. This assumes that every patient meets the 4-hour target, and that there is no variation in the time it takes people to be processed through ED/SDEC, so it is not realistic, but serves as a starting point. 


```python
title = 'Average number of beds needed this hour on weekdays,\nif each patient is admitted exactly four hours after arrival, starting at 8 am'
plot_arrival_rates(weekdays,
                   title, 
                   time_interval=60, 
                   start_plot_index=start_plot_index, 
                   file_prefix = '2_', 
                   lagged_by=4,
                    media_file_path=media_file_path_weekdays,
                   num_days=num_weekdays,
)

title = 'Average number of beds needed this hour on weekends,\nif each patient is admitted exactly four hours after arrival, starting at 8 am'
plot_arrival_rates(weekends,
                   title, 
                   time_interval=60, 
                   start_plot_index=start_plot_index, 
                   file_prefix = '2_', 
                   lagged_by=4,
                    media_file_path=media_file_path_weekends,
                   num_days=num_weekends,
)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_26_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_26_1.png)
    


## Plot cumulative arrival rates

We can show the same information as above, counting the beds needed cumulatively over the day.   


```python
from patientflow.viz.arrival_rates import plot_cumulative_arrival_rates
title = f'Cumulative number of beds needed on weekdays, by hour of the day,\n if each incoming patient is admitted exactly four hours after arrival, starting at 8 am'
plot_cumulative_arrival_rates(
    weekdays,
    title,
    start_plot_index=8,
    media_file_path=media_file_path_weekdays,
    file_prefix='3_',
    num_days=num_weekdays
)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_28_0.png)
    


## Now considering the discharge window

Here we are interested in bringing forward the number of beds vacated at each hour, such that the necessary number of beds are vacated within the influenceable window before decision-makers leave the wards

We first define a dictionary of scenarios we'd like to plot. (This is simply to reduce lines of code.)


```python
scenarios = [
    {
        'data': weekdays,
        'file_path': media_file_path_weekdays,
        'label': 'weekdays',
        'num_days': num_weekdays
    },
    {
        'data': weekends,
        'file_path': media_file_path_weekends,
        'label': 'weekends',
        'num_days': num_weekends,        
    }
]


```


```python
from patientflow.viz.arrival_rates import plot_cumulative_arrival_rates

start_of_influencable_window = 8
end_of_influencable_window = 17


def plot_average_beds_needed(scenarios):
    """
    Generate bed occupancy plots for both weekday and weekend data.
    """
    
    for scenario in scenarios:
        # First plot: 17:00 end of influencable window
        title = f'Cumulative number of beds to be vacated on {scenario["label"]}, by hour of the day,\n if all beds for overnight admissions are to be vacated by {17}00'
        plot_cumulative_arrival_rates(
            scenario['data'],
            title,
            start_plot_index=8,
            draw_window=(start_of_influencable_window, 17),
            media_file_path=scenario["file_path"],
            file_prefix='4_',
            set_y_lim=55,
            num_days=scenario["num_days"]
        )
        
        # Second plot: 20:00 end of influencable window
        title = f'Cumulative number of beds to be vacated on {scenario["label"]}, by hour of the day,\n if all beds for overnight admissions are to be vacated by {20}00'
        plot_cumulative_arrival_rates(
            scenario['data'],
            title,
            start_plot_index=8,
            draw_window=(start_of_influencable_window, 20),
            media_file_path=scenario["file_path"],
            file_prefix='4_',
            set_y_lim=55,
            hour_lines=[12,17,20],
            num_days=scenario["num_days"]
        )


plot_average_beds_needed(
    scenarios,
)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_31_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_31_1.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_31_2.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_31_3.png)
    



## Introducing an aspirational approach

As noted above, this is not very realistic as it assumes that every patient meets the 4-hour target, and that there is no variation in the time it takes people to be processed through ED/SDEC. Ideally patients would be processed sooner than 4 hours in most cases. 

We can instead use a probabilistic approach to determine whether any patient will be admitted within 4 hours. The probability is shown in the plot below.


```python
import matplotlib.pyplot as plt
from patientflow.viz.aspirational_curve_plot import plot_curve

figsize = (6,3)
title = 'Aspirational curve reflecting a ' + str(int(x1)) + ' hour target for ' + str(int(y1*100)) + \
        '% of patients\nand a '+ str(int(x2)) + ' hour target for ' + str(int(y2*100)) + '% of patients'

plot_curve(
    title = title,
    x1 = x1,
    y1 = y1,
    x2 = x2,
    y2 = y2,
    figsize = (10,6),
    include_titles=True,
    text_size=14,
    media_file_path=media_file_path_weekends,
    file_name=title.replace(" ", "_"),
)

```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_33_0.png)
    


## Plot the time beds would be needed after applying the aspirational curve

Here, the aspirational curve has been applied. It has the effect of smoothing out the times at which beds are needed (the solid line falls between the time people arrived and the line that is lagged by 4 hours)


```python

title = f'Number of beds needed for admitted patients by hour on weekdays\nassuming ED targets of {int(y1*100)}% in {int(x1)} hours are hit, starting at 8 am'
plot_arrival_rates(weekdays, title, start_plot_index = 8, 
                   lagged_by=4, 
                   curve_params=(x1, y1, x2, y2), 
                   file_prefix = '5_',
                   media_file_path=media_file_path_weekdays,
                  num_days=num_weekdays)


# title = f'Number of beds needed for admitted patients by hour on weekends\nassuming ED targets of {int(y1*100)}% in {int(x1)} hours are hit, starting at 8 am'
# plot_arrival_rates(weekends, title, start_plot_index = 8, 
#                    lagged_by=4, 
#                    curve_params=(x1, y1, x2, y2), 
#                    file_prefix = '5_',
#                     media_file_path=media_file_path_weekends,
                  # num_days=num_weekends)



# plot for both
title = f'Number of beds needed for admitted patients by hour for all days\nassuming ED targets of {int(y1*100)}% in {int(x1)} hours are hit, starting at 8 am'
plot_arrival_rates(inpatient_arrivals=weekdays,
                        inpatient_arrivals_2=weekends, 
                    labels=('Weekdays', 'Weekends'),
                   title=title, 
                   # lagged_by=4, 
                   curve_params=(x1, y1, x2, y2), 
                    start_plot_index=start_plot_index, 
                   file_prefix = '5_',
                   media_file_path=media_file_path_weekends,
                  num_days=num_weekdays,
                   num_days_2=num_weekends
                   )
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_35_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_35_1.png)
    


We can then plot the cumulative numbers required, applying the aspirational window as before


```python
def plot_average_beds_needed_using_aspirational_curve(scenarios):
    """
    Generate bed occupancy plots for both weekday and weekend data, after applying aspirational curve.
    """
    
    for scenario in scenarios:
        # First plot: 4-hour admission delay
        title = f'Cumulative number of beds needed on {scenario["label"]}, by hour of the day,\n after applying aspirational ED performance curve, starting at 8 am'
        plot_cumulative_arrival_rates(
            scenario['data'],
            title,
            curve_params=(x1, y1, x2, y2),
            start_plot_index=8,
            media_file_path=scenario["file_path"],
            file_prefix='6_',
            num_days=scenario["num_days"]
        )
        
        # Second plot: 17:00 vacancy requirement
        title = f'Cumulative number of beds to be vacated on {scenario["label"]}, by hour of the day, after applying aspirational ED performance curve,\n if all beds for overnight admissions are to be vacated by {17}00'
        plot_cumulative_arrival_rates(
            scenario['data'],
            title,
            curve_params=(x1, y1, x2, y2),
            start_plot_index=8,
            draw_window=(start_of_influencable_window, 17),
            media_file_path=scenario["file_path"],
            file_prefix='7_',
            set_y_lim=55,
            num_days=scenario["num_days"]
        )
        
        # Second plot: 20:00 vacancy requirement
        title = f'Cumulative number of beds to be vacated on {scenario["label"]}, by hour of the day, after applying aspirational ED performance curve,\n if all beds for overnight admissions are to be vacated by {20}00'
        plot_cumulative_arrival_rates(
            scenario['data'],
            title,
            curve_params=(x1, y1, x2, y2),
            start_plot_index=8,
            draw_window=(start_of_influencable_window, 20),
            media_file_path=scenario["file_path"],
            file_prefix='7_',
            set_y_lim=55,
            hour_lines=[12,17,20],
            num_days=scenario["num_days"]
        )

plot_average_beds_needed_using_aspirational_curve(scenarios)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_1.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_2.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_3.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_4.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_37_5.png)
    


## Cumulative plot with probabilities

Up to now, we have worked with the average number of beds needed. If this number of beds were ready, the hospital would be equipped to hit its ED targets on a day with average arrival rates, but not on a day that exceeds the average. On such days, the performance against 4-hour targets would deteriorate. 

Here we allow for the idea that a hospital might set an aspiration to hit its ED targets on (say) 90% of days.  

The code below shows how to plot a chart that explores how the number of beds needed by a given hour of the day will change if 4-hour targets are to be met on 90% of days. The red line is the same as above (the  beds that need to be vacated to meet average demand), and the blue dotted line shows the number of empty beds would need to be to ensure that enough capacity is available on 90% of days. 


```python


# plot showing just 90% centile for weekends
title = f'Cumulative number of beds needed on weekends, by hour of day, with 90% probability of hitting ED targets of {int(y1*100)}% in {int(x1)} hours\non any single day if this number of beds were available'
plot_cumulative_arrival_rates(
   weekends,
    title,
    curve_params=(x1, y1, x2, y2),
    lagged_by=None,
    time_interval=60,
    start_plot_index=8,
    draw_window=(start_of_influencable_window, 20),
    x_margin=0.5,
    file_prefix='9_',
    set_y_lim=None,
    hour_lines=[12,20],
    annotation_prefix='To hit target',
    line_colour='red',
    plot_centiles=True,
    highlight_centile=0.9,
    centiles=[ 0.9],
    markers=['o'],
    line_styles_centiles=['-.', '--', ':', '-', '-'],
    bed_type_spec='',
    media_file_path=media_file_path_weekends,
    num_days=num_weekends


)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_39_0.png)
    



```python


# plot showing just 90% centile for weekends, with influenceable window
title = f'Cumulative number of beds needed on weekends, by hour of day, with 90% probability of hitting ED targets of {int(y1*100)}% in {int(x1)} hours\non any single day if this number of beds were available'
plot_cumulative_arrival_rates(
   weekends,
    title,
    curve_params=(x1, y1, x2, y2),
    lagged_by=None,
    time_interval=60,
    start_plot_index=8,
    draw_window=None,
    x_margin=0.5,
    file_prefix='9_',
    set_y_lim=None,
    hour_lines=[12,17,20],
    annotation_prefix='To hit target',
    line_colour='red',
    plot_centiles=True,
    highlight_centile=0.9,
    centiles=[ 0.9],
    markers=['o'],
    line_styles_centiles=['-.', '--', ':', '-', '-'],
    bed_type_spec='',
    media_file_path=media_file_path_weekends,
    num_days=num_weekends


)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_40_0.png)
    


For completeness, we show here how you could plot any number of centiles of probability. 


```python
title = f'Cumulative number of beds needed on weekdays, by hour of day, with probability of hitting ED targets of {int(y1*100)}% in {int(x1)} hours\non any single day if this number of beds were available'
plot_cumulative_arrival_rates(
   weekdays,
    title,
    curve_params=(x1, y1, x2, y2),
    lagged_by=None,
    time_interval=60,
    start_plot_index=8,
    draw_window=None,
    x_margin=0.5,
    file_prefix='9_',
    set_y_lim=None,
    hour_lines=[12, 17],
    line_styles={12: '--', 17: ':'},
    annotation_prefix='On average',
    line_colour='red',
    plot_centiles=True,
    highlight_centile=0.9,
    centiles=[0.3, 0.5, 0.7, 0.9, 0.99],
    markers=['D', 's', '^', 'o', 'v'],
    line_styles_centiles=['-.', '--', ':', '-', '-'],
    bed_type_spec='',
    media_file_path=media_file_path_weekdays,
    num_days=num_weekdays

)

title = f'Cumulative number of beds needed on weekends, by hour of day, with probability of hitting ED targets of {int(y1*100)}% in {int(x1)} hours\non any single day if this number of beds were available'
plot_cumulative_arrival_rates(
   weekends,
    title,
    curve_params=(x1, y1, x2, y2),
    lagged_by=None,
    time_interval=60,
    start_plot_index=8,
    draw_window=None,
    x_margin=0.5,
    file_prefix='9_',
    set_y_lim=None,
    hour_lines=[12, 17],
    line_styles={12: '--', 17: ':'},
    annotation_prefix='On average',
    line_colour='red',
    plot_centiles=True,
    highlight_centile=0.9,
    centiles=[0.3, 0.5, 0.7, 0.9, 0.99],
    markers=['D', 's', '^', 'o', 'v'],
    line_styles_centiles=['-.', '--', ':', '-', '-'],
    bed_type_spec='',
    media_file_path=media_file_path_weekends,
    num_days=num_weekends


)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_42_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_42_1.png)
    


## Plot by specialty

Below we show a breakdown by specialty


```python
from patientflow.viz.arrival_rates import plot_cumulative_arrival_rates

for _spec in ['medical', 'surgical', 'haem/onc', 'paediatric']:
    inpatient_arrivals_spec = weekends[(~weekends.specialty.isnull()) & (weekends.specialty == _spec)]
    __spec = _spec.replace('/', '_')
    
    title = f'Cumulative number of {__spec} beds needed on weekends, by hour of day,\n with 90% probability of hitting ED targets on any single day if this number of beds were available'

    plot_cumulative_arrival_rates(
    inpatient_arrivals_spec,
        title,
        curve_params=(x1, y1, x2, y2),
        lagged_by=None,
        time_interval=60,
        start_plot_index=8,
        draw_window=None,
        x_margin=0.5,
        file_prefix='A_',
        set_y_lim=None,
        hour_lines=[12,17,20],
        annotation_prefix='To hit targets on 90% of days',
        line_colour='red',
        plot_centiles=True,
        highlight_centile=0.9,
        centiles=[0.9],
        markers=['o'],
        line_styles_centiles=['-.', '--', ':', '-', '-'],
        bed_type_spec=__spec,

        text_y_offset=0.5,
        media_file_path=media_file_path_weekends,
        num_days=num_weekends


    )
    

```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_44_0.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_44_1.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_44_2.png)
    



    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_44_3.png)
    


## Conclusion

The charts above have used in presentations with UCLH's patient flow improvement group, as part of a wider project on Emergency Patient Pathways. They help to focus attention on a very intractable problem. Hospitals are full, most of the time. Beds are vacated at certain times of day, typically afternoons and early evening. People show up at the ED/SDEC all through the day, not just at times that suit the way the hospital works. This makes it very difficult to hit ED targets when hospitals are full, unless proactive steps are taken. 

Note that all the charts have been created using only a very simple set of inputs


```python
from patientflow.viz.undelayed_demand import generate_plot

generate_plot(weekdays, x1 = 4, y1 = 0.8, x2 = 12, y2 = 0.99)
```


    
![png](7_Visualise_un-delayed_demand_files/7_Visualise_un-delayed_demand_46_0.png)
    



```python
f"Date Range: {weekdays.index.date.min()} to {weekdays.index.date.max()}"
```




    'Date Range: 2031-03-03 to 2031-12-31'




```python

```
