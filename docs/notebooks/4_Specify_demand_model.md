# 4. Specify requirements for demand prediction

## From building blocks to a production pipeline

In the 3x_ notebooks I introduced the building blocks provided in `patientflow` to make predictions of bed counts within a prediction window:

- **3a/3b**: Creating group snapshots from patient snapshots and evaluating the resulting bed count distributions
- **3c/3d**: Disaggregating predictions by hospital service and evaluating specialty-level predictions
- **3e/3f**: Predicting demand from patients yet to arrive and evaluating those predictions

In this series of notebooks, I show how we assembled these building blocks into a production system at University College London Hospital (UCLH) to predict demand for beds, by specialty, over the next 8 hours. The notebooks in this series cover:

- **4a**: The production data structures (`FlowInputs`, `DemandPredictor`, `PredictionBundle`, etc.) that make the pipeline configurable 
- **4b**: Stratifying predictions by patient subgroups (e.g. children vs adults vs older adults, men vs women) using `MultiSubgroupPredictor`
- **4c**: The full prediction pipeline, combining patients currently in the ED with those yet to arrive
- **4d**: Evaluating the production model components 
- **4e**: Extending to hierarchical reporting at multiple organisational levels (e.g. specialty, division, hospital)

First, a brief recap on the requirements of our models. 

## Recap on the requirements of our users (this refers to the legacy version of our application)

In notebook 1 I introduced bed managers and their work. Through working closely with them over five years, we have developed an understanding of their requirements for emergency demand predictions. 

* They want information at specific times of day to coincide with their flow huddles, with an 8-hour view of incoming demand at these times
* The 8-hour view needs to take account of patients who are yet to arrive, who should be admitted within that time
* The predictions should be based on the assumption that the ED is meeting its 4-hour targets for performance (for example, the target that 80% of patients are to be admitted or discharged within 4 hours of arriving at the front door)
* The predictions should exclude patients who already have decisions to admit under a specialty; these should be counted as known demand for that specialty
* The predictions should be provided as numbers of beds needed (rather than predictions of whether any individual patient will be admitted) and should be grouped by speciality of admission, since a specialty breakdown helps to inform targeted actions
* The predictions should be sent by email, with a spreadsheet attached
* The output should communicate a low threshold number of beds needed with 90% probability, and with 70% probability

For more information about these requirements, and how we tailored the UCLH application to meet them, check out this talk by me, with Craig Wood, bed manager at UCLH, at the Health and Care Analytics Conference 2023:

<a href="https://www.youtube.com/watch?v=1V1IzWmOyX8" target="_blank">
    <img src="https://img.youtube.com/vi/1V1IzWmOyX8/0.jpg" alt="Watch the video" width="600"/>
</a>


## The output from our initial UCLH deployment

The annotated figure below shows the output from the initial deployment of our application at UCLH.


```python
from IPython.display import Image
Image(filename='img/thumbnail_UCLH_application.jpg')
```




    
![jpeg](4_Specify_demand_model_files/4_Specify_demand_model_1_0.jpg)
    



The modelling output (which is now legacy code at UCLH)

- Differentiates between patients with a decision to admit (columns B:C) and those without (columns D:G)
- Provides separate predictions for patients in the ED and SDEC now (columns D:E), and those yet to arrive (columns F:G)
- Breaks down the output by speciality (rows 4:7); in this initial version, this is done at a high level — medical, surgical, haematology/oncology and paediatric. Notebook 4e shows how predictions can be generated at more detailed specialty levels using a hierarchical approach.
- Shows the minimum number of beds needed with 90% probability (columns D and F) and with 70% probability (columns E and G)

## The new requirements of our users 

* They still want information at specific times of day to coincide with their flow huddles
* They would like the prediction window to end at 8 am the following day (meaning that the length of the window varies depending on the time of day predictions are made)
* The view needs to take account of patients who are yet to arrive, who should be admitted within that time including both emergency and elective patients
* The view should also take account of expected discharges among current inpatients
* The predictions should be based on the assumption that the ED is meeting its 4-hour targets for performance (for example, the target that 80% of patients are to be admitted or discharged within 4 hours of arriving at the front door)
* Patients in ED/SDEC who already have decisions to admit under a specialty should be counted as part of the inpatient demand for that specialty
* The predictions should be provided as numbers of beds needed (rather than predictions of whether any individual patient will be admitted) 
* The predictions should be grouped by speciality of admission, since a specialty breakdown helps to inform targeted actions, as before, but now the specialty breakdown should be available at any level of the hospital's reporting hierarchy
* The predictions should be sent by email, with a spreadsheet attached for now, but ultimately should be displayed in Epic

The new output at UCLH 

- serves the legacy predictions as the first in a multi-tab spreadsheet
- provides a tab for predictions at board, division, reporting unit and subspecialty levels (each a lower level of granularity)
- shows inflows, outflows and net flow for each entity, split into emergency and elective streams
- will show the current census of patients under each entity, and the agreed bed base

The following notebooks show the implementation in code, starting with the data structures that organise predictions for production use.


 
