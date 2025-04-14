# 4a. Specify requirements for emergency demand prediction

In previous notebooks I have introduced all of the building blocks provided in `patientflow` to make predictions of bed counts within a prediction window. In the next notebook I show how we used these building blocks in our application at University College London Hospital (UCLH) to predict emergency demand for beds, by specialty, over the next 8 hours. The predictions are aspirational; they assume that ED four-hour targets are met.

First, a brief recap on the requirements of our models.

## Recap on the requirements of our users

In [the first notebook](1_Meet_the_users_of_our_predictions.md) I introduced bed managers and their work. Through working closely with them over five years, we have developed an understanding their requirements for emergency demand predictions.

- They want information at specific times of day to coincide with their flow huddles, with an 8-hour view of incoming demand at these times
- The 8-hour view needs to take account of patients who are yet to arrive, who should be admitted within that time
- The predictions should be based on the assumption that the ED is meeting its 4-hour targets for performance (for example, the target that 80% of patients are to be admitted or discharged within 4 hours of arriving at the front door)
- The predictions should exclude patients who already have decisions to admit under a specialty; these should be counted as known demand for that specialty
- The predictions should be provided as numbers of bed needed (rather than predictions of whether any individual patient will be admitted) and should be grouped by speciality of admission, since a specialty breakdown helps to inform targeted actions
- The predictions should be sent by email, with a spreadsheet attached
- The output should communicate a low threshold number of beds needed with 90% probability, and with 70% probability

For more information about these requirements, and how we tailored the UCLH application to meet them, check out this talk by me, with Craig Wood, bed manager at UCLH, at the Health and Care Analytics Conference 2023:

<a href="https://www.youtube.com/watch?v=1V1IzWmOyX8" target="_blank">
    <img src="https://img.youtube.com/vi/1V1IzWmOyX8/0.jpg" alt="Watch the video" width="600"/>
</a>

## The current output from the UCLH application

The annotated figure below shows the output that our application currently generates at UCLH

```python
from IPython.display import Image
Image(filename='img/thumbnail_UCLH_application.jpg')
```

![jpeg](4a_Specify_emergency_demand_model_files/4a_Specify_emergency_demand_model_1_0.jpg)

The modelling output

- Differentiates between patients with a decision to admit (columns B:C) and those without (columns D:G)
- Provides separate predictions for patients in the ED and SDEC now (columns D:E), and those yet to arrive (columns F:G)
- Breaks down the output by speciality (rows 4:7); this is currently done at a high level - medical, surgical, haematology/oncology and paediatric - but a future version will provide predictions at detailed specialty level
- Shows the minimum number of beds needed with 90% probability (columns D and F) and with 70% probability (columns E and G)

The next notebook will show the implementation in code.
