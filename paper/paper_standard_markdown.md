# patientflow: A Python package for predicting demand for hospital beds using real-time data

**Authors:**  
Zella King (ORCID: 0000-0001-7389-1527)¹  
Jon Gillham (ORCID: 0009-0007-4110-7284)²  
Martin Utley (ORCID: 0000-0001-9928-1516)¹  
Matt Graham¹  
Sonya Crowe (ORCID: 0000-0003-1882-5476)¹

**Affiliations:**  
¹ Clinical Operational Research Unit (CORU), University College London, United Kingdom  
² Institute of Health Informatics, University College London, United Kingdom

**Tags:** Python, patient, hospital, bed, demand, real-time, electronic health records  
**Date:** June 2 2025

## Summary

patientflow, a Python package available on PyPI (King et al., 2025), converts patient-level predictions into output that is useful for bed managers in hospitals. If you want to predict a non-clinical outcome for a patient, such as admission or discharge from hospital, you can use patientflow to create bed count distributions for a cohort of patients. The package was developed for University College London Hospitals (UCLH) NHS Trust to predict the number of emergency admissions using real-time data. The methods generalise to any problem where it is useful to predict non-clinical outcomes for a cohort of patients at a point in time. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.

## Statement of need

Hospital bed managers monitor whether they have sufficient beds to meet demand. At specific points during the day they predict numbers of inpatients likely to leave, and numbers of new admissions. These point-in-time predictions are important because, if bed managers anticipate a shortage of beds, they must take swift action to mitigate the situation. Commonly, bed managers use simple heuristics based on past admission and discharge patterns. Electronic Health Record (EHR) systems can offer superior predictions, grounded in real-time knowledge about patients coming in and those expected to leave. Many published studies demonstrate the use of EHR data to predict individual patient outcomes, but few convert these predictions into numbers of beds needed. It is that second step that makes predictions meaningful for bed managers (King et al., 2022).

This package is intended to make it easier for you, as a researcher, to create point-in-time predictions. Its central tenet is the structuring of data into 'snapshots' of a hospital, where a patient snapshot captures an individual patient's state at a specific moment, and a cohort snapshot represents a collection of patient snapshots, for aggregate predictions. If your data is structured into snapshots, you can use your own patient-level models with convenient aggregation methods, based on symbolic mathematics, to produce cohort-level predictions. Use validation tools such as MADCAP plots and QQ plots to compare predicted cohort distributions against observed outcomes. There are functions for comparing prediction accuracy of your models against baseline approaches, such as the heuristics used by bed managers.

Our intention is that the package will help you demonstrate the practical value of your predictive models for hospital management. Step through a series of notebooks to learn how to use the package and see examples based on fake and synthetic data (King et al., 2024). You also have the option to download real patient data from Zenodo to use with the notebooks (King and Crowe, 2025). patientflow includes a fully worked example of how the package has been used in a live application at University College London Hospital to predict demand for emergency beds.

## Acknowledgements

The py-pi template developed by Tom Monks inspired us to create a Python package. This repository is based on a template developed by the Centre for Advanced Research Computing, University College London. We are grateful to Lawrence Lai for creation of the synthetic dataset.

The development of this repository/package was funded by UCL's QR Policy Support Fund, which is funded by Research England.

## References

King, Z., Farrington, J., Utley, M. et al. (2022). Machine learning for real-time aggregated prediction of hospital admission for emergency patients. npj Digital Medicine, 5(104). https://doi.org/10.1038/s41746-022-00649-y

King, Z., Gillham, J., Utley, M., Graham, M., and Crowe, S. (2025). patientflow (Version 0.4.3) [Computer software]. Python Package Index. https://pypi.org/project/patientflow/

King, Z., Gillham, J., Utley, M., Graham, M., and Crowe, S. (2024). patientflow: Code and explanatory notebooks for predicting short-term hospital bed demand using real-time data. GitHub. https://github.com/ucl-coru/patientflow

King, Z. and Crowe, S. (2025). Patient visits to the Emergency Department of an Acute Hospital; dataset to accompany the patientflow repository. Zenodo. https://doi.org/10.5281/zenodo.14866056 