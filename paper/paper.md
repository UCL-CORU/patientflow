---
title: 'patientflow: A Python package for predicting demand for hospital beds using real-time data'
tags:
  - Python
  - patient
  - hospital
  - bed
  - demand
  - real-time
  - electronic health records
authors:
  - name: Zella King
    orcid: 0000-0001-7389-1527
    affiliation: "1"
  - name: Jon Gillham
    orcid: 0009-0007-4110-7284
    affiliation: "2"
  - name: Martin Utley
    orcid: 0000-0001-9928-1516
    affiliation: "1"
  - name: Matt Graham
    affiliation: "1"
  - name: Sonya Crowe
    orcid: 0000-0003-1882-5476
    affiliation: "1"
affiliations:
 - name: Clinical Operational Research Unit (CORU), University College London, United Kingdom
   index: 1
 - name: Institute of Health Informatics, University College London, United Kingdom
   index: 2
date: 2024-03-19
bibliography: paper.bib

---

# Summary

patientflow, a Python package available on PyPi[@patientflow], converts patient-level predictions into output that is useful for bed managers in hospitals. If you want to predict a non-clinical outcome for a patient, such as admission or discharge from hospital, you can use patientflow to create bed count distributions for a cohort of patients. The package was developed for University College London Hospitals (UCLH) NHS Trust to predict the number of emergency admissions using real-time data. The methods generalise to any problem where it is useful to predict non-clinical outcomes for a cohort of patients at a point in time. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.	

# Statement of need

Hospital bed managers monitor whether they have sufficient beds to meet demand. At specific points during the day they predict numbers of inpatients likely to leave, and numbers of new admissions. These point-in-time predictions are important because, if bed managers anticipate a shortage of beds, they must take swift action to mitigate the situation. Commonly, bed managers use simple heuristics based on past admission and discharge patterns. Electronic Health Record (EHR) systems can offer superior predictions, grounded in real-time knowledge about patients coming in and those expected to leave. Many published studies demonstrate the use of EHR data to predict individual patient outcomes, but few convert these predictions into numbers of beds needed. It is that second step that makes predictions meaningful for bed managers [@king2022machine].

This package is intended to make it easier for you, as a researcher, to create point-in-time predictions. Its central tenet is the structuring of data into 'snapshots' of a hospital, where a patient snapshot captures an individual patient's state at a specific moment, and a cohort snapshot represents a collection of patient snapshots, for aggregate predictions. If your data is structured into snapshots, you can use your own patient-level models with convenient aggregation methods, based on symbolic mathematics, to produce cohort-level predictions. Use validation tools such as MADCAP plots and QQ plots to compare predicted cohort distributions against observed outcomes. There are functions for comparing prediction accuracy of your models against baseline approaches, such as the heuristics used by bed managers.

Our intention is that the package will help you demonstrate the practical value of your predictive models for hospital management. Step through a series of notebooks to learn how to use the package and see examples based on fake and synthetic data [@patientflow_github]. You also have the option to download real patient data from Zenodo to use with the notebooks [@patientflow_data]. patientflow includes a fully worked example of how the package has been used in a live application at University College London Hospital to predict demand for emergency beds. 

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References