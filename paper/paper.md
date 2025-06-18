---
title: "patientflow: a Python package for real-time prediction of hospital bed demand from current and incoming patients"
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
    orcid: 0000-0001-9104-7960
    affiliation: "1"
  - name: Sonya Crowe
    orcid: 0000-0003-1882-5476
    affiliation: "1"
affiliations:
  - name: Clinical Operational Research Unit (CORU), University College London, United Kingdom
    index: 1
  - name: Institute of Health Informatics, University College London, United Kingdom
    index: 2
date: 2025-06-02
bibliography: paper.bib
---

# patientflow: a Python package for real-time prediction of hospital bed demand from current and incoming patients

# Summary

patientflow is a Python package available on PyPi[@patientflow] for real-time prediction of hospital bed demand from current and incoming patients. It creates output that is useful for bed managers in hospitals, enabling researchers to easily develop predictive models and demonstrate their utility to practitioners. Researchers can use it to prepare data sets for predictive modelling, generate patient level predictions of admission, discharge or transfer, and then combine patient-level predictions at different levels of aggregation to give output that is useful for bed managers. The package was developed for University College London Hospitals (UCLH) NHS Trust to predict the number of emergency admissions using real-time data. The methods generalise to any problem where it is useful to predict non-clinical outcomes for a cohort of patients at a point in time. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.

# Statement of need

Hospital bed managers monitor whether they have sufficient beds to meet demand. At specific points during the day they predict numbers of inpatients likely to leave, and numbers of new admissions. These point-in-time predictions are important because, if bed managers anticipate a shortage of beds, they must take swift action to mitigate the situation. Commonly, bed managers use simple heuristics based on past admission and discharge patterns. Electronic Health Record (EHR) systems can offer superior predictions, grounded in real-time knowledge about patients coming in and those expected to leave. 

In prior research, many studies demonstrate the use of EHR data to predict individual patient outcomes, but few convert these predictions into numbers of beds needed within a short time horizon. It is that second step that makes predictions meaningful for bed managers [@king2022machine].

This package is intended to make it easier for researchers to create such point-in-time predictions. Its central tenet is the structuring of data into 'snapshots' of a hospital, where a patient snapshot captures a current patient's state at a specific moment, and a cohort snapshot represents a collection of patient snapshots, for aggregate predictions. Notebooks in the Github repository demonstrate how to use the package to create patient and group snapshots from EHR data. Once data is structured into snapshots, researchers can use their own patient-level models with convenient aggregation methods, based on symbolic mathematics, to produce cohort-level predictions. The package provides validation tools to compare predicted cohort distributions against observed outcomes. 

Our intention is that the patientflow package will help researchers demonstrate the practical value of their predictive models for hospital management. Notebooks in the accompanying repository show examples based on fake and synthetic data [@patientflow_github]. Researchers also have the option to download real patient data from Zenodo to use with the notebooks [@patientflow_data]. The repository includes a fully worked example of how the package has been used in a live application at University College London Hospital to predict demand for emergency beds. 

# Related software

Simulation is a common approach for modelling patient flow, and there are various packages to support that, such as PathSimR for R [@tyler2022improving] and sim-tools [@monks2023sim] and ActaPatientFlow [@szabo2024patient] for Python.

To our knowledge, there are no packages that support the use of real-time patient data with a specific focus on output that can help healthcare managers respond to changes as they arise. Our intention for patientflow is to support the use of real-time data from an EHR, and to show how, with appropriate mathematical assumptions, exact probability distributions for bed demand can be computed analytically rather than approximated through simulation. 

# Acknowledgements

The py-pi template developed by Tom Monks inspired us to create a Python package. This repository is based on a template developed by the Centre for Advanced Research Computing, University College London. We are grateful to Lawrence Lai for creation of the synthetic dataset, and to Sara Lundell for her extensive work piloting the package for use in Sahlgrenska University Hospital, Gothenburg, Sweden. 

The development of this repository/package was funded by UCL's QR Policy Support Fund, which is funded by Research England.

# References
