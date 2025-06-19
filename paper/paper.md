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
  - name: Sara Lundell
    orcid:
    affiliation: "3"
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
  - name: Sahlgrenska University Hospital, Göteborg, Sweden
    index: 3
date: 2025-06-02
bibliography: paper.bib
---

# patientflow: a Python package for real-time predictions of hospital bed demand from current and incoming patients

# Summary

patientflow is a Python package available on PyPi[@patientflow] for real-time prediction of hospital bed demand from current and incoming patients. It enables researchers to easily develop predictive models and demonstrate their utility to practitioners. Researchers can use it to prepare data sets for predictive modelling, generate patient level predictions of admission, discharge or transfer, and then combine patient-level predictions at different levels of aggregation to give output that is useful for bed managers. The package was developed for University College London Hospitals (UCLH) NHS Trust to predict demand for emergency beds using real-time data. The methods generalise to any problem where it is useful to predict non-clinical outcomes for a cohort of patients at a point in time. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.

# Statement of need

Hospital bed managers monitor whether they have sufficient beds to meet demand. At specific points during the day they predict numbers of inpatients likely to leave and numbers of new admissions. These predictions are important because, if bed managers anticipate a shortage of beds, they must take swift action to mitigate the situation. Commonly, bed managers use simple heuristics based on past admission and discharge patterns. Electronic Health Record (EHR) systems can offer superior predictions, grounded in real-time knowledge about patients currently in the hospital.

Many studies demonstrate the use of EHR data to predict individual patient outcomes, but few harness such predictive models to methods for estimating aggregate outcomes for cohorts of patients. In the context of predicting bed demand, it is this aggregate level that is most meaningful for bed managers [@king2022machine]. Note that by design, we provide methods to estimate unfettered demand for beds to inform decision-making [@worthington2019infinite].

This package is intended to make it easier for researchers to create such predictions. Its central tenet is the structuring of data into 'snapshots' of a hospital, where a patient snapshot captures what data are available on a current patient's state at a specific moment, and a cohort snapshot represents a collection of patient snapshots, for aggregate predictions. Notebooks in the Github repository demonstrate how to use the package to create patient and group snapshots from EHR data. Once data is structured into snapshots, researchers can use their own patient-level models with our analytical methods to produce cohort-level predictions. The package provides tools to compare predicted distributions against observations.

Our intention is that the patientflow package will help researchers demonstrate the practical value of their predictive models for hospital management. Notebooks in the accompanying repository show examples based on fake and synthetic data [@patientflow_github]. Researchers also have the option to download real patient data from Zenodo to use with the notebooks [@patientflow_data]. The repository includes a fully worked example of how the package has been used in a live application at University College London Hospital to predict demand for emergency beds.

# Related software

Simulation is a common approach for modelling patient flow, and there are various packages to support that, such as PathSimR for R [@tyler2022improving] and sim-tools [@monks2023improving] and ActaPatientFlow [@szabo2024patient] for Python.

To our knowledge, there are no packages that support the use of real-time patient data with a specific focus on output that can help healthcare managers respond to changes as they arise. Our intention for patientflow is to support the development of patient level predictive models and the use of real-time data, conbined with a mathematical approach to calculating distributions of aggregate demand. Taking a mathematical approach provides quicker and more accurate results than deploying simulation for the same task.

# Acknowledgements

The PyPi template developed by Tom Monks inspired us to create a Python package. This repository is based on a template developed by the Centre for Advanced Research Computing, University College London. We are grateful to Lawrence Lai for creation of the synthetic dataset.

The development of this repository/package was funded by UCL's QR Policy Support Fund, which is funded by Research England.

# References
