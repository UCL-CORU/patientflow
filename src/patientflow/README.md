# PatientFlow: A Python package for converting patient-level predictions into output that is useful for bed managers in hospitals

The package will support predictions of bed demand and discharges by providing functions that

- predict patient-level probabilities of admission and discharge, by specialty
- create probability distributions predicting number of beds needed for or vacated by those patients, at different levels of aggregation
- return a net bed position by combining predictions of demand and supply of beds
- evaluate and provide visualisation of the performance of these predictions

The package is intended to serve as a wrapper of the functions typically used for such purposes in the `sklearn` and `scipy` python packages, with additional context to support their application and evaluation in bed management in healthcare.

For the full documentation, see the [API reference](https://ucl-coru.github.io/patientflow/)

## Modules Overview (in order of their use in a typical modelling workflow)

- `load`: A module for loading configuration files, saved data and trained models
- `prepare`: A module for preparing saved data prior to input into model training
- `train`: A module and submodules for training predictive models
- `calculate`: A module for calculating time-varying arrival rates, and probability of admission within a prediction window
- `predictors`: A module and submodules containing custom predictors developed for the `patientflow` package
- `predict`: A module using trained models for predicting various aspects of bed demand and discharges
- `aggregate`: A module that turns patient-level probabilities into aggregate distributions of bed numbers
- `evaluate`: A module that provides convenient functions for evaluating and comparing prediction models
- `viz`: A module containing convenient plotting functions to examine the outputs from the above functions

Two modules provide supporting

- `model_artifacts`: Defines a set of data classes to organise results from model training processes
- `errors` : Custom exception classes for model loading and validation

The following module has been used in the preparation of this repository, but are not core to the package:

- `convert`: Used for converting data from UCLH into the public dataset that is available from [Zenodo](https://zenodo.org/records/14866057)
- `generate` : Functions to generate fake datasets for patient visits to an emergency department (ED); used for illustrative purposes in some of the notebooks

Other modules may follow in future

## Deployment

This package is designed for use in hospital data projects analysing patient flow and bed capacity in short time horizons. The modules can be customised to align with specific hospital requirements
