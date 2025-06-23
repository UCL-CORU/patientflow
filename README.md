# patientflow: a Python package for real-time predictions of hospital bed demand from current and incoming patients

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![Documentation][docs-badge]][docs-link]
[![License][license-badge]](./LICENSE.md)
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15722296.svg)](https://doi.org/10.5281/zenodo.15722296)

[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7389--1527-green.svg)](https://orcid.org/0000-0001-7389-1527)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--4110--7284-green.svg)](https://orcid.org/0009-0007-4110-7284)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--9928--1516-green.svg)](https://orcid.org/0000-0001-9928-1516)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--9104--7960-green.svg)](https://orcid.org/0000-0001-9104-7960)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--1882--5476-green.svg)](https://orcid.org/0000-0003-1882-5476)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/zmek/patientflow/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/zmek/patientflow/actions/workflows/tests.yml
[linting-badge]:            https://github.com/zmek/patientflow/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/zmek/patientflow/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/zmek/patientflow/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/zmek/patientflow/actions/workflows/docs.yml
[docs-badge]:               https://img.shields.io/badge/docs-ucl--coru.github.io-blue
[docs-link]:                https://ucl-coru.github.io/patientflow/
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
[pypi-link]:                https://pypi.org/project/patientflow/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/patientflow
[pypi-version]:             https://img.shields.io/pypi/v/patientflow
<!-- prettier-ignore-end -->

## Summary

`patientflow`, a Python package for real-time prediction of hospital bed demand from current and incoming patients, creates output that is useful for bed managers in hospitals, allowing researchers to easily develop predictive models and demonstrate their utility to practitioners.

We originally developed this code for University College London Hospitals (UCLH) to predict the number of emergency admissions they should expect within the next eight hours. Our method used real-time data from their Electronic Health Record (EHR) system. We wrote code to convert patient-level data, extracted from the EHR at a point in time, into predicted numbers of admissions in the following hours. We also wrote code to help us evaluate the predictions.

We have created the `patientflow` python package to make it convenient for others to adopt our approach. Its purpose is to predict bed demand for groups of hospital patients at a point in time. The package is organised around the following concepts:

- Prediction time: A moment in the day at which predictions are to be made, for example 09:30.
- Patient snapshot: A summary of data from the EHR capturing what is known about a current patient at the prediction time. Each patient snapshot has a date and a prediction time associated with it.
- Group snapshot: The set of snapshots for a defined group of current patients. Each group snapshot has a date and a prediction time associated with it.
- Prediction window: A time period that begins at the prediction time.

For **current patients**, the package includes functions to create patient and group snapshots, to generate patient-level predictions, and to aggregate patient-level predictions into predicted bed counts for a group snapshots. The aggregation functions in `patientflow` are designed to receive a group snapshot as an input, and to predict something about that group's demand for beds between the prediction moment and the end of the prediction window. For example, that group could be the patients currently in the Emergency Department (ED), and the predictions could be the number of beds needed by those patients in the prediction window. The snapshot-based approach to predicting demand generalises to other aspects of patient flow in hospitals, such as predictions of how many current patients will be discharged from a clinical specialty.

For **incoming patients**, whose visits are not yet recorded in the EHR data (such as future arrivals to the ED) the aggregation functions make predictions based on past patterns of arrivals.

In both cases the output is a probability distribution over the number of beds needed. It is possible to create output at different levels of aggregation (for example by sex, or by clinical area), which bed managers find more actionable than whole-hospital predictions. The package includes functions to visualise the predicted probability distributions, and to evaluate them.

A series of notebooks demonstrates the use of the package. I show how to prepare your data and train models based on a snapshot approach. The repository includes a synthetic dataset, and an anonymised patient dataset, based on real data from UCLH is available on [Zenodo](https://zenodo.org/records/14866057). Both the synthetic and the real dataset have been prepared in a snapshot structure.

## Citation

If you use this software in your research, please cite it as:

**King, Zella. (2025). PatientFlow: Code and training materials for predicting short-term hospital bed capacity using real-time data (v1.0.3). Zenodo. https://doi.org/10.5281/zenodo.15722296**

BibTeX:

```bibtex
@software{king_patientflow_2025,
  title = {PatientFlow: Code and training materials for predicting short-term hospital bed capacity using real-time data},
  author = {King, Zella},
  year = {2025},
  month = {6},
  version = {v1.0.3},
  doi = {10.5281/zenodo.15722296},
  url = {https://doi.org/10.5281/zenodo.15722296},
  publisher = {Zenodo}
}
```

### Dataset

The accompanying dataset is available at: [https://doi.org/10.5281/zenodo.15311282](https://doi.org/10.5281/zenodo.15311282)

**King, Zella, University College London Hospitals NHS Foundation Trust, & Crowe, Sonya. (2025). Patient visits to the Emergency Department of an Acute Hospital; dataset to accompany the patientflow repository (Version 1.1.1). Zenodo. https://doi.org/10.5281/zenodo.15311282**

BibTeX:

```bibtex
@dataset{king_patient_visits_2025,
  title = {Patient visits to the Emergency Department of an Acute Hospital; dataset to accompany the patientflow repository},
  author = {King, Zella and Crowe, Sonya},
  year = {2025},
  month = {4},
  version = {1.1.1},
  doi = {10.5281/zenodo.15311282},
  url = {https://doi.org/10.5281/zenodo.15311282},
  publisher = {Zenodo}
}
```

## Documentation

Documentation is available at [ucl-coru.github.io/patientflow](https://ucl-coru.github.io/patientflow/). The full API reference is [here](https://ucl-coru.github.io/patientflow/api/).

## What `patientflow` is for:

- Predicting patient flow in hospitals: The package can be used by researchers or analysts who want to predict numbers of emergency admissions, discharges, transfers between units or combinations of these
- Short-term operational planning: The predictions produced by this package are designed for bed managers who need to make decisions within a short timeframe (up to 24 hours, but not days or weeks).
- Working with real-time data: The design assumes that data from an electronic health record (EHR) is available in real-time, or near to real-time.
- Point-in-time analysis: For cohorts of hospital patients at different stages of a hospital visit, the package can be used to make mid-visit predictions about whether a non-clinical event like admission or discharge will occur within a short time horizon.

## What `patientflow` is NOT for:

- Long-term capacity planning: The package focuses on short-term operational demand (hours ahead), not strategic planning over weeks or months.
- Making decisions about individual patients: The package relies on data entered into the EHR by clinical staff looking after patients, but the patient-level predictions it generates cannot and should not be used to influence their decision-making.
- Predicting what happens _after_ a hospital visit has finished: While historical data might train underlying models, the package itself focuses on patients currently in the hospital or soon to arrive.
- Replacing human judgment: The predictions are meant to augment the information available to bed managers, but not to automate bed management decisions.

## This package will help you if you want to:

- Make predictions for unfinished patient visits, using real-time data.
- Attract the attention of hospital managers in your predictions; since the output is bed numbers, a currency they use daily, they may find it more actionable than typical predictive modelling output, especially if you can break it down by clinical area.
- Develop your own emergency bed modelling application - the repository includes a fully worked example of how we have used the package at UCLH - or an adjacent appplication such as one predicting how many patients will be discharged.

## This package will NOT help you if:

- You work with time series data: `patientflow` works with snapshots of a hospital visit summarising what is in the patient record up to that point in time. It would need modification to accept time series data formats.
- You want to predict clinical outcomes: the approach is designed for the management of hospital sites, not the management of patient care.

## Mathematical assumptions underlying the conversion from individual to group predictions:

- Independence of patient journeys: The package assumes that an individual patient's presence in one part of ahospital system is independent of patients elsewhere
- Bernoulli outcome model: Each patient outcome is modelled as a Bernoulli trial with its own probability, and the package computes a probability distribution for the sum of these independent trials.
- Different levels of aggregation: The package can calculate probabilities for compound events (such as the probability of a patient being admitted, assigned to a specific specialty if admitted, and being admitted within the prediction window) and separate distributions for patient subgroups (like distributions by age or gender). In all cases, the independence assumption between patients is maintained.

## Getting started

- Exploration: Start with the [notebooks README](https://github.com/UCL-CORU/patientflow/blob/main/notebooks/README.md) to get an outline of what is included in the notebooks, and read the [package README](https://github.com/UCL-CORU/patientflow/tree/main/src/patientflow#readme) or the [documentation](https://ucl-coru.github.io/patientflow) for an overview of the Python package.
- Installation: Follow the instructions below to set up the environment and install necessary dependencies in your own environment.
- Configuration: Repurpose config.yaml to configure the package to your own data and user requirements.

### Prerequisites

`patientflow` requires Python 3.10.

### Installation

You can install `patientflow` directly from PyPI:

```sh
pip install patientflow
```

To access the example notebooks and synthetic data, clone the repository:

```sh
git clone https://github.com/ucl-coru/patientflow.git
cd patientflow
```

### Development Installation (optional)

If you want to contribute or modify the code, or run documentation locally, install the development and documentation dependencies:

```sh
# For contributors (includes development tools, documentation, and testing)
pip install -e ".[dev,docs,test]"

# For specific purposes only:
# For development tools (linting, formatting, etc.)
pip install -e ".[dev]"

# For building documentation
pip install -e ".[docs]"

# For running tests
pip install -e ".[test]"
```

Navigate to the patientflow folder and run tests to confirm that the installation worked correctly. This command will only work from the root repository. (To date, this has only been tested on Linux and Mac OS machines. If you are running Windows, there may be errors we don't know about. Please raise an issue on Github in that case.)

```sh
pytest
```

If you get errors running the pytest command, there may be other installations needed on your local machine.

### Building and Viewing Documentation (optional)

After installing the documentation dependencies, you can build and view the documentation locally:

```sh
# Build and serve the documentation with live reloading
mkdocs serve

# Or just build the documentation
mkdocs build
```

The documentation will be available at http://127.0.0.1:8000/ when using `mkdocs serve`.

## Using the notebooks in this repository

The notebooks in this repository demonstrate the use of some of the functions provided in `patientflow`. The cell output shows the results of running the notebooks. If you want to run them yourself, you have two options

- step through the notebooks using the real patient datasets that were used to prepare them. For this you need to request access on [Zenodo](https://doi.org/10.5281/zenodo.15311282) to real patient data
- step through the notebooks using synthetic data. You will need to copy the two csv files from `data-synthetic`into your `data-public` folder or change the source in the each notebook. If you use synthetic data, you will not see the same cell output.

## About the UCLH implementation

This repository includes a set of notebooks (prefixed with 4) that show a fully worked example of the implementation of the patientflow package at University College London Hospitals (UCLH). As noted above, please request access to the UCLH dataset via [Zenodo](https://doi.org/10.5281/zenodo.15311282).

There is also a Python script that illustrates the training of the models that predict emergency demand at UCLH and saves them in your local environment using following commands (by default this will run with the synthetic data in its current location; change the `data_folder_name` parameter if you have downloaded the Zenodo dataset in `data-public`)

```sh
cd src
python -m patientflow.train.emergency_demand --data_folder_name=data-synthetic
```

The `data_folder_name`argument specifies the name of the folder containing data. The function expects this folder to be directly below the root of the repository.

## Contributing to PatientFlow

We welcome contributions to the patientflow project. To contribute, follow the instructions below.

### Development Workflow

1. Fork the repository on GitHub
2. Clone your fork locally and set up your development environment following the [installation instructions](#installation) above, making sure to install the development dependencies.
3. Create a new branch for your changes:
   ```sh
   git checkout -b feature/your-feature-name
   ```
4. Make your changes following the code style guidelines
5. Run tests as described in the installation section to ensure your changes don't break existing functionality
6. Update documentation if needed using the documentation tools mentioned above
7. Commit your changes with a descriptive message

### Code Style Guidelines

- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate
- Write docstrings for all functions, classes, and modules
- Add unit tests for new functionality

### Submitting Your Contribution

1. Push your changes to your forked repository:
   ```sh
   git push origin feature/your-feature-name
   ```
2. Open a pull request from your fork to the main repository
   - Provide a clear title and description
   - Reference any relevant issues
   - Include screenshots if applicable
3. Address any feedback or review comments

### Reporting Issues

If you find a bug or have a suggestion:

1. Check existing issues to avoid duplicates
2. Open a new issue describing:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce
   - Your environment details (OS, Python version, etc.)

Thank you for contributing!

## Roadmap

- [x] Initial Research
- [x] Minimum viable product
- [x] Alpha Release (PyPI Package) <-- You are Here
- [ ] Feature-Complete Release

### Project Team

- [Dr Zella King](https://github.com/zmek), Clinical Operational Research Unit (CORU), University College London (UCL)([zella.king@ucl.ac.uk](mailto:zella.king@ucl.ac.uk))
- [Jon Gillham](https://github.com/jongillham), Institute of Health Informatics, UCL
- Professor Martin Utley, Clinical Operational Research Unit, UCL
- Matt Graham, Advanced Research Computing, UCL
- Professor Sonya Crowe, Clinical Operational Research Unit, UCL

## Acknowledgements

The [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by [Tom Monks](https://github.com/TomMonks) inspired us to create a Python package. This repository is based on a template developed by the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London. We are grateful to [Lawrence Lai](https://github.com/lawrencelai) for creation of the synthetic dataset, and to Sara Lundell for her extensive work piloting the package for use at Sahlgrenska University Hospital, Gothenburg, Sweden.

The development of this repository/package was funded by UCL's QR Policy Support Fund, which is funded by [Research England](https://www.ukri.org/councils/research-england/).
