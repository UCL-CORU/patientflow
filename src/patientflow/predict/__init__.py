"""Prediction module for patient flow forecasting.

This module provides functions for making predictions about future patient flow,
including emergency demand forecasting and other predictive analytics.
"""

from patientflow.predict.subspecialty import (
    build_subspecialty_data,
    SubspecialtyPredictionInputs,
    compute_transfer_arrivals,
    scale_pmf_by_probability,
    convolve_pmfs,
)

__all__ = [
    "build_subspecialty_data",
    "SubspecialtyPredictionInputs",
    "compute_transfer_arrivals",
    "scale_pmf_by_probability",
    "convolve_pmfs",
]
