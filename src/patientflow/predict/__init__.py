"""Prediction module for patient flow forecasting.

This module provides functions for making predictions about future patient flow,
including emergency demand forecasting and other predictive analytics.
"""

from patientflow.predict.service import (
    build_service_data,
    ServicePredictionInputs,
    compute_transfer_arrivals,
)

__all__ = [
    "build_service_data",
    "ServicePredictionInputs",
    "compute_transfer_arrivals",
]
