"""
PatientFlow: A package for predicting short-term hospital bed demand.

This package provides tools and models for analysing patient flow data and
making predictions about emergency demand, elective demand, and hospital discharges.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"  # Fallback for when not installed

__author__ = "Zella King"
__email__ = "zella.king@ucl.ac.uk"
