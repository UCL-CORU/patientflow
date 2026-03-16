"""Shared constants for evaluation workflows.

Currently this module defines reliability thresholds used when recording
scalar outputs for classifier and distribution diagnostics.
"""

from typing import Dict


RELIABILITY_THRESHOLDS: Dict[str, int] = {
    "classifier_positive_cases": 50,
    "distribution_snapshots": 30,
    "transition_transfers": 10,
}
