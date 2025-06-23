"""
Model training results containers.

This module defines a set of data classes to organise results from
model training, including hyperparameter tuning, cross-validation fold metrics,
and final trained classifier artifacts. These classes serve as structured containers for
various types of model evaluation outputs and metadata.

Classes
-------
HyperParameterTrial
    Container for storing hyperparameter tuning trial results.

FoldResults
    Stores evaluation metrics from a single cross-validation fold.

TrainingResults
    Encapsulates comprehensive evaluation metrics and metadata from model training.

TrainedClassifier
    Container for a trained model and associated training results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, Optional, Tuple
from sklearn.pipeline import Pipeline


@dataclass
class HyperParameterTrial:
    """
    Container for a single hyperparameter tuning trial.

    Attributes
    ----------
    parameters : dict of str to Any
        Dictionary of hyperparameters used in the trial.
    cv_results : dict of str to float
        Cross-validation metrics obtained using the specified parameters.
    """

    parameters: Dict[str, Any]
    cv_results: Dict[str, float]


@dataclass
class FoldResults:
    """
    Store evaluation metrics for a single fold.

    Attributes
    ----------
    auc : float
        Area Under the ROC Curve (AUC) for this fold.
    logloss : float
        Logarithmic loss (cross-entropy loss) for this fold.
    auprc : float
        Area Under the Precision-Recall Curve (AUPRC) for this fold.
    """

    auc: float
    logloss: float
    auprc: float


@dataclass
class TrainingResults:
    """
    Store comprehensive evaluation metrics and metadata from model training.

    Attributes
    ----------
    prediction_time : tuple of int
        Start and end time of prediction, represented as UNIX timestamps.
    training_info : dict of str to Any, optional
        Metadata or logs collected during training.
    calibration_info : dict of str to Any, optional
        Information about model calibration, if applicable.
    test_results : dict of str to float, optional
        Evaluation metrics computed on the test dataset. None if test evaluation was not performed.
    balance_info : dict of str to bool or int or float, optional
        Information related to class balance (e.g., whether data was balanced, class ratios).
    """

    prediction_time: Tuple[int, int]
    training_info: Dict[str, Any] = field(default_factory=dict)
    calibration_info: Dict[str, Any] = field(default_factory=dict)
    test_results: Optional[Dict[str, float]] = None
    balance_info: Dict[str, Union[bool, int, float]] = field(default_factory=dict)


@dataclass
class TrainedClassifier:
    """
    Container for trained model artifacts and their associated information.

    Attributes
    ----------
    training_results : TrainingResults
        Evaluation metrics and training metadata for the classifier.
    pipeline : sklearn.pipeline.Pipeline or None, optional
        The scikit-learn pipeline representing the trained classifier.
    calibrated_pipeline : sklearn.pipeline.Pipeline or None, optional
        The calibrated version of the pipeline, if model calibration was performed.
    """

    training_results: TrainingResults
    pipeline: Optional[Pipeline] = None
    calibrated_pipeline: Optional[Pipeline] = None
