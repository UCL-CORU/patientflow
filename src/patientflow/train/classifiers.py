"""
Machine learning classifiers for patient flow prediction.

This module provides functions for training and evaluating machine learning
classifiers for patient admission prediction. It includes utilities for
data preparation, model training, hyperparameter tuning, and evaluation
using time series cross-validation.

Functions
---------
evaluate_predictions
    Calculate multiple metrics (AUC, log loss, AUPRC) for given predictions
chronological_cross_validation
    Perform time series cross-validation with multiple metrics
initialise_model
    Initialize a model with given hyperparameters
create_column_transformer
    Create a column transformer for a dataframe with dynamic column handling
calculate_class_balance
    Calculate class balance ratios for target labels
get_feature_metadata
    Extract feature names and importances from pipeline
get_dataset_metadata
    Get dataset sizes and class balances
create_balance_info
    Create a dictionary with balance information
evaluate_model
    Evaluate model on test set
train_classifier
    Train a single model including data preparation and balancing
train_multiple_classifiers
    Train admission prediction models for multiple prediction times
"""

from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, Type
import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier
from pandas import DataFrame, Series
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn import __version__ as sk_version


from patientflow.prepare import prepare_patient_snapshots
from patientflow.load import get_model_key
from patientflow.model_artifacts import (
    HyperParameterTrial,
    FoldResults,
    TrainingResults,
    TrainedClassifier,
)


def evaluate_predictions(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.float64]
) -> FoldResults:
    """Calculate multiple metrics for given predictions.

    Parameters
    ----------
    y_true : npt.NDArray[np.int_]
        True binary labels
    y_pred : npt.NDArray[np.float64]
        Predicted probabilities

    Returns
    -------
    FoldResults
        Object containing AUC, log loss, and AUPRC metrics
    """
    return FoldResults(
        auc=roc_auc_score(y_true, y_pred),
        logloss=log_loss(y_true, y_pred),
        auprc=average_precision_score(y_true, y_pred),
    )


def chronological_cross_validation(
    pipeline: Pipeline, X: DataFrame, y: Series, n_splits: int = 5
) -> Dict[str, float]:
    """Perform time series cross-validation with multiple metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline to evaluate
    X : DataFrame
        Feature matrix
    y : Series
        Target labels
    n_splits : int, optional
        Number of time series splits, by default 5

    Returns
    -------
    Dict[str, float]
        Dictionary containing training and validation metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    train_metrics: List[FoldResults] = []
    valid_metrics: List[FoldResults] = []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict_proba(X_train)[:, 1]
        valid_preds = pipeline.predict_proba(X_valid)[:, 1]

        train_metrics.append(evaluate_predictions(y_train, train_preds))
        valid_metrics.append(evaluate_predictions(y_valid, valid_preds))

    def aggregate_metrics(metrics_list: List[FoldResults]) -> Dict[str, float]:
        return {
            field: np.mean([getattr(m, field) for m in metrics_list])
            for field in FoldResults.__dataclass_fields__
        }

    train_means = aggregate_metrics(train_metrics)
    valid_means = aggregate_metrics(valid_metrics)

    return {f"train_{metric}": value for metric, value in train_means.items()} | {
        f"valid_{metric}": value for metric, value in valid_means.items()
    }


def initialise_model(
    model_class: Type,
    params: Dict[str, Any],
    xgb_specific_params: Dict[str, Any] = {
        "n_jobs": -1,
        "eval_metric": "logloss",
        "enable_categorical": True,
    },
) -> Any:
    """
    Initialize a model with given hyperparameters.

    Parameters
    ----------
    model_class : Type
        The classifier class to instantiate
    params : Dict[str, Any]
        Model-specific parameters to set
    xgb_specific_params : Dict[str, Any], optional
        XGBoost-specific default parameters

    Returns
    -------
    Any
        Initialized model instance
    """
    if model_class == XGBClassifier:
        model = model_class(**xgb_specific_params)
        model.set_params(**params)
    else:
        model = model_class(**params)

    return model


def create_column_transformer(
    df: DataFrame, ordinal_mappings: Optional[Dict[str, List[Any]]] = None
) -> ColumnTransformer:
    """Create a column transformer for a dataframe with dynamic column handling.

    Parameters
    ----------
    df : DataFrame
        Input dataframe
    ordinal_mappings : Dict[str, List[Any]], optional
        Mappings for ordinal categorical features, by default None

    Returns
    -------
    ColumnTransformer
        Configured column transformer
    """
    transformers: List[
        Tuple[str, Union[OrdinalEncoder, OneHotEncoder, StandardScaler], List[str]]
    ] = []

    if ordinal_mappings is None:
        ordinal_mappings = {}

    for col in df.columns:
        if col in ordinal_mappings:
            transformers.append(
                (
                    col,
                    OrdinalEncoder(
                        categories=[ordinal_mappings[col]],
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                    ),
                    [col],
                )
            )
        elif df[col].dtype == "object" or (
            df[col].dtype == "bool" or df[col].nunique() == 2
        ):
            transformers.append((col, OneHotEncoder(handle_unknown="ignore"), [col]))
        else:
            transformers.append((col, StandardScaler(), [col]))

    return ColumnTransformer(transformers)


def calculate_class_balance(y: Series) -> Dict[Any, float]:
    """Calculate class balance ratios for target labels.

    Parameters
    ----------
    y : Series
        Target labels

    Returns
    -------
    Dict[Any, float]
        Dictionary mapping each class to its proportion
    """
    counter = Counter(y)
    total = len(y)
    return {cls: count / total for cls, count in counter.items()}


class FeatureMetadata(TypedDict):
    feature_names: List[str]
    feature_importances: List[float]


class DatasetMetadata(TypedDict):
    train_valid_test_set_no: Dict[str, Optional[int]]
    train_valid_test_class_balance: Dict[str, Optional[Dict[Any, float]]]


def get_feature_metadata(pipeline: Pipeline) -> FeatureMetadata:
    """
    Extract feature names and importances from pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline containing feature transformer and classifier

    Returns
    -------
    FeatureMetadata
        Dictionary containing feature names and their importance scores (if available)

    Raises
    ------
    AttributeError
        If the classifier doesn't support feature importance
    """
    transformed_cols = pipeline.named_steps[
        "feature_transformer"
    ].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    # Try different common feature importance attributes
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = (
            np.abs(classifier.coef_[0])
            if classifier.coef_.ndim > 1
            else np.abs(classifier.coef_)
        )
    else:
        raise AttributeError("Classifier doesn't provide feature importance scores")

    return {
        "feature_names": [col.split("__")[-1] for col in transformed_cols],
        "feature_importances": importances.tolist(),
    }


def get_dataset_metadata(
    X_train: DataFrame,
    X_valid: DataFrame,
    y_train: Series,
    y_valid: Series,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[Series] = None,
) -> DatasetMetadata:
    """Get dataset sizes and class balances.

    Parameters
    ----------
    X_train : DataFrame
        Training features
    X_valid : DataFrame
        Validation features
    y_train : Series
        Training labels
    y_valid : Series
        Validation labels
    X_test : DataFrame, optional
        Test features. If None, test set information will be set to None.
    y_test : Series, optional
        Test labels. If None, test set information will be set to None.

    Returns
    -------
    DatasetMetadata
        Dictionary containing dataset sizes and class balances
    """
    metadata: DatasetMetadata = {
        "train_valid_test_set_no": {
            "train_set_no": len(X_train),
            "valid_set_no": len(X_valid),
            "test_set_no": len(X_test) if X_test is not None else None,
        },
        "train_valid_test_class_balance": {
            "y_train_class_balance": calculate_class_balance(y_train),
            "y_valid_class_balance": calculate_class_balance(y_valid),
            "y_test_class_balance": calculate_class_balance(y_test)
            if y_test is not None
            else None,
        },
    }

    return metadata


def create_balance_info(
    is_balanced: bool,
    original_size: int,
    balanced_size: int,
    original_positive_rate: float,
    balanced_positive_rate: float,
    majority_to_minority_ratio: float,
) -> Dict[str, Union[bool, int, float]]:
    """Create a dictionary with balance information.

    Parameters
    ----------
    is_balanced : bool
        Whether the dataset was balanced
    original_size : int
        Original dataset size
    balanced_size : int
        Size after balancing
    original_positive_rate : float
        Positive class rate before balancing
    balanced_positive_rate : float
        Positive class rate after balancing
    majority_to_minority_ratio : float
        Ratio of majority to minority class samples

    Returns
    -------
    Dict[str, Union[bool, int, float]]
        Dictionary containing balance information
    """
    return {
        "is_balanced": is_balanced,
        "original_size": original_size,
        "balanced_size": balanced_size,
        "original_positive_rate": original_positive_rate,
        "balanced_positive_rate": balanced_positive_rate,
        "majority_to_minority_ratio": majority_to_minority_ratio,
    }


def evaluate_model(
    pipeline: Pipeline, X_test: DataFrame, y_test: Series
) -> Dict[str, float]:
    """Evaluate model on test set.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels

    Returns
    -------
    Dict[str, float]
        Dictionary containing test metrics
    """
    y_test_pred = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_auc": float(roc_auc_score(y_test, y_test_pred)),
        "test_logloss": float(log_loss(y_test, y_test_pred)),
        "test_auprc": float(average_precision_score(y_test, y_test_pred)),
    }


def train_classifier(
    train_visits: DataFrame,
    valid_visits: DataFrame,
    prediction_time: Tuple[int, int],
    exclude_from_training_data: List[str],
    grid: Dict[str, List[Any]],
    ordinal_mappings: Dict[str, List[Any]],
    test_visits: Optional[DataFrame] = None,
    visit_col: Optional[str] = None,
    model_class: Type = XGBClassifier,
    use_balanced_training: bool = True,
    majority_to_minority_ratio: float = 1.0,
    calibrate_probabilities: bool = True,
    calibration_method: str = "sigmoid",
    single_snapshot_per_visit: bool = True,
    label_col: str = "is_admitted",
    evaluate_on_test: bool = False,
) -> TrainedClassifier:
    """
    Train a single model including data preparation and balancing.

    Parameters
    ----------
    train_visits : DataFrame
        Training visits dataset
    valid_visits : DataFrame
        Validation visits dataset
    prediction_time : Tuple[int, int]
        The prediction time point to use
    exclude_from_training_data : List[str]
        Columns to exclude from training
    grid : Dict[str, List[Any]]
        Parameter grid for hyperparameter tuning
    ordinal_mappings : Dict[str, List[Any]]
        Mappings for ordinal categorical features
    test_visits : DataFrame, optional
        Test visits dataset. Required only when evaluate_on_test=True.
    visit_col : str, optional
        Name of the visit column. Required if single_snapshot_per_visit is True.
    model_class : Type, optional
        The classifier class to use. Must be sklearn-compatible with fit() and predict_proba().
        Defaults to XGBClassifier.
    use_balanced_training : bool, default=True
        Whether to use balanced training data
    majority_to_minority_ratio : float, default=1.0
        Ratio of majority to minority class samples
    calibrate_probabilities : bool, default=True
        Whether to apply probability calibration to the best model
    calibration_method : str, default='sigmoid'
        Method for probability calibration ('isotonic' or 'sigmoid')
    single_snapshot_per_visit : bool, default=True
        Whether to select only one snapshot per visit. If True, visit_col must be provided.
    label_col : str, default="is_admitted"
        Name of the column containing the target labels
    evaluate_on_test : bool, default=False
        Whether to evaluate the final model on the test set. Set to True only when
        satisfied with validation performance to avoid test set contamination.

    Returns
    -------
    TrainedClassifier
        Trained model, including metrics, and feature information

    """
    if single_snapshot_per_visit and visit_col is None:
        raise ValueError(
            "visit_col must be provided when single_snapshot_per_visit is True"
        )

    if evaluate_on_test and test_visits is None:
        raise ValueError("test_visits must be provided when evaluate_on_test=True")

    # Get snapshots for each set
    X_train, y_train = prepare_patient_snapshots(
        train_visits,
        prediction_time,
        exclude_from_training_data,
        visit_col=visit_col,
        single_snapshot_per_visit=single_snapshot_per_visit,
        label_col=label_col,
    )
    X_valid, y_valid = prepare_patient_snapshots(
        valid_visits,
        prediction_time,
        exclude_from_training_data,
        visit_col=visit_col,
        single_snapshot_per_visit=single_snapshot_per_visit,
        label_col=label_col,
    )

    # Only prepare test data if evaluation is requested
    if evaluate_on_test:
        X_test, y_test = prepare_patient_snapshots(
            test_visits,
            prediction_time,
            exclude_from_training_data,
            visit_col=visit_col,
            single_snapshot_per_visit=single_snapshot_per_visit,
            label_col=label_col,
        )
    else:
        X_test, y_test = None, None

    # Get dataset metadata before any balancing
    dataset_metadata = get_dataset_metadata(
        X_train, X_valid, y_train, y_valid, X_test, y_test
    )

    # Store original size and positive rate before any balancing
    original_size = len(X_train)
    original_positive_rate = y_train.mean()

    if use_balanced_training:
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index

        n_pos = len(pos_indices)
        n_neg = int(n_pos * majority_to_minority_ratio)

        neg_indices_sampled = np.random.choice(
            neg_indices, size=min(n_neg, len(neg_indices)), replace=False
        )

        train_balanced_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(train_balanced_indices)

        X_train = X_train.loc[train_balanced_indices]
        y_train = y_train.loc[train_balanced_indices]

    # Create balance info after any balancing is done
    balance_info = create_balance_info(
        is_balanced=use_balanced_training,
        original_size=original_size,
        balanced_size=len(X_train),
        original_positive_rate=original_positive_rate,
        balanced_positive_rate=y_train.mean(),
        majority_to_minority_ratio=majority_to_minority_ratio
        if use_balanced_training
        else 1.0,
    )

    # Initialize best training results with default values
    best_training = TrainingResults(
        prediction_time=prediction_time,
        balance_info=balance_info,
        # Other fields will use their default empty dictionaries
    )

    # Initialize best model container
    best_model = TrainedClassifier(
        training_results=best_training,
        pipeline=None,
        calibrated_pipeline=None,
    )

    trials_list: List[HyperParameterTrial] = []
    best_logloss = float("inf")

    for params in ParameterGrid(grid):
        # Initialize model based on provided class
        model = initialise_model(model_class, params)

        column_transformer = create_column_transformer(X_train, ordinal_mappings)
        pipeline = Pipeline(
            [("feature_transformer", column_transformer), ("classifier", model)]
        )

        cv_results = chronological_cross_validation(
            pipeline, X_train, y_train, n_splits=5
        )
        # Store trial results
        trials_list.append(
            HyperParameterTrial(
                parameters=params.copy(),  # Make a copy to ensure immutability
                cv_results=cv_results,
            )
        )

        if cv_results["valid_logloss"] < best_logloss:
            best_logloss = cv_results["valid_logloss"]
            best_model.pipeline = pipeline

            # Get feature metadata if available
            try:
                feature_metadata = get_feature_metadata(pipeline)
                has_feature_importance = True
            except (AttributeError, NotImplementedError):
                feature_metadata = {
                    "feature_names": column_transformer.get_feature_names_out().tolist(),
                    "feature_importances": [],
                }
                has_feature_importance = False

            # Update training results
            best_training.training_info = {
                "cv_trials": trials_list,
                "features": {
                    "names": feature_metadata["feature_names"],
                    "importances": feature_metadata["feature_importances"],
                    "has_importance_values": has_feature_importance,
                },
                "dataset_info": dataset_metadata,
            }

            if calibrate_probabilities:
                best_training.calibration_info = {"method": calibration_method}

    # Apply probability calibration to the best model if requested
    if calibrate_probabilities and best_model.pipeline is not None:
        best_feature_transformer = best_model.pipeline.named_steps[
            "feature_transformer"
        ]
        best_classifier = best_model.pipeline.named_steps["classifier"]

        X_valid_transformed = best_feature_transformer.transform(X_valid)

        if sk_version >= "1.6.0":
            from sklearn.frozen import FrozenEstimator

            calibrated_classifier = CalibratedClassifierCV(
                estimator=FrozenEstimator(best_classifier),
                method=calibration_method,
            )
        else:
            calibrated_classifier = CalibratedClassifierCV(
                estimator=best_classifier, method=calibration_method, cv="prefit"
            )
        calibrated_classifier.fit(X_valid_transformed, y_valid)

        calibrated_pipeline = Pipeline(
            [
                ("feature_transformer", best_feature_transformer),
                ("classifier", calibrated_classifier),
            ]
        )

        best_model.calibrated_pipeline = calibrated_pipeline

        # Only evaluate on test set if requested
        if evaluate_on_test:
            best_training.test_results = evaluate_model(
                calibrated_pipeline, X_test, y_test
            )
        else:
            best_training.test_results = None

    else:
        # Only evaluate on test set if requested
        if evaluate_on_test:
            best_training.test_results = evaluate_model(
                best_model.pipeline, X_test, y_test
            )
        else:
            best_training.test_results = None

    return best_model


def train_multiple_classifiers(
    train_visits: DataFrame,
    valid_visits: DataFrame,
    grid: Dict[str, List[Any]],
    exclude_from_training_data: List[str],
    ordinal_mappings: Dict[str, List[Any]],
    prediction_times: List[Tuple[int, int]],
    test_visits: Optional[DataFrame] = None,
    model_name: str = "admissions",
    visit_col: str = "visit_number",
    calibrate_probabilities: bool = True,
    calibration_method: str = "isotonic",
    use_balanced_training: bool = True,
    majority_to_minority_ratio: float = 1.0,
    label_col: str = "is_admitted",
    evaluate_on_test: bool = False,
) -> Dict[str, TrainedClassifier]:
    """Train admission prediction models for multiple prediction times.

    Parameters
    ----------
    train_visits : DataFrame
        Training visits dataset
    valid_visits : DataFrame
        Validation visits dataset
    grid : Dict[str, List[Any]]
        Parameter grid for hyperparameter tuning
    exclude_from_training_data : List[str]
        Columns to exclude from training
    ordinal_mappings : Dict[str, List[Any]]
        Mappings for ordinal categorical features
    prediction_times : List[Tuple[int, int]]
        List of prediction time points
    test_visits : DataFrame, optional
        Test visits dataset, by default None
    model_name : str, optional
        Name prefix for models, by default "admissions"
    visit_col : str, optional
        Name of the visit column, by default "visit_number"
    calibrate_probabilities : bool, optional
        Whether to calibrate probabilities, by default True
    calibration_method : str, optional
        Calibration method, by default "isotonic"
    use_balanced_training : bool, optional
        Whether to use balanced training, by default True
    majority_to_minority_ratio : float, optional
        Ratio for class balancing, by default 1.0
    label_col : str, optional
        Name of the label column, by default "is_admitted"
    evaluate_on_test : bool, optional
        Whether to evaluate on test set, by default False

    Returns
    -------
    Dict[str, TrainedClassifier]
        Dictionary mapping model keys to trained classifiers
    """
    if evaluate_on_test and test_visits is None:
        raise ValueError("test_visits must be provided when evaluate_on_test=True")

    trained_models: Dict[str, TrainedClassifier] = {}

    for prediction_time in prediction_times:
        print(f"\nProcessing: {prediction_time}")
        model_key = get_model_key(model_name, prediction_time)

        # Train model with the new simplified interface
        best_model = train_classifier(
            train_visits,
            valid_visits,
            prediction_time,
            exclude_from_training_data,
            grid,
            ordinal_mappings,
            test_visits,
            visit_col,
            use_balanced_training=use_balanced_training,
            majority_to_minority_ratio=majority_to_minority_ratio,
            calibrate_probabilities=calibrate_probabilities,
            calibration_method=calibration_method,
            label_col=label_col,
            evaluate_on_test=evaluate_on_test,
        )

        trained_models[model_key] = best_model

    return trained_models
