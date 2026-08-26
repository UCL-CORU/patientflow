"""Prefit probability calibration that does not depend on sklearn version APIs.

``CalibratedClassifierCV`` changed meaning between sklearn 1.4 (``cv="prefit"``)
and 1.6+ (``FrozenEstimator`` plus internal cross-validation). This module owns
the prefit path: score an already-fitted classifier, then fit isotonic or Platt
sigmoid on those scores. See patientflow#204.

Functions
---------
fit_platt_sigmoid
    Fit Platt (1999) A, B on binary scores and labels.
predict_platt_sigmoid
    Apply a fitted Platt map: ``1 / (1 + exp(A f + B))``.

Classes
-------
PrefitProbabilityCalibrator
    Sklearn-compatible wrapper around a fitted base classifier plus a calibrator.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import fmin_bfgs
from scipy.special import expit, xlogy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

CalibrationMethod = Literal["isotonic", "sigmoid"]


def fit_platt_sigmoid(
    scores: npt.ArrayLike,
    y: npt.ArrayLike,
) -> tuple[float, float]:
    """Fit Platt sigmoid parameters on binary classifier scores.

    Uses the Bayesian-prior target construction from Platt (1999, §2.2) and
    BFGS on the Bernoulli log-likelihood. This is a local copy of the mapping
    sklearn used under ``CalibratedClassifierCV(..., method="sigmoid",
    cv="prefit")``; it does not import sklearn's private ``_SigmoidCalibration``.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Uncalibrated positive-class scores, typically ``predict_proba`` column 1.
    y : array-like of shape (n_samples,)
        Binary labels. Values ``> 0`` are treated as the positive class.

    Returns
    -------
    a : float
        Slope.
    b : float
        Intercept.

    References
    ----------
    Platt, J. (1999). Probabilistic outputs for support vector machines.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    if scores.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores and y must have the same length; got {scores.shape[0]} and "
            f"{y.shape[0]}"
        )

    prior0 = float(np.sum(y <= 0))
    prior1 = float(y.shape[0] - prior0)
    if prior0 == 0 or prior1 == 0:
        raise ValueError("Platt sigmoid calibration requires both classes in y")

    # Platt §2.2: smoothed targets instead of hard 0/1 labels.
    targets = np.empty_like(scores)
    targets[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    targets[y <= 0] = 1.0 / (prior0 + 2.0)
    targets_neg = 1.0 - targets

    def objective(ab: npt.NDArray[np.float64]) -> float:
        pred = expit(-(ab[0] * scores + ab[1]))
        loss = -(xlogy(targets, pred) + xlogy(targets_neg, 1.0 - pred))
        return float(loss.sum())

    def grad(ab: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pred = expit(-(ab[0] * scores + ab[1]))
        residual = targets - pred
        return np.array(
            [float(np.dot(residual, scores)), float(residual.sum())],
            dtype=np.float64,
        )

    ab0 = np.array([0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))], dtype=np.float64)
    fitted = fmin_bfgs(objective, ab0, fprime=grad, disp=False)
    return float(fitted[0]), float(fitted[1])


def predict_platt_sigmoid(
    scores: npt.ArrayLike,
    a: float,
    b: float,
) -> npt.NDArray[np.float64]:
    """Apply a fitted Platt map ``p = 1 / (1 + exp(A f + B))``.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Uncalibrated positive-class scores.
    a : float
        Fitted slope.
    b : float
        Fitted intercept.

    Returns
    -------
    ndarray of shape (n_samples,)
        Calibrated positive-class probabilities in ``[0, 1]``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    return np.clip(expit(-(a * scores + b)), 0.0, 1.0)


class PrefitProbabilityCalibrator(ClassifierMixin, BaseEstimator):
    """Calibrate an already-fitted binary classifier without refitting it.

    On ``fit``, scores the supplied estimator with ``predict_proba`` and fits
    either isotonic regression or Platt sigmoid on those scores. The base
    estimator is not cloned or refit.

    Parameters
    ----------
    estimator : object
        Already-fitted classifier implementing ``predict_proba``.
    method : {'isotonic', 'sigmoid'}, default='sigmoid'
        Calibration map. ``'isotonic'`` uses
        ``IsotonicRegression(out_of_bounds="clip")``. ``'sigmoid'`` uses Platt
        scaling (see ``fit_platt_sigmoid``).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels, taken from ``estimator.classes_`` when present.
    calibrator_ : IsotonicRegression or tuple of float
        Fitted isotonic model, or ``(a, b)`` Platt parameters.
    """

    def __init__(
        self,
        estimator: Any,
        method: CalibrationMethod = "sigmoid",
    ):
        self.estimator = estimator
        self.method = method

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> "PrefitProbabilityCalibrator":
        """Fit the calibration map on ``estimator.predict_proba(X)[:, 1]``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration features in the space the base estimator expects
            (already transformed, if used inside a pipeline).
        y : array-like of shape (n_samples,)
            Binary calibration labels.

        Returns
        -------
        self : PrefitProbabilityCalibrator
            Fitted calibrator.
        """
        if self.method not in ("isotonic", "sigmoid"):
            raise ValueError(
                "method must be 'isotonic' or 'sigmoid', " f"got {self.method!r}"
            )

        y_arr = np.asarray(y).ravel()
        scores = np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)

        if hasattr(self.estimator, "classes_"):
            self.classes_ = np.asarray(self.estimator.classes_)
        else:
            self.classes_ = np.unique(y_arr)

        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(scores, y_arr)
        else:
            self.calibrator_ = fit_platt_sigmoid(scores, y_arr)

        return self

    def _positive_proba(self, X: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "calibrator_")
        scores = np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)
        if self.method == "isotonic":
            return np.clip(self.calibrator_.predict(scores), 0.0, 1.0)
        a, b = self.calibrator_
        return predict_platt_sigmoid(scores, a, b)

    def predict_proba(self, X: Any) -> npt.NDArray[np.float64]:
        """Return calibrated class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features in the space the base estimator expects.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns are ``[1 - p, p]`` for the negative and positive class.
        """
        positive = self._positive_proba(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Return the class with the higher calibrated probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features in the space the base estimator expects.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, "classes_")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
