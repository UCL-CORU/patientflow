"""Tests for PrefitProbabilityCalibrator (patientflow#204)."""

from __future__ import annotations

from io import BytesIO

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from patientflow.train.probability_calibrator import (
    PrefitProbabilityCalibrator,
    fit_platt_sigmoid,
    predict_platt_sigmoid,
)


def _fitted_base(seed: int = 0):
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        random_state=seed,
    )
    base = LogisticRegression(max_iter=1000, solver="lbfgs")
    base.fit(X, y)
    return base, X, y


class _ScoreProbe:
    """Already-fitted stub: predict_proba from the first feature; counts fit calls."""

    def __init__(self, n_fits: int = 1):
        self.n_fits = n_fits
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.n_fits += 1
        return self

    def predict_proba(self, X):
        positive = np.clip(np.asarray(X, dtype=np.float64)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_predict_proba_is_valid(method):
    base, X, y = _fitted_base()
    calibrator = PrefitProbabilityCalibrator(estimator=base, method=method)
    calibrator.fit(X, y)
    proba = calibrator.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_joblib_round_trip(method):
    base, X, y = _fitted_base()
    calibrator = PrefitProbabilityCalibrator(estimator=base, method=method)
    calibrator.fit(X, y)
    expected = calibrator.predict_proba(X)

    buffer = BytesIO()
    joblib.dump(calibrator, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)

    np.testing.assert_allclose(restored.predict_proba(X), expected)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_fit_does_not_refit_base_estimator(method):
    rng = np.random.default_rng(1)
    X = rng.random((80, 3))
    y = (X[:, 0] > 0.5).astype(int)
    y[0] = 0
    y[1] = 1
    probe = _ScoreProbe(n_fits=3)
    PrefitProbabilityCalibrator(estimator=probe, method=method).fit(X, y)
    assert probe.n_fits == 3


def test_matches_hand_rolled_isotonic():
    base, X, y = _fitted_base()
    scores = base.predict_proba(X)[:, 1]
    reference = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
    expected = np.clip(reference.predict(scores), 0.0, 1.0)

    calibrator = PrefitProbabilityCalibrator(estimator=base, method="isotonic")
    calibrator.fit(X, y)
    np.testing.assert_allclose(calibrator.predict_proba(X)[:, 1], expected)


def test_matches_hand_rolled_platt():
    base, X, y = _fitted_base()
    scores = base.predict_proba(X)[:, 1]
    a, b = fit_platt_sigmoid(scores, y)
    expected = predict_platt_sigmoid(scores, a, b)

    calibrator = PrefitProbabilityCalibrator(estimator=base, method="sigmoid")
    calibrator.fit(X, y)
    np.testing.assert_allclose(calibrator.predict_proba(X)[:, 1], expected)


def test_invalid_method_raises():
    base, X, y = _fitted_base()
    with pytest.raises(ValueError, match="isotonic"):
        PrefitProbabilityCalibrator(estimator=base, method="temperature").fit(X, y)
