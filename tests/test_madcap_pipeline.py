"""MADCAP plots with pipelines that select their own columns."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from patientflow.model_artifacts import TrainedClassifier, TrainingResults
from patientflow.train.classifiers import FeatureColumnTransformer
from patientflow.viz.madcap import plot_madcap


def _feature_columns_model(prediction_time: tuple[int, int]) -> TrainedClassifier:
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = (x > 0).astype(int)
    pipeline = Pipeline(
        [
            ("feature_columns", FeatureColumnTransformer()),
            ("classifier", LogisticRegression()),
        ]
    )
    pipeline.fit(pd.DataFrame({"x": x}), y)
    return TrainedClassifier(
        training_results=TrainingResults(prediction_time=prediction_time),
        pipeline=pipeline,
    )


def _visits(prediction_times: list[tuple[int, int]]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for pt in prediction_times:
        for _ in range(20):
            x = rng.normal()
            rows.append({"prediction_time": pt, "x": x, "is_admitted": int(x > 0)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("prediction_times", [[(6, 0)], [(6, 0), (15, 30)]])
def test_plot_madcap_skips_add_missing_columns_for_feature_columns_pipeline(
    prediction_times,
):
    models = [_feature_columns_model(pt) for pt in prediction_times]
    fig = plot_madcap(models, _visits(prediction_times), return_figure=True)
    assert fig is not None
    plt.close(fig)
