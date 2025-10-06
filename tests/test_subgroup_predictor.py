import unittest
import pandas as pd
import numpy as np
import pytest

from patientflow.predictors.subgroup_predictor import (
    MultiSubgroupPredictor,
)
from patientflow.predictors.sequence_to_outcome_predictor import (
    SequenceToOutcomePredictor,
)


class TestMultiSubgroupPredictor(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Create small synthetic dataset
        # Two exclusive subgroups based on sex
        self.subgroup_functions = {
            "male": lambda row: row.get("sex") == "M",
            "female": lambda row: row.get("sex") == "F",
        }

        # Training data with minimal columns
        # Use custom names to ensure naming is respected throughout
        self.input_var = "input_var"
        self.grouping_var = "grouping_var"
        self.outcome_var = "outcome_var"

        input_sequences = [
            ("A",),
            ("B",),
            ("A",),
            ("B", "C"),
            (),
            ("A", "C"),
        ]
        grouping_sequences = [
            ("A", "C"),
            ("B", "C"),
            ("A",),
            ("B", "C"),
            (),
            ("A", "C"),
        ]
        outcomes = ["X", "Y", "X", "Y", "X", "X"]
        sexes = ["M", "F", "M", "F", "M", "F"]

        self.train_df = pd.DataFrame(
            {
                "snapshot_date": pd.date_range(
                    "2023-01-01", periods=len(input_sequences), freq="h"
                ),
                self.input_var: input_sequences,
                self.grouping_var: grouping_sequences,
                self.outcome_var: outcomes,
                "sex": sexes,
                # admitted column included but not used since special filtering is disabled in subgroup models
                "is_admitted": [True] * len(input_sequences),
            }
        )

        # Validation dataframe for predictions
        self.valid_df = pd.DataFrame(
            {
                self.input_var: [
                    ("A",),
                    ("B",),
                    ("B", "C"),
                    None,  # missing sequence
                ],
                "sex": ["M", "F", "F", "M"],
            }
        )

    def _fit_model(self, min_samples: int = 1) -> MultiSubgroupPredictor:
        model = MultiSubgroupPredictor(
            subgroup_functions=self.subgroup_functions,
            base_predictor_class=SequenceToOutcomePredictor,
            input_var=self.input_var,
            grouping_var=self.grouping_var,
            outcome_var=self.outcome_var,
            min_samples=min_samples,
        )
        model.fit(self.train_df)
        return model

    def test_predict_dataframe_returns_series(self):
        model = self._fit_model()
        series = model.predict_dataframe(self.valid_df)
        self.assertIsInstance(series, pd.Series)
        self.assertTrue(series.index.equals(self.valid_df.index))

        # First row male with sequence ("A",) should yield a dict
        self.assertTrue(
            isinstance(series.iloc[0], dict)
            or (isinstance(series.iloc[0], float) and np.isnan(series.iloc[0]))
        )
        # Fourth row has None sequence -> NaN
        self.assertTrue(isinstance(series.iloc[3], float) and np.isnan(series.iloc[3]))

        # For rows with dicts, probabilities should be between 0 and 1 and sum to ~1
        dict_rows = series[series.apply(lambda x: isinstance(x, dict))]
        for d in dict_rows:
            self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in d.values()))
            total = sum(d.values())
            # allow float tolerance; empty dicts are not expected from subgroup predictor
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_missing_input_column_raises(self):
        model = self._fit_model()
        df_missing = self.valid_df.drop(columns=[self.input_var])
        with self.assertRaises(ValueError):
            _ = model.predict_dataframe(df_missing)

    def test_overlapping_masks_raise(self):
        # Create overlapping subgroup functions (both True)
        overlapping = {
            "g1": lambda row: True,
            "g2": lambda row: True,
        }
        model = MultiSubgroupPredictor(
            subgroup_functions=overlapping,
            base_predictor_class=SequenceToOutcomePredictor,
            input_var=self.input_var,
            grouping_var=self.grouping_var,
            outcome_var=self.outcome_var,
            min_samples=1,
        )
        model.fit(self.train_df)
        with self.assertRaises(ValueError):
            _ = model.predict_dataframe(self.valid_df)

    def test_rows_with_untrained_subgroup_return_nan(self):
        # Set min_samples large so one subgroup is skipped
        with pytest.warns(UserWarning) as record:
            model = self._fit_model(min_samples=1000)
        # Expect skipping warnings for subgroups with insufficient samples
        assert any("Skipping male" in str(w.message) for w in record)
        assert any("Skipping female" in str(w.message) for w in record)
        # At least one subgroup should have no model
        self.assertTrue(len(model.models) < len(self.subgroup_functions))
        series = model.predict_dataframe(self.valid_df)
        # Ensure some entries are NaN (for subgroup without a trained model)
        self.assertTrue(series.isna().any())

    def test_respects_configured_names(self):
        # Change names and ensure still works
        other_input = "consultation_sequence"
        other_group = "all_consultations"
        other_outcome = "specialty"

        df_train = self.train_df.rename(
            columns={
                self.input_var: other_input,
                self.grouping_var: other_group,
                self.outcome_var: other_outcome,
            }
        )
        df_valid = self.valid_df.rename(columns={self.input_var: other_input})

        model = MultiSubgroupPredictor(
            subgroup_functions=self.subgroup_functions,
            base_predictor_class=SequenceToOutcomePredictor,
            input_var=other_input,
            grouping_var=other_group,
            outcome_var=other_outcome,
            min_samples=1,
        )
        model.fit(df_train)
        series = model.predict_dataframe(df_valid)
        self.assertIsInstance(series, pd.Series)


if __name__ == "__main__":
    unittest.main()
