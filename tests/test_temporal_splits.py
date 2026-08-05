"""Tests for temporal splitting and patient-level set assignment.

Covers the three-way legacy behaviour and the optional four-way split with a
chronological calibration window (issue #228).
"""

import unittest
from datetime import date

import pandas as pd
import numpy as np

from patientflow.prepare import assign_patient_ids, create_temporal_splits

START_TRAIN = date(2023, 1, 1)
START_CALIBRATION = date(2023, 2, 1)
START_VALID = date(2023, 3, 1)
START_TEST = date(2023, 4, 1)
END_TEST = date(2023, 5, 1)


def make_visits(n_patients=200, visits_per_patient=3, seed=0):
    """Build synthetic visits spanning all four windows.

    Each patient has several visits at random datetimes across the full
    period, so many patients straddle window boundaries.
    """
    rng = np.random.default_rng(seed)
    rows = []
    encounter = 0
    span_days = (END_TEST - START_TRAIN).days
    for patient in range(n_patients):
        for _ in range(visits_per_patient):
            offset_days = int(rng.integers(0, span_days))
            offset_hours = int(rng.integers(0, 24))
            rows.append(
                {
                    "mrn": f"patient_{patient}",
                    "encounter": encounter,
                    "arrival_datetime": pd.Timestamp(START_TRAIN)
                    + pd.Timedelta(days=offset_days, hours=offset_hours),
                }
            )
            encounter += 1
    return pd.DataFrame(rows)


class TestCreateTemporalSplitsThreeWay(unittest.TestCase):
    def setUp(self):
        self.df = make_visits()

    def test_returns_three_splits(self):
        splits = create_temporal_splits(
            self.df,
            START_TRAIN,
            START_VALID,
            START_TEST,
            END_TEST,
            verbose=False,
        )
        self.assertEqual(len(splits), 3)

    def test_reproducible_with_seed(self):
        first = create_temporal_splits(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST, verbose=False
        )
        second = create_temporal_splits(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST, verbose=False
        )
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)

    def test_windows_bound_each_split(self):
        train, valid, test = create_temporal_splits(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST, verbose=False
        )
        for split, start, end in [
            (train, START_TRAIN, START_VALID),
            (valid, START_VALID, START_TEST),
            (test, START_TEST, END_TEST),
        ]:
            dates = split["arrival_datetime"].dt.date
            self.assertTrue((dates >= start).all())
            self.assertTrue((dates < end).all())

    def test_no_patient_in_more_than_one_split(self):
        train, valid, test = create_temporal_splits(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST, verbose=False
        )
        train_ids = set(train["mrn"])
        valid_ids = set(valid["mrn"])
        test_ids = set(test["mrn"])
        self.assertEqual(train_ids & valid_ids, set())
        self.assertEqual(valid_ids & test_ids, set())
        self.assertEqual(train_ids & test_ids, set())


class TestCreateTemporalSplitsFourWay(unittest.TestCase):
    def setUp(self):
        self.df = make_visits()

    def make_four_way(self, df=None):
        return create_temporal_splits(
            self.df if df is None else df,
            START_TRAIN,
            START_VALID,
            START_TEST,
            END_TEST,
            verbose=False,
            start_calibration=START_CALIBRATION,
        )

    def test_returns_four_splits_in_chronological_order(self):
        splits = self.make_four_way()
        self.assertEqual(len(splits), 4)
        train, calibration, valid, test = splits
        for split, start, end in [
            (train, START_TRAIN, START_CALIBRATION),
            (calibration, START_CALIBRATION, START_VALID),
            (valid, START_VALID, START_TEST),
            (test, START_TEST, END_TEST),
        ]:
            dates = split["arrival_datetime"].dt.date
            self.assertGreater(len(split), 0)
            self.assertTrue((dates >= start).all())
            self.assertTrue((dates < end).all())

    def test_boundary_dates_belong_to_later_window(self):
        boundary_df = pd.DataFrame(
            {
                "mrn": ["a", "b", "c", "d"],
                "encounter": [1, 2, 3, 4],
                "arrival_datetime": [
                    pd.Timestamp(START_TRAIN),
                    pd.Timestamp(START_CALIBRATION),
                    pd.Timestamp(START_VALID),
                    pd.Timestamp(START_TEST),
                ],
            }
        )
        train, calibration, valid, test = self.make_four_way(boundary_df)
        self.assertEqual(list(train["mrn"]), ["a"])
        self.assertEqual(list(calibration["mrn"]), ["b"])
        self.assertEqual(list(valid["mrn"]), ["c"])
        self.assertEqual(list(test["mrn"]), ["d"])

    def test_no_patient_in_more_than_one_split(self):
        splits = self.make_four_way()
        id_sets = [set(split["mrn"]) for split in splits]
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                self.assertEqual(id_sets[i] & id_sets[j], set())

    def test_calibration_date_must_fall_between_train_and_valid(self):
        for bad_date in [
            START_TRAIN,
            START_VALID,
            date(2022, 12, 1),
            date(2023, 3, 15),
        ]:
            with self.assertRaises(ValueError):
                create_temporal_splits(
                    self.df,
                    START_TRAIN,
                    START_VALID,
                    START_TEST,
                    END_TEST,
                    verbose=False,
                    start_calibration=bad_date,
                )

    def test_four_way_covers_same_rows_as_three_way_train(self):
        """The four-way train + calibration frames partition the three-way train window."""
        train3, _, _ = create_temporal_splits(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST, verbose=False
        )
        train4, calibration4, _, _ = self.make_four_way()
        combined_dates = pd.concat([train4, calibration4])["arrival_datetime"].dt.date
        self.assertTrue((combined_dates >= START_TRAIN).all())
        self.assertTrue((combined_dates < START_VALID).all())
        three_way_dates = train3["arrival_datetime"].dt.date
        self.assertTrue((three_way_dates < START_VALID).all())


class TestAssignPatientIds(unittest.TestCase):
    def setUp(self):
        self.df = make_visits()

    def test_three_way_labels(self):
        assignment = assign_patient_ids(
            self.df, START_TRAIN, START_VALID, START_TEST, END_TEST
        )
        self.assertTrue(
            set(assignment["training_validation_test"]).issubset(
                {"train", "valid", "test"}
            )
        )
        self.assertNotIn("calibration_set", assignment.columns)

    def test_four_way_labels_and_membership_columns(self):
        assignment = assign_patient_ids(
            self.df,
            START_TRAIN,
            START_VALID,
            START_TEST,
            END_TEST,
            start_calibration_set=START_CALIBRATION,
        )
        self.assertIn("calibration_set", assignment.columns)
        labels = set(assignment["training_validation_test"])
        self.assertTrue(labels.issubset({"train", "calibration", "valid", "test"}))
        # With visits spread across the whole period, all four labels occur
        self.assertEqual(labels, {"train", "calibration", "valid", "test"})

    def test_each_patient_assigned_to_exactly_one_set(self):
        assignment = assign_patient_ids(
            self.df,
            START_TRAIN,
            START_VALID,
            START_TEST,
            END_TEST,
            start_calibration_set=START_CALIBRATION,
        )
        self.assertEqual(assignment.index.nunique(), len(assignment))

    def test_patient_only_assigned_to_window_with_visits(self):
        """A patient whose visits all fall in one window is assigned to it."""
        df = pd.DataFrame(
            {
                "mrn": ["cal_only"] * 2,
                "encounter": [1, 2],
                "arrival_datetime": [
                    pd.Timestamp("2023-02-05"),
                    pd.Timestamp("2023-02-20"),
                ],
            }
        )
        assignment = assign_patient_ids(
            df,
            START_TRAIN,
            START_VALID,
            START_TEST,
            END_TEST,
            start_calibration_set=START_CALIBRATION,
        )
        self.assertEqual(
            assignment.loc["cal_only", "training_validation_test"], "calibration"
        )

    def test_invalid_calibration_date_raises(self):
        with self.assertRaises(ValueError):
            assign_patient_ids(
                self.df,
                START_TRAIN,
                START_VALID,
                START_TEST,
                END_TEST,
                start_calibration_set=START_VALID,
            )


if __name__ == "__main__":
    unittest.main()
