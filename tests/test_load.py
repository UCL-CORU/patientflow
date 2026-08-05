"""Tests for configuration loading, focusing on modelling_dates handling."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import yaml

from patientflow.load import load_config_file

BASE_CONFIG = {
    "prediction_times": [[6, 0], [12, 0]],
}


def write_config(tmp_dir, modelling_dates):
    config = dict(BASE_CONFIG)
    # Unquoted YAML dates parse as datetime.date, as in the real config.yaml
    config["modelling_dates"] = [date.fromisoformat(d) for d in modelling_dates]
    path = Path(tmp_dir) / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


class TestLoadConfigModellingDates(unittest.TestCase):
    def test_four_dates_loads_without_calibration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_config(
                tmp_dir, ["2031-03-01", "2031-09-01", "2031-10-01", "2032-01-01"]
            )
            params = load_config_file(path)
        self.assertIsNotNone(params)
        self.assertNotIn("start_calibration_set", params)
        self.assertEqual(str(params["start_training_set"]), "2031-03-01")
        self.assertEqual(str(params["start_validation_set"]), "2031-09-01")
        self.assertEqual(str(params["start_test_set"]), "2031-10-01")
        self.assertEqual(str(params["end_test_set"]), "2032-01-01")

    def test_five_dates_loads_with_calibration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_config(
                tmp_dir,
                [
                    "2031-03-01",
                    "2031-08-01",
                    "2031-09-01",
                    "2031-10-01",
                    "2032-01-01",
                ],
            )
            params = load_config_file(path)
        self.assertIsNotNone(params)
        self.assertEqual(str(params["start_training_set"]), "2031-03-01")
        self.assertEqual(str(params["start_calibration_set"]), "2031-08-01")
        self.assertEqual(str(params["start_validation_set"]), "2031-09-01")
        self.assertEqual(str(params["start_test_set"]), "2031-10-01")
        self.assertEqual(str(params["end_test_set"]), "2032-01-01")

    def test_wrong_number_of_dates_returns_none(self):
        for dates in [
            ["2031-03-01", "2031-09-01", "2031-10-01"],
            [
                "2031-03-01",
                "2031-08-01",
                "2031-09-01",
                "2031-10-01",
                "2032-01-01",
                "2032-02-01",
            ],
        ]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = write_config(tmp_dir, dates)
                self.assertIsNone(load_config_file(path))

    def test_out_of_order_dates_return_none(self):
        # Four-date form with validation before training
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_config(
                tmp_dir, ["2031-09-01", "2031-03-01", "2031-10-01", "2032-01-01"]
            )
            self.assertIsNone(load_config_file(path))
        # Five-date form with the calibration date in the wrong slot
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_config(
                tmp_dir,
                [
                    "2031-03-01",
                    "2031-09-15",
                    "2031-09-01",
                    "2031-10-01",
                    "2032-01-01",
                ],
            )
            self.assertIsNone(load_config_file(path))


if __name__ == "__main__":
    unittest.main()
