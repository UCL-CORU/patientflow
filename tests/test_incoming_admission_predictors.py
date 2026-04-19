import unittest
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from patientflow.predictors.incoming_admission_predictors import (
    ParametricIncomingAdmissionPredictor,
    EmpiricalIncomingAdmissionPredictor,
    DirectAdmissionPredictor,
)


class TestIncomingAdmissionPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across multiple tests."""
        np.random.seed(42)

        n_patients = 1000
        start_date = datetime(2024, 1, 1)

        arrival_times = []
        departure_times = []
        specialties = []

        for i in range(n_patients):
            days_offset = np.random.randint(0, 31)
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)

            arrival_time = start_date + timedelta(
                days=days_offset, hours=hours_offset, minutes=minutes_offset
            )

            time_to_admission = np.random.exponential(6)
            time_to_admission = max(0.5, min(48, time_to_admission))

            departure_time = arrival_time + timedelta(hours=time_to_admission)

            arrival_times.append(arrival_time)
            departure_times.append(departure_time)

            specialty = np.random.choice(
                ["medical", "surgical", "haem/onc", "paediatric"]
            )
            specialties.append(specialty)

        cls.test_df = pd.DataFrame(
            {
                "arrival_datetime": arrival_times,
                "departure_datetime": departure_times,
                "specialty": specialties,
            }
        ).set_index("arrival_datetime")

        cls.prediction_window = timedelta(hours=8)
        cls.yta_time_interval = timedelta(minutes=30)
        cls.prediction_times = [(8, 0), (12, 0), (16, 0), (20, 0)]
        cls.num_days = 31

        cls.filters = {
            "medical": {"specialty": "medical"},
            "surgical": {"specialty": "surgical"},
            "haem_onc": {"specialty": "haem/onc"},
            "paediatric": {"specialty": "paediatric"},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fit_new_api(self, predictor):
        """Fit using the new keyword-only API (no deprecation warnings)."""
        predictor.fit(
            self.test_df,
            yta_time_interval=self.yta_time_interval,
            num_days=self.num_days,
        )
        return predictor

    # ------------------------------------------------------------------
    # Basic identities / unit-level checks
    # ------------------------------------------------------------------
    def test_parametric_simplified_poisson_equivalence(self):
        """Parametric predictor should match a single Poisson with mu = sum(lambda * theta)."""
        arrival_rates = np.array([1.0, 2.0, 1.5])
        theta = np.array([0.5, 0.6, 0.7])
        mu = float(np.sum(arrival_rates * theta))
        from scipy.stats import poisson as _poisson

        mv = 50
        expected = _poisson.pmf(np.arange(mv), mu)
        expected = expected[expected > 1e-10]
        expected = expected / expected.sum()

        self.assertGreater(mu, 0.0)
        self.assertTrue(np.isclose(expected.sum(), 1.0, atol=1e-12))

    def test_snap_to_interval_boundary(self):
        """_snap_to_interval_boundary should round to the nearest yta_time_interval."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        # Exact boundary: no change.
        self.assertEqual(predictor._snap_to_interval_boundary((12, 0)), (12, 0))
        self.assertEqual(predictor._snap_to_interval_boundary((12, 30)), (12, 30))

        # Between boundaries: round to nearest.
        self.assertEqual(predictor._snap_to_interval_boundary((12, 10)), (12, 0))
        self.assertEqual(predictor._snap_to_interval_boundary((12, 20)), (12, 30))

        # 24-hour wrap-around.
        self.assertEqual(predictor._snap_to_interval_boundary((23, 50)), (0, 0))

    def test_incoming_admission_predictor_base_class(self):
        """Test the base IncomingAdmissionPredictor class via a concrete subclass."""
        predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters, verbose=True
        )
        self.assertEqual(predictor.filters, self.filters)
        self.assertTrue(predictor.verbose)

        filtered_df = predictor.filter_dataframe(self.test_df, {"specialty": "medical"})
        self.assertTrue(all(filtered_df["specialty"] == "medical"))

        def is_medical(specialty):
            return specialty == "medical"

        filtered_df_func = predictor.filter_dataframe(
            self.test_df, {"specialty": is_medical}
        )
        self.assertTrue(all(filtered_df_func["specialty"] == "medical"))

    # ------------------------------------------------------------------
    # Parametric predictor
    # ------------------------------------------------------------------
    def test_parametric_predictor_initialization(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self.assertEqual(predictor.filters, self.filters)
        self.assertFalse(predictor.verbose)

    def test_parametric_predictor_fit(self):
        """Fit stores the full 24-hour arrival-rate dictionary per filter key."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        self.assertIsInstance(predictor.weights, dict)
        self.assertTrue(len(predictor.weights) > 0)

        for key in self.filters.keys():
            self.assertIn(key, predictor.weights)
            self.assertIn("arrival_rates_dict", predictor.weights[key])
            # 24 hours at 30-minute granularity = 48 entries
            self.assertEqual(len(predictor.weights[key]["arrival_rates_dict"]), 48)

    def test_parametric_predictor_fit_validation(self):
        """yta_time_interval and num_days are required; types are validated."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)

        with self.assertRaises(TypeError):
            predictor.fit(self.test_df)  # missing yta_time_interval

        with self.assertRaises(TypeError):
            predictor.fit(
                self.test_df,
                yta_time_interval="30 minutes",
                num_days=self.num_days,
            )

        with self.assertRaises(TypeError):
            predictor.fit(
                self.test_df,
                yta_time_interval=self.yta_time_interval,
            )  # missing num_days

        with self.assertRaises(ValueError):
            predictor.fit(
                self.test_df,
                yta_time_interval=timedelta(seconds=0),
                num_days=self.num_days,
            )

    def test_parametric_predictor_predict(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        predictions = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys=["medical", "surgical"],
            x1=2,
            y1=0.5,
            x2=4,
            y2=0.9,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        for pred_df in predictions.values():
            self.assertIsInstance(pred_df, pd.DataFrame)
            self.assertIn("agg_proba", pred_df.columns)
            self.assertTrue(np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10))

        # Missing curve parameters must raise
        with self.assertRaises(ValueError):
            predictor.predict(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
                filter_keys=["medical"],
                x1=2,
                y1=0.5,
            )

        # Unknown filter key
        with self.assertRaises(ValueError):
            predictor.predict(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
                filter_keys=["invalid_key"],
                x1=2,
                y1=0.5,
                x2=4,
                y2=0.9,
            )

        # filter_keys required when weights has multiple keys (e.g. services)
        with self.assertRaises(ValueError):
            predictor.predict(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
                x1=2,
                y1=0.5,
                x2=4,
                y2=0.9,
            )

    def test_predict_requires_prediction_window(self):
        """predict() raises ValueError if prediction_window cannot be resolved."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        with self.assertRaises(ValueError):
            predictor.predict(
                prediction_time=(8, 0),
                filter_keys="medical",
                x1=2,
                y1=0.5,
                x2=4,
                y2=0.9,
            )

    def test_predict_mean_requires_prediction_window(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        with self.assertRaises(ValueError):
            predictor.predict_mean(
                prediction_time=(8, 0),
                filter_key="medical",
                x1=2,
                y1=0.5,
                x2=4,
                y2=0.9,
            )

    def test_predict_mean_with_prediction_window(self):
        """predict_mean returns a positive float when given prediction_window."""
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        mean = predictor.predict_mean(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_key="medical",
        )
        self.assertIsInstance(mean, float)
        self.assertGreater(mean, 0.0)

    # ------------------------------------------------------------------
    # Empirical predictor
    # ------------------------------------------------------------------
    def test_empirical_predictor_initialization(self):
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        self.assertEqual(predictor.filters, self.filters)
        self.assertIsNone(predictor.survival_df)

    def test_empirical_predictor_fit(self):
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        self.assertIsNotNone(predictor.survival_df)
        self.assertIsInstance(predictor.survival_df, pd.DataFrame)
        self.assertIn("time_hours", predictor.survival_df.columns)
        self.assertIn("survival_probability", predictor.survival_df.columns)

        self.assertIsInstance(predictor.weights, dict)
        self.assertTrue(len(predictor.weights) > 0)
        for key in self.filters.keys():
            self.assertIn("arrival_rates_dict", predictor.weights[key])

    def test_empirical_predictor_get_survival_curve(self):
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        with self.assertRaises(RuntimeError):
            predictor.get_survival_curve()

        self._fit_new_api(predictor)

        survival_curve = predictor.get_survival_curve()
        self.assertIsInstance(survival_curve, pd.DataFrame)
        self.assertIn("time_hours", survival_curve.columns)
        self.assertIn("survival_probability", survival_curve.columns)

    def test_empirical_predictor_predict(self):
        predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        predictions = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys=["medical", "surgical"],
            max_value=25,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        for pred_df in predictions.values():
            self.assertIsInstance(pred_df, pd.DataFrame)
            self.assertIn("agg_proba", pred_df.columns)
            self.assertTrue(np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10))

        predictor_no_survival = EmpiricalIncomingAdmissionPredictor(
            filters=self.filters
        )
        with self.assertRaises(RuntimeError):
            predictor_no_survival.predict(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
                filter_keys=["medical", "surgical"],
            )

    def test_predictor_comparison(self):
        """Parametric and empirical predictors both produce valid distributions."""
        parametric_predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters
        )
        empirical_predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        self._fit_new_api(parametric_predictor)
        self._fit_new_api(empirical_predictor)

        parametric_pred = parametric_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
            x1=2,
            y1=0.5,
            x2=4,
            y2=0.9,
        )
        empirical_pred = empirical_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
            max_value=25,
        )

        for pred_dict in [parametric_pred, empirical_pred]:
            for pred_df in pred_dict.values():
                self.assertTrue(np.all(pred_df["agg_proba"] >= 0))
                self.assertTrue(
                    np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10)
                )

    # ------------------------------------------------------------------
    # Edge cases / metrics / weights
    # ------------------------------------------------------------------
    def test_edge_cases(self):
        """Empty filters fall back to an 'unfiltered' entry; short windows still work."""
        predictor = ParametricIncomingAdmissionPredictor(filters={})
        self._fit_new_api(predictor)
        self.assertIn("unfiltered", predictor.weights)

        short_window = timedelta(minutes=30)
        short_predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(short_predictor)

        predictions = short_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=short_window,
            filter_keys="medical",
            x1=2,
            y1=0.5,
            x2=4,
            y2=0.9,
        )
        self.assertIn("medical", predictions)

    def test_get_weights(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        weights = predictor.get_weights()
        self.assertIsInstance(weights, dict)
        self.assertEqual(weights, predictor.weights)

    def test_metrics_storage(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        self.assertIsInstance(predictor.metrics, dict)
        self.assertIn("train_dttm", predictor.metrics)
        self.assertIn("train_set_no", predictor.metrics)
        self.assertIn("start_date", predictor.metrics)
        self.assertIn("end_date", predictor.metrics)
        self.assertIn("num_days", predictor.metrics)

        self.assertEqual(predictor.metrics["train_set_no"], len(self.test_df))
        self.assertEqual(predictor.metrics["num_days"], self.num_days)

    def test_snap_warning_on_non_boundary_prediction_time(self):
        """Requesting a prediction_time off the interval grid emits a warning and still predicts."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        # yta_time_interval is 30 minutes; (14, 15) is off-grid
        with self.assertWarns(UserWarning) as warning_context:
            predictions = predictor.predict(
                prediction_time=(14, 15),
                prediction_window=self.prediction_window,
                filter_keys="medical",
                x1=2,
                y1=0.5,
                x2=4,
                y2=0.9,
            )

        warning_messages = [str(w.message) for w in warning_context.warnings]
        self.assertTrue(any("snapping to" in msg for msg in warning_messages))
        self.assertIn("medical", predictions)

    # ------------------------------------------------------------------
    # Direct predictor
    # ------------------------------------------------------------------
    def test_direct_predictor_initialization(self):
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self.assertEqual(predictor.filters, self.filters)
        self.assertFalse(predictor.verbose)

        verbose_predictor = DirectAdmissionPredictor(filters=self.filters, verbose=True)
        self.assertTrue(verbose_predictor.verbose)

        no_filter_predictor = DirectAdmissionPredictor(filters=None)
        self.assertEqual(no_filter_predictor.filters, {})

    def test_direct_predictor_fit(self):
        """Direct predictor stores the full 24-hour arrival-rate dictionary per filter."""
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        self.assertIsInstance(predictor.weights, dict)
        self.assertTrue(len(predictor.weights) > 0)

        for filter_key in self.filters.keys():
            self.assertIn(filter_key, predictor.weights)
            self.assertIn("arrival_rates_dict", predictor.weights[filter_key])
            arrival_rates_dict = predictor.weights[filter_key]["arrival_rates_dict"]
            self.assertIsInstance(arrival_rates_dict, dict)
            # 24 hours at 30-minute granularity
            self.assertEqual(len(arrival_rates_dict), 48)

    def test_direct_predictor_predict(self):
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        predictions = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys=["medical", "surgical"],
            max_value=30,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn("medical", predictions)
        self.assertIn("surgical", predictions)

        for pred_df in predictions.values():
            self.assertIsInstance(pred_df, pd.DataFrame)
            self.assertIn("agg_proba", pred_df.columns)
            prob_sum = pred_df["agg_proba"].sum()
            self.assertLessEqual(prob_sum, 1.0)
            self.assertGreater(prob_sum, 0.99)
            self.assertTrue(np.all(pred_df["agg_proba"] >= 0))

        # Custom max_value
        predictions_small = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys=["medical", "surgical"],
            max_value=10,
        )
        for pred_df in predictions_small.values():
            self.assertLessEqual(pred_df.index.max(), 9)
            prob_sum = pred_df["agg_proba"].sum()
            self.assertLessEqual(prob_sum, 1.0)
            self.assertGreater(prob_sum, 0.5)

        # Invalid filter key
        with self.assertRaises(ValueError):
            predictor.predict(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
                filter_keys=["invalid_key"],
            )

    def test_direct_predictor_mathematical_correctness(self):
        """Direct predictor's expected value matches the total arrival rate over the window."""
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        # Slice arrival rates for the requested window from the stored full-cycle dict
        arrival_rates_dict = predictor.weights["medical"]["arrival_rates_dict"]
        Ntimes = int(self.prediction_window / self.yta_time_interval)
        medical_arrival_rates = [
            arrival_rates_dict[
                (datetime(1970, 1, 1, 8, 0) + i * self.yta_time_interval).time()
            ]
            for i in range(Ntimes)
        ]
        total_expected_arrivals = sum(medical_arrival_rates)

        predictions = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
            max_value=50,
        )

        pred_df = predictions["medical"]
        expected_value = sum(pred_df.index * pred_df["agg_proba"])
        self.assertAlmostEqual(expected_value, total_expected_arrivals, places=2)

    def test_direct_predictor_consistency(self):
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)

        pred1 = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
        )
        pred2 = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
        )

        np.testing.assert_array_almost_equal(
            pred1["medical"]["agg_proba"].values, pred2["medical"]["agg_proba"].values
        )

    def test_direct_predictor_without_filters(self):
        predictor = DirectAdmissionPredictor(filters=None)
        self._fit_new_api(predictor)

        self.assertIn("unfiltered", predictor.weights)

        predictions = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
        )

        self.assertIn("unfiltered", predictions)
        pred_df = predictions["unfiltered"]
        prob_sum = pred_df["agg_proba"].sum()
        self.assertLessEqual(prob_sum, 1.0)
        self.assertGreater(prob_sum, 0.99)

    def test_direct_predictor_comparison_with_others(self):
        """Direct predictor has the highest expected value (100% admission rate)."""
        direct_predictor = DirectAdmissionPredictor(filters=self.filters)
        parametric_predictor = ParametricIncomingAdmissionPredictor(
            filters=self.filters
        )
        empirical_predictor = EmpiricalIncomingAdmissionPredictor(filters=self.filters)

        self._fit_new_api(direct_predictor)
        self._fit_new_api(parametric_predictor)
        self._fit_new_api(empirical_predictor)

        direct_pred = direct_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
        )
        parametric_pred = parametric_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
            x1=2,
            y1=0.5,
            x2=4,
            y2=0.9,
        )
        empirical_pred = empirical_predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
            max_value=25,
        )

        for pred_dict in [direct_pred, parametric_pred, empirical_pred]:
            for pred_df in pred_dict.values():
                self.assertTrue(np.all(pred_df["agg_proba"] >= 0))
                self.assertTrue(
                    np.allclose(pred_df["agg_proba"].sum(), 1.0, atol=1e-10)
                )

        direct_expected = sum(
            direct_pred["medical"].index * direct_pred["medical"]["agg_proba"]
        )
        parametric_expected = sum(
            parametric_pred["medical"].index * parametric_pred["medical"]["agg_proba"]
        )
        empirical_expected = sum(
            empirical_pred["medical"].index * empirical_pred["medical"]["agg_proba"]
        )

        self.assertGreaterEqual(direct_expected, parametric_expected)
        self.assertGreaterEqual(direct_expected, empirical_expected)

    # ------------------------------------------------------------------
    # Deprecation path
    # ------------------------------------------------------------------
    def test_fit_emits_deprecation_warning_for_prediction_window(self):
        """Passing prediction_window to fit() still works but emits DeprecationWarning."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor.fit(
                self.test_df,
                prediction_window=self.prediction_window,
                yta_time_interval=self.yta_time_interval,
                num_days=self.num_days,
            )
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "prediction_window" in str(w.message)
                for w in caught
            )
        )
        # Legacy attributes populated for back-compat
        self.assertEqual(predictor.prediction_window, self.prediction_window)

    def test_fit_emits_deprecation_warning_for_prediction_times(self):
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor.fit(
                self.test_df,
                yta_time_interval=self.yta_time_interval,
                prediction_times=self.prediction_times,
                num_days=self.num_days,
            )
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "prediction_times" in str(w.message)
                for w in caught
            )
        )
        self.assertEqual(predictor.prediction_times, self.prediction_times)

    def test_predict_falls_back_to_fit_time_prediction_window(self):
        """If prediction_window is omitted at predict(), the fit-time value is used with a warning."""
        predictor = DirectAdmissionPredictor(filters=self.filters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            predictor.fit(
                self.test_df,
                prediction_window=self.prediction_window,
                yta_time_interval=self.yta_time_interval,
                num_days=self.num_days,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictions = predictor.predict(
                prediction_time=(8, 0), filter_keys="medical"
            )

        self.assertIn("medical", predictions)
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "prediction_window" in str(w.message)
                for w in caught
            )
        )

    def test_fit_legacy_positional_call_still_works(self):
        """The legacy positional call `(df, window, interval, times, num_days)` still works."""
        predictor = ParametricIncomingAdmissionPredictor(filters=self.filters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            predictor.fit(
                self.test_df,
                self.prediction_window,
                self.yta_time_interval,
                self.prediction_times,
                self.num_days,
            )
        self.assertIn("medical", predictor.weights)
        self.assertIn("arrival_rates_dict", predictor.weights["medical"])

    def test_deprecated_prediction_context_dict_warns(self):
        """Legacy prediction_context emits DeprecationWarning and matches new API results."""
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)
        legacy_ctx = {"medical": {"prediction_time": (8, 0)}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_pred = predictor.predict(
                prediction_context=legacy_ctx,
                prediction_window=self.prediction_window,
            )
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "prediction_context" in str(w.message).lower()
                for w in caught
            )
        )
        new_pred = predictor.predict(
            prediction_time=(8, 0),
            prediction_window=self.prediction_window,
            filter_keys="medical",
        )
        np.testing.assert_array_almost_equal(
            legacy_pred["medical"]["agg_proba"].values,
            new_pred["medical"]["agg_proba"].values,
        )

    def test_legacy_prediction_context_mismatched_times_raise(self):
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)
        bad_ctx = {
            "medical": {"prediction_time": (8, 0)},
            "surgical": {"prediction_time": (12, 0)},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError) as cm:
                predictor.predict(
                    prediction_context=bad_ctx,
                    prediction_window=self.prediction_window,
                )
        self.assertIn("same prediction_time", str(cm.exception).lower())

    def test_predict_mean_requires_filter_key_when_multiple_services(self):
        predictor = DirectAdmissionPredictor(filters=self.filters)
        self._fit_new_api(predictor)
        with self.assertRaises(ValueError) as cm:
            predictor.predict_mean(
                prediction_time=(8, 0),
                prediction_window=self.prediction_window,
            )
        self.assertIn("filter_key", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
