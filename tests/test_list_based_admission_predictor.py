import itertools
import pickle
import unittest
import warnings
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates_by_weekday,
    weekday_occurrences_in_span,
)
from patientflow.predict.distribution import Distribution
from patientflow.predictors.incoming_admission_predictors import (
    DirectAdmissionPredictor,
)
from patientflow.predictors.list_based_admission_predictor import (
    ListBasedAdmissionPredictor,
)

FILTERS = {"A": {"service": "A"}, "B": {"service": "B"}}


def _rows_to_frames(arrival_rows, list_rows):
    train_df = (
        pd.DataFrame(arrival_rows, columns=["arrival_datetime", "encounter", "service"])
        .sort_values("arrival_datetime")
        .set_index("arrival_datetime")
    )
    list_df = pd.DataFrame(list_rows, columns=["encounter", "service", "intended_date"])
    return train_df, list_df


def _fixture_hourly(n_days=30):
    """Ten patients listed at A each day, plus unlisted electives at A and B."""
    start = date(2024, 1, 1)
    arrivals, listed = [], []
    for d in range(n_days):
        day = start + timedelta(days=d)
        at = lambda h, m=0, dd=0: datetime.combine(  # noqa: E731
            day + timedelta(days=dd), time(h, m)
        )
        for i in range(10):
            enc = f"L{d}-{i}"
            listed.append((enc, "A", day))
            if i < 4:
                arrivals.append((at(8, 30), enc, "A"))
            elif i < 6:
                arrivals.append((at(14, 10), enc, "A"))
            elif i == 6:
                arrivals.append((at(10, 0), enc, "B"))
            elif i == 7:
                pass  # never arrives
            elif i == 8:
                arrivals.append((at(20, 0, -1), enc, "A"))  # arrived the evening before
            else:
                arrivals.append((at(9, 0, 1), enc, "A"))  # moved to the next day
        arrivals.append((at(12, 0), f"UA{d}", "A"))
        arrivals.append((at(11, 0), f"UB{d}", "B"))
    return _rows_to_frames(arrivals, listed)


def _fixture_worked_example():
    """Worked example from the decision comment on issue 249 (30-minute grid)."""
    day = date(2024, 1, 10)  # a Wednesday
    at = lambda h, m=0: datetime.combine(day, time(h, m))  # noqa: E731
    arrivals, listed = [], []
    spec = [
        (40, "A", at(8)),  # at A before 09:30
        (5, "B", at(8)),  # at B before 09:30
        (5, "B", at(12)),  # at B after 09:30
        (20, "A", at(12)),  # at A inside the window
        (10, "A", at(18)),  # at A after the window
        (20, None, None),  # never
    ]
    n = 0
    for count, svc, ts in spec:
        for _ in range(count):
            enc = f"W{n}"
            n += 1
            listed.append((enc, "A", day))
            if svc is not None:
                arrivals.append((ts, enc, svc))
    return _rows_to_frames(arrivals, listed)


def _not_admitted_frame(rows):
    return (
        pd.DataFrame(rows, columns=["arrival_datetime", "encounter"])
        .sort_values("arrival_datetime")
        .set_index("arrival_datetime")
    )


def _fixture_not_admitted_example():
    """Worked example from the s_k comment on issue 249 (30-minute grid)."""
    day = date(2024, 1, 10)  # a Wednesday
    at = lambda h, m=0: datetime.combine(day, time(h, m))  # noqa: E731
    arrivals, not_admitted, listed = [], [], []
    spec = [
        (40, "A", at(8)),  # at A, needing a bed, before 09:30
        (20, "A", at(12)),  # at A, needing a bed, inside the window
        (25, None, at(8)),  # without needing a bed, before 09:30
        (5, None, at(11)),  # without needing a bed, after 09:30
        (10, None, None),  # never
    ]
    n = 0
    for count, svc, ts in spec:
        for _ in range(count):
            enc = f"N{n}"
            n += 1
            listed.append((enc, "A", day))
            if svc is not None:
                arrivals.append((ts, enc, svc))
            elif ts is not None:
                not_admitted.append((ts, enc))
    train_df, list_df = _rows_to_frames(arrivals, listed)
    return train_df, list_df, _not_admitted_frame(not_admitted)


def _fit(train_df, list_df, **kwargs):
    kwargs.setdefault("yta_time_interval", timedelta(hours=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ListBasedAdmissionPredictor(filters=FILTERS).fit(
            train_df, list_df=list_df, **kwargs
        )


def _day_inputs(train_df, list_df, day):
    day_list = list_df[list_df["intended_date"] == day]
    return day_list, train_df.reset_index()


class TestPoissonBinomial(unittest.TestCase):
    def test_matches_brute_force_with_unequal_q(self):
        q = [0.1, 0.5, 0.9, 0.3]
        expected = np.zeros(len(q) + 1)
        for outcome in itertools.product([0, 1], repeat=len(q)):
            prob = np.prod([qi if o else 1 - qi for qi, o in zip(q, outcome)])
            expected[sum(outcome)] += prob
        got = Distribution.from_bernoulli_probs(q).probabilities
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_equal_q_matches_binomial(self):
        got = Distribution.from_bernoulli_probs([0.3] * 12).probabilities
        np.testing.assert_allclose(got, binom(12, 0.3).pmf(np.arange(13)), atol=1e-12)

    def test_empty_and_degenerate(self):
        np.testing.assert_array_equal(
            Distribution.from_bernoulli_probs([]).probabilities, [1.0]
        )
        np.testing.assert_allclose(
            Distribution.from_bernoulli_probs([1.0, 0.0, 1.0]).probabilities,
            [0, 0, 1, 0],
        )
        with self.assertRaises(ValueError):
            Distribution.from_bernoulli_probs([0.5, 1.5])


class TestDateSpan(unittest.TestCase):
    def setUp(self):
        idx = pd.to_datetime(["2024-01-01 09:00", "2024-01-08 09:00"])
        self.df = pd.DataFrame(index=idx)
        self.interval = timedelta(hours=1)

    def test_default_unchanged(self):
        rates = time_varying_arrival_rates_by_weekday(self.df, self.interval)
        self.assertEqual(rates[0][time(9)], 2 / 2)

    def test_date_span_changes_divisors(self):
        rates = time_varying_arrival_rates_by_weekday(
            self.df,
            self.interval,
            date_span=(date(2024, 1, 1), date(2024, 1, 22)),
        )
        occ = weekday_occurrences_in_span(date(2024, 1, 1), date(2024, 1, 22))
        self.assertEqual(rates[0][time(9)], 2 / occ[0])
        self.assertEqual(occ[0], 4)

    def test_invalid_span(self):
        with self.assertRaises(ValueError):
            time_varying_arrival_rates_by_weekday(
                self.df, self.interval, date_span=(date(2024, 2, 1), date(2024, 1, 1))
            )


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.list_df = _fixture_hourly()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.model = ListBasedAdmissionPredictor(filters=FILTERS).fit(
                cls.train_df,
                list_df=cls.list_df,
                yta_time_interval=timedelta(hours=1),
                num_days=30,
            )
        cls.fit_warnings = [str(w.message) for w in caught]

    def test_list_probabilities_recovered(self):
        w = self.model.weights["A"]
        p, r = w["list_interval_probs"], w["list_other_service_probs"]
        self.assertEqual(w["list_count"], 270)
        self.assertEqual(len(p), 24)
        expected_p = np.zeros(24)
        expected_p[8], expected_p[14] = 4 / 9, 2 / 9
        expected_r = np.zeros(24)
        expected_r[10] = 1 / 9
        np.testing.assert_allclose(p, expected_p)
        np.testing.assert_allclose(r, expected_r)
        # Without arrivals_not_admitted, s_k is all zeros.
        np.testing.assert_array_equal(w["list_not_admitted_probs"], np.zeros(24))
        self.assertLessEqual(float(np.sum(p + r)), 1.0)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_off_list_rates_exclude_listed_arrivals(self):
        res = self.model.weights["A"]["off_list_arrival_rates_dict"]
        expected = {time(9): 1.0, time(12): 1.0, time(20): 1.0}
        for t, rate in res.items():
            self.assertAlmostEqual(rate, expected.get(t, 0.0), msg=str(t))

    def test_all_elective_rates_match_direct(self):
        direct = DirectAdmissionPredictor(filters=FILTERS).fit(
            self.train_df, yta_time_interval=timedelta(hours=1), num_days=30
        )
        for key in ("A", "B"):
            self.assertEqual(
                self.model.weights[key]["arrival_rates_dict"],
                direct.weights[key]["arrival_rates_dict"],
            )
            self.assertEqual(
                self.model.weights[key]["arrival_rates_by_weekday"],
                direct.weights[key]["arrival_rates_by_weekday"],
            )

    def test_weekday_off_list_rates_use_full_span(self):
        full_df = self.train_df[self.train_df["service"] == "A"]
        occ = weekday_occurrences_in_span(
            full_df.index.date.min(), full_df.index.date.max()
        )
        by_weekday = self.model.weights["A"]["off_list_arrival_rates_by_weekday"]
        unlisted_days = pd.to_datetime(
            [d for d in full_df.index[full_df.index.hour == 12].date]
        )
        for d in range(7):
            n_days = int((unlisted_days.weekday == d).sum())
            self.assertAlmostEqual(by_weekday[d][time(12)], n_days / occ[d])

    def test_service_without_list_falls_back(self):
        w = self.model.weights["B"]
        self.assertNotIn("list_interval_probs", w)
        self.assertEqual(w["off_list_arrival_rates_dict"], w["arrival_rates_dict"])
        # B has no listed patients at all: counted, but no warning.
        self.assertEqual(self.model.no_list_services, ["B"])
        self.assertEqual(self.model.no_list_filter_count, 1)
        self.assertEqual(self.model.list_fallback_services, [])
        self.assertEqual(self.fit_warnings_about_min_count, [])

    @property
    def fit_warnings_about_min_count(self):
        return [m for m in self.fit_warnings if "min_list_count" in m]

    def test_min_list_count_threshold(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ListBasedAdmissionPredictor(filters=FILTERS).fit(
                self.train_df,
                list_df=self.list_df,
                yta_time_interval=timedelta(hours=1),
                num_days=30,
                min_list_count=271,
            )
        self.assertNotIn("list_interval_probs", model.weights["A"])
        # Some listed patients but too few: warned about once, named.
        self.assertEqual(model.list_fallback_services, ["A"])
        self.assertEqual(model.low_list_count_filter_count, 1)
        self.assertEqual(model.weights["A"]["list_count"], 270)
        messages = [
            str(w.message) for w in caught if "min_list_count" in str(w.message)
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("'A'", messages[0])
        self.assertNotIn("'B'", messages[0])
        model = _fit(self.train_df, self.list_df, num_days=30, min_list_count=270)
        self.assertIn("list_interval_probs", model.weights["A"])

    def test_no_list_df_is_direct_fit(self):
        model = _fit(self.train_df, None, num_days=30)
        direct = DirectAdmissionPredictor(filters=FILTERS).fit(
            self.train_df, yta_time_interval=timedelta(hours=1), num_days=30
        )
        self.assertEqual(model.weights, direct.weights)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            _fit(self.train_df, self.list_df.drop(columns=["service"]))


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.list_df = _fixture_hourly()
        cls.model = _fit(cls.train_df, cls.list_df, num_days=30)
        cls.day = date(2024, 1, 15)  # Monday
        cls.day_list, cls.arrivals = _day_inputs(cls.train_df, cls.list_df, cls.day)
        cls.day_list = cls.day_list[cls.day_list["service"] == "A"]

    def _off_list_rate(self, key, day, start_hour, n_hours):
        w = self.model.weights[key]
        lam = 0.0
        for i in range(n_hours):
            h = start_hour + i
            if h < 24:
                lam += w["off_list_arrival_rates_by_weekday"][day.weekday()][time(h)]
            else:
                nxt = day + timedelta(days=1)
                lam += w["arrival_rates_by_weekday"][nxt.weekday()][time(h - 24)]
        return lam

    def test_without_day_list_identical_to_direct(self):
        direct = DirectAdmissionPredictor(filters=FILTERS).fit(
            self.train_df, yta_time_interval=timedelta(hours=1), num_days=30
        )
        kwargs = dict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            prediction_date=self.day,
        )
        got = self.model.predict(filter_keys=["A", "B"], **kwargs)
        want = direct.predict(filter_keys=["A", "B"], **kwargs)
        for key in ("A", "B"):
            pd.testing.assert_frame_equal(got[key], want[key])
            self.assertEqual(
                self.model.predict_mean(filter_key=key, **kwargs),
                direct.predict_mean(filter_key=key, **kwargs),
            )

    def test_empty_day_list_equals_off_list_poisson(self):
        empty = self.day_list.iloc[0:0]
        lam = self._off_list_rate("A", self.day, 9, 8)
        pmf = self.model.predict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            filter_keys="A",
            prediction_date=self.day,
            day_list=empty,
            arrivals=self.arrivals,
        )["A"]["agg_proba"].to_numpy()
        np.testing.assert_allclose(pmf.sum(), 1.0)
        np.testing.assert_allclose(pmf[:5], poisson.pmf(np.arange(5), lam), atol=1e-12)

    def test_everyone_arrived_equals_off_list_poisson(self):
        arrived = self.day_list[self.day_list["encounter"].str.endswith(("-0", "-1"))]
        lam = self._off_list_rate("A", self.day, 15, 12)  # past midnight
        pmf = self.model.predict(
            prediction_time=(15, 0),
            prediction_window=timedelta(hours=12),
            filter_keys="A",
            prediction_date=self.day,
            day_list=arrived,
            arrivals=self.arrivals,
        )["A"]["agg_proba"].to_numpy()
        np.testing.assert_allclose(pmf[:5], poisson.pmf(np.arange(5), lam), atol=1e-12)

    def test_midnight_crossing_uses_all_elective_rates_after_midnight(self):
        empty = self.day_list.iloc[0:0]
        expected = self._off_list_rate("A", self.day, 20, 16)
        mean = self.model.predict_mean(
            prediction_time=(20, 0),
            prediction_window=timedelta(hours=16),
            filter_key="A",
            prediction_date=self.day,
            day_list=empty,
            arrivals=self.arrivals,
        )
        self.assertAlmostEqual(mean, expected)
        # Listed arrivals are only in the all-elective rates, so using the
        # off-list rates after midnight would give a smaller number.
        w = self.model.weights["A"]
        tue = (self.day + timedelta(days=1)).weekday()
        self.assertGreater(
            w["arrival_rates_by_weekday"][tue][time(8)],
            w["off_list_arrival_rates_by_weekday"][tue][time(8)],
        )

    def test_mean_is_sum_of_q_plus_lambda(self):
        w = self.model.weights["A"]
        p, r = w["list_interval_probs"], w["list_other_service_probs"]
        t, n_window = 7, 8
        q = p[t : t + n_window].sum() / (1 - (p[:t] + r[:t]).sum())
        arrived = set(
            self.arrivals.loc[
                self.arrivals["arrival_datetime"]
                <= datetime.combine(self.day, time(7)),
                "encounter",
            ]
        )
        n_pending = int((~self.day_list["encounter"].isin(arrived)).sum())
        lam = self._off_list_rate("A", self.day, 7, 8)
        mean = self.model.predict_mean(
            prediction_time=(7, 0),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.day_list,
            arrivals=self.arrivals,
        )
        self.assertAlmostEqual(mean, n_pending * q + lam)
        pmf = self.model.predict(
            prediction_time=(7, 0),
            prediction_window=timedelta(hours=8),
            filter_keys="A",
            prediction_date=self.day,
            day_list=self.day_list,
            arrivals=self.arrivals,
        )["A"]
        values = pmf.index.to_numpy()
        np.testing.assert_allclose(pmf["agg_proba"].sum(), 1.0)
        self.assertAlmostEqual(float((values * pmf["agg_proba"]).sum()), mean, places=5)

    def test_conditioned_q_equals_unconditioned_at_midnight(self):
        w = self.model.weights["A"]
        p = w["list_interval_probs"]
        n = len(self.day_list) - 1  # i8 arrived the evening before
        mean = self.model.predict_mean(
            prediction_time=(0, 0),
            prediction_window=timedelta(hours=12),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.day_list,
            arrivals=self.arrivals,
        )
        lam = self._off_list_rate("A", self.day, 0, 12)
        self.assertAlmostEqual(mean, n * p[:12].sum() + lam)

    def test_q_is_capped_when_denominator_is_zero(self):
        train_df, list_df = _fixture_hourly()
        # Everyone listed arrives at 08:30 at A.
        list_df = list_df[list_df["encounter"].str.endswith("-0")]
        train_df = train_df[
            train_df["encounter"].str.endswith("-0")
            | train_df["encounter"].str.startswith("U")
        ]
        model = _fit(train_df, list_df, num_days=30, min_list_count=1)
        day_list = pd.DataFrame(
            {"encounter": ["pending"], "service": ["A"], "intended_date": [self.day]}
        )
        empty_arrivals = train_df.reset_index().iloc[0:0]
        mean = model.predict_mean(
            prediction_time=(12, 0),
            prediction_window=timedelta(hours=4),
            filter_key="A",
            prediction_date=self.day,
            day_list=day_list,
            arrivals=empty_arrivals,
        )
        lam = model.weights["A"]["off_list_arrival_rates_by_weekday"][0]
        lam = sum(lam[time(h)] for h in range(12, 16))
        self.assertGreaterEqual(mean, lam)
        self.assertLessEqual(mean, lam + 1.0)

    def test_service_without_list_gets_poisson_with_day_list(self):
        direct = DirectAdmissionPredictor(filters=FILTERS).fit(
            self.train_df, yta_time_interval=timedelta(hours=1), num_days=30
        )
        day_list = pd.DataFrame(
            {"encounter": ["x"], "service": ["B"], "intended_date": [self.day]}
        )
        kwargs = dict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            prediction_date=self.day,
        )
        got = self.model.predict(
            filter_keys="B", day_list=day_list, arrivals=self.arrivals, **kwargs
        )["B"]
        pd.testing.assert_frame_equal(
            got, direct.predict(filter_keys="B", **kwargs)["B"]
        )
        self.assertEqual(
            self.model.predict_mean(
                filter_key="B", day_list=day_list, arrivals=self.arrivals, **kwargs
            ),
            direct.predict_mean(filter_key="B", **kwargs),
        )

    def test_requires_prediction_date_and_arrivals(self):
        common = dict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            filter_keys="A",
            day_list=self.day_list,
        )
        with self.assertRaises(ValueError):
            self.model.predict(arrivals=self.arrivals, **common)
        with self.assertRaises(ValueError):
            self.model.predict(prediction_date=self.day, **common)

    def test_pooled_fallback_warns_and_strict_raises(self):
        model = _fit(
            self.train_df, self.list_df, num_days=30, stratify_by_weekday=False
        )
        kwargs = dict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.day_list,
            arrivals=self.arrivals,
        )
        with self.assertWarns(UserWarning):
            model.predict_mean(**kwargs)
        with self.assertRaises(ValueError):
            model.predict_mean(strict_prediction_date=True, **kwargs)

    def test_pickle_round_trip(self):
        restored = pickle.loads(pickle.dumps(self.model))
        kwargs = dict(
            prediction_time=(9, 0),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.day_list,
            arrivals=self.arrivals,
        )
        self.assertEqual(
            restored.predict_mean(**kwargs), self.model.predict_mean(**kwargs)
        )

    def test_pickled_direct_predictor_still_runs(self):
        direct = DirectAdmissionPredictor(filters=FILTERS).fit(
            self.train_df, yta_time_interval=timedelta(hours=1), num_days=30
        )
        restored = pickle.loads(pickle.dumps(direct))
        self.assertEqual(
            restored.predict_mean(
                prediction_time=(9, 0),
                prediction_window=timedelta(hours=8),
                filter_key="A",
            ),
            direct.predict_mean(
                prediction_time=(9, 0),
                prediction_window=timedelta(hours=8),
                filter_key="A",
            ),
        )


class TestWorkedExample(unittest.TestCase):
    """Cross-service arrivals before and after t (issue 249 decision comment)."""

    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.list_df = _fixture_worked_example()
        cls.model = _fit(
            cls.train_df,
            cls.list_df,
            yta_time_interval=timedelta(minutes=30),
        )
        cls.day = date(2024, 1, 10)
        cls.arrivals = cls.train_df.reset_index()

    def test_stored_probabilities(self):
        w = self.model.weights["A"]
        self.assertEqual(w["list_count"], 100)
        self.assertAlmostEqual(w["list_interval_probs"].sum(), 0.70)
        self.assertAlmostEqual(w["list_other_service_probs"].sum(), 0.10)
        self.assertLessEqual(
            (w["list_interval_probs"] + w["list_other_service_probs"]).sum(), 1.0
        )

    def test_expected_arrivals_match_truth(self):
        kwargs = dict(
            prediction_time=(9, 30),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.list_df,
            arrivals=self.arrivals,
        )
        # 55 patients pending at 09:30 and 20 of them arrive at A in the window.
        self.assertAlmostEqual(self.model.predict_mean(**kwargs), 20.0)
        pmf = self.model.predict(
            prediction_time=(9, 30),
            prediction_window=timedelta(hours=8),
            filter_keys="A",
            prediction_date=self.day,
            day_list=self.list_df,
            arrivals=self.arrivals,
        )["A"]["agg_proba"].to_numpy()
        np.testing.assert_allclose(
            pmf[:56], binom(55, 0.2 / 0.55).pmf(np.arange(56)), atol=1e-12
        )

    def test_dropping_ignores_later_arrivals_and_other_services(self):
        # Arrivals after the moment do not change the answer.
        later_only = self.arrivals[
            self.arrivals["arrival_datetime"] > datetime(2024, 1, 10, 9, 30)
        ]
        mean_all = self.model.predict_mean(
            prediction_time=(9, 30),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.list_df,
            arrivals=self.arrivals,
        )
        mean_later = self.model.predict_mean(
            prediction_time=(9, 30),
            prediction_window=timedelta(hours=8),
            filter_key="A",
            prediction_date=self.day,
            day_list=self.list_df,
            arrivals=later_only,
        )
        # With no one dropped, all 100 are pending and the same q applies.
        self.assertAlmostEqual(mean_later, 100 * 0.2 / 0.55)
        self.assertLess(mean_all, mean_later)


class TestNotAdmittedWorkedExample(unittest.TestCase):
    """Listed patients arriving without needing a bed (issue 249 s_k comment)."""

    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.list_df, cls.not_admitted = _fixture_not_admitted_example()
        cls.model = _fit(
            cls.train_df,
            cls.list_df,
            arrivals_not_admitted=cls.not_admitted,
            yta_time_interval=timedelta(minutes=30),
        )
        cls.day = date(2024, 1, 10)
        # At live inference, arrivals include patients who did not need a bed.
        cls.arrivals = pd.concat(
            [cls.train_df.reset_index(), cls.not_admitted.reset_index()],
            ignore_index=True,
        )
        cls.kwargs = dict(
            prediction_time=(9, 30),
            prediction_window=timedelta(hours=8),
            prediction_date=cls.day,
            day_list=cls.list_df,
            arrivals=cls.arrivals,
        )

    def test_stored_probabilities(self):
        w = self.model.weights["A"]
        p, r, s = (
            w["list_interval_probs"],
            w["list_other_service_probs"],
            w["list_not_admitted_probs"],
        )
        self.assertEqual(w["list_count"], 100)
        self.assertEqual(len(s), len(p))
        self.assertAlmostEqual(p.sum(), 0.60)
        self.assertAlmostEqual(r.sum(), 0.0)
        self.assertAlmostEqual(s[16], 0.25)  # 08:00
        self.assertAlmostEqual(s[22], 0.05)  # 11:00
        self.assertLessEqual((p + r + s).sum(), 1.0)

    def test_expected_arrivals_match_truth(self):
        # 35 patients pending at 09:30 and 20 of them arrive at A in the window.
        self.assertAlmostEqual(
            self.model.predict_mean(filter_key="A", **self.kwargs), 20.0
        )
        pmf = self.model.predict(filter_keys="A", **self.kwargs)["A"][
            "agg_proba"
        ].to_numpy()
        np.testing.assert_allclose(
            pmf[:36], binom(35, 0.20 / 0.35).pmf(np.arange(36)), atol=1e-12
        )

    def test_without_arrivals_not_admitted_q_is_too_low(self):
        model = _fit(
            self.train_df, self.list_df, yta_time_interval=timedelta(minutes=30)
        )
        np.testing.assert_array_equal(
            model.weights["A"]["list_not_admitted_probs"], np.zeros(48)
        )
        self.assertAlmostEqual(
            model.predict_mean(filter_key="A", **self.kwargs), 35 * 0.20 / 0.60
        )


class TestNotAdmittedFit(unittest.TestCase):
    """Precedence, arrivals before the day, and rates unchanged by s_k."""

    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.list_df = _fixture_hourly()
        start = date(2024, 1, 1)
        rows = []
        for d in range(30):
            day = start + timedelta(days=d)
            if d < 15:
                # i7 (never in train_df) arrives without needing a bed on the day.
                rows.append((datetime.combine(day, time(7)), f"L{d}-7"))
            else:
                # ... or the evening before.
                rows.append(
                    (datetime.combine(day - timedelta(days=1), time(20)), f"L{d}-7")
                )
            # i0 is in train_df at 08:30, so this row is ignored.
            rows.append((datetime.combine(day, time(6)), f"L{d}-0"))
            # Not on any list, so this row is ignored.
            rows.append((datetime.combine(day, time(10)), f"X{d}"))
        cls.not_admitted = _not_admitted_frame(rows)
        cls.model = _fit(
            cls.train_df,
            cls.list_df,
            arrivals_not_admitted=cls.not_admitted,
            num_days=30,
        )
        cls.baseline = _fit(cls.train_df, cls.list_df, num_days=30)

    def test_list_probabilities(self):
        w = self.model.weights["A"]
        n = 270 - 15  # i7 arriving the evening before leaves the denominator
        self.assertEqual(w["list_count"], n)
        expected_p = np.zeros(24)
        expected_p[8], expected_p[14] = 120 / n, 60 / n
        expected_r = np.zeros(24)
        expected_r[10] = 30 / n
        expected_s = np.zeros(24)
        expected_s[7] = 15 / n
        np.testing.assert_allclose(w["list_interval_probs"], expected_p)
        np.testing.assert_allclose(w["list_other_service_probs"], expected_r)
        np.testing.assert_allclose(w["list_not_admitted_probs"], expected_s)

    def test_rates_unchanged(self):
        for key in ("A", "B"):
            for rates in (
                "arrival_rates_dict",
                "arrival_rates_by_weekday",
                "off_list_arrival_rates_dict",
                "off_list_arrival_rates_by_weekday",
            ):
                self.assertEqual(
                    self.model.weights[key][rates],
                    self.baseline.weights[key][rates],
                    msg=f"{key} {rates}",
                )

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            _fit(self.train_df, None, arrivals_not_admitted=self.not_admitted)
        with self.assertRaises(TypeError):
            _fit(
                self.train_df,
                self.list_df,
                arrivals_not_admitted=self.not_admitted.reset_index(),
            )
        with self.assertRaises(ValueError):
            _fit(
                self.train_df,
                self.list_df,
                arrivals_not_admitted=self.not_admitted.rename(
                    columns={"encounter": "id"}
                ),
            )


if __name__ == "__main__":
    unittest.main()
