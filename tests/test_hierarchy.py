"""Tests for hierarchical demand prediction functionality."""

import numpy as np

from patientflow.predict.hierarchy import (
    DemandPredictor,
    DemandPrediction,
)
from patientflow.predict.subspecialty import (
    SubspecialtyPredictionInputs,
    FlowInputs,
)


class TestDemandPrediction:
    """Test the DemandPrediction dataclass."""

    def test_with_negative_offset(self):
        """Test DemandPrediction with negative offset (net flow case)."""
        pred = DemandPrediction(
            entity_id="test_net",
            entity_type="net_flow",
            probabilities=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            expected_value=-0.5,
            percentiles={50: 0, 75: 1, 90: 2, 95: 2, 99: 2},
            offset=-2,  # Support starts at -2
        )

        assert pred.offset == -2
        assert pred.expected_value == -0.5


class TestDemandPredictor:
    """Test the DemandPredictor class."""

    def test_convolution(self):
        """Test basic convolution of two distributions."""
        predictor = DemandPredictor()

        p1 = np.array([0.5, 0.5])  # 0 or 1 with equal probability
        p2 = np.array([0.3, 0.7])  # 0 or 1 with different probabilities

        result = predictor._convolve(p1, p2)

        # Result should have support [0, 2]
        assert len(result) == 3
        assert np.allclose(result.sum(), 1.0)
        # P(sum=0) = 0.5 * 0.3 = 0.15
        assert np.isclose(result[0], 0.15)

    def test_expected_value_without_offset(self):
        """Test expected value calculation without offset."""
        predictor = DemandPredictor()

        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Mean should be 2.0
        expected = predictor._expected_value(p, offset=0)

        assert np.isclose(expected, 2.0)

    def test_expected_value_with_offset(self):
        """Test expected value calculation with negative offset."""
        predictor = DemandPredictor()

        # Same distribution but now support is [-2, -1, 0, 1, 2]
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        expected = predictor._expected_value(p, offset=-2)

        # Mean should be 0.0 (symmetric around 0)
        assert np.isclose(expected, 0.0)

    def test_percentiles_with_offset(self):
        """Test percentile calculation with negative offset."""
        predictor = DemandPredictor()

        # Support: [-2, -1, 0, 1, 2]
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        percentiles = predictor._percentiles(p, [50], offset=-2)

        # Median should account for offset
        assert percentiles[50] == 0  # Index 2, offset -2 -> value 0

    def test_net_flow_pmf_deterministic(self):
        """Test net flow PMF with deterministic arrivals and departures."""
        predictor = DemandPredictor()

        # Always 3 arrivals, always 2 departures -> net flow always 1
        p_arrivals = np.array([0.0, 0.0, 0.0, 1.0])
        p_departures = np.array([0.0, 0.0, 1.0])

        p_net, offset = predictor._compute_net_flow_pmf(p_arrivals, p_departures)

        # Should have a single peak at net flow = 1
        assert np.isclose(p_net.sum(), 1.0, atol=1e-6)
        mode_idx = np.argmax(p_net)
        mode_value = mode_idx + offset
        assert mode_value == 1
        assert np.isclose(p_net[mode_idx], 1.0)

    def test_net_flow_pmf_negative_result(self):
        """Test net flow PMF when result is negative."""
        predictor = DemandPredictor()

        # Always 1 arrival, always 3 departures -> net flow always -2
        p_arrivals = np.array([0.0, 1.0])
        p_departures = np.array([0.0, 0.0, 0.0, 1.0])

        p_net, offset = predictor._compute_net_flow_pmf(p_arrivals, p_departures)

        assert np.isclose(p_net.sum(), 1.0, atol=1e-6)
        mode_idx = np.argmax(p_net)
        mode_value = mode_idx + offset
        assert mode_value == -2

    def test_net_flow_pmf_stochastic(self):
        """Test net flow PMF with stochastic distributions."""
        predictor = DemandPredictor()

        # Simple stochastic case
        p_arrivals = np.array([0.3, 0.5, 0.2])  # E[A] = 0.9
        p_departures = np.array([0.2, 0.6, 0.2])  # E[D] = 1.0

        p_net, offset = predictor._compute_net_flow_pmf(p_arrivals, p_departures)

        # Should sum to 1
        assert np.isclose(p_net.sum(), 1.0, atol=1e-6)

        # Expected value should be approximately E[A] - E[D] = -0.1
        expected = predictor._expected_value(p_net, offset)
        assert np.isclose(expected, -0.1, atol=1e-10)

    def test_predict_subspecialty(self):
        """Test subspecialty-level prediction."""
        predictor = DemandPredictor()

        inputs = SubspecialtyPredictionInputs(
            subspecialty_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current",
                    flow_type="pmf",
                    distribution=np.array([0.5, 0.3, 0.2]),
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.5
                ),
            },
            outflows={
                "departures": FlowInputs(
                    flow_id="departures", flow_type="poisson", distribution=2.0
                ),
            },
        )

        bundle = predictor.predict_subspecialty("cardio", inputs)

        assert bundle.entity_id == "cardio"
        assert bundle.entity_type == "subspecialty"

        # Net flow expected should match arrivals - departures
        expected_diff = (
            bundle.arrivals.expected_value - bundle.departures.expected_value
        )
        assert np.isclose(bundle.net_flow.expected_value, expected_diff, atol=1e-6)
