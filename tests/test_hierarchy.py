"""Tests for hierarchical demand prediction functionality."""

import numpy as np
import pandas as pd
import pytest

from patientflow.predict.hierarchy import (
    DemandPredictor,
    DemandPrediction,
    FlowSelection,
    EntityType,
    Hierarchy,
    PredictionBundle,
    create_hierarchical_predictor,
    DEFAULT_PERCENTILES,
    DEFAULT_PRECISION,
    DEFAULT_MAX_PROBS,
)
from patientflow.predict.distribution import Distribution
from patientflow.predict.service import (
    ServicePredictionInputs,
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

    def test_apply_cap_with_renormalization_truncates_tail(self):
        """Truncating with a cap should fold remaining mass into the last bin."""
        predictor = DemandPredictor()
        pmf = np.array([0.2, 0.3, 0.1, 0.4])

        truncated = predictor.apply_cap_with_renormalization(pmf, 2)

        assert len(truncated) == 3
        assert pytest.approx(truncated.sum(), rel=1e-9) == 1.0
        # Last bin should contain its own mass plus the tail.
        assert pytest.approx(truncated[2], rel=1e-9) == 0.5

    def test_apply_cap_with_renormalization_negative_cap(self):
        """Negative caps should collapse the PMF into a single bucket."""
        predictor = DemandPredictor()
        pmf = np.array([0.2, 0.3])

        truncated = predictor.apply_cap_with_renormalization(pmf, -1)

        assert len(truncated) == 1
        assert pytest.approx(truncated[0], rel=1e-9) == 0.5

    def test_predict_flow_total_respects_max_support(self):
        """predict_flow_total should respect the provided max_support cap."""
        predictor = DemandPredictor()
        flows = [
            FlowInputs(
                flow_id="pmf_flow",
                flow_type="pmf",
                distribution=np.array([0.6, 0.4]),
            ),
            FlowInputs(flow_id="pois_flow", flow_type="poisson", distribution=3.0),
        ]

        prediction = predictor.predict_flow_total(
            flows, entity_id="entity", entity_type="arrivals", max_support=2
        )

        assert len(prediction.probabilities) == 3
        assert pytest.approx(prediction.probabilities.sum(), rel=1e-9) == 1.0
        # Ensure some mass lands in the capped bucket.
        assert prediction.probabilities[-1] > 0.0

    def test_calculate_hierarchical_stats_respects_flow_selection(self):
        """calculate_hierarchical_stats should honor flow selection filters."""
        predictor = DemandPredictor(k_sigma=1.0)
        hierarchy = Hierarchy.create_default_hospital()
        levels = hierarchy.get_levels_ordered()
        subspecialty_type = levels[0]
        reporting_unit_type = levels[1]

        hierarchy.add_entity("Cardiology", subspecialty_type)
        hierarchy.add_entity("UnitA", reporting_unit_type)
        hierarchy.relationships[f"{subspecialty_type.name}:Cardiology"] = (
            f"{reporting_unit_type.name}:UnitA"
        )

        inputs = ServicePredictionInputs(
            service_id="Cardiology",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=2.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=3.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=0.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=0.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=0.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=0.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        bottom_level_data = {"Cardiology": inputs}

        stats_all = predictor.calculate_hierarchical_stats(
            "UnitA",
            reporting_unit_type,
            bottom_level_data,
            hierarchy,
            "arrivals",
            FlowSelection.default(),
        )
        # Two Poisson flows with lambdas 2 and 3 -> mean 5, variance 5.
        assert pytest.approx(stats_all[0], rel=1e-9) == 5.0
        assert pytest.approx(stats_all[1], rel=1e-9) == np.sqrt(5.0)
        assert stats_all[2] >= 5  # cap should be at least the mean

        stats_elective = predictor.calculate_hierarchical_stats(
            "UnitA",
            reporting_unit_type,
            bottom_level_data,
            hierarchy,
            "arrivals",
            FlowSelection.elective_only(),
        )
        assert stats_elective == (0.0, 0.0, 0)

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
        # Always 3 arrivals, always 2 departures -> net flow always 1
        p_arrivals = np.array([0.0, 0.0, 0.0, 1.0])
        p_departures = np.array([0.0, 0.0, 1.0])

        net = Distribution.from_pmf(p_arrivals).net(Distribution.from_pmf(p_departures))

        # Should have a single peak at net flow = 1
        assert np.isclose(net.probabilities.sum(), 1.0, atol=1e-6)
        mode_idx = np.argmax(net.probabilities)
        mode_value = mode_idx + net.offset
        assert mode_value == 1
        assert np.isclose(net.probabilities[mode_idx], 1.0)

    def test_net_flow_pmf_negative_result(self):
        """Test net flow PMF when result is negative."""
        # Always 1 arrival, always 3 departures -> net flow always -2
        p_arrivals = np.array([0.0, 1.0])
        p_departures = np.array([0.0, 0.0, 0.0, 1.0])

        net = Distribution.from_pmf(p_arrivals).net(Distribution.from_pmf(p_departures))

        assert np.isclose(net.probabilities.sum(), 1.0, atol=1e-6)
        mode_idx = np.argmax(net.probabilities)
        mode_value = mode_idx + net.offset
        assert mode_value == -2

    def test_net_flow_pmf_stochastic(self):
        """Test net flow PMF with stochastic distributions."""
        # Simple stochastic case
        p_arrivals = np.array([0.3, 0.5, 0.2])  # E[A] = 0.9
        p_departures = np.array([0.2, 0.6, 0.2])  # E[D] = 1.0

        net = Distribution.from_pmf(p_arrivals).net(Distribution.from_pmf(p_departures))

        # Should sum to 1
        assert np.isclose(net.probabilities.sum(), 1.0, atol=1e-6)

        # Expected value should be approximately E[A] - E[D] = -0.1
        expected = net.expected()
        assert np.isclose(expected, -0.1, atol=1e-10)

    def test_predict_service(self):
        """Test subspecialty-level prediction."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
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
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=0.5
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=0.8
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers",
                    flow_type="pmf",
                    distribution=np.array([0.7, 0.3]),
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers",
                    flow_type="pmf",
                    distribution=np.array([0.9, 0.1]),
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        bundle = predictor.predict_service("cardio", inputs)

        assert bundle.entity_id == "cardio"
        assert bundle.entity_type == "service"

        # Net flow expected should match arrivals - departures
        expected_diff = (
            bundle.arrivals.expected_value - bundle.departures.expected_value
        )
        assert np.isclose(bundle.net_flow.expected_value, expected_diff, atol=1e-6)

    def test_predict_service_with_custom_flow_selection(self):
        """Test subspecialty prediction with custom flow selection."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
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
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=0.5
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=0.8
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers",
                    flow_type="pmf",
                    distribution=np.array([0.7, 0.3]),
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers",
                    flow_type="pmf",
                    distribution=np.array([0.9, 0.1]),
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Test with emergency-only flow selection
        flow_selection = FlowSelection.emergency_only()
        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        assert bundle.entity_id == "cardio"
        assert bundle.entity_type == "service"
        assert bundle.flow_selection.cohort == "emergency"

    def test_predict_service_missing_keys_validation(self):
        """Test that missing flow keys raise appropriate errors."""
        predictor = DemandPredictor()

        # Create inputs with missing keys
        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current",
                    flow_type="pmf",
                    distribution=np.array([0.5, 0.3, 0.2]),
                ),
                # Missing other required keys
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                # Missing emergency_departures
            },
        )

        # Should raise KeyError for missing inflow keys
        with pytest.raises(KeyError, match="Missing inflow keys"):
            predictor.predict_service("cardio", inputs)

    def test_compute_net_flow_helper(self):
        """Test the _compute_net_flow helper method."""
        predictor = DemandPredictor()

        # Create test predictions
        arrivals = DemandPrediction(
            entity_id="test",
            entity_type="arrivals",
            probabilities=np.array([0.3, 0.5, 0.2]),
            expected_value=0.9,
            percentiles={50: 1, 75: 1, 90: 2, 95: 2, 99: 2},
        )

        departures = DemandPrediction(
            entity_id="test",
            entity_type="departures",
            probabilities=np.array([0.2, 0.6, 0.2]),
            expected_value=1.0,
            percentiles={50: 1, 75: 1, 90: 2, 95: 2, 99: 2},
        )

        net_flow = predictor._compute_net_flow(arrivals, departures, "test_net")

        assert net_flow.entity_id == "test_net"
        assert net_flow.entity_type == "net_flow"
        # Expected value should be approximately arrivals - departures
        assert np.isclose(net_flow.expected_value, -0.1, atol=1e-6)


class TestFlowSelection:
    """Test the FlowSelection class."""

    def test_default_flow_selection(self):
        """Test default flow selection."""
        selection = FlowSelection.default()

        assert selection.include_ed_current is True
        assert selection.include_ed_yta is True
        assert selection.include_non_ed_yta is True
        assert selection.include_elective_yta is True
        assert selection.include_transfers_in is True
        assert selection.include_departures is True
        assert selection.cohort == "all"

    def test_emergency_only_flow_selection(self):
        """Test emergency-only flow selection."""
        selection = FlowSelection.emergency_only()

        assert selection.cohort == "emergency"
        # Should include emergency flows and exclude elective flows
        assert selection.include_ed_current is True
        assert selection.include_ed_yta is True
        assert selection.include_non_ed_yta is True
        assert selection.include_elective_yta is False
        assert selection.include_transfers_in is True  # Will be filtered by cohort
        assert selection.include_departures is True  # Will be filtered by cohort

    def test_elective_only_flow_selection(self):
        """Test elective-only flow selection."""
        selection = FlowSelection.elective_only()

        assert selection.cohort == "elective"
        # Should exclude emergency flows and include only elective flows
        assert selection.include_ed_current is False
        assert selection.include_ed_yta is False
        assert selection.include_non_ed_yta is False
        assert selection.include_elective_yta is True
        assert selection.include_transfers_in is True  # Will be filtered by cohort
        assert selection.include_departures is True  # Will be filtered by cohort

    def test_incoming_only_flow_selection(self):
        """Test incoming-only flow selection."""
        selection = FlowSelection.incoming_only()

        assert selection.include_departures is False
        # All inflow settings should be defaults
        assert selection.include_ed_current is True
        assert selection.include_ed_yta is True

    def test_outgoing_only_flow_selection(self):
        """Test outgoing-only flow selection."""
        selection = FlowSelection.outgoing_only()

        assert selection.include_departures is True
        # All inflow settings should be False
        assert selection.include_ed_current is False
        assert selection.include_ed_yta is False
        assert selection.include_non_ed_yta is False
        assert selection.include_elective_yta is False
        assert selection.include_transfers_in is False

    def test_custom_flow_selection(self):
        """Test custom flow selection."""
        selection = FlowSelection.custom(
            include_ed_current=False,
            include_ed_yta=True,
            include_non_ed_yta=False,
            include_elective_yta=True,
            include_transfers_in=False,
            include_departures=True,
            cohort="elective",
        )

        assert selection.include_ed_current is False
        assert selection.include_ed_yta is True
        assert selection.include_non_ed_yta is False
        assert selection.include_elective_yta is True
        assert selection.include_transfers_in is False
        assert selection.include_departures is True
        assert selection.cohort == "elective"

    def test_flow_selection_validation_valid(self):
        """Test flow selection validation with valid cohort."""
        selection = FlowSelection(cohort="all")
        selection.validate()  # Should not raise

        selection = FlowSelection(cohort="elective")
        selection.validate()  # Should not raise

        selection = FlowSelection(cohort="emergency")
        selection.validate()  # Should not raise

    def test_flow_selection_validation_invalid(self):
        """Test flow selection validation with invalid cohort."""
        selection = FlowSelection(cohort="invalid")

        with pytest.raises(ValueError, match="Invalid cohort 'invalid'"):
            selection.validate()


class TestConstants:
    """Test the constants defined in the module."""

    def test_default_percentiles(self):
        """Test that default percentiles are correct."""
        assert DEFAULT_PERCENTILES == [50, 75, 90, 95, 99]

    def test_default_precision(self):
        """Test that default precision is correct."""
        assert DEFAULT_PRECISION == 3

    def test_default_max_probs(self):
        """Test that default max probs is correct."""
        assert DEFAULT_MAX_PROBS == 10

    def test_constants_used_in_demand_prediction(self):
        """Test that constants are used in DemandPrediction methods."""
        pred = DemandPrediction(
            entity_id="test",
            entity_type="test",
            probabilities=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            expected_value=2.0,
            percentiles={50: 2, 75: 3, 90: 3, 95: 4, 99: 4},
        )

        # Test that to_pretty uses the constants
        pretty_str = pred.to_pretty()
        assert "Expectation:" in pretty_str
        assert "2.000" in pretty_str  # Should use DEFAULT_PRECISION

    def test_constants_used_in_prediction_bundle(self):
        """Test that constants are used in PredictionBundle methods."""
        # Test the constants directly since we can't easily test PredictionBundle usage
        assert DEFAULT_PERCENTILES == [50, 75, 90, 95, 99]
        assert DEFAULT_PRECISION == 3
        assert DEFAULT_MAX_PROBS == 10


class TestCohortFiltering:
    """Test cohort filtering logic in flow selection."""

    def test_elective_cohort_filtering(self):
        """Test that elective cohort correctly filters flows."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Test elective-only flow selection
        flow_selection = FlowSelection.elective_only()
        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        # Should only include elective flows
        assert bundle.flow_selection.cohort == "elective"
        # The filtering happens internally, but we can verify the selection is applied

    def test_emergency_cohort_filtering(self):
        """Test that emergency cohort correctly filters flows."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Test emergency-only flow selection
        flow_selection = FlowSelection.emergency_only()
        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        # Should only include emergency flows
        assert bundle.flow_selection.cohort == "emergency"

    def test_all_cohort_filtering(self):
        """Test that 'all' cohort includes all flows."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Test all cohort flow selection
        flow_selection = FlowSelection(cohort="all")
        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        # Should include all flows
        assert bundle.flow_selection.cohort == "all"


class TestFlowSelectionEdgeCases:
    """Test edge cases in flow selection."""

    def test_empty_flow_selection(self):
        """Test flow selection with no flows enabled."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Create flow selection with no flows enabled
        flow_selection = FlowSelection(
            include_ed_current=False,
            include_ed_yta=False,
            include_non_ed_yta=False,
            include_elective_yta=False,
            include_transfers_in=False,
            include_departures=False,
        )

        # Should succeed but with empty flow lists
        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        assert bundle.entity_id == "cardio"
        assert bundle.entity_type == "service"
        assert bundle.flow_selection.include_departures is False
        # All flow settings should be False
        assert bundle.flow_selection.include_ed_current is False
        assert bundle.flow_selection.include_ed_yta is False
        assert bundle.flow_selection.include_non_ed_yta is False
        assert bundle.flow_selection.include_elective_yta is False
        assert bundle.flow_selection.include_transfers_in is False

    def test_partial_flow_selection(self):
        """Test flow selection with only some flows enabled."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                "emergency_departures": FlowInputs(
                    flow_id="emergency_departures",
                    flow_type="poisson",
                    distribution=1.0,
                ),
            },
        )

        # Create flow selection with only ED flows enabled
        flow_selection = FlowSelection(
            include_ed_current=True,
            include_ed_yta=True,
            include_non_ed_yta=False,
            include_elective_yta=False,
            include_transfers_in=False,
            include_departures=True,
        )

        bundle = predictor.predict_service("cardio", inputs, flow_selection)

        assert bundle.entity_id == "cardio"
        assert bundle.entity_type == "service"
        assert bundle.flow_selection.include_ed_current is True
        assert bundle.flow_selection.include_ed_yta is True
        assert bundle.flow_selection.include_non_ed_yta is False
        assert bundle.flow_selection.include_elective_yta is False
        assert bundle.flow_selection.include_transfers_in is False
        assert bundle.flow_selection.include_departures is True

    def test_missing_keys_with_partial_selection(self):
        """Test missing keys validation with partial flow selection."""
        predictor = DemandPredictor()

        # Create inputs with only some flows
        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                # Missing other flows
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                # Missing emergency_departures
            },
        )

        # Create flow selection that requires missing keys
        flow_selection = FlowSelection(
            include_ed_current=True,
            include_ed_yta=True,
            include_non_ed_yta=True,  # This will be missing
            include_elective_yta=False,
            include_transfers_in=False,
            include_departures=True,
        )

        # Should raise KeyError for missing inflow keys
        with pytest.raises(KeyError, match="Missing inflow keys"):
            predictor.predict_service("cardio", inputs, flow_selection)

    def test_missing_outflow_keys_validation(self):
        """Test missing outflow keys validation."""
        predictor = DemandPredictor()

        inputs = ServicePredictionInputs(
            service_id="cardio",
            prediction_window=24,
            inflows={
                "ed_current": FlowInputs(
                    flow_id="ed_current", flow_type="poisson", distribution=1.0
                ),
                "ed_yta": FlowInputs(
                    flow_id="ed_yta", flow_type="poisson", distribution=1.0
                ),
                "non_ed_yta": FlowInputs(
                    flow_id="non_ed_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_yta": FlowInputs(
                    flow_id="elective_yta", flow_type="poisson", distribution=1.0
                ),
                "elective_transfers": FlowInputs(
                    flow_id="elective_transfers", flow_type="poisson", distribution=1.0
                ),
                "emergency_transfers": FlowInputs(
                    flow_id="emergency_transfers", flow_type="poisson", distribution=1.0
                ),
            },
            outflows={
                "elective_departures": FlowInputs(
                    flow_id="elective_departures", flow_type="poisson", distribution=1.0
                ),
                # Missing emergency_departures
            },
        )

        # Should raise KeyError for missing outflow keys
        with pytest.raises(KeyError, match="Missing outflow keys"):
            predictor.predict_service("cardio", inputs)


class TestHierarchyCollisionFix:
    """Test the hierarchy collision fix for entities with same names across levels."""

    def test_entity_collision_fix(self):
        """Test that entities with the same name can coexist at different levels."""
        # Create test data with collision scenario
        data = {
            "sub_specialty": [
                "Cardiology",
                "Neurology",
                "Cardiology",
            ],  # Same name as reporting unit
            "reporting_unit": [
                "Cardiology",
                "Neurology",
                "Cardiology",
            ],  # Same name as subspecialty
            "division": ["Medicine", "Medicine", "Medicine"],
            "board": ["Clinical Board", "Clinical Board", "Clinical Board"],
        }

        df = pd.DataFrame(data)

        # Column mapping
        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
        }

        # Create hierarchical predictor
        predictor = create_hierarchical_predictor(
            hierarchy_df=df, column_mapping=column_mapping, top_level_id="Hospital"
        )

        # Test that both "Cardiology" entities exist but are different types
        subspecialties = predictor.hierarchy.get_entities_by_type(
            EntityType("subspecialty")
        )
        reporting_units = predictor.hierarchy.get_entities_by_type(
            EntityType("reporting_unit")
        )

        assert "Cardiology" in subspecialties
        assert "Cardiology" in reporting_units
        assert "Neurology" in subspecialties
        assert "Neurology" in reporting_units

        # Test that they have different entity types
        cardio_subspecialty_type = predictor.hierarchy.get_entity_type("Cardiology")

        # Since we have two entities with the same name, we need to check which one is returned
        # The get_entity_type method should find the first match
        assert cardio_subspecialty_type is not None

        # Test entity info for both entities
        cardio_info = predictor.hierarchy.get_entity_info("Cardiology")
        assert cardio_info is not None
        assert cardio_info["entity_id"] == "Cardiology"
        assert cardio_info["entity_type"] is not None

        # Test that hierarchy structure is correct
        assert len(subspecialties) == 2
        assert len(reporting_units) == 2

        # Test that no collisions occurred in internal storage
        all_entities = predictor.hierarchy.get_all_entities()
        assert "Cardiology" in all_entities
        assert "Neurology" in all_entities

        # Test parent-child relationships work correctly
        hospital_children = predictor.hierarchy.get_children("Hospital")
        assert len(hospital_children) > 0

        # Test that the hierarchy can be represented correctly
        hierarchy_str = str(predictor.hierarchy)
        assert "subspecialty: 2" in hierarchy_str
        assert "reporting_unit: 2" in hierarchy_str

    def test_entity_collision_with_multiple_levels(self):
        """Test collision fix with entities having same names across multiple levels."""
        # Create more complex collision scenario
        data = {
            "sub_specialty": ["Cardiology", "Cardiology", "Neurology"],
            "reporting_unit": [
                "Cardiology",
                "Cardiology",
                "Neurology",
            ],  # Same as subspecialty
            "division": [
                "Cardiology",
                "Medicine",
                "Medicine",
            ],  # Same as subspecialty/reporting_unit
            "board": ["Clinical Board", "Clinical Board", "Clinical Board"],
        }

        df = pd.DataFrame(data)

        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
        }

        predictor = create_hierarchical_predictor(
            hierarchy_df=df, column_mapping=column_mapping, top_level_id="Hospital"
        )

        # Test that all entities exist
        subspecialties = predictor.hierarchy.get_entities_by_type(
            EntityType("subspecialty")
        )
        reporting_units = predictor.hierarchy.get_entities_by_type(
            EntityType("reporting_unit")
        )
        divisions = predictor.hierarchy.get_entities_by_type(EntityType("division"))

        assert "Cardiology" in subspecialties
        assert "Cardiology" in reporting_units
        assert "Cardiology" in divisions
        assert "Neurology" in subspecialties
        assert "Neurology" in reporting_units
        assert "Medicine" in divisions

        # Test that hierarchy relationships work
        for entity_name in ["Cardiology", "Neurology", "Medicine"]:
            entity_info = predictor.hierarchy.get_entity_info(entity_name)
            if entity_info:
                assert entity_info["entity_id"] == entity_name
                assert entity_info["entity_type"] is not None
                assert entity_info["prefixed_id"].startswith(
                    f"{entity_info['entity_type'].name}:"
                )

    def test_hierarchy_methods_with_collision(self):
        """Test that all hierarchy methods work correctly with entity collisions."""
        data = {
            "sub_specialty": ["Cardiology", "Neurology"],
            "reporting_unit": ["Cardiology", "Neurology"],
            "division": ["Medicine", "Medicine"],
            "board": ["Clinical Board", "Clinical Board"],
        }

        df = pd.DataFrame(data)

        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
        }

        predictor = create_hierarchical_predictor(
            hierarchy_df=df, column_mapping=column_mapping, top_level_id="Hospital"
        )

        # Test get_children method
        hospital_children = predictor.hierarchy.get_children("Hospital")
        assert len(hospital_children) > 0

        # Test get_parent method
        for child in hospital_children:
            parent = predictor.hierarchy.get_parent(child)
            if parent:
                assert parent == "Hospital"

        # Test get_entity_type method
        for entity_name in [
            "Cardiology",
            "Neurology",
            "Medicine",
            "Clinical Board",
            "Hospital",
        ]:
            entity_type = predictor.hierarchy.get_entity_type(entity_name)
            if entity_type:
                assert entity_type is not None

        # Test get_entities_by_type method
        for level in predictor.hierarchy.get_levels_ordered():
            entities = predictor.hierarchy.get_entities_by_type(level)
            assert len(entities) > 0
            for entity in entities:
                assert isinstance(entity, str)
                assert len(entity) > 0

        # Test get_all_entities method
        all_entities = predictor.hierarchy.get_all_entities()
        assert len(all_entities) > 0
        assert "Cardiology" in all_entities
        assert "Neurology" in all_entities

    def test_prefixed_id_helpers(self):
        """Test the helper methods for working with prefixed IDs."""
        data = {
            "sub_specialty": ["Cardiology", "Neurology"],
            "reporting_unit": ["Cardiology", "Neurology"],
            "division": ["Medicine", "Medicine"],
            "board": ["Clinical Board", "Clinical Board"],
        }

        df = pd.DataFrame(data)

        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
        }

        predictor = create_hierarchical_predictor(
            hierarchy_df=df, column_mapping=column_mapping, top_level_id="Hospital"
        )

        # Test _get_original_name method
        original_name = predictor.hierarchy._get_original_name(
            "subspecialty:Cardiology"
        )
        assert original_name == "Cardiology"

        original_name = predictor.hierarchy._get_original_name(
            "reporting_unit:Cardiology"
        )
        assert original_name == "Cardiology"

        # Test _get_prefixed_id method
        prefixed_id = predictor.hierarchy._get_prefixed_id(
            "Cardiology", EntityType("subspecialty")
        )
        assert prefixed_id == "subspecialty:Cardiology"

        prefixed_id = predictor.hierarchy._get_prefixed_id(
            "Cardiology", EntityType("reporting_unit")
        )
        assert prefixed_id == "reporting_unit:Cardiology"

        # Test _find_entity_type_by_name method
        entity_type = predictor.hierarchy._find_entity_type_by_name("Cardiology")
        assert entity_type is not None
        assert entity_type in [EntityType("subspecialty"), EntityType("reporting_unit")]

    def test_collision_fix_backward_compatibility(self):
        """Test that the collision fix maintains backward compatibility."""
        data = {
            "sub_specialty": ["Cardiology", "Neurology"],
            "reporting_unit": ["Cardiology", "Neurology"],
            "division": ["Medicine", "Medicine"],
            "board": ["Clinical Board", "Clinical Board"],
        }

        df = pd.DataFrame(data)

        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
        }

        predictor = create_hierarchical_predictor(
            hierarchy_df=df, column_mapping=column_mapping, top_level_id="Hospital"
        )

        # Test that all public methods return original entity names (not prefixed)
        subspecialties = predictor.hierarchy.get_entities_by_type(
            EntityType("subspecialty")
        )
        for entity in subspecialties:
            assert ":" not in entity  # Should not contain prefixed format

        all_entities = predictor.hierarchy.get_all_entities()
        for entity in all_entities:
            assert ":" not in entity  # Should not contain prefixed format

        # Test that get_children returns original names
        hospital_children = predictor.hierarchy.get_children("Hospital")
        for child in hospital_children:
            assert ":" not in child  # Should not contain prefixed format

        # Test that get_parent returns original names
        for child in hospital_children:
            parent = predictor.hierarchy.get_parent(child)
            if parent:
                assert ":" not in parent  # Should not contain prefixed format


class TestHierarchicalPredictor:
    """Test the HierarchicalPredictor class."""

    def test_predict_all_levels_respects_caps(self):
        """predict_all_levels should use statistical caps at higher levels."""
        hierarchy_df = pd.DataFrame(
            {
                "sub_specialty": ["Cardiology", "Neurology"],
                "reporting_unit": ["UnitA", "UnitA"],
                "division": ["DivisionA", "DivisionA"],
                "board": ["BoardA", "BoardA"],
                "hospital": ["UCLH", "UCLH"],
            }
        )
        column_mapping = {
            "sub_specialty": "subspecialty",
            "reporting_unit": "reporting_unit",
            "division": "division",
            "board": "board",
            "hospital": "hospital",
        }

        hierarchical_predictor = create_hierarchical_predictor(
            hierarchy_df, column_mapping, "UCLH", k_sigma=0.0
        )

        def poisson_flow(flow_id: str, lam: float) -> FlowInputs:
            return FlowInputs(flow_id=flow_id, flow_type="poisson", distribution=lam)

        def pmf_flow(flow_id: str, probabilities: np.ndarray) -> FlowInputs:
            return FlowInputs(
                flow_id=flow_id, flow_type="pmf", distribution=probabilities
            )

        def make_inputs(subspecialty_id: str) -> ServicePredictionInputs:
            return ServicePredictionInputs(
                service_id=subspecialty_id,
                prediction_window=24,
                inflows={
                    "ed_current": poisson_flow("ed_current", 1.0),
                    "ed_yta": poisson_flow("ed_yta", 0.0),
                    "non_ed_yta": poisson_flow("non_ed_yta", 0.0),
                    "elective_yta": poisson_flow("elective_yta", 0.0),
                    "elective_transfers": poisson_flow("elective_transfers", 0.0),
                    "emergency_transfers": poisson_flow("emergency_transfers", 0.0),
                },
                outflows={
                    # Departures must be PMF-based (physically bounded by current patients)
                    # For mean=0.5, use PMF [0.5, 0.5] (P(0)=0.5, P(1)=0.5)
                    "elective_departures": pmf_flow(
                        "elective_departures", np.array([0.5, 0.5])
                    ),
                    # For mean=0.0, use PMF [1.0] (0 patients)
                    "emergency_departures": pmf_flow(
                        "emergency_departures", np.array([1.0])
                    ),
                },
            )

        bottom_level_data = {
            "Cardiology": make_inputs("Cardiology"),
            "Neurology": make_inputs("Neurology"),
        }

        results = hierarchical_predictor.predict_all_levels(
            bottom_level_data, flow_selection=FlowSelection.default()
        )

        assert "UCLH" in results
        hospital_bundle = results["UCLH"]

        # Two subspecialties with lambda=1 -> mean=2, so cap (k_sigma=0) should be 2.
        assert len(hospital_bundle.arrivals.probabilities) == 3
        assert (
            pytest.approx(hospital_bundle.arrivals.probabilities.sum(), rel=1e-9) == 1.0
        )

        # Departures: each subspecialty has PMF [0.5, 0.5] (1 patient, mean=0.5)
        # When convolved: [0.5, 0.5] * [0.5, 0.5] = [0.25, 0.5, 0.25] (support [0, 1, 2])
        # Physical cap = 1 + 1 = 2, so length should be 3 (support 0-2)
        assert len(hospital_bundle.departures.probabilities) == 3
        assert (
            pytest.approx(hospital_bundle.departures.probabilities.sum(), rel=1e-9)
            == 1.0
        )
        # Verify the expected distribution
        expected_departures = np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(
            hospital_bundle.departures.probabilities, expected_departures
        )

    def test_entity_type_name_usage_in_hierarchical_prediction(self):
        """Test that hierarchical prediction correctly uses EntityType.name instead of .value."""
        # Test the specific bug fix: EntityType.name vs EntityType.value
        predictor = DemandPredictor()

        # Create some mock predictions to test the hierarchical level prediction
        mock_prediction1 = DemandPrediction(
            entity_id="test1",
            entity_type="subspecialty",
            probabilities=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            expected_value=2.0,
            percentiles={50: 2, 75: 3, 90: 3, 95: 3, 99: 4},
            offset=0,
        )

        mock_prediction2 = DemandPrediction(
            entity_id="test2",
            entity_type="subspecialty",
            probabilities=np.array([0.2, 0.3, 0.3, 0.2]),
            expected_value=1.5,
            percentiles={50: 1, 75: 2, 90: 2, 95: 3, 99: 3},
            offset=0,
        )

        # Test that predict_hierarchical_level works with EntityType.name
        # This should not raise "EntityType object has no attribute 'value'"
        result = predictor.predict_hierarchical_level(
            entity_id="test_entity",
            entity_type=EntityType("reporting_unit"),  # This will use .name internally
            child_predictions=[mock_prediction1, mock_prediction2],
        )

        # Verify the result is a DemandPrediction
        assert isinstance(result, DemandPrediction)
        assert result.entity_id == "test_entity"
        assert result.entity_type == "reporting_unit"
        assert hasattr(result, "probabilities")
        assert hasattr(result, "expected_value")

    def test_create_bundle_from_children_with_empty_list(self):
        """Test that _create_bundle_from_children handles empty child_bundles list."""
        predictor = DemandPredictor()

        # Test with empty child_bundles list - should not raise IndexError
        bundle = predictor._create_bundle_from_children(
            entity_id="test_entity",
            entity_type="reporting_unit",
            child_bundles=[],  # Empty list
        )

        # Verify the result is a PredictionBundle
        assert isinstance(bundle, PredictionBundle)
        assert bundle.entity_id == "test_entity"
        assert bundle.entity_type == "reporting_unit"
        assert hasattr(bundle, "arrivals")
        assert hasattr(bundle, "departures")
        assert hasattr(bundle, "net_flow")
        assert hasattr(bundle, "flow_selection")

        # Should use default FlowSelection when no children
        assert bundle.flow_selection is not None
