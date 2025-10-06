import numpy as np
import pandas as pd

from patientflow.predict.hierarchy import (
    DemandPredictor,
    HospitalHierarchy,
    populate_hierarchy_from_dataframe,
    populate_hierarchical_predictor_from_dataframe,
    create_hierarchical_predictor,
)


def test_poisson_max_and_pmf_respect_epsilon():
    predictor = DemandPredictor(epsilon=1e-8)
    max_k = predictor._calculate_poisson_max(3.0)
    pmf = predictor._poisson_pmf(3.0, max_k)
    assert len(pmf) == max_k + 1
    assert np.isclose(pmf.sum(), 1.0, atol=1e-10)


def test_poisson_lambda_zero():
    predictor = DemandPredictor()
    assert predictor._calculate_poisson_max(0.0) == 0
    pmf = predictor._poisson_pmf(0.0, 0)
    assert np.array_equal(pmf, np.array([1.0]))


def test_truncate_enforces_tail_mass_bound():
    predictor = DemandPredictor(epsilon=1e-6)
    base = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    truncated = predictor._truncate(base)
    assert truncated.sum() >= 1 - predictor.epsilon
    assert truncated.sum() <= 1.0


def test_convolve_and_multiple_preserve_mass():
    predictor = DemandPredictor()
    p = np.array([0.6, 0.4])
    q = np.array([0.7, 0.3])
    r = predictor._convolve(p, q)
    assert len(r) == len(p) + len(q) - 1
    assert np.isclose(r.sum(), 1.0, atol=1e-12)
    s = predictor._convolve_multiple([p, q, p])
    assert np.isclose(s.sum(), 1.0, atol=1e-12)


def test_expected_value_and_percentiles_basic():
    predictor = DemandPredictor()
    pmf = np.array([0.25, 0.5, 0.25])  # mean = 1.0
    mean = predictor._expected_value(pmf)
    assert np.isclose(mean, 1.0, atol=1e-12)
    pct = predictor._percentiles(pmf, [50, 90])
    # CDF: [0.25, 0.75, 1.0] -> 50% at 1, 90% at 2
    assert pct[50] == 1
    assert pct[90] == 2


def test_predict_subspecialty_expected_value_matches_components():
    predictor = DemandPredictor()
    # ED distribution with mean 0.5: P(0)=0.5, P(1)=0.5
    ed_pmf = np.array([0.5, 0.5])
    out = predictor.predict_subspecialty(
        "cardiology",
        prob_admission_pats_in_ed=ed_pmf,
        lambda_ed_yta=0.7,
        lambda_non_ed_yta=0.2,
        lambda_elective_yta=0.1,
    )
    assert out.entity_id == "cardiology"
    assert out.entity_type == "subspecialty"
    # Expected value: ED mean (0.5) + combined lambda (0.7+0.2+0.1 = 1.0) = 1.5
    assert np.isclose(out.expected_value, 1.5, atol=1e-6)
    assert np.isclose(out.probabilities.sum(), 1.0, atol=1e-10)


def test_aggregations_combine_child_pmfs():
    predictor = DemandPredictor()
    a = predictor._create_prediction("A", "subspecialty", np.array([0.8, 0.2]))
    b = predictor._create_prediction("B", "subspecialty", np.array([0.7, 0.3]))
    ru = predictor.predict_reporting_unit("RU", [a, b])
    assert ru.entity_id == "RU"
    assert ru.entity_type == "reporting_unit"
    assert np.isclose(ru.probabilities.sum(), 1.0, atol=1e-12)


def test_hospital_hierarchy_add_and_get_relationships():
    h = HospitalHierarchy()
    h.add_subspecialty("S1", "R1")
    h.add_reporting_unit("R1", "D1")
    h.add_division("D1", "B1")
    h.add_board("B1", "H1")
    assert h.get_subspecialties_for_reporting_unit("R1") == ["S1"]
    assert h.get_reporting_units_for_division("D1") == ["R1"]
    assert h.get_divisions_for_board("B1") == ["D1"]
    assert h.get_boards_for_hospital("H1") == ["B1"]


def test_populate_hierarchy_from_dataframe_builds_levels_and_deduplicates():
    df = pd.DataFrame(
        {
            "board": ["B1", "B1", "B1"],
            "division": ["D1", "D1", "D1"],
            "reporting_unit": ["R1", "R1", "R1"],
            "sub_specialty": ["S1", "S1", "S1"],
        }
    )
    h = populate_hierarchy_from_dataframe(df, hospital_id="H1")
    assert h.subspecialties["S1"] == "R1"
    assert h.reporting_units["R1"] == "D1"
    assert h.divisions["D1"] == "B1"
    assert set(h.get_boards_for_hospital("H1")) == {"B1"}


def test_create_and_populate_hierarchical_predictor_wires_epsilon_and_hospital():
    df = pd.DataFrame(
        {
            "board": ["B1"],
            "division": ["D1"],
            "reporting_unit": ["R1"],
            "sub_specialty": ["S1"],
        }
    )
    hp = create_hierarchical_predictor(df, hospital_id="H1", epsilon=1e-6)
    assert hp.predictor.epsilon == 1e-6
    hp2 = populate_hierarchical_predictor_from_dataframe(
        df, "H1", subspecialty_data={}, epsilon=1e-5
    )
    assert hp2.predictor.epsilon == 1e-5


def test_predict_all_levels_outputs_every_entity_and_caches():
    df = pd.DataFrame(
        {
            "board": ["B1"],
            "division": ["D1"],
            "reporting_unit": ["R1"],
            "sub_specialty": ["S1"],
        }
    )
    hp = create_hierarchical_predictor(df, hospital_id="H1", epsilon=1e-7)
    subspecialty_data = {
        "S1": {
            "prob_admission_pats_in_ed": np.array([0.6, 0.4]),
            "lambda_ed_yta": 0.5,
            "lambda_non_ed_yta": 0.2,
            "lambda_elective_yta": 0.3,
        }
    }
    results = hp.predict_all_levels("H1", subspecialty_data)
    for key in ["S1", "R1", "D1", "B1", "H1"]:
        assert key in results
        assert hp.get_prediction(key) is results[key]


