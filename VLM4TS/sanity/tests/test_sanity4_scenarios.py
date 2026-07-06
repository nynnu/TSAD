from run_sanity4 import SCENARIOS, _scenario
from run_sanity4 import _candidate_metrics

import pandas as pd


def test_sanity4_scenarios_cover_valid_and_invalid():
    labels = [_scenario(0, scenario)["ground_truth_label"] for scenario in SCENARIOS]
    assert "valid" in labels
    assert "invalid" in labels
    assert len(labels) == 7


def test_sanity4_shifted_c3_candidate_is_invalid():
    case = _scenario(0, "V3")
    assert case["source_case"] == "C3"
    assert (case["candidate_start"], case["candidate_end"]) == (0, 100)
    assert (case["source_break_start"], case["source_break_end"]) == (100, 200)
    assert case["ground_truth_label"] == "invalid"


def test_candidate_metrics_use_valid_as_positive_label():
    df = pd.DataFrame([
        {"parse_status": "OK", "ground_truth_label": "valid", "model_answer": "valid"},
        {"parse_status": "OK", "ground_truth_label": "invalid", "model_answer": "invalid"},
        {"parse_status": "OK", "ground_truth_label": "valid", "model_answer": "invalid"},
    ])
    out = _candidate_metrics(df)
    assert out["accuracy"] == 2 / 3
    assert out["precision_valid"] == 1.0
    assert out["recall_valid"] == 0.5
