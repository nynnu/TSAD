import pandas as pd

from metrics import classification_metrics, confidence_analysis


def test_confidence_empty_subset():
    df = pd.DataFrame([
        {"parse_status": "OK", "ground_truth_label": "broken", "model_answer": "broken", "model_confidence": 0.9},
    ])
    out = confidence_analysis(df)
    assert out["mean_confidence_correct"] == 0.9
    assert out["mean_confidence_incorrect"] is None


def test_classification_grouped():
    df = pd.DataFrame([
        {"case_type": "C1", "parse_status": "OK", "ground_truth_label": "broken", "model_answer": "broken", "model_confidence": 0.9},
        {"case_type": "C1", "parse_status": "OK", "ground_truth_label": "broken", "model_answer": "maintained", "model_confidence": 0.6},
    ])
    out = classification_metrics(df, "case_type")
    assert out["C1"]["n"] == 2
