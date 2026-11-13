import pandas as pd

from metrics import interval_iou, localization_metrics


def test_interval_iou_exact_partial_none():
    assert interval_iou(100, 200, 100, 200) == 1.0
    assert interval_iou(100, 200, 150, 250) == 50 / 150
    assert interval_iou(100, 200, 210, 250) == 0.0


def test_localization_metrics_false_localization():
    df = pd.DataFrame([
        {
            "parse_status": "OK",
            "ground_truth_label": "broken",
            "model_answer": "broken",
            "expected_break_start": 100,
            "expected_break_end": 200,
            "predicted_break_start": 100,
            "predicted_break_end": 200,
        },
        {
            "parse_status": "OK",
            "ground_truth_label": "maintained",
            "model_answer": "maintained",
            "expected_break_start": None,
            "expected_break_end": None,
            "predicted_break_start": 10,
            "predicted_break_end": 20,
        },
    ])
    out = localization_metrics(df)
    assert out["interval_iou_mean"] == 1.0
    assert out["hit_iou_0.5"] == 1.0
    assert out["false_localization_rate"] == 1.0
