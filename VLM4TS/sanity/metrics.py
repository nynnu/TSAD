import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["parse_status"] == "OK") & df["model_answer"].notna()]


def classification_metrics(df: pd.DataFrame, group_col: str | None = None) -> dict:
    def one(sub: pd.DataFrame) -> dict:
        scored = _valid(sub)
        if scored.empty:
            return {
                "n": 0,
                "accuracy": None,
                "precision_broken": None,
                "recall_broken": None,
                "f1_broken": None,
                "f1_macro": None,
            }
        y_true = scored["ground_truth_label"]
        y_pred = scored["model_answer"]
        return {
            "n": int(len(scored)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_broken": float(precision_score(y_true, y_pred, pos_label="broken", zero_division=0)),
            "recall_broken": float(recall_score(y_true, y_pred, pos_label="broken", zero_division=0)),
            "f1_broken": float(f1_score(y_true, y_pred, pos_label="broken", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    if group_col:
        return {str(k): one(v) for k, v in df.groupby(group_col)}
    return one(df)


def confidence_analysis(df: pd.DataFrame, group_col: str | None = None) -> dict:
    def mean_or_none(series):
        return None if series.empty else float(series.mean())

    def one(sub: pd.DataFrame) -> dict:
        scored = _valid(sub).copy()
        if scored.empty:
            return {"mean_confidence": None, "mean_confidence_correct": None, "mean_confidence_incorrect": None}
        scored["correct"] = scored["ground_truth_label"] == scored["model_answer"]
        return {
            "mean_confidence": mean_or_none(scored["model_confidence"].dropna()),
            "mean_confidence_correct": mean_or_none(scored.loc[scored["correct"], "model_confidence"].dropna()),
            "mean_confidence_incorrect": mean_or_none(scored.loc[~scored["correct"], "model_confidence"].dropna()),
        }

    if group_col:
        return {str(k): one(v) for k, v in df.groupby(group_col)}
    return one(df)


def interval_iou(true_start, true_end, pred_start, pred_end) -> float | None:
    if pd.isna(true_start) or pd.isna(true_end):
        return None
    if pd.isna(pred_start) or pd.isna(pred_end):
        return 0.0
    ts, te = int(true_start), int(true_end)
    ps, pe = int(pred_start), int(pred_end)
    inter = max(0, min(te, pe) - max(ts, ps))
    union = max(te, pe) - min(ts, ps)
    return 0.0 if union <= 0 else float(inter / union)


def localization_metrics(df: pd.DataFrame) -> dict:
    scored = df[(df["parse_status"] == "OK") & df["model_answer"].notna()].copy()
    broken = scored[scored["ground_truth_label"] == "broken"].copy()
    maintained = scored[scored["ground_truth_label"] == "maintained"].copy()

    ious = []
    start_errors = []
    end_errors = []
    for _, row in broken.iterrows():
        iou = interval_iou(
            row["expected_break_start"],
            row["expected_break_end"],
            row["predicted_break_start"],
            row["predicted_break_end"],
        )
        if iou is not None:
            ious.append(iou)
        if not pd.isna(row["predicted_break_start"]):
            start_errors.append(abs(int(row["predicted_break_start"]) - int(row["expected_break_start"])))
        if not pd.isna(row["predicted_break_end"]):
            end_errors.append(abs(int(row["predicted_break_end"]) - int(row["expected_break_end"])))

    false_localized = 0
    if not maintained.empty:
        false_localized = int(
            maintained["predicted_break_start"].notna().sum()
            + maintained["predicted_break_end"].notna().sum()
        )

    return {
        "n_broken": int(len(broken)),
        "interval_iou_mean": None if not ious else float(pd.Series(ious).mean()),
        "interval_iou_median": None if not ious else float(pd.Series(ious).median()),
        "hit_iou_0.3": None if not ious else float((pd.Series(ious) >= 0.3).mean()),
        "hit_iou_0.5": None if not ious else float((pd.Series(ious) >= 0.5).mean()),
        "start_mae": None if not start_errors else float(pd.Series(start_errors).mean()),
        "end_mae": None if not end_errors else float(pd.Series(end_errors).mean()),
        "n_maintained": int(len(maintained)),
        "false_localization_rate": None if maintained.empty else float(false_localized / (2 * len(maintained))),
    }
