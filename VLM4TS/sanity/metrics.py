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
