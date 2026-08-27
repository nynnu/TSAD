"""Effect-size re-analysis of the DINO Relation Mini Sanity Check.

Follow-up to dino_relation_check.py: the role-based (cascade/mild/unrelated)
Kruskal-Wallis was significant (large n=1800 pairs) but the raw mean deltas
are only ~0.001-0.007 apart -- that gap could be statistically significant
and practically noise-level at the same time. This script reports effect
sizes (Cohen's d, AUC-equivalent via Mann-Whitney U) instead of just
p-values, tests "breakdown_type match" (the existing `homogeneous` column,
which is True/False for cascade+mild and NaN for unrelated -- i.e. True iff
channel break_type == root break_type) as a direct 2-group alternative to
the 3-way role split, and quantifies how often the page-8 failure pattern
(an unrelated channel's delta exceeds cascade's) occurs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

SANITY_DIR = Path(__file__).resolve().parent
DINO_RESULTS = SANITY_DIR / "results" / "dino_relation9"
PAIRS_CSV = DINO_RESULTS / "raw" / "pairs.csv"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def auc_effect(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """AUC-equivalent probability that a random `a` value exceeds a random
    `b` value (0.5 = no separation, 1.0 = perfect separation), via the
    Mann-Whitney U statistic (U = AUC * n1 * n2). Returns (auc, p_value)."""
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    auc = float(stat / (len(a) * len(b)))
    return auc, float(p)


def two_group_row(label: str, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    auc, p = auc_effect(a, b)
    return {
        "comparison": label,
        f"mean_{name_a}": float(a.mean()),
        f"mean_{name_b}": float(b.mean()),
        "cohens_d": cohens_d(a, b),
        "auc": auc,
        "p_value": p,
        f"n_{name_a}": int(len(a)),
        f"n_{name_b}": int(len(b)),
    }


def analyze_representation(df: pd.DataFrame, col: str) -> dict:
    cascade = df[df["role"] == "cascade"][col].values
    mild = df[df["role"] == "mild"][col].values
    unrelated = df[df["role"] == "unrelated"][col].values

    # 3-way role Kruskal + epsilon-squared effect size
    H, p3 = kruskal(cascade, mild, unrelated)
    n_total = len(cascade) + len(mild) + len(unrelated)
    epsilon_sq = float((H - 3 + 1) / (n_total - 3))  # epsilon-squared for Kruskal-Wallis

    pairwise = [
        two_group_row("cascade vs unrelated (role)", cascade, unrelated, "cascade", "unrelated"),
        two_group_row("mild vs unrelated (role)", mild, unrelated, "mild", "unrelated"),
        two_group_row("cascade vs mild (role)", cascade, mild, "cascade", "mild"),
    ]

    # type_match: True iff channel break_type == root break_type (homogeneous
    # column is True/False for cascade+mild, NaN for unrelated -> NaN becomes
    # "no match", same as heterogeneous/unrelated channels never matching).
    type_match = (df["homogeneous"] == True)  # noqa: E712 (NaN compares False, which is what we want)
    match_true = df[type_match][col].values
    match_false = df[~type_match][col].values
    type_match_row = two_group_row("type_match True vs False (all pairs)", match_true, match_false, "match", "no_match")

    # Apples-to-apples version restricted to affected (cascade+mild) channels
    # only -- excludes unrelated entirely, matching the original homogeneous
    # vs heterogeneous comparison that showed a ~2x mean difference.
    affected = df[df["role"].isin(["cascade", "mild"])]
    homog = affected[affected["homogeneous"] == True][col].values  # noqa: E712
    heterog = affected[affected["homogeneous"] == False][col].values  # noqa: E712
    homog_affected_only_row = two_group_row(
        "homogeneous vs heterogeneous (affected only)", homog, heterog, "homogeneous", "heterogeneous"
    )

    return {
        "role_kruskal": {"H": float(H), "p": float(p3), "epsilon_squared": epsilon_sq, "n": n_total},
        "role_pairwise": pairwise,
        "type_match_all_pairs": type_match_row,
        "type_match_affected_only": homog_affected_only_row,
    }


def failure_rate(df: pd.DataFrame, col: str) -> dict:
    """Case-level: fraction of samples (with >=1 affected AND >=1 unrelated
    channel present) where mean(affected delta) < mean(unrelated delta) --
    the page-8 failure pattern, generalized beyond a single eyeballed case."""
    case_results = []
    for case_id, g in df.groupby("case_id"):
        affected = g[g["role"].isin(["cascade", "mild"])][col]
        unrelated = g[g["role"] == "unrelated"][col]
        if affected.empty or unrelated.empty:
            continue
        case_results.append(affected.mean() < unrelated.mean())
    n = len(case_results)
    n_fail = int(sum(case_results))
    return {"n_applicable_cases": n, "n_failure_pattern": n_fail, "failure_rate": n_fail / n if n else None}


def main() -> None:
    df = pd.read_csv(PAIRS_CSV)

    summary = {
        "delta_cls": analyze_representation(df, "delta_cls"),
        "delta_patch": analyze_representation(df, "delta_patch"),
        "failure_rate_cls": failure_rate(df, "delta_cls"),
        "failure_rate_patch": failure_rate(df, "delta_patch"),
    }

    out_path = DINO_RESULTS / "dino_relation9_effect_sizes.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
