"""Fast, LLM-free screening of two channel-grouping methods on Sanity-9's
360 non-NORMAL samples: does correlation (A) or DINOv2 last-layer attention
similarity (B) separate "related" channel pairs (both root/cascade/mild)
from "unrelated" pairs well enough to bother carrying either into the SMD
(real dataset) stage?

Ground truth edge(X, Y) = 1 iff both X and Y are in {root, cascade, mild};
0 if either is unrelated. NORMAL samples (no root) are excluded.

Method A (correlation): |Pearson r| between raw channel values, computed
both on the full [0:300] window and the post-onset [100:300] window.

Method B (attention): cosine similarity between the last-layer CLS->patch
attention maps already extracted in dino_relation_layers.py (post-onset
window), reusing the cached embeddings -- no new DINOv2 forward passes.

Channel arrays are regenerated via sanity9_data_gen.generate_sample with the
exact same (n_affected, lag, seed) triples run_sanity9.py used, so they are
byte-identical to the original 780-case data (no new/different samples).
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SANITY_DIR = Path(__file__).resolve().parent
DINO_RESULTS = SANITY_DIR / "results" / "dino_relation9"
sys.path.insert(0, str(SANITY_DIR))

from sanity9_data_gen import generate_sample  # noqa: E402

N_AFFECTED_VALUES = [0, 1, 2, 3, 4, 5]
LAG_VALUES = [10, 50]
N_PER_SCENARIO = 30
T = 300
ROOT_ONSET = 100
CHANNEL_NAMES = [str(i) for i in range(1, 7)]
THRESHOLDS = [0.3, 0.5, 0.7]


def _seed(n_affected: int, lag: int, index: int) -> int:
    return n_affected * 10_000 + lag * 100 + index


def _scenarios() -> list[tuple[int, int, str]]:
    return [(n, lag, f"NA{n}_LAG{lag}") for n in N_AFFECTED_VALUES for lag in LAG_VALUES]


def _load_attn_lookup() -> dict:
    meta = json.loads((DINO_RESULTS / "raw" / "image_meta.json").read_text(encoding="utf-8"))
    npz = np.load(DINO_RESULTS / "raw" / "layer_embeddings.npz")
    attn = npz["attn"]
    lookup = {}
    for i, m in enumerate(meta):
        lookup[(m["case_id"], m["channel"], m["window"])] = attn[i]
    return lookup


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_pair_table() -> pd.DataFrame:
    attn_lookup = _load_attn_lookup()
    pairs = list(itertools.combinations(CHANNEL_NAMES, 2))
    rows = []

    for n_affected, lag, label in _scenarios():
        for index in range(N_PER_SCENARIO):
            seed = _seed(n_affected, lag, index)
            channels, meta = generate_sample(n_affected=n_affected, lag=lag, seed=seed, t=T)
            root = meta["root_cause_channel"]
            if root is None:
                continue
            case_id = f"{label}_{index:03d}"
            roles = meta["roles"]

            for ch_i, ch_j in pairs:
                role_i, role_j = roles[ch_i]["role"], roles[ch_j]["role"]
                gt_edge = int(role_i in ("root", "cascade", "mild") and role_j in ("root", "cascade", "mild"))

                full_i, full_j = channels[ch_i], channels[ch_j]
                post_i, post_j = full_i[ROOT_ONSET:], full_j[ROOT_ONSET:]
                corr_full = abs(float(np.corrcoef(full_i, full_j)[0, 1]))
                corr_post = abs(float(np.corrcoef(post_i, post_j)[0, 1]))

                key_i, key_j = (case_id, ch_i, "post"), (case_id, ch_j, "post")
                attn_sim = _cosine_sim(attn_lookup[key_i], attn_lookup[key_j]) if key_i in attn_lookup and key_j in attn_lookup else np.nan

                bt_i, bt_j = roles[ch_i].get("break_type"), roles[ch_j].get("break_type")
                pair_type_match = (bt_i is not None and bt_i == bt_j) if gt_edge else None

                rows.append({
                    "case_id": case_id, "scenario": label, "n_affected": n_affected, "lag": lag,
                    "ch_i": ch_i, "ch_j": ch_j, "role_i": role_i, "role_j": role_j,
                    "gt_edge": gt_edge, "corr_full": corr_full, "corr_post": corr_post,
                    "attn_sim": attn_sim, "pair_type_match": pair_type_match,
                })
    return pd.DataFrame(rows)


def prf1(gt: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall)) else (0.0 if (tp + fp + fn) else np.nan)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def edge_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    methods = {"Corr (full)": "corr_full", "Corr (post-onset)": "corr_post", "Attention (post-onset)": "attn_sim"}
    rows = []
    for name, col in methods.items():
        for th in THRESHOLDS:
            pred = (df[col] > th).astype(int)
            m = prf1(df["gt_edge"].values, pred.values)
            rows.append({"method": name, "threshold": th, **m})
    return pd.DataFrame(rows)


def best_threshold(metrics_df: pd.DataFrame, method: str) -> float:
    sub = metrics_df[metrics_df["method"] == method]
    return float(sub.loc[sub["f1"].idxmax(), "threshold"])


def component_purity(df: pd.DataFrame, col: str, threshold: float) -> dict:
    """Root's connected component under predicted edges vs GT {root}+affected."""
    jaccards, leak_rates = [], []
    for case_id, g in df.groupby("case_id"):
        nodes = set(CHANNEL_NAMES)
        adj = {n: set() for n in nodes}
        for _, row in g.iterrows():
            if row[col] > threshold:
                adj[row["ch_i"]].add(row["ch_j"])
                adj[row["ch_j"]].add(row["ch_i"])
        root = g[g["role_i"] == "root"]["ch_i"].iloc[0] if (g["role_i"] == "root").any() else g[g["role_j"] == "root"]["ch_j"].iloc[0]

        seen, stack = {root}, [root]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        predicted_component = seen

        affected = set(g[g["role_i"].isin(["cascade", "mild"])]["ch_i"]) | set(g[g["role_j"].isin(["cascade", "mild"])]["ch_j"])
        true_set = affected | {root}
        union = predicted_component | true_set
        jaccard = len(predicted_component & true_set) / len(union) if union else 1.0
        jaccards.append(jaccard)

        unrelated = nodes - true_set
        if unrelated:
            leaked = len(predicted_component & unrelated)
            leak_rates.append(leaked / len(unrelated))

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "median_jaccard": float(np.median(jaccards)),
        "mean_unrelated_leak_rate": float(np.mean(leak_rates)) if leak_rates else None,
        "n_samples": len(jaccards),
    }


def n_affected_breakdown(df: pd.DataFrame, methods: dict[str, tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for n in N_AFFECTED_VALUES:
        sub = df[df["n_affected"] == n]
        row = {"n_affected": n, "n_pairs": len(sub), "n_positive": int(sub["gt_edge"].sum())}
        for name, (col, th) in methods.items():
            pred = (sub[col] > th).astype(int)
            row[f"{name}_f1"] = prf1(sub["gt_edge"].values, pred.values)["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def homogeneous_breakdown(df: pd.DataFrame, methods: dict[str, tuple[str, float]]) -> pd.DataFrame:
    positives = df[df["gt_edge"] == 1].copy()
    rows = []
    for match in [True, False]:
        sub = positives[positives["pair_type_match"] == match]
        row = {"type_match": match, "n_pairs": len(sub)}
        for name, (col, th) in methods.items():
            recall = float(((sub[col] > th).astype(int) == 1).mean()) if len(sub) else np.nan
            row[f"{name}_recall"] = recall
        rows.append(row)
    return pd.DataFrame(rows)


def failure_examples(df: pd.DataFrame, methods: dict[str, tuple[str, float]], n: int = 3) -> dict:
    (col_a, th_a) = methods["A"]
    (col_b, th_b) = methods["B"]
    pred_a = (df[col_a] > th_a).astype(int)
    pred_b = (df[col_b] > th_b).astype(int)
    correct_a = pred_a == df["gt_edge"]
    correct_b = pred_b == df["gt_edge"]

    out = {}
    for name, mask in [
        ("A만 틀림 (B는 맞음)", (~correct_a) & correct_b),
        ("B만 틀림 (A는 맞음)", correct_a & (~correct_b)),
        ("둘 다 틀림", (~correct_a) & (~correct_b)),
    ]:
        sub = df[mask].head(n)
        examples = []
        for _, row in sub.iterrows():
            examples.append({
                "case_id": row["case_id"], "pair": f"{row['ch_i']}-{row['ch_j']}",
                "role_i": row["role_i"], "role_j": row["role_j"], "gt_edge": int(row["gt_edge"]),
                "corr_full": round(row["corr_full"], 3), "corr_post": round(row["corr_post"], 3),
                "attn_sim": round(row["attn_sim"], 3) if not np.isnan(row["attn_sim"]) else None,
            })
        out[name] = {"n_total": int(mask.sum()), "examples": examples}
    return out


def main() -> None:
    print("Regenerating 360 samples and computing pairwise correlation + attention similarity...")
    df = build_pair_table()
    out_dir = DINO_RESULTS.parent / "channel_grouping9"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pair_table.csv", index=False)
    print(f"Pair table: {len(df)} rows ({df['case_id'].nunique()} samples x 15 pairs)")

    metrics_df = edge_metrics_table(df)
    metrics_df.to_csv(out_dir / "edge_metrics.csv", index=False)
    print("\n=== Edge-level Precision/Recall/F1 by threshold ===")
    print(metrics_df.to_string(index=False))

    best_a_full = best_threshold(metrics_df, "Corr (full)")
    best_a_post = best_threshold(metrics_df, "Corr (post-onset)")
    best_b = best_threshold(metrics_df, "Attention (post-onset)")
    best_a_col, best_a_name, best_a_th = ("corr_post", "Corr (post-onset)", best_a_post) if \
        metrics_df[(metrics_df.method == "Corr (post-onset)") & (metrics_df.threshold == best_a_post)]["f1"].iloc[0] >= \
        metrics_df[(metrics_df.method == "Corr (full)") & (metrics_df.threshold == best_a_full)]["f1"].iloc[0] else \
        ("corr_full", "Corr (full)", best_a_full)
    print(f"\nBest thresholds -> A ({best_a_name}): {best_a_th}, B (Attention): {best_b}")

    methods_best = {"A": (best_a_col, best_a_th), "B": ("attn_sim", best_b)}

    purity_a = component_purity(df, best_a_col, best_a_th)
    purity_b = component_purity(df, "attn_sim", best_b)
    print("\n=== Group purity (root's connected component vs GT affected+root) ===")
    print(f"A ({best_a_name} @ {best_a_th}): {purity_a}")
    print(f"B (Attention @ {best_b}): {purity_b}")

    n_aff_df = n_affected_breakdown(df, {"A": methods_best["A"], "B": methods_best["B"]})
    print("\n=== n_affected breakdown (F1) ===")
    print(n_aff_df.to_string(index=False))
    n_aff_df.to_csv(out_dir / "n_affected_breakdown.csv", index=False)

    homog_df = homogeneous_breakdown(df, {"A": methods_best["A"], "B": methods_best["B"]})
    print("\n=== homogeneous/heterogeneous breakdown (recall among true edges) ===")
    print(homog_df.to_string(index=False))
    homog_df.to_csv(out_dir / "homogeneous_breakdown.csv", index=False)

    fails = failure_examples(df, methods_best)
    (out_dir / "failure_examples.json").write_text(json.dumps(fails, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Failure examples ===")
    print(json.dumps(fails, indent=2, ensure_ascii=False))

    summary = {
        "best_thresholds": {"A": {"name": best_a_name, "threshold": best_a_th}, "B": {"threshold": best_b}},
        "purity": {"A": purity_a, "B": purity_b},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
