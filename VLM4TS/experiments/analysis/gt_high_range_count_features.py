"""
Section 49: "threshold-exceeding channel count" features for high-GT-range
(9+) segment detection. Reuses cached Step0 raw scores from section 46-48
(no new DINOv2 computation, no VLM/GPT calls).

Inputs:
  experiments/results_adaptive_vs_fixed/step0_scores_cache.json      (74 segs, source=74)
  experiments/results_gt_range_predictability/step0_scores_cache_49.json (49 segs, source=46/47)
  experiments/results_gt_range_predictability/feature_table.csv      (n_gt, bucket labels)

Outputs:
  experiments/results_gt_high_range_count/count_feature_table.csv
  experiments/results_gt_high_range_count/anova_posthoc_summary.json
"""
import json
import csv
import statistics
from pathlib import Path
from itertools import combinations

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CACHE_74 = ROOT / "experiments/results_adaptive_vs_fixed/step0_scores_cache.json"
CACHE_49 = ROOT / "experiments/results_gt_range_predictability/step0_scores_cache_49.json"
FEATURE_TABLE = ROOT / "experiments/results_gt_range_predictability/feature_table.csv"
OUT_DIR = ROOT / "experiments/results_gt_high_range_count"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_MAP = {"1-3": "low", "4-8": "mid", "9+": "high"}


def bucket_key(raw: str) -> str:
    # feature_table.csv bucket strings are mojibake Korean (encoding lost),
    # but the ASCII digit range suffix survives intact -> use that.
    for k, v in BUCKET_MAP.items():
        if k in raw:
            return v
    raise ValueError(f"unrecognized bucket: {raw!r}")


def sigma_threshold_counts(vals):
    mu = statistics.fmean(vals)
    sigma = statistics.pstdev(vals)
    out = {}
    for k in (0.5, 1.0, 1.5, 2.0):
        thr = mu + k * sigma
        out[f"count_thr_{k}sigma"] = sum(1 for v in vals if v > thr)
    return out


def median_ratio_count(vals, ratio=2.0):
    med = statistics.median(vals)
    if med <= 1e-9:
        # degenerate (near-zero median): fall back to a tiny epsilon so the
        # ratio doesn't explode to "everything counts" or divide-by-zero.
        med = 1e-9
    return sum(1 for v in vals if v > ratio * med)


def load_scores():
    d = {}
    d.update(json.loads(CACHE_74.read_text(encoding="utf-8")))
    d.update(json.loads(CACHE_49.read_text(encoding="utf-8")))
    return d


def main():
    scores_by_seg = load_scores()
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))

    out_rows = []
    missing = []
    for r in rows:
        seg = r["segment_id"]
        if seg not in scores_by_seg:
            missing.append(seg)
            continue
        vals = list(scores_by_seg[seg].values())
        feats = sigma_threshold_counts(vals)
        feats["count_thr_median_2x"] = median_ratio_count(vals, ratio=2.0)
        out_rows.append({
            "segment_id": seg,
            "entity": r["entity"],
            "source": r["source"],
            "n_gt": int(r["n_gt"]),
            "bucket": bucket_key(r["bucket"]),
            **feats,
        })

    if missing:
        print(f"WARNING: {len(missing)} segments in feature_table.csv not found in score caches: {missing}")

    feature_names = ["count_thr_0.5sigma", "count_thr_1.0sigma", "count_thr_1.5sigma",
                      "count_thr_2.0sigma", "count_thr_median_2x"]

    # write CSV
    csv_path = OUT_DIR / "count_feature_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id", "entity", "source", "n_gt", "bucket", *feature_names])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"wrote {csv_path} ({len(out_rows)} rows)")

    # 3-way ANOVA + pairwise t-tests per feature
    groups = {"low": [], "mid": [], "high": []}
    for row in out_rows:
        groups[row["bucket"]].append(row)

    print(f"group sizes: low={len(groups['low'])} mid={len(groups['mid'])} high={len(groups['high'])}")

    summary = {"group_sizes": {k: len(v) for k, v in groups.items()}, "features": {}}
    for feat in feature_names:
        by_group = {k: [row[feat] for row in v] for k, v in groups.items()}
        f_stat, p_anova = stats.f_oneway(by_group["low"], by_group["mid"], by_group["high"])
        means = {k: statistics.fmean(v) for k, v in by_group.items()}
        stdevs = {k: (statistics.pstdev(v) if len(v) > 1 else 0.0) for k, v in by_group.items()}

        posthoc = {}
        for a, b in combinations(["low", "mid", "high"], 2):
            t_stat, p_t = stats.ttest_ind(by_group[a], by_group[b], equal_var=False)
            posthoc[f"{a}_vs_{b}"] = {"t_stat": t_stat, "p_value": p_t}

        summary["features"][feat] = {
            "means": means,
            "stdevs": stdevs,
            "anova_f": f_stat,
            "anova_p": p_anova,
            "posthoc": posthoc,
        }

    json_path = OUT_DIR / "anova_posthoc_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    # console report focused on mid vs high
    print("\n=== mid(4-8) vs high(9+) pairwise p-values (the key question) ===")
    for feat in feature_names:
        p = summary["features"][feat]["posthoc"]["mid_vs_high"]["p_value"]
        m_mid = summary["features"][feat]["means"]["mid"]
        m_high = summary["features"][feat]["means"]["high"]
        sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{feat:22s}  mid_mean={m_mid:6.2f}  high_mean={m_high:6.2f}  p={p:.4f}  {sig}")


def demo():
    """Smallest runnable check: sigma/median-ratio counters behave sanely."""
    vals = [1, 1, 1, 1, 10, 10, 20]  # mean~6.3, clear high outliers
    counts = sigma_threshold_counts(vals)
    assert counts["count_thr_0.5sigma"] >= counts["count_thr_2.0sigma"], "higher sigma thr must count <= fewer channels"
    assert median_ratio_count(vals, ratio=2.0) == sum(1 for v in vals if v > 2 * statistics.median(vals))
    # degenerate near-zero median doesn't explode/divide-by-zero
    assert median_ratio_count([0, 0, 0, 1e-10, 5]) >= 0
    print("demo OK")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
