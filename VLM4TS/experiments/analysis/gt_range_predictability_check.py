"""
§48: GT 채널 수 구간(저1-3/중4-8/고9+)을 Step0(DINOv2 raw score)만으로
VLM 호출 전에 예측할 수 있는지 확인. API 비용 없음, 로컬 DINOv2 연산만.

데이터 소스:
  1) 기존 74개: results_adaptive_vs_fixed/step0_scores_cache.json (raw score 있음)
     + GT는 results_smd_3way_baseline/results_3way_broad/results_full_smd_3way
       checkpoint.json의 f"{seg_id}_gt" 병합
  2) §46 23개: results_heldout_diagnosis/segment_results.csv (n_gt, bucket 있음)
     raw score 없음 -> 재계산
  3) §47 26개: results_valtest_holdout26/segment_results_holdout26.csv (n_gt_dims 있음)
     raw score 없음 -> 재계산

재계산은 full_smd_3way.step0_rank_original_mechanism과 완전히 동일한 로직이되
정렬 전 raw scores dict를 리턴하는 변형을 사용, 새 캐시 파일에 저장.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)
from smd_3way_baseline_comparison import load_smd_test  # noqa: E402

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
EXP = BASE / "experiments"
OUT_DIR = EXP / "results_gt_range_predictability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_49 = OUT_DIR / "step0_scores_cache_49.json"


def bucket_of(n_gt):
    if n_gt <= 3:
        return "저(1-3)"
    if n_gt <= 8:
        return "중(4-8)"
    return "고(9+)"


# ---------------------------------------------------------------------------
# 1) 기존 74개 로드
# ---------------------------------------------------------------------------

def load_74():
    scores_cache = json.loads((EXP / "results_adaptive_vs_fixed" / "step0_scores_cache.json").read_text(encoding="utf-8"))

    gt_by_seg = {}
    for ckpt_name in ["results_smd_3way_baseline", "results_3way_broad", "results_full_smd_3way"]:
        ckpt = json.loads((EXP / ckpt_name / "checkpoint.json").read_text(encoding="utf-8"))
        for k, v in ckpt.items():
            if k.endswith("_gt"):
                seg_id = k[:-3]
                gt_by_seg[seg_id] = v["gt_channels"]

    rows = []
    for seg_id, scores in scores_cache.items():
        if seg_id not in gt_by_seg:
            continue  # gt 못 찾으면 스킵
        entity = seg_id.rsplit("_", 2)[0]
        rows.append({
            "segment_id": seg_id, "entity": entity, "source": "74",
            "n_gt": len(gt_by_seg[seg_id]),
            "scores": {int(c): v for c, v in scores.items()},
        })
    return rows


# ---------------------------------------------------------------------------
# 2/3) §46/§47 세그먼트 목록 + raw step0 재계산
# ---------------------------------------------------------------------------

def parse_seg_id(seg_id, entity):
    rest = seg_id[len(entity) + 1:]
    cs_s, ce_s = rest.split("_")
    return int(cs_s), int(ce_s)


def load_46():
    path = EXP / "results_heldout_diagnosis" / "segment_results.csv"
    with path.open(encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    out = []
    for row in r:
        seg_id, entity = row["segment_id"], row["entity"]
        cs, ce = parse_seg_id(seg_id, entity)
        out.append({"segment_id": seg_id, "entity": entity, "source": "46", "n_gt": int(row["n_gt"]), "cs": cs, "ce": ce})
    return out


def load_47():
    path = EXP / "results_valtest_holdout26" / "segment_results_holdout26.csv"
    with path.open(encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    out = []
    for row in r:
        seg_id, entity = row["segment_id"], row["entity"]
        cs, ce = parse_seg_id(seg_id, entity)
        out.append({"segment_id": seg_id, "entity": entity, "source": "47", "n_gt": int(row["n_gt_dims"]), "cs": cs, "ce": ce})
    return out


def step0_raw_scores(train, test, cs, ce):
    """full_smd_3way.step0_rank_original_mechanism과 동일 로직, raw dict(정렬 전) 리턴."""
    s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)
    scores = {}
    for c in range(N_CHANNELS):
        static_emb = embed_channel_window(train[:, c], s_static, e_static)
        dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
        scores[c] = cosine_dist(static_emb, dynamic_emb)
    return scores


def compute_49(segments_46, segments_47):
    cache = {}
    if CACHE_49.exists():
        cache = json.loads(CACHE_49.read_text(encoding="utf-8"))

    entity_train, entity_test = {}, {}
    all_segs = segments_46 + segments_47
    for i, seg in enumerate(all_segs):
        seg_id = seg["segment_id"]
        if seg_id in cache:
            continue
        entity = seg["entity"]
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        print(f"[{i+1}/{len(all_segs)}] computing step0 raw scores: {seg_id}", flush=True)
        scores = step0_raw_scores(entity_train[entity], entity_test[entity], seg["cs"], seg["ce"])
        cache[seg_id] = scores
        CACHE_49.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    for seg in all_segs:
        seg["scores"] = {int(c): v for c, v in cache[seg["segment_id"]].items()}
    return all_segs


# ---------------------------------------------------------------------------
# 특성 계산
# ---------------------------------------------------------------------------

def compute_features(scores_dict):
    """scores_dict: {channel_int: raw_cosine_dist}. 값이 클수록 '이상' 신호."""
    vals = np.array([scores_dict[c] for c in range(N_CHANNELS)])
    order = np.argsort(-vals)  # 내림차순 (제일 이상한 채널 먼저)
    sorted_vals = vals[order]

    top8 = sorted_vals[:8]
    rest = sorted_vals[8:]
    top3 = sorted_vals[:3]

    std_top8 = float(np.std(top8))
    gap_8_9_abs = float(sorted_vals[7] - sorted_vals[8]) if len(sorted_vals) > 8 else np.nan
    full_range = float(sorted_vals[0] - sorted_vals[-1])
    gap_8_9_norm = float(gap_8_9_abs / full_range) if full_range > 1e-12 else np.nan

    skew = float(stats.skew(vals))
    kurt = float(stats.kurtosis(vals))

    top3_mean = float(np.mean(top3))
    top8_mean = float(np.mean(top8))
    top3_top8_diff_ratio = float((top3_mean - top8_mean) / top8_mean) if abs(top8_mean) > 1e-12 else np.nan

    top8_range = float(top8[0] - top8[-1])
    top8_range_ratio = float(top8_range / full_range) if full_range > 1e-12 else np.nan

    # 정규화 후 엔트로피 (음수/0 처리 위해 min-shift 후 확률분포화)
    shifted = vals - vals.min() + 1e-9
    p = shifted / shifted.sum()
    entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))  # 0~1 정규화

    return {
        "std_top8": std_top8,
        "gap_8_9_abs": gap_8_9_abs,
        "gap_8_9_norm": gap_8_9_norm,
        "skew": skew,
        "kurtosis": kurt,
        "top3_top8_diff_ratio": top3_top8_diff_ratio,
        "top8_range_ratio": top8_range_ratio,
        "entropy_norm": entropy,
    }


FEATURE_NAMES = [
    "std_top8", "gap_8_9_abs", "gap_8_9_norm", "skew", "kurtosis",
    "top3_top8_diff_ratio", "top8_range_ratio", "entropy_norm",
]


def anova_f(groups):
    """groups: list of arrays (per bucket), NaN 제외. 값 부족하면 None."""
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None, None
    f, p = stats.f_oneway(*groups)
    return float(f), float(p)


def main():
    rows_74 = load_74()
    segs_46 = load_46()
    segs_47 = load_47()
    rows_49 = compute_49(segs_46, segs_47)

    all_rows = rows_74 + rows_49
    print(f"\n총 세그먼트 수: {len(all_rows)} (목표 123: 74+{len(segs_46)}+{len(segs_47)})")

    table = []
    for r in all_rows:
        feats = compute_features(r["scores"])
        n_gt = r["n_gt"]
        row = {
            "segment_id": r["segment_id"], "entity": r["entity"], "source": r["source"],
            "n_gt": n_gt, "bucket": bucket_of(n_gt),
        }
        row.update(feats)
        table.append(row)

    csv_path = OUT_DIR / "feature_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_id", "entity", "source", "n_gt", "bucket"] + FEATURE_NAMES
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(table)
    print(f"Saved: {csv_path}")

    n_gt_arr = np.array([r["n_gt"] for r in table], dtype=float)
    bucket_arr = np.array([r["bucket"] for r in table])

    correlations = {}
    bucket_stats = {}
    for fname in FEATURE_NAMES:
        fvals = np.array([r[fname] for r in table], dtype=float)
        mask = ~np.isnan(fvals)
        pear = stats.pearsonr(fvals[mask], n_gt_arr[mask]) if mask.sum() > 2 else (np.nan, np.nan)
        spear = stats.spearmanr(fvals[mask], n_gt_arr[mask]) if mask.sum() > 2 else (np.nan, np.nan)
        correlations[fname] = {
            "pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
            "spearman_r": float(spear[0]), "spearman_p": float(spear[1]),
            "n": int(mask.sum()),
        }

        groups = {}
        for bname in ["저(1-3)", "중(4-8)", "고(9+)"]:
            bmask = (bucket_arr == bname) & mask
            groups[bname] = {
                "mean": float(np.mean(fvals[bmask])) if bmask.sum() else None,
                "std": float(np.std(fvals[bmask])) if bmask.sum() else None,
                "n": int(bmask.sum()),
            }
        f_val, p_val = anova_f([fvals[(bucket_arr == b) & mask] for b in ["저(1-3)", "중(4-8)", "고(9+)"]])
        bucket_stats[fname] = {"groups": groups, "anova_F": f_val, "anova_p": p_val}

    summary = {
        "n_total": len(table),
        "n_by_source": {s: sum(1 for r in table if r["source"] == s) for s in ["74", "46", "47"]},
        "n_by_bucket": {b: int(np.sum(bucket_arr == b)) for b in ["저(1-3)", "중(4-8)", "고(9+)"]},
        "correlations_with_n_gt": correlations,
        "bucket_group_stats": bucket_stats,
    }
    summary_path = OUT_DIR / "correlation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\n=== 특성별 GT 채널수 상관관계 (Pearson / Spearman) ===")
    for fname in FEATURE_NAMES:
        c = correlations[fname]
        print(f"  {fname:25s}: pearson r={c['pearson_r']:+.3f} (p={c['pearson_p']:.3f})  "
              f"spearman r={c['spearman_r']:+.3f} (p={c['spearman_p']:.3f})  n={c['n']}")

    print("\n=== 버킷별 그룹 평균 ± std, ANOVA F ===")
    for fname in FEATURE_NAMES:
        bs = bucket_stats[fname]
        g = bs["groups"]
        line = f"  {fname:25s}: "
        for b in ["저(1-3)", "중(4-8)", "고(9+)"]:
            m, s, n = g[b]["mean"], g[b]["std"], g[b]["n"]
            if m is None:
                line += f"{b}=N/A  "
            else:
                line += f"{b}={m:.3f}±{s:.3f}(n={n})  "
        fstr = f"F={bs['anova_F']:.2f} p={bs['anova_p']:.3f}" if bs["anova_F"] is not None else "F=N/A"
        line += fstr
        print(line)


if __name__ == "__main__":
    main()
