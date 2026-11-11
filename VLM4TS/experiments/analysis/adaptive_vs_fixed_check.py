"""
화면혼잡 이득이 고정 임계값인지 세그먼트별 연속적 관계인지 확인.
VLM 호출 없음(DINOv2 로컬 재계산만, VLM 클라이언트 모듈 import 없음).

Step0 실제 점수값이 체크포인트엔 랭킹(채널 인덱스 순서)만 저장돼 있고 원시
점수가 없어서, 74개 세그먼트에 대해 원본 Step0 메커니즘(단일 static 참조,
final-layer, 파라미터 변경 없음)으로 로컬 재계산 -- API 비용 0.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
import sys
sys.path.insert(0, str(ANALYSIS_DIR))
from step1v3_dino_graph_smd import (  # noqa: E402  (DINOv2 전용, VLM 클라이언트 아님)
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_adaptive_vs_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORE_CACHE = OUT_DIR / "step0_scores_cache.json"

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
SEG_ID_RE = re.compile(r"^(machine-\d+-\d+)_(\d+)_(\d+)$")


def prf(pred: set, true: set) -> tuple[float, float]:
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def load_all_segments():
    segments = {}
    ckpts = []
    for p in CKPT_PATHS:
        ck = json.loads(p.read_text(encoding="utf-8"))
        ckpts.append(ck)
        for key, val in ck.items():
            if key.endswith("_gt") and isinstance(val, dict) and "gt_channels" in val:
                seg_id = key[:-3]
                m = SEG_ID_RE.match(seg_id)
                if not m:
                    continue
                entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
                segments[seg_id] = (entity, cs, ce, set(val["gt_channels"]))
    return segments, ckpts


def find_ckpt_for(seg_id, ckpts):
    for ck in ckpts:
        if f"{seg_id}_A_vlm4ts_own" in ck:
            return ck
    return None


def compute_step0_scores(train, test, cs, ce):
    s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)
    scores = {}
    for c in range(N_CHANNELS):
        static_emb = embed_channel_window(train[:, c], s_static, e_static)
        dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
        scores[c] = cosine_dist(static_emb, dynamic_emb)
    return scores


def run():
    segments, ckpts = load_all_segments()
    print(f"세그먼트 수: {len(segments)}")

    score_cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8")) if SCORE_CACHE.exists() else {}
    entity_data = {}

    rows = []
    for seg_id, (entity, cs, ce, gt_channels) in segments.items():
        k = len(gt_channels)

        if seg_id not in score_cache:
            if entity not in entity_data:
                entity_data[entity] = load_smd(entity)
            train, test = entity_data[entity]
            print(f"[Step0] {seg_id} 재계산 중 ...", flush=True)
            score_cache[seg_id] = compute_step0_scores(train, test, cs, ce)
            SCORE_CACHE.write_text(json.dumps(score_cache, indent=2), encoding="utf-8")
        scores = {int(c): v for c, v in score_cache[seg_id].items()}

        ranked = sorted(scores, key=lambda c: -scores[c])
        ranked_scores = [scores[c] for c in ranked]
        overall_std = float(np.std(ranked_scores))
        overall_mean = float(np.mean(ranked_scores))

        # gap at true k: score[k-1](마지막 포함) - score[k](첫 제외)
        gap_at_k = ranked_scores[k - 1] - ranked_scores[k] if k < N_CHANNELS else 0.0
        gap_norm = gap_at_k / overall_std if overall_std > 1e-9 else 0.0

        # F1 확보 (A vs B)
        ck = find_ckpt_for(seg_id, ckpts)
        f1_a = f1_b = None
        if ck is not None:
            a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
            if a_entry.get("status") == "OK":
                p, r = prf(set(a_entry["pred"] or []), gt_channels)
                f1_a = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            b_pred, b_ok = set(), True
            for gi in range(4):
                ge = ck.get(f"{seg_id}_B_group{gi}", {})
                if ge.get("status") == "OK" and ge.get("pred"):
                    b_pred |= set(ge["pred"])
                if ge.get("status") != "OK":
                    b_ok = False
            if b_ok:
                p, r = prf(b_pred, gt_channels)
                f1_b = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        rows.append({
            "segment_id": seg_id, "entity": entity, "k_gt": k,
            "step0_overall_std": overall_std, "step0_overall_mean": overall_mean,
            "gap_at_k": gap_at_k, "gap_normalized": gap_norm,
            "f1_vlm4ts_own": f1_a, "f1_canmmlm_naive": f1_b,
        })

    with (OUT_DIR / "segment_features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- (1)(2) 분포 요약 ----
    k_vals = [r["k_gt"] for r in rows]
    gap_vals = [r["gap_normalized"] for r in rows]
    print("\n" + "=" * 70)
    print("(1)(2) 세그먼트별 GT 채널 수, Step0 밀집도(gap_normalized) 분포")
    print("=" * 70)
    print(f"GT 채널 수: min={min(k_vals)} max={max(k_vals)} mean={np.mean(k_vals):.2f} std={np.std(k_vals):.2f}")
    print(f"gap_normalized(클수록 뚜렷하게 갈림, 작을수록 밀집): "
          f"min={min(gap_vals):.3f} max={max(gap_vals):.3f} mean={np.mean(gap_vals):.3f} std={np.std(gap_vals):.3f}")
    print(f"  25/50/75 백분위수: {np.percentile(gap_vals, [25,50,75])}")

    # ---- (3) canmmlm 승 vs vlm4ts 승 그룹 비교 ----
    print("\n" + "=" * 70)
    print("(3) canmmlm_naive가 나은 세그먼트 vs vlm4ts_own이 나은 세그먼트, 특성 비교")
    print("=" * 70)
    both = [r for r in rows if r["f1_vlm4ts_own"] is not None and r["f1_canmmlm_naive"] is not None]
    b_better = [r for r in both if r["f1_canmmlm_naive"] > r["f1_vlm4ts_own"]]
    a_better = [r for r in both if r["f1_vlm4ts_own"] > r["f1_canmmlm_naive"]]
    tie = [r for r in both if r["f1_vlm4ts_own"] == r["f1_canmmlm_naive"]]
    print(f"canmmlm_naive 승: {len(b_better)}개  vlm4ts_own 승: {len(a_better)}개  동점: {len(tie)}개")

    def summarize(group, name):
        if not group:
            print(f"  {name}: (empty)")
            return {}
        ks = [g["k_gt"] for g in group]
        gaps = [g["gap_normalized"] for g in group]
        print(f"  {name} (n={len(group)}): GT수 mean={np.mean(ks):.2f}  gap_norm mean={np.mean(gaps):.3f} median={np.median(gaps):.3f}")
        return {"n": len(group), "k_gt_mean": float(np.mean(ks)), "gap_norm_mean": float(np.mean(gaps)), "gap_norm_median": float(np.median(gaps))}

    b_summary = summarize(b_better, "canmmlm_naive 승")
    a_summary = summarize(a_better, "vlm4ts_own 승")

    summary = {
        "n_segments": len(rows),
        "k_gt_distribution": {"min": min(k_vals), "max": max(k_vals), "mean": float(np.mean(k_vals)), "std": float(np.std(k_vals))},
        "gap_normalized_distribution": {
            "min": min(gap_vals), "max": max(gap_vals), "mean": float(np.mean(gap_vals)), "std": float(np.std(gap_vals)),
            "p25_p50_p75": np.percentile(gap_vals, [25, 50, 75]).tolist(),
        },
        "canmmlm_better_group": b_summary,
        "vlm4ts_better_group": a_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
