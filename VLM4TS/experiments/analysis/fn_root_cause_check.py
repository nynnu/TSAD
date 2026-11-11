"""
FN 채널 원인 2가지 가설 검증 (신규 실험 없음, 기존 코드/데이터 대조만).

배경: vlm4ts_own_channel_diagnosis.py 결과 -- FN 채널 신호 강도가 TP보다 큼
("미묘해서 놓친다" 가설 기각), entity별 반복 FN 채널 존재.

가설 A (point-wise 미스매치): FN 채널은 라벨 구간 내 짧게만 튀는 point성 신호,
  TP 채널은 구간 내내 유지되는 range성 신호인가?
가설 B (Step0 스코어링 문제): 반복 FN 채널은 DINOv2 Step0 점수 자체도 낮은가,
  아니면 Step0는 잡았는데 VLM이 놓쳤나?

Step0 스코어링은 저장된 전체 38채널 점수가 없어(top10만 캐시됨), 기존
step1v3_dino_graph_smd.py의 Step0 메커니즘(단일 static 참조, WIN=224,
DINOv2 patch-mean cosine distance)을 파라미터 변경 없이 그대로 재사용해
9개 세그먼트에 대해서만 재계산. VLM 호출 없음, API 비용 0.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import BASE, SEGMENTS  # noqa: E402
from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)

SMD_DIR = BASE / "mv_data" / "SMD"
DIAG_DIR = BASE / "experiments" / "results_vlm4ts_own_diagnosis"
OUT_DIR = BASE / "experiments" / "results_fn_root_cause"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Z_THRESHOLD = 3.0  # 표준 anomaly threshold (task 명시 fallback)

# entity별 반복 FN 채널 (직전 진단에서 확인된 목록, 하드코딩 아님 -- 진단 결과 그대로 인용)
REPEATED_FN_CHANNELS = {
    "machine-1-1": [8, 9, 11, 12, 13, 14],
    "machine-1-2": [9, 10, 12],
    "machine-1-5": [1, 2, 3, 6, 23, 25],
}


def load_channel_level():
    rows = []
    with (DIAG_DIR / "channel_level.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["channel"] = int(row["channel"])
            row["is_gt"] = row["is_gt"] == "True"
            row["is_pred"] = row["is_pred"] == "True"
            rows.append(row)
    return rows


def point_ratio(test: np.ndarray, cs: int, ce: int, c: int, mu: float, sigma: float) -> float:
    if sigma < 1e-9:
        return 0.0
    seg = test[cs:ce, c]
    z = np.abs(seg - mu) / sigma
    return float(np.mean(z > Z_THRESHOLD))


def compute_step0_scores(train, test, cs, ce):
    """Exact same mechanism as step1v3_dino_graph_smd.py's Step0 (unmodified)."""
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
    channel_rows = load_channel_level()

    seg_meta = {f"{entity}_{cs}_{ce}": (entity, cs, ce, [d - 1 for d in gt_dims])
                for entity, (cs, ce), gt_dims in SEGMENTS}

    entity_data = {}
    step0_cache_path = OUT_DIR / "step0_full_scores.json"
    step0_scores_by_seg = {}
    if step0_cache_path.exists():
        step0_scores_by_seg = json.loads(step0_cache_path.read_text(encoding="utf-8"))
        print(f"[reuse] loaded cached Step0 full scores from {step0_cache_path}")

    out_rows = []

    for row in channel_rows:
        seg_id = row["segment_id"]
        entity, cs, ce, gt_channels = seg_meta[seg_id]
        c = row["channel"]
        category = row["category"]

        if category not in ("TP", "FN"):
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        # --- 가설 A: point ratio ---
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        pr = point_ratio(test, cs, ce, c, mu, sigma)

        # --- 가설 B: Step0 rank ---
        if seg_id not in step0_scores_by_seg:
            print(f"[Step0] computing full 38-channel scores for {seg_id} ...", flush=True)
            step0_scores_by_seg[seg_id] = compute_step0_scores(train, test, cs, ce)
            step0_cache_path.write_text(json.dumps(step0_scores_by_seg, indent=2), encoding="utf-8")
        scores = step0_scores_by_seg[seg_id]
        scores = {int(k): v for k, v in scores.items()}
        ranked = sorted(scores, key=lambda ch: -scores[ch])
        rank = ranked.index(c) + 1  # 1-based
        k_gt = len(gt_channels)
        step0_caught = rank <= k_gt  # Step0 top-k(=|GT|) 안에 있었는지, recall@k와 동일 기준

        is_repeated_fn_channel = c in REPEATED_FN_CHANNELS.get(entity, [])

        out_rows.append({
            "segment_id": seg_id, "entity": entity, "channel": c, "category": category,
            "label_span_len": ce - cs,
            "point_ratio_z_gt_3": round(pr, 4),
            "step0_rank": rank, "step0_rank_of_38": f"{rank}/38",
            "n_gt_in_segment": k_gt, "step0_within_topk": step0_caught,
            "is_repeated_fn_channel": is_repeated_fn_channel,
        })
        print(f"  {seg_id} ch{c} [{category}] point_ratio={pr:.3f} step0_rank={rank}/38 "
              f"(top-{k_gt} cutoff, within={step0_caught}) repeated_fn={is_repeated_fn_channel}", flush=True)

    with (OUT_DIR / "fn_tp_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- 가설 A 집계 ----
    tp_pr = [r["point_ratio_z_gt_3"] for r in out_rows if r["category"] == "TP"]
    fn_pr = [r["point_ratio_z_gt_3"] for r in out_rows if r["category"] == "FN"]
    fn_repeated_pr = [r["point_ratio_z_gt_3"] for r in out_rows if r["category"] == "FN" and r["is_repeated_fn_channel"]]

    # ---- 가설 B 집계 ----
    fn_rows = [r for r in out_rows if r["category"] == "FN"]
    fn_repeated_rows = [r for r in fn_rows if r["is_repeated_fn_channel"]]
    tp_rows = [r for r in out_rows if r["category"] == "TP"]

    def b_summary(rows):
        n = len(rows)
        n_step0_caught = sum(1 for r in rows if r["step0_within_topk"])
        return n, n_step0_caught, n - n_step0_caught

    fn_n, fn_step0_in, fn_step0_out = b_summary(fn_rows)
    fn_rep_n, fn_rep_step0_in, fn_rep_step0_out = b_summary(fn_repeated_rows)
    tp_n, tp_step0_in, tp_step0_out = b_summary(tp_rows)

    md = []
    md.append("## 가설 A -- point-wise vs range-wise (|z|>3 초과 비율)\n")
    md.append(f"- TP (n={len(tp_pr)}): mean={np.mean(tp_pr):.3f}, median={np.median(tp_pr):.3f}")
    md.append(f"- FN (n={len(fn_pr)}): mean={np.mean(fn_pr):.3f}, median={np.median(fn_pr):.3f}")
    md.append(f"- FN, 반복 FN 채널만 (n={len(fn_repeated_pr)}): mean={np.mean(fn_repeated_pr):.3f}, median={np.median(fn_repeated_pr):.3f}\n")

    md.append("## 가설 B -- Step0(DINOv2) 랭킹 (해당 세그먼트 GT 개수 = k 기준 top-k 안에 들었는지)\n")
    md.append(f"- FN 전체 (n={fn_n}): Step0도 top-k 밖(=B 해당) {fn_step0_out}개 / Step0는 top-k 안(=VLM만 놓침) {fn_step0_in}개")
    md.append(f"- FN, 반복 FN 채널만 (n={fn_rep_n}): Step0도 top-k 밖 {fn_rep_step0_out}개 / Step0는 top-k 안 {fn_rep_step0_in}개")
    md.append(f"- TP 참고 (n={tp_n}): Step0 top-k 안 {tp_step0_in}개 / 밖인데도 VLM이 잡음 {tp_step0_out}개\n")

    md_text = "\n".join(md)
    (OUT_DIR / "summary.md").write_text(md_text, encoding="utf-8")
    print("\n" + "=" * 70)
    print(md_text)

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "hypothesis_A": {
            "tp_mean": float(np.mean(tp_pr)), "tp_median": float(np.median(tp_pr)),
            "fn_mean": float(np.mean(fn_pr)), "fn_median": float(np.median(fn_pr)),
            "fn_repeated_mean": float(np.mean(fn_repeated_pr)) if fn_repeated_pr else None,
            "fn_repeated_median": float(np.median(fn_repeated_pr)) if fn_repeated_pr else None,
        },
        "hypothesis_B": {
            "fn_total": fn_n, "fn_step0_also_missed": fn_step0_out, "fn_step0_caught_vlm_missed": fn_step0_in,
            "fn_repeated_total": fn_rep_n, "fn_repeated_step0_also_missed": fn_rep_step0_out,
            "fn_repeated_step0_caught_vlm_missed": fn_rep_step0_in,
            "tp_total": tp_n, "tp_step0_within_topk": tp_step0_in,
        },
    }, indent=2), encoding="utf-8")

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
