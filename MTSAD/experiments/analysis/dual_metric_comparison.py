"""
Max-F1(논문 정의) 재현 가능성 판단 + 계산 가능한 대안 지표를 recall@k(채널 F1)와 나란히 비교.

1단계 결론 (이 스크립트 실행 전에 조사로 확정됨, 코드로 재확인 없음):
  - I8/K2/canmmlm_naive/vlm4ts_own/our_design-top4 체크포인트는 전부 GT 이상구간에
    센터링된 세그먼트뿐 -- 정상(negative) 세그먼트가 하나도 없다.
  - 예측은 {"anomalous_channels": [...], "confidence": "low/medium/high"} 카테고리형이라
    연속 anomaly score가 없다 (threshold sweep 불가).
  - compute_pa_f1.py/ablation_pa_f1.py 스타일 Max-F1(전체 시계열 + 연속 점수 + PA)은
    이 데이터로 원천적으로 계산 불가. 억지 근사치를 "Max-F1"이라 부르지 않는다.

이 스크립트가 계산하는 것 (전부 recall@k와 나란히, 정확한 이름으로):
  A. recall@k 채널 F1/precision/recall (기존 방식 재확인, 세그먼트 평균)
  B. segment-level detection recall = pred가 비어있지 않은 세그먼트 비율
     (정상 세그먼트가 없어 precision/F1 정의 불가 -- recall만 있는 반쪽 지표)
  C. confidence(low/medium/high) 분포 -- 순서형 근사, 이것도 정상 세그먼트 없어 반쪽
  D. GT 채널 수 구간(저 1-3 / 중 4-8 / 고 9+)별 A/B 분해

비용 0: 전부 기존 체크포인트/캐시 재사용, VLM 재호출 없음.
"""
import csv
import json
import re
from pathlib import Path

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_dual_metric_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
I8_CKPT = BASE / "experiments" / "results_heatmap_overlay_74" / "checkpoint_I8.json"
K2_CKPT = BASE / "experiments" / "results_pilot_reduced_index" / "checkpoint_K2.json"
FEATURE_TABLE = BASE / "experiments" / "results_gt_range_predictability" / "feature_table.csv"

N_GROUPS = 4  # canmmlm_naive group count (B_group0..3)


def prf(pred: set, true: set):
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    return p, r


def f1_of(pred, gt):
    p, r = prf(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def confidence_of(entry):
    m = re.search(r'"confidence":\s*"(\w+)"', entry.get("raw") or "")
    return m.group(1) if m else None


def load_gt_map():
    gt = {}
    for path in GT_CKPT_PATHS:
        ck = json.loads(path.read_text(encoding="utf-8"))
        for k, v in ck.items():
            if k.endswith("_gt"):
                gt[k[:-3]] = set(v["gt_channels"])
    return gt


def load_bucket_map():
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))
    m = {}
    for r in rows:
        b = r["bucket"]
        if "1-3" in b:
            label = "low(1-3)"
        elif "4-8" in b:
            label = "mid(4-8)"
        else:
            label = "high(9+)"
        m[r["segment_id"]] = label
    return m


def collect_full_3way():
    """vlm4ts_own(A) / canmmlm_naive(B, union of 4 groups) / our_design-top4(C) 의 74개 세그먼트."""
    merged = {}
    for path in GT_CKPT_PATHS:
        merged.update(json.loads(path.read_text(encoding="utf-8")))
    seg_ids = sorted({k[:-3] for k in merged if k.endswith("_gt")})

    out = {"vlm4ts_own": {}, "canmmlm_naive": {}, "our_design-top4": {}}
    for seg_id in seg_ids:
        a = merged.get(f"{seg_id}_A_vlm4ts_own")
        if a and a.get("status") == "OK":
            out["vlm4ts_own"][seg_id] = a

        b_pred, b_ok, b_confs = set(), True, []
        for gi in range(N_GROUPS):
            ge = merged.get(f"{seg_id}_B_group{gi}")
            if not ge or ge.get("status") != "OK":
                b_ok = False
                break
            b_pred |= set(ge.get("pred") or [])
            c = confidence_of(ge)
            if c:
                b_confs.append(c)
        if b_ok:
            # union prediction across 4 calls; confidence = highest seen (ordinal) for the approx table
            order = {"low": 0, "medium": 1, "high": 2}
            worst_first_conf = min(b_confs, key=lambda x: order.get(x, 1)) if b_confs else None
            out["canmmlm_naive"][seg_id] = {"pred": sorted(b_pred), "status": "OK", "raw": f'"confidence": "{worst_first_conf}"'}

        c = merged.get(f"{seg_id}_C_our_design")
        if c and c.get("status") == "OK":
            out["our_design-top4"][seg_id] = c

    return out


def main():
    gt_map = load_gt_map()
    bucket_map = load_bucket_map()

    conditions = collect_full_3way()
    conditions["I8"] = json.loads(I8_CKPT.read_text(encoding="utf-8"))
    conditions["K2"] = json.loads(K2_CKPT.read_text(encoding="utf-8"))

    detail_rows = []
    for cond_name, ckpt in conditions.items():
        for seg_id, entry in ckpt.items():
            if entry.get("status") != "OK" or seg_id not in gt_map:
                continue
            gt = gt_map[seg_id]
            pred = set(entry.get("pred") or [])
            p, r = prf(pred, gt)
            detail_rows.append({
                "condition": cond_name,
                "segment_id": seg_id,
                "n_gt": len(gt),
                "gt_bucket": bucket_map.get(seg_id, "unknown"),
                "n_pred": len(pred),
                "pred_nonempty": int(len(pred) > 0),
                "precision": p,
                "recall": r,
                "f1": f1_of(pred, gt),
                "confidence": confidence_of(entry),
            })

    with (OUT_DIR / "segment_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    # ---- aggregate per condition ----
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    summary = {}
    for cond_name in conditions:
        rows = [r for r in detail_rows if r["condition"] == cond_name]
        if not rows:
            continue
        conf_counts = {}
        for r in rows:
            c = r["confidence"] or "unknown"
            conf_counts[c] = conf_counts.get(c, 0) + 1
        by_bucket = {}
        for bucket in ("low(1-3)", "mid(4-8)", "high(9+)"):
            brows = [r for r in rows if r["gt_bucket"] == bucket]
            if not brows:
                continue
            by_bucket[bucket] = {
                "n": len(brows),
                "recall_at_k_f1": mean(x["f1"] for x in brows),
                "recall_at_k_precision": mean(x["precision"] for x in brows),
                "recall_at_k_recall": mean(x["recall"] for x in brows),
                "segment_detection_recall": mean(x["pred_nonempty"] for x in brows),
            }
        summary[cond_name] = {
            "n": len(rows),
            "recall_at_k_f1": mean(r["f1"] for r in rows),
            "recall_at_k_precision": mean(r["precision"] for r in rows),
            "recall_at_k_recall": mean(r["recall"] for r in rows),
            "segment_detection_recall": mean(r["pred_nonempty"] for r in rows),
            "segment_detection_recall_NOTE": "정상(negative) 세그먼트가 없어 precision/F1 정의 불가 -- recall만 있는 반쪽 지표",
            "confidence_distribution": conf_counts,
            "confidence_NOTE": "정상 세그먼트 없어 confidence 기반 sweep도 precision/F1 계산 불가 -- 분포만 참고용",
            "by_gt_bucket": by_bucket,
            "max_f1_paper_definition": None,
            "max_f1_NOTE": (
                "계산 불가: 연속 anomaly score 없음(카테고리 예측만 존재) + 정상 세그먼트 없음(전부 GT 이상구간). "
                "compute_pa_f1.py/ablation_pa_f1.py는 별도 파이프라인(Stage1 DINOv2 overlay 연속 점수, 전체 "
                "시계열)을 전제로 하며 이 GPT 카테고리 예측 데이터에는 적용 불가."
            ),
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console table ----
    print("=" * 100)
    print("Max-F1(논문 정의) 계산 가능성: 불가 (연속 점수 없음 + 정상 세그먼트 없음)")
    print("=" * 100)
    print(f"{'condition':<18}{'n':>5}{'recall@k F1':>14}{'recall@k P':>13}{'recall@k R':>13}{'seg-detect R':>14}")
    for cond_name, s in summary.items():
        print(f"{cond_name:<18}{s['n']:>5}{s['recall_at_k_f1']:>14.4f}{s['recall_at_k_precision']:>13.4f}"
              f"{s['recall_at_k_recall']:>13.4f}{s['segment_detection_recall']:>14.4f}")
    print(f"\nSaved: {OUT_DIR}")


def demo():
    """최소 자기 점검: prf/f1_of 로직이 깨지지 않았는지."""
    assert prf(set(), set()) == (1.0, 1.0)
    assert prf({1, 2}, {2, 3}) == (0.5, 0.5)
    assert f1_of({1, 2}, {2, 3}) == 0.5
    assert f1_of(set(), {1}) == 0.0
    print("demo OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
