"""
vlm4ts_own (baseline, no top-k cutoff) 채널 단위 실패 진단.

새 실험/새 방법론 없음 -- §29 3-way 비교(smd_3way_baseline_comparison.py)의
기존 캐시된 결과(experiments/results_smd_3way_baseline/checkpoint.json)만
재분석. VLM 재호출 없음(vlm4ts_own 9개 세그먼트 전부 이미 status=OK로 캐시돼
있음, 직접 확인).

SEGMENTS/GT는 원본 스크립트에서 그대로 import(하드코딩 금지).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import BASE, SEGMENTS, load_smd_test  # noqa: E402

RESULTS_DIR = BASE / "experiments" / "results_smd_3way_baseline"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
SMD_DIR = BASE / "mv_data" / "SMD"

OUT_DIR = BASE / "experiments" / "results_vlm4ts_own_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train(entity):
    return np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")


def run():
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))

    channel_rows = []  # segment_id, entity, channel, is_gt, is_pred, category, max_abs_z
    segment_rows = []  # summary table rows
    entity_test_cache = {}
    entity_train_cache = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        entry = checkpoint.get(f"{seg_id}_A_vlm4ts_own")
        if entry is None or entry.get("status") != "OK":
            print(f"[MISSING] {seg_id}: vlm4ts_own not cached/OK -- skipping (should not happen)")
            continue
        pred_channels = set(entry["pred"] or [])

        if entity not in entity_test_cache:
            entity_test_cache[entity] = load_smd_test(entity)
            entity_train_cache[entity] = load_train(entity)
        test = entity_test_cache[entity]
        train = entity_train_cache[entity]

        tp = sorted(pred_channels & gt_channels)
        fp = sorted(pred_channels - gt_channels)
        fn = sorted(gt_channels - pred_channels)

        segment_rows.append({
            "segment_id": seg_id, "entity": entity, "cs": cs, "ce": ce,
            "n_gt": len(gt_channels), "n_pred": len(pred_channels),
            "tp": tp, "fp": fp, "fn": fn,
            "n_tp": len(tp), "n_fp": len(fp), "n_fn": len(fn),
            "raw_response": entry["raw"],
        })

        # channel-level rows: every channel 0..37 (SMD is 38-dim for all entities used here)
        n_channels = test.shape[1]
        for c in range(n_channels):
            is_gt = c in gt_channels
            is_pred = c in pred_channels
            if is_gt and is_pred:
                category = "TP"
            elif is_pred and not is_gt:
                category = "FP"
            elif is_gt and not is_pred:
                category = "FN"
            else:
                category = "TN"

            mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
            seg_vals = test[cs:ce, c]
            mz = float(np.max(np.abs(seg_vals - mu)) / sigma) if sigma > 1e-9 else 0.0

            channel_rows.append({
                "segment_id": seg_id, "entity": entity, "channel": c,
                "is_gt": is_gt, "is_pred": is_pred, "category": category,
                "max_abs_z_in_labeled_span": round(mz, 3),
            })

    # ---- CSV outputs ----
    with (OUT_DIR / "channel_level.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
        w.writeheader()
        w.writerows(channel_rows)

    with (OUT_DIR / "segment_level.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_id", "entity", "cs", "ce", "n_gt", "n_pred", "n_tp", "n_fp", "n_fn", "tp", "fp", "fn"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in segment_rows:
            row = {k: r[k] for k in fieldnames}
            w.writerow(row)

    # ---- Markdown summary table ----
    md_lines = ["| segment_id | GT 채널(개수) | 예측 채널(개수) | TP | FP | FN |",
                "|---|---|---|---|---|---|"]
    for r in segment_rows:
        md_lines.append(
            f"| {r['segment_id']} | {r['n_gt']} | {r['n_pred']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} |"
        )
    (OUT_DIR / "segment_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))

    # ---- FN / FP repeated-channel + max_abs_z pattern check ----
    print("\n" + "=" * 70)
    print("FN(놓친) 채널 vs TP(잡은) 채널의 max_abs_z 분포 비교")
    print("=" * 70)
    tp_rows = [r for r in channel_rows if r["category"] == "TP"]
    fn_rows = [r for r in channel_rows if r["category"] == "FN"]
    fp_rows = [r for r in channel_rows if r["category"] == "FP"]
    tp_z = [r["max_abs_z_in_labeled_span"] for r in tp_rows]
    fn_z = [r["max_abs_z_in_labeled_span"] for r in fn_rows]
    fp_z = [r["max_abs_z_in_labeled_span"] for r in fp_rows]
    print(f"TP (n={len(tp_z)}): mean max|z|={np.mean(tp_z):.3f}, median={np.median(tp_z):.3f}")
    print(f"FN (n={len(fn_z)}): mean max|z|={np.mean(fn_z):.3f}, median={np.median(fn_z):.3f}")
    print(f"FP (n={len(fp_z)}): mean max|z|={np.mean(fp_z):.3f}, median={np.median(fp_z):.3f}")

    print("\nFN 채널 상세 (segment, channel, max|z|):")
    for r in sorted(fn_rows, key=lambda r: r["max_abs_z_in_labeled_span"]):
        print(f"  {r['segment_id']:>28} ch{r['channel']:>2}  max|z|={r['max_abs_z_in_labeled_span']}")

    print("\nFP 채널 상세 (segment, channel, max|z|):")
    for r in sorted(fp_rows, key=lambda r: -r["max_abs_z_in_labeled_span"]):
        print(f"  {r['segment_id']:>28} ch{r['channel']:>2}  max|z|={r['max_abs_z_in_labeled_span']}")

    # repeated FN/FP channel index within same entity across its 3 segments
    print("\n같은 entity 내에서 반복되는 FN 채널 인덱스:")
    from collections import defaultdict
    fn_by_entity = defaultdict(list)
    for r in fn_rows:
        fn_by_entity[r["entity"]].append(r["channel"])
    for entity, chans in fn_by_entity.items():
        counts = {c: chans.count(c) for c in set(chans)}
        repeated = {c: n for c, n in counts.items() if n > 1}
        print(f"  {entity}: {dict(sorted(counts.items()))}  (반복(>1): {repeated if repeated else '없음'})")

    print("\n같은 entity 내에서 반복되는 FP 채널 인덱스:")
    fp_by_entity = defaultdict(list)
    for r in fp_rows:
        fp_by_entity[r["entity"]].append(r["channel"])
    for entity, chans in fp_by_entity.items():
        counts = {c: chans.count(c) for c in set(chans)}
        repeated = {c: n for c, n in counts.items() if n > 1}
        print(f"  {entity}: {dict(sorted(counts.items()))}  (반복(>1): {repeated if repeated else '없음'})")

    print("\n" + "=" * 70)
    print("raw VLM 응답 샘플 (원문 그대로, 참고: 이 프롬프트는 reasoning을 요청하지 않음 -- anomalous_channels 리스트 + confidence만 반환)")
    print("=" * 70)
    for r in segment_rows:
        print(f"\n[{r['segment_id']}] GT={sorted(r['tp']+r['fn'])} raw={r['raw_response']}")

    (OUT_DIR / "summary_stats.json").write_text(json.dumps({
        "tp_mean_max_abs_z": float(np.mean(tp_z)) if tp_z else None,
        "fn_mean_max_abs_z": float(np.mean(fn_z)) if fn_z else None,
        "fp_mean_max_abs_z": float(np.mean(fp_z)) if fp_z else None,
        "n_tp": len(tp_z), "n_fn": len(fn_z), "n_fp": len(fp_z),
    }, indent=2), encoding="utf-8")

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
