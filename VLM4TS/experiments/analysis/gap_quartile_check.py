"""
gap_normalized(Step0 밀집도) 3분위별 canmmlm_naive vs vlm4ts_own 비교.
VLM 재호출 없음, VLM 클라이언트 모듈 import 없음 -- 기존 checkpoint 예측 +
adaptive_vs_fixed_check.py가 이미 계산해둔 segment_features.csv만 재사용.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_gap_quartile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = BASE / "experiments" / "results_adaptive_vs_fixed" / "segment_features.csv"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
SEG_ID_RE = re.compile(r"^(machine-\d+-\d+)_(\d+)_(\d+)$")


def load_gt_and_ckpts():
    segments = {}
    ckpts = []
    for p in CKPT_PATHS:
        ck = json.loads(p.read_text(encoding="utf-8"))
        ckpts.append(ck)
        for key, val in ck.items():
            if key.endswith("_gt") and isinstance(val, dict) and "gt_channels" in val:
                seg_id = key[:-3]
                if SEG_ID_RE.match(seg_id):
                    segments[seg_id] = set(val["gt_channels"])
    return segments, ckpts


def find_ckpt_for(seg_id, ckpts):
    for ck in ckpts:
        if f"{seg_id}_A_vlm4ts_own" in ck:
            return ck
    return None


def run():
    gt_map, ckpts = load_gt_and_ckpts()
    features = list(csv.DictReader(FEATURES_CSV.open(encoding="utf-8")))
    print(f"세그먼트 수: {len(features)}")

    rows = []
    for r in features:
        seg_id = r["segment_id"]
        gt_channels = gt_map.get(seg_id)
        ck = find_ckpt_for(seg_id, ckpts)
        if gt_channels is None or ck is None:
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        if a_entry.get("status") != "OK":
            continue
        a_pred = set(a_entry["pred"] or [])
        b_pred, b_ok = set(), True
        for gi in range(4):
            ge = ck.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") == "OK" and ge.get("pred"):
                b_pred |= set(ge["pred"])
            if ge.get("status") != "OK":
                b_ok = False
        if not b_ok:
            continue

        fp_a = len(a_pred - gt_channels)
        fp_b = len(b_pred - gt_channels)

        rows.append({
            "segment_id": seg_id,
            "gap_normalized": float(r["gap_normalized"]),
            "k_gt": int(r["k_gt"]),
            "f1_A": float(r["f1_vlm4ts_own"]) if r["f1_vlm4ts_own"] not in ("", "None") else None,
            "f1_B": float(r["f1_canmmlm_naive"]) if r["f1_canmmlm_naive"] not in ("", "None") else None,
            "fp_A": fp_a, "fp_B": fp_b,
        })

    rows = [r for r in rows if r["f1_A"] is not None and r["f1_B"] is not None]
    rows.sort(key=lambda r: r["gap_normalized"])
    n = len(rows)
    q1_cut = n // 4
    q3_cut = n - n // 4

    quartiles = {
        "Q1_밀집(하위25%)": rows[:q1_cut],
        "Q2Q3_중간50%": rows[q1_cut:q3_cut],
        "Q4_뚜렷(상위25%)": rows[q3_cut:],
    }

    print("\n" + "=" * 70)
    print("gap_normalized 3분위별 F1/FP 비교")
    print("=" * 70)
    summary = {}
    for name, grp in quartiles.items():
        f1_a = np.mean([g["f1_A"] for g in grp])
        f1_b = np.mean([g["f1_B"] for g in grp])
        fp_a = np.mean([g["fp_A"] for g in grp])
        fp_b = np.mean([g["fp_B"] for g in grp])
        gap_range = (grp[0]["gap_normalized"], grp[-1]["gap_normalized"])
        b_wins = sum(1 for g in grp if g["f1_B"] > g["f1_A"])
        a_wins = sum(1 for g in grp if g["f1_A"] > g["f1_B"])
        canmmlm_advantage = f1_b - f1_a
        summary[name] = {
            "n": len(grp), "gap_range": gap_range,
            "f1_vlm4ts_own": float(f1_a), "f1_canmmlm_naive": float(f1_b),
            "canmmlm_advantage_f1": float(canmmlm_advantage),
            "fp_vlm4ts_own": float(fp_a), "fp_canmmlm_naive": float(fp_b),
            "n_canmmlm_wins": b_wins, "n_vlm4ts_wins": a_wins,
        }
        print(f"\n{name} (n={len(grp)}, gap {gap_range[0]:.3f}~{gap_range[1]:.3f}):")
        print(f"  F1: vlm4ts_own={f1_a:.4f}  canmmlm_naive={f1_b:.4f}  canmmlm 우위={canmmlm_advantage:+.4f}")
        print(f"  FP: vlm4ts_own={fp_a:.2f}  canmmlm_naive={fp_b:.2f}")
        print(f"  승수: canmmlm {b_wins} vs vlm4ts {a_wins} (동점 {len(grp)-b_wins-a_wins})")

    with (OUT_DIR / "gap_quartile_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 핵심 판정 ----
    q1_adv = summary["Q1_밀집(하위25%)"]["canmmlm_advantage_f1"]
    q4_adv = summary["Q4_뚜렷(상위25%)"]["canmmlm_advantage_f1"]
    print("\n" + "=" * 70)
    print("핵심 판정: 밀집도에 따라 canmmlm_naive 우위가 달라지는가")
    print("=" * 70)
    print(f"Q1(밀집) canmmlm 우위: {q1_adv:+.4f}")
    print(f"Q4(뚜렷) canmmlm 우위: {q4_adv:+.4f}")
    trend_clear = (q4_adv - q1_adv) > 0.05  # 임의 기준: 5%p 이상 벌어지면 "뚜렷한 경향"으로 판단
    both_positive_similar = q1_adv > 0 and q4_adv > 0 and abs(q4_adv - q1_adv) <= 0.05

    if q1_adv <= 0 < q4_adv:
        verdict = "밀집(Q1)에서는 canmmlm_naive 우위가 사라지거나 역전, 뚜렷(Q4)에서는 우위 유지 -- 가설 지지"
    elif trend_clear:
        verdict = "밀집도가 낮을수록(Q1) canmmlm_naive 우위가 뚜렷하게 작음 -- 가설 지지, 적응형 검증 가치 있음"
    elif both_positive_similar:
        verdict = "Q1/Q4 모두 canmmlm_naive가 비슷한 크기로 우위 -- 밀집도가 우위 크기를 가르지 않음, 적응형 방향 보류 권고"
    else:
        verdict = "뚜렷한 패턴 없음 -- 적응형 방향 보류 권고"

    print(f"\n판정: {verdict}")

    summary["verdict"] = verdict
    summary["q1_vs_q4_advantage_diff"] = float(q4_adv - q1_adv)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
