"""
SoM 파일럿(조건 G)의 F1 개선이 hedging(38채널 전부/거의 전부 응답) 아티팩트인지 확인.
새 VLM 호출 없음. VLM 클라이언트 모듈 import 없음 -- checkpoint_G.json + 기존 A/F/GT만 재사용.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_som_hedging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
SOM_CKPT = BASE / "experiments" / "results_pilot_som" / "checkpoint_G.json"
LAYOUT_CKPT = BASE / "experiments" / "results_pilot_layout" / "checkpoint_F.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
HEDGE_THRESHOLD = 35  # 35개 이상(38개 중) 응답 -> hedging으로 판정


def prf(pred: set, true: set) -> tuple[float, float]:
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def f1_of(pred, gt):
    p, r = prf(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    g_ckpt = json.loads(SOM_CKPT.read_text(encoding="utf-8"))
    f_ckpt = json.loads(LAYOUT_CKPT.read_text(encoding="utf-8"))
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]

    def find_gt(seg_id):
        for ck in ckpts:
            v = ck.get(f"{seg_id}_gt")
            if v:
                return set(v["gt_channels"])
        return None

    def find_ckpt_for(seg_id):
        for ck in ckpts:
            if f"{seg_id}_A_vlm4ts_own" in ck:
                return ck
        return None

    rows = []
    hedge_table = []
    for s in selected:
        seg_id, quartile = s["segment_id"], s["quartile"]
        gt = find_gt(seg_id)
        ck = find_ckpt_for(seg_id)
        g_entry = g_ckpt.get(seg_id, {})
        f_entry = f_ckpt.get(seg_id, {})
        if gt is None or ck is None or g_entry.get("status") != "OK" or f_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        if a_entry.get("status") != "OK":
            continue

        a_pred = set(a_entry["pred"] or [])
        f_pred = set(f_entry["pred"] or [])
        g_pred = set(g_entry["pred"] or [])

        is_hedge_a = len(a_pred) >= HEDGE_THRESHOLD
        is_hedge_g = len(g_pred) >= HEDGE_THRESHOLD

        rows.append({
            "segment_id": seg_id, "quartile": quartile,
            "n_pred_A": len(a_pred), "n_pred_G": len(g_pred),
            "hedge_A": is_hedge_a, "hedge_G": is_hedge_g,
            "f1_A": f1_of(a_pred, gt), "f1_F": f1_of(f_pred, gt), "f1_G": f1_of(g_pred, gt),
        })
        if is_hedge_g:
            hedge_table.append({
                "segment_id": seg_id, "n_pred_A": len(a_pred), "hedge_A": is_hedge_a,
                "n_pred_G": len(g_pred), "new_hedge_in_G": is_hedge_g and not is_hedge_a,
            })

    with (OUT_DIR / "hedging_reclassified.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (OUT_DIR / "hedge_segments_detail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hedge_table[0].keys()))
        w.writeheader()
        w.writerows(hedge_table)

    n_total = len(rows)
    n_hedge_g = sum(1 for r in rows if r["hedge_G"])
    n_new_hedge = sum(1 for h in hedge_table if h["new_hedge_in_G"])
    n_already_hedge_in_a = n_hedge_g - n_new_hedge

    print("=" * 70)
    print(f"전체 파일럿 n={n_total}, G에서 hedging(>={HEDGE_THRESHOLD}채널)으로 분류된 세그먼트: {n_hedge_g}개")
    print(f"  - 이 중 A에서도 이미 hedging이었던 경우: {n_already_hedge_in_a}개")
    print(f"  - 이 중 A는 정상이었는데 G에서 새로 생긴 hedging: {n_new_hedge}개")
    print("\n[3] hedging 세그먼트별 A 상태:")
    for h in hedge_table:
        tag = "NEW(라벨이 유발)" if h["new_hedge_in_G"] else "기존(A도 hedging)"
        print(f"  {h['segment_id']}: A n_pred={h['n_pred_A']} (hedge={h['hedge_A']})  G n_pred={h['n_pred_G']}  -> {tag}")

    # ---- [1][2] hedging 세그먼트 제외 후 재계산 ----
    clean = [r for r in rows if not r["hedge_G"]]
    n_clean = len(clean)
    mean_a = np.mean([r["f1_A"] for r in clean])
    mean_f = np.mean([r["f1_F"] for r in clean])
    mean_g = np.mean([r["f1_G"] for r in clean])

    print("\n" + "=" * 70)
    print(f"[1][2] hedging({n_hedge_g}개) 제외 후 나머지 n={n_clean}에서 A/F/G 재계산")
    print("=" * 70)
    print(f"F1: A={mean_a:.4f}  F={mean_f:.4f}  G={mean_g:.4f}")
    diff_g_a = mean_g - mean_a
    diff_g_f = mean_g - mean_f
    print(f"G - A = {diff_g_a:+.4f}   G - F = {diff_g_f:+.4f}")
    if diff_g_a > 0.01:
        verdict = "hedging 제외 후에도 G가 A보다 낫다 -- binding 개선 효과가 일부 실재함"
    elif diff_g_a < -0.01:
        verdict = "hedging 제외 후 G가 오히려 A보다 못하다 -- 원래의 개선은 대부분 hedging 아티팩트"
    else:
        verdict = "hedging 제외 후 G와 A가 사실상 동일 -- 52.3% 회복은 대부분 hedging 아티팩트, 순수 binding 개선 효과는 거의 없음"
    print(f"\n판정: {verdict}")

    # 전체(18개, hedging 포함) 대비 재확인
    mean_a_full = np.mean([r["f1_A"] for r in rows])
    mean_g_full = np.mean([r["f1_G"] for r in rows])
    print(f"\n(참고) hedging 포함 전체 n={n_total}: A={mean_a_full:.4f}  G={mean_g_full:.4f}  차이={mean_g_full-mean_a_full:+.4f}")

    summary = {
        "hedge_threshold": HEDGE_THRESHOLD,
        "n_total": n_total,
        "n_hedge_in_G": n_hedge_g,
        "n_already_hedge_in_A": n_already_hedge_in_a,
        "n_new_hedge_caused_by_label": n_new_hedge,
        "clean_n": n_clean,
        "clean_mean_f1": {"A": float(mean_a), "F": float(mean_f), "G": float(mean_g)},
        "clean_diff_G_minus_A": float(diff_g_a),
        "clean_diff_G_minus_F": float(diff_g_f),
        "full_mean_f1_for_reference": {"A": float(mean_a_full), "G": float(mean_g_full)},
        "verdict": verdict,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
