"""
Q1(밀집) 분위에서 heatmap 색상 대비(38채널 간 Step0 점수 분산)가 실제로 Q4(뚜렷)보다
작은지 확인, 그리고 Q1 안에서도 I8이 B에 진 세그먼트만 따로 떼서 대비가 Q1 평균보다도
더 낮은 극단 케이스인지 확인. VLM 재호출 없음, 기존 캐시(segment_features.csv,
gap_quartile_detail.csv, results_heatmap_overlay_74)만 재사용.

"38채널 간 값의 분산" = step0_overall_std(adaptive_vs_fixed_check.py가 이미 계산해둔,
세그먼트별 38채널 Step0 점수의 표준편차) -- heatmap 색상이 실제로 인코딩하는 것과
가장 가까운 정량 지표. gap_normalized(=gap_at_k/std)는 Q1/Q4 분위를 정의하는 데
이미 쓰인 값이라 순환 논리를 피하려고 std 자체를 별도로 확인.
"""

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_q1_contrast_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = BASE / "experiments" / "results_adaptive_vs_fixed" / "segment_features.csv"
GAP_DETAIL_CSV = BASE / "experiments" / "results_gap_quartile" / "gap_quartile_detail.csv"
I8_COMPARISON_CSV = BASE / "experiments" / "results_heatmap_overlay_74" / "comparison_I8.csv"


def load_quartiles():
    rows = list(csv.DictReader(GAP_DETAIL_CSV.open(encoding="utf-8")))
    for r in rows:
        r["gap_normalized"] = float(r["gap_normalized"])
    rows.sort(key=lambda r: r["gap_normalized"])
    n = len(rows)
    q1_cut, q3_cut = n // 4, n - n // 4
    quartile_of = {}
    for i, r in enumerate(rows):
        if i < q1_cut:
            quartile_of[r["segment_id"]] = "Q1_밀집"
        elif i < q3_cut:
            quartile_of[r["segment_id"]] = "Q2Q3_중간"
        else:
            quartile_of[r["segment_id"]] = "Q4_뚜렷"
    return quartile_of


def run():
    quartile_of = load_quartiles()
    features = {r["segment_id"]: r for r in csv.DictReader(FEATURES_CSV.open(encoding="utf-8"))}
    i8_rows = {r["segment_id"]: r for r in csv.DictReader(I8_COMPARISON_CSV.open(encoding="utf-8"))}

    rows = []
    for seg_id, q in quartile_of.items():
        if seg_id not in features:
            continue
        f = features[seg_id]
        row = {
            "segment_id": seg_id, "quartile": q,
            "step0_overall_std": float(f["step0_overall_std"]),
            "gap_normalized": float(f["gap_normalized"]),
        }
        if seg_id in i8_rows:
            ir = i8_rows[seg_id]
            row["f1_I8"] = float(ir["f1_I8"])
            row["f1_B"] = float(ir["f1_B"])
            row["I8_loses_to_B"] = float(ir["f1_I8"]) < float(ir["f1_B"])
        rows.append(row)

    with (OUT_DIR / "contrast_by_quartile.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- [1] Q1 vs Q4 채널 간 표준편차(색상 대비의 정량 지표) ----
    q1_std = [r["step0_overall_std"] for r in rows if r["quartile"] == "Q1_밀집"]
    q4_std = [r["step0_overall_std"] for r in rows if r["quartile"] == "Q4_뚜렷"]
    q23_std = [r["step0_overall_std"] for r in rows if r["quartile"] == "Q2Q3_중간"]
    print("=" * 70)
    print("[1] 38채널 간 Step0 점수 표준편차(대비) -- Q1 vs Q2Q3 vs Q4")
    print("=" * 70)
    print(f"Q1(밀집): mean={np.mean(q1_std):.4f}  median={np.median(q1_std):.4f}  (n={len(q1_std)})")
    print(f"Q2Q3(중간): mean={np.mean(q23_std):.4f}  median={np.median(q23_std):.4f}  (n={len(q23_std)})")
    print(f"Q4(뚜렷): mean={np.mean(q4_std):.4f}  median={np.median(q4_std):.4f}  (n={len(q4_std)})")
    ratio = np.mean(q1_std) / np.mean(q4_std) if np.mean(q4_std) > 0 else None
    print(f"\nQ1/Q4 대비 비율: {ratio:.3f}" if ratio else "")
    q1_lower = np.mean(q1_std) < np.mean(q4_std)
    print(f"Q1이 Q4보다 대비(표준편차) 실제로 작음: {q1_lower}")

    # ---- [2] Q1 안에서 I8이 B에 진 세그먼트 vs 나머지 Q1 ----
    q1_rows = [r for r in rows if r["quartile"] == "Q1_밀집" and "f1_I8" in r]
    q1_lose = [r for r in q1_rows if r["I8_loses_to_B"]]
    q1_not_lose = [r for r in q1_rows if not r["I8_loses_to_B"]]
    print("\n" + "=" * 70)
    print(f"[2] Q1({len(q1_rows)}개) 중 I8이 B에 진 세그먼트({len(q1_lose)}개) vs 나머지({len(q1_not_lose)}개)")
    print("=" * 70)
    if q1_lose:
        lose_std = [r["step0_overall_std"] for r in q1_lose]
        print(f"I8<B 세그먼트 대비: mean={np.mean(lose_std):.4f}  median={np.median(lose_std):.4f}")
    if q1_not_lose:
        notlose_std = [r["step0_overall_std"] for r in q1_not_lose]
        print(f"I8>=B 세그먼트 대비: mean={np.mean(notlose_std):.4f}  median={np.median(notlose_std):.4f}")
    print(f"Q1 전체 평균 대비: {np.mean(q1_std):.4f}")
    extreme_case = q1_lose and np.mean(lose_std) < np.mean(q1_std)
    print(f"\nI8<B 세그먼트가 Q1 평균보다도 대비가 더 낮은 극단 케이스인가: {extreme_case}")

    for r in q1_lose:
        print(f"  {r['segment_id']}: std={r['step0_overall_std']:.4f}  f1_I8={r['f1_I8']:.3f}  f1_B={r['f1_B']:.3f}")

    summary = {
        "q1_std_mean": float(np.mean(q1_std)), "q1_std_median": float(np.median(q1_std)),
        "q23_std_mean": float(np.mean(q23_std)),
        "q4_std_mean": float(np.mean(q4_std)), "q4_std_median": float(np.median(q4_std)),
        "q1_lower_contrast_than_q4": bool(q1_lower),
        "n_q1": len(q1_rows), "n_q1_i8_loses_to_b": len(q1_lose),
        "q1_lose_std_mean": float(np.mean(lose_std)) if q1_lose else None,
        "q1_not_lose_std_mean": float(np.mean(notlose_std)) if q1_not_lose else None,
        "extreme_low_contrast_case": bool(extreme_case) if q1_lose else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
