"""
top-8 경계(8등-9등 점수 차이)가 I8 성패와 실제로 상관있는지 확인 -- 기존 캐시만 재사용,
VLM 호출 없음. gap-텍스트를 프롬프트에 넣는 실험(신규 VLM 콜 필요)의 사전 타당성 체크.
"""

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT_DIR = BASE / "experiments" / "results_gap_at_8_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
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
    score_cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
    features = {r["segment_id"]: r for r in csv.DictReader(FEATURES_CSV.open(encoding="utf-8"))}
    i8_rows = {r["segment_id"]: r for r in csv.DictReader(I8_COMPARISON_CSV.open(encoding="utf-8"))}
    quartile_of = load_quartiles()

    rows = []
    for seg_id, scores_raw in score_cache.items():
        if seg_id not in i8_rows or seg_id not in features:
            continue
        scores = {int(c): v for c, v in scores_raw.items()}
        ranked_scores = sorted(scores.values(), reverse=True)
        overall_std = float(features[seg_id]["step0_overall_std"])

        gap8_raw = ranked_scores[7] - ranked_scores[8]  # 8등 - 9등
        gap8_norm = gap8_raw / overall_std if overall_std > 1e-9 else 0.0

        ir = i8_rows[seg_id]
        f1_i8, f1_b = float(ir["f1_I8"]), float(ir["f1_B"])
        rows.append({
            "segment_id": seg_id, "quartile": quartile_of.get(seg_id, "?"),
            "gap8_normalized": gap8_norm,
            "f1_I8": f1_i8, "f1_B": f1_b,
            "I8_loses_to_B": f1_i8 < f1_b,
        })

    with (OUT_DIR / "gap8_vs_outcome.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 전체 74개: gap8과 f1_I8 상관관계 ----
    gap8_all = np.array([r["gap8_normalized"] for r in rows])
    f1_all = np.array([r["f1_I8"] for r in rows])
    corr = float(np.corrcoef(gap8_all, f1_all)[0, 1])
    print("=" * 70)
    print(f"[전체 74개] gap8_normalized(8등-9등 점수차) vs f1_I8 상관계수: {corr:.3f}")
    print("=" * 70)

    # ---- Q1 안에서 I8이 진 세그먼트 vs 나머지 ----
    q1_rows = [r for r in rows if r["quartile"] == "Q1_밀집"]
    q1_lose = [r for r in q1_rows if r["I8_loses_to_B"]]
    q1_not_lose = [r for r in q1_rows if not r["I8_loses_to_B"]]
    print(f"\nQ1({len(q1_rows)}개) 중 I8이 B에 진 세그먼트({len(q1_lose)}개) vs 나머지({len(q1_not_lose)}개)")
    if q1_lose:
        print(f"  I8<B: gap8_normalized mean={np.mean([r['gap8_normalized'] for r in q1_lose]):.4f}")
    if q1_not_lose:
        print(f"  I8>=B: gap8_normalized mean={np.mean([r['gap8_normalized'] for r in q1_not_lose]):.4f}")

    # ---- 전체에서도 I8이 진 세그먼트 vs 이긴 세그먼트 ----
    lose_all = [r for r in rows if r["I8_loses_to_B"]]
    notlose_all = [r for r in rows if not r["I8_loses_to_B"]]
    print(f"\n[전체 74개] I8이 B에 진 세그먼트({len(lose_all)}개) vs 나머지({len(notlose_all)}개)")
    print(f"  I8<B: gap8_normalized mean={np.mean([r['gap8_normalized'] for r in lose_all]):.4f}  median={np.median([r['gap8_normalized'] for r in lose_all]):.4f}")
    print(f"  I8>=B: gap8_normalized mean={np.mean([r['gap8_normalized'] for r in notlose_all]):.4f}  median={np.median([r['gap8_normalized'] for r in notlose_all]):.4f}")

    summary = {
        "n": len(rows),
        "corr_gap8_vs_f1_I8": corr,
        "q1_lose_gap8_mean": float(np.mean([r["gap8_normalized"] for r in q1_lose])) if q1_lose else None,
        "q1_notlose_gap8_mean": float(np.mean([r["gap8_normalized"] for r in q1_not_lose])) if q1_not_lose else None,
        "all_lose_gap8_mean": float(np.mean([r["gap8_normalized"] for r in lose_all])),
        "all_notlose_gap8_mean": float(np.mean([r["gap8_normalized"] for r in notlose_all])),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
