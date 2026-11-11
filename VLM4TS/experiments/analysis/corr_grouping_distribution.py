"""
상관계수 임계값 기반 채널 그룹핑의 그룹 수/크기 분포 확인 (VLM 호출 없음).

colab_multivariate_v2.py의 build_channel_groups()는 고정 그룹 크기(seed+top-k)
방식이라 이번 실험(임계값 이상이면 묶고 아니면 단독 그룹)과는 다름 -- "참고"만 하고
임계값 기반 connected-component 그룹핑으로 새로 구현. 상관행렬은 entity의 train
데이터 전체(정상 구간)로 계산(콜드 실험이라 세그먼트 window 대신 entity 단위 고정
참조 사용 -- Step0 참조 메커니즘과 동일한 선택), entity당 1회 계산 후 그 entity의
모든 세그먼트에 재사용.

74개 세그먼트는 full_smd_3way.py 3-way 비교와 동일 표본(3개 checkpoint 병합),
gap 분위 라벨은 results_gap_quartile/gap_quartile_detail.csv 재사용.
"""

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_corr_grouping"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
GAP_DETAIL_CSV = BASE / "experiments" / "results_gap_quartile" / "gap_quartile_detail.csv"
THRESHOLDS = [0.5, 0.7, 0.85, 0.9]
BIG_BLOB_SIZE = 20  # 이 이상이면 "큰 덩어리"로 간주


def get_74_segments():
    merged = {}
    for p in CKPT_PATHS:
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return sorted({k[:-3] for k in merged if k.endswith("_gt")})


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


def build_channel_groups_threshold(train_data: np.ndarray, threshold: float) -> list[list[int]]:
    """임계값 이상 |corr|인 채널끼리 connected-component로 묶음. 안 되면 단독 그룹."""
    C = train_data.shape[1]
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    adj = np.abs(corr) >= threshold

    visited = [False] * C
    groups = []
    for i in range(C):
        if visited[i]:
            continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(C):
                if not visited[v] and adj[u, v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(sorted(comp))
    return groups


def run():
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from step1v3_dino_graph_smd import load_smd

    seg_ids = get_74_segments()
    quartile_of = load_quartiles()
    print(f"세그먼트: {len(seg_ids)}개, 임계값: {THRESHOLDS}")

    entity_train = {}
    rows = []
    for seg_id in seg_ids:
        entity = seg_id.rsplit("_", 2)[0]
        if entity not in entity_train:
            train, _ = load_smd(entity)
            entity_train[entity] = train
        train = entity_train[entity]
        quartile = quartile_of.get(seg_id, "?")

        for t in THRESHOLDS:
            groups = build_channel_groups_threshold(train, t)
            sizes = [len(g) for g in groups]
            rows.append({
                "segment_id": seg_id, "entity": entity, "quartile": quartile,
                "threshold": t, "n_groups": len(groups), "max_group_size": max(sizes),
                "min_group_size": min(sizes), "mean_group_size": float(np.mean(sizes)),
                "n_singletons": sum(1 for s in sizes if s == 1),
                "has_big_blob": max(sizes) >= BIG_BLOB_SIZE,
            })

    with (OUT_DIR / "grouping_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- [1] 전체(74개) 임계값별 요약 ----
    print("\n" + "=" * 70)
    print("[1] 전체(74개) 임계값별 그룹 수 / 최대 그룹 크기 분포")
    print("=" * 70)
    overall_summary = {}
    for t in THRESHOLDS:
        grp = [r for r in rows if r["threshold"] == t]
        n_groups_vals = [r["n_groups"] for r in grp]
        max_size_vals = [r["max_group_size"] for r in grp]
        big_blob_frac = np.mean([r["has_big_blob"] for r in grp])
        overall_summary[t] = {
            "n_groups_mean": float(np.mean(n_groups_vals)), "n_groups_min": int(min(n_groups_vals)),
            "n_groups_max": int(max(n_groups_vals)),
            "max_group_size_mean": float(np.mean(max_size_vals)), "max_group_size_max": int(max(max_size_vals)),
            "big_blob_frac": float(big_blob_frac),
        }
        print(f"  threshold={t}: n_groups mean={np.mean(n_groups_vals):.2f} "
              f"(min={min(n_groups_vals)}, max={max(n_groups_vals)})  "
              f"max_group_size mean={np.mean(max_size_vals):.1f} (max={max(max_size_vals)})  "
              f"20+채널 덩어리 비율={big_blob_frac*100:.1f}%")

    # ---- [2] gap 분위별 ----
    print("\n" + "=" * 70)
    print("[2] gap 분위별 임계값별 그룹 수 / 최대 그룹 크기")
    print("=" * 70)
    quartile_summary = {}
    for q in ["Q1_밀집", "Q2Q3_중간", "Q4_뚜렷"]:
        quartile_summary[q] = {}
        print(f"\n{q}:")
        for t in THRESHOLDS:
            grp = [r for r in rows if r["threshold"] == t and r["quartile"] == q]
            if not grp:
                continue
            n_groups_vals = [r["n_groups"] for r in grp]
            max_size_vals = [r["max_group_size"] for r in grp]
            big_blob_frac = np.mean([r["has_big_blob"] for r in grp])
            quartile_summary[q][t] = {
                "n": len(grp), "n_groups_mean": float(np.mean(n_groups_vals)),
                "max_group_size_mean": float(np.mean(max_size_vals)), "max_group_size_max": int(max(max_size_vals)),
                "big_blob_frac": float(big_blob_frac),
            }
            print(f"  threshold={t}: n={len(grp)}  n_groups mean={np.mean(n_groups_vals):.2f}  "
                  f"max_group_size mean={np.mean(max_size_vals):.1f} (max={max(max_size_vals)})  "
                  f"20+채널 덩어리 비율={big_blob_frac*100:.1f}%")

    # ---- [3] canmmlm_naive 규모(3~5그룹)와 비슷한 임계값 ----
    print("\n" + "=" * 70)
    print("[3] canmmlm_naive 규모(고정 4그룹)와 비슷한(3~5그룹) 임계값")
    print("=" * 70)
    for t in THRESHOLDS:
        frac_in_range = np.mean([3 <= r["n_groups"] <= 5 for r in rows if r["threshold"] == t])
        print(f"  threshold={t}: 3~5그룹인 세그먼트 비율={frac_in_range*100:.1f}%")

    # ---- [4] 임계값 민감도 ----
    print("\n" + "=" * 70)
    print("[4] 임계값 민감도 (세그먼트별 n_groups 변화)")
    print("=" * 70)
    by_seg = {}
    for r in rows:
        by_seg.setdefault(r["segment_id"], {})[r["threshold"]] = r["n_groups"]
    for t1, t2 in zip(THRESHOLDS, THRESHOLDS[1:]):
        deltas = [by_seg[s][t2] - by_seg[s][t1] for s in by_seg]
        print(f"  {t1}->{t2}: n_groups 변화 mean={np.mean(deltas):+.2f}  std={np.std(deltas):.2f}  "
              f"(min={min(deltas)}, max={max(deltas)})")

    summary = {
        "n_segments": len(seg_ids), "thresholds": THRESHOLDS,
        "overall": {str(t): v for t, v in overall_summary.items()},
        "by_quartile": {q: {str(t): v for t, v in d.items()} for q, d in quartile_summary.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
