"""
진짜 held-out 26개 세그먼트(11개 entity, entity당 상한 3개 층화추출)에 I8/K2를
파라미터 변경 없이 재적용 -- 74개(I8) / 18개 파일럿(K2) 결과가 과적합인지 확인.

26개는 74개 선정 시 이미 후보였지만(GT 채널 4~8개 조건 충족) 뽑히지 않은 세그먼트들
중에서 추출됨. 걸친 11개 entity는 전부 74개 세트가 이미 사용한 entity(새 entity 아님)
-- 이 실험이 검증하는 건 "미지 entity 일반화"가 아니라 "같은 entity 안에서 우연히
유리한 세그먼트만 골랐는지"임.

Step0 랭킹은 full_smd_3way.py의 step0_rank_original_mechanism 그대로 재사용(로컬, 콜 없음).
I8은 pilot_heatmap_overlay_nsweep.py의 render_heatmap_overlay(n=8)/build_prompt_heatmap_overlay,
K2는 pilot_reduced_index_on_i8.py의 build_prompt_reduced_index(n_points=25) + 위 heatmap 렌더.
call_vlm/parse_response는 smd_3way_baseline_comparison.py 재사용. 파라미터 전부 고정.

비교는 "같은 세그먼트, 다른 조건"이 아니라 "다른 세그먼트 묶음"이므로 반드시
two-sample bootstrap(독립표본) 사용 -- paired/cluster 부트스트랩(조건 내 대응비교용) 금지.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from step1v3_dino_graph_smd import N_CHANNELS, WIN, _centered_window, load_smd  # noqa: E402
from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response  # noqa: E402
from full_smd_3way import step0_rank_original_mechanism  # noqa: E402
from pilot_heatmap_overlay_nsweep import render_heatmap_overlay, build_prompt_heatmap_overlay  # noqa: E402
from pilot_reduced_index_on_i8 import build_prompt_reduced_index, N_POINTS_PER_CHANNEL  # noqa: E402

OUT_DIR = BASE / "experiments" / "results_valtest_holdout26"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

I8_74_CSV = BASE / "experiments" / "results_heatmap_overlay_74" / "comparison_I8.csv"
K2_18_SUMMARY = BASE / "experiments" / "results_pilot_reduced_index" / "pilot_summary_K2.json"

N = 8
N_BOOT = 1000
SEED = 42
ALL_CHANNELS = list(range(N_CHANNELS))

# 정확히 이 26개만. GT dims는 1-indexed(원본 표기 그대로), 0-indexed 변환은 아래에서.
SEGMENTS = [
    ("machine-1-6", (5708, 5713), [9, 10, 11, 13]),
    ("machine-1-6", (16491, 16499), [1, 2, 3, 9, 10, 11, 13, 15]),
    ("machine-1-6", (16972, 16976), [9, 12, 13, 15]),
    ("machine-1-7", (15239, 15270), [1, 6, 9, 12, 13, 14, 15, 16]),
    ("machine-1-8", (2725, 2728), [9, 10, 13, 14]),
    ("machine-1-8", (7100, 7104), [6, 7, 11, 16]),
    ("machine-1-8", (8657, 8660), [10, 11, 13, 16]),
    ("machine-2-2", (15630, 16502), [10, 11, 12, 13]),
    ("machine-2-2", (17090, 17942), [10, 11, 12, 13]),
    ("machine-2-4", (5099, 5380), [1, 10, 12, 13, 14]),
    ("machine-2-4", (17965, 18350), [1, 9, 10, 12, 13, 14]),
    ("machine-2-5", (13616, 13619), [9, 13, 14, 15]),
    ("machine-2-5", (20846, 20854), [9, 13, 14, 15]),
    ("machine-2-5", (22283, 22294), [9, 13, 14, 15]),
    ("machine-2-7", (4984, 4987), [9, 13, 14, 15]),
    ("machine-2-7", (12099, 12127), [1, 2, 3, 4, 24, 26]),
    ("machine-2-7", (15063, 15066), [9, 13, 14, 15]),
    ("machine-2-9", (20615, 20845), [9, 10, 11, 12, 13, 14, 15, 16]),
    ("machine-3-10", (2155, 2228), [9, 12, 13, 14, 15]),
    ("machine-3-10", (6486, 6498), [9, 12, 13, 14]),
    ("machine-3-10", (7891, 7901), [9, 12, 13, 14]),
    ("machine-3-3", (6499, 6502), [10, 11, 12, 13]),
    ("machine-3-3", (12259, 12262), [10, 11, 12, 13]),
    ("machine-3-3", (18019, 18022), [10, 11, 12, 13]),
    ("machine-3-6", (8177, 8211), [1, 2, 3, 4, 34, 33]),
    ("machine-3-6", (18054, 18134), [1, 10, 12, 13, 14, 15]),
]

assert len(SEGMENTS) == 26, f"expected 26 segments, got {len(SEGMENTS)}"


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prf(pred: set, true: set):
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def f1_of(pred, gt):
    p, r = prf(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def two_sample_bootstrap_diff(sample_new, sample_old, n_boot=N_BOOT, seed=SEED):
    """독립표본(다른 세그먼트 묶음) 평균차 부트스트랩: 각 표본을 자신의 크기로 복원추출."""
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_new, dtype=float)
    b = np.asarray(sample_old, dtype=float)
    na, nb = len(a), len(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=na, replace=True)
        rb = rng.choice(b, size=nb, replace=True)
        diffs[i] = ra.mean() - rb.mean()
    obs = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs, lo, hi


def run():
    checkpoint = load_checkpoint()
    entity_test, entity_train = {}, {}

    n_calls_expected = len(SEGMENTS) * 2
    print(f"대상 세그먼트: {len(SEGMENTS)}개, 조건 I8/K2 x 1콜씩 = {n_calls_expected}콜 예상\n")

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)
        checkpoint[f"{seg_id}_gt"] = {"gt_channels": sorted(gt_channels)}

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        train = entity_train[entity]

        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        step0_key = f"{seg_id}_step0_ranked"
        if step0_key not in checkpoint:
            print(f"  [Step0] {seg_id} 랭킹 계산 중 ...", flush=True)
            ranked = step0_rank_original_mechanism(train, test, cs, ce)
            checkpoint[step0_key] = ranked
            save_checkpoint(checkpoint)
        ranked = checkpoint[step0_key]
        top8 = ranked[:N]

        # ---- I8 ----
        case_i8 = f"{seg_id}_I8"
        if checkpoint.get(case_i8, {}).get("status") != "OK":
            img = render_heatmap_overlay(window, ranked, N)
            prompt = build_prompt_heatmap_overlay(ALL_CHANNELS, top8)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_i8] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_i8]['status']}] {case_i8}: pred={pred}", flush=True)

        # ---- K2 ----
        case_k2 = f"{seg_id}_K2"
        if checkpoint.get(case_k2, {}).get("status") != "OK":
            img = render_heatmap_overlay(window, ranked, N)
            prompt = build_prompt_reduced_index(ALL_CHANNELS, top8, window, train, N_POINTS_PER_CHANNEL)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_k2] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_k2]['status']}] {case_k2}: pred={pred}", flush=True)

    # ---- 세그먼트별 결과 수집 ----
    rows = []
    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(checkpoint[f"{seg_id}_gt"]["gt_channels"])
        i8_entry = checkpoint.get(f"{seg_id}_I8", {})
        k2_entry = checkpoint.get(f"{seg_id}_K2", {})
        if i8_entry.get("status") != "OK" or k2_entry.get("status") != "OK":
            print(f"[WARN] {seg_id} 미완료, 스킵")
            continue
        i8_pred = set(i8_entry["pred"] or [])
        k2_pred = set(k2_entry["pred"] or [])
        rows.append({
            "segment_id": seg_id, "entity": entity, "n_gt_dims": len(gt_dims),
            "f1_I8": f1_of(i8_pred, gt_channels), "f1_K2": f1_of(k2_pred, gt_channels),
            "fp_I8": len(i8_pred - gt_channels), "fp_K2": len(k2_pred - gt_channels),
            "n_pred_I8": len(i8_pred), "n_pred_K2": len(k2_pred),
        })

    with (OUT_DIR / "segment_results_holdout26.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    f1_i8_new = np.array([r["f1_I8"] for r in rows])
    f1_k2_new = np.array([r["f1_K2"] for r in rows])
    print("\n" + "=" * 70)
    print(f"held-out 26개(완료 {n}개) mean F1: I8={f1_i8_new.mean():.4f}  K2={f1_k2_new.mean():.4f}")
    print("=" * 70)

    # ---- I8: 74개와 two-sample bootstrap 비교 ----
    old_i8_rows = list(csv.DictReader(I8_74_CSV.open(encoding="utf-8")))
    f1_i8_old = np.array([float(r["f1_I8"]) for r in old_i8_rows])
    obs_i8, lo_i8, hi_i8 = two_sample_bootstrap_diff(f1_i8_new, f1_i8_old, seed=SEED)
    i8_sig = not (lo_i8 <= 0 <= hi_i8)
    print(f"\n[I8] 새 26개(mean={f1_i8_new.mean():.4f}) vs 기존 74개(mean={f1_i8_old.mean():.4f}, n={len(f1_i8_old)})")
    print(f"  two-sample bootstrap diff(new-old) = {obs_i8:+.4f}  95% CI=[{lo_i8:+.4f}, {hi_i8:+.4f}]  "
          f"{'유의미한 차이' if i8_sig else '유의한 차이 없음'}")

    # ---- K2: 18개 파일럿과 two-sample bootstrap 비교 ----
    k2_18_summary = json.loads(K2_18_SUMMARY.read_text(encoding="utf-8"))
    k2_18_csv = BASE / "experiments" / "results_pilot_reduced_index" / "pilot_comparison_K2.csv"
    old_k2_rows = list(csv.DictReader(k2_18_csv.open(encoding="utf-8")))
    f1_k2_old = np.array([float(r["f1_K2"]) for r in old_k2_rows])
    obs_k2, lo_k2, hi_k2 = two_sample_bootstrap_diff(f1_k2_new, f1_k2_old, seed=SEED)
    k2_sig = not (lo_k2 <= 0 <= hi_k2)
    print(f"\n[K2] 새 26개(mean={f1_k2_new.mean():.4f}) vs 기존 파일럿 18개(mean={f1_k2_old.mean():.4f}, n={len(f1_k2_old)})")
    print(f"  ** 표본 크기 작음(18) -- 통계적 검정력 약함, 참고용 **")
    print(f"  two-sample bootstrap diff(new-old) = {obs_k2:+.4f}  95% CI=[{lo_k2:+.4f}, {hi_k2:+.4f}]  "
          f"{'유의미한 차이' if k2_sig else '유의한 차이 없음'}")

    # ---- entity 참고 ----
    entities_26 = sorted(set(r["entity"] for r in rows))
    print(f"\n26개가 걸친 entity({len(entities_26)}개, 전부 74개 세트가 이미 사용): {entities_26}")

    verdict_i8 = ("과적합 아님(유의한 차이 없음)" if not i8_sig else
                  ("과적합 가능성(새 표본이 유의미하게 낮음)" if obs_i8 < 0 else "새 표본이 유의미하게 높음(원인 불명, 사실만 보고)"))
    verdict_k2 = ("과적합 아님(유의한 차이 없음, 단 검정력 약함)" if not k2_sig else
                  ("과적합 가능성(새 표본이 유의미하게 낮음, 단 검정력 약함)" if obs_k2 < 0 else "새 표본이 유의미하게 높음(검정력 약함)"))

    summary = {
        "n_segments": n,
        "entities_covered": entities_26,
        "mean_f1_new26": {"I8": float(f1_i8_new.mean()), "K2": float(f1_k2_new.mean())},
        "I8_vs_74": {"mean_old": float(f1_i8_old.mean()), "n_old": len(f1_i8_old),
                     "obs_diff": obs_i8, "ci_lo": lo_i8, "ci_hi": hi_i8, "significant": i8_sig, "verdict": verdict_i8},
        "K2_vs_pilot18": {"mean_old": float(f1_k2_old.mean()), "n_old": len(f1_k2_old),
                          "obs_diff": obs_k2, "ci_lo": lo_k2, "ci_hi": hi_k2, "significant": k2_sig, "verdict": verdict_k2,
                          "note": "n_old=18 -> 통계적 검정력 약함"},
    }
    (OUT_DIR / "summary_holdout26.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\n판정:\n  I8: {verdict_i8}\n  K2: {verdict_k2}")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
