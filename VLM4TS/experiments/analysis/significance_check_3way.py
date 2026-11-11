"""
3-way 비교(18개 세그먼트, results_3way_broad)의 통계적 유의성 확인.
새 실험/VLM 호출 없음 -- 기존 원자료 재분석만.
"""

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from smd_3way_baseline_comparison import BASE, N_GROUPS, prf  # noqa: E402
from reverify_3way_broad_sample import NEW_SEGMENTS, ORIG_SEGMENTS  # noqa: E402

OUT_DIR = BASE / "experiments" / "results_significance_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG_CKPT_PATH = BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json"
NEW_CKPT_PATH = BASE / "experiments" / "results_3way_broad" / "checkpoint.json"

N_BOOT = 1000
RNG = np.random.default_rng(0)


def segment_f1s(segments, ckpt):
    """Returns list of (seg_id, f1_A, f1_B, f1_C) for segments fully scored in ckpt."""
    rows = []
    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        a_entry = ckpt.get(f"{seg_id}_A_vlm4ts_own", {})
        if a_entry.get("status") != "OK":
            continue
        p, r = prf(set(a_entry["pred"] or []), gt_channels)
        f1_a = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        b_pred, b_ok = set(), True
        for gi in range(N_GROUPS):
            ge = ckpt.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") == "OK" and ge.get("pred"):
                b_pred |= set(ge["pred"])
            if ge.get("status") != "OK":
                b_ok = False
        if not b_ok:
            continue
        p, r = prf(b_pred, gt_channels)
        f1_b = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        c_entry = ckpt.get(f"{seg_id}_C_our_design", {})
        if c_entry.get("status") != "OK":
            continue
        p, r = prf(set(c_entry["pred"] or []), gt_channels)
        f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        rows.append((seg_id, f1_a, f1_b, f1_c))
    return rows


def bootstrap_ci(values: np.ndarray, n_boot=N_BOOT):
    n = len(values)
    boot_means = np.array([RNG.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.mean(values)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT):
    """Paired (same segment index resampled together) bootstrap of mean(a)-mean(b)."""
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    obs_diff = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    includes_zero = lo <= 0 <= hi
    return obs_diff, lo, hi, includes_zero


def run():
    orig_ckpt = json.loads(ORIG_CKPT_PATH.read_text(encoding="utf-8"))
    new_ckpt = json.loads(NEW_CKPT_PATH.read_text(encoding="utf-8"))

    rows = segment_f1s(ORIG_SEGMENTS, orig_ckpt) + segment_f1s(NEW_SEGMENTS, new_ckpt)
    print(f"n segments (fully scored, all 3 methods) = {len(rows)}")
    for seg_id, fa, fb, fc in rows:
        print(f"  {seg_id}: A={fa:.3f} B={fb:.3f} C={fc:.3f}")

    f1_a = np.array([r[1] for r in rows])
    f1_b = np.array([r[2] for r in rows])
    f1_c = np.array([r[3] for r in rows])
    n = len(rows)

    # ---- 1) 방법별 단독 95% CI ----
    print("\n" + "=" * 70)
    print("방법별 단독 F1 95% CI (부트스트랩)")
    print("=" * 70)
    ci_results = {}
    for name, vals in [("vlm4ts_own", f1_a), ("canmmlm_naive", f1_b), ("our_design", f1_c)]:
        mean, lo, hi = bootstrap_ci(vals)
        ci_results[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        print(f"  {name}: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    # ---- 2) 쌍별 페어드 부트스트랩 차이 ----
    print("\n" + "=" * 70)
    print("방법 쌍별 F1 차이, 95% CI (페어드 부트스트랩, 세그먼트 단위 리샘플)")
    print("=" * 70)
    pair_results = {}
    for name1, v1, name2, v2 in [
        ("vlm4ts_own", f1_a, "our_design", f1_c),
        ("vlm4ts_own", f1_a, "canmmlm_naive", f1_b),
        ("our_design", f1_c, "canmmlm_naive", f1_b),
    ]:
        obs, lo, hi, incl0 = paired_bootstrap_diff(v1, v2)
        key = f"{name1}_minus_{name2}"
        pair_results[key] = {"observed_diff": obs, "ci_lo": lo, "ci_hi": hi, "ci_includes_zero": incl0}
        sig = "유의미하지 않음(0 포함)" if incl0 else "유의미함(0 안 포함)"
        print(f"  {name1} - {name2}: 관측차이={obs:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  -> {sig}")

    # ---- 3) 러프 파워 계산 ----
    print("\n" + "=" * 70)
    print("러프 파워 계산: 0.05 차이를 95% 신뢰도로 검출하려면 세그먼트 몇 개?")
    print("=" * 70)
    power_results = {}
    for name1, v1, name2, v2 in [
        ("vlm4ts_own", f1_a, "our_design", f1_c),
        ("vlm4ts_own", f1_a, "canmmlm_naive", f1_b),
        ("our_design", f1_c, "canmmlm_naive", f1_b),
    ]:
        diff_per_seg = v1 - v2
        sigma = float(np.std(diff_per_seg, ddof=1))
        # rough: n needed for 95% CI half-width <= 0.05 on the mean difference,
        # using normal approx: half_width = 1.96 * sigma / sqrt(n) <= target
        target = 0.05
        if sigma > 0:
            n_needed = int(np.ceil((1.96 * sigma / target) ** 2))
        else:
            n_needed = None
        key = f"{name1}_vs_{name2}"
        power_results[key] = {"sigma_of_paired_diff": sigma, "n_needed_for_0.05_precision": n_needed,
                               "current_n": n}
        print(f"  {name1} vs {name2}: 세그먼트별 차이 표준편차={sigma:.3f}  "
              f"0.05 정밀도(95%CI 반폭)에 필요한 대략 n={n_needed} (현재 n={n})")

    # ---- 저장 ----
    summary = {
        "n_segments": n,
        "per_method_ci": ci_results,
        "pairwise_diff_ci": pair_results,
        "rough_power": power_results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    with (OUT_DIR / "segment_f1_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "f1_vlm4ts_own", "f1_canmmlm_naive", "f1_our_design"])
        for seg_id, fa, fb, fc in rows:
            w.writerow([seg_id, fa, fb, fc])

    n_significant = sum(1 for v in pair_results.values() if not v["ci_includes_zero"])
    print(f"\n{'='*70}\n최종 결론\n{'='*70}")
    print(f"3개 쌍 중 통계적으로 유의미한(0을 포함 안 하는) 차이: {n_significant}개")
    if n_significant == 0:
        print("-> 현재 n=18로는 세 방법 중 어느 것도 다른 것보다 낫다고 통계적으로 주장할 수 없음.")

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
