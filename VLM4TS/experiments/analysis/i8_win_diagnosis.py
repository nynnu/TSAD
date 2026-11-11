"""
I8(heatmap+overlay, N=8) 승리 원인 채널 단위 진단. VLM 재호출 없음, 기존 체크포인트/
로컬 SMD 데이터만 사용.

파트1: A/B 둘 다 놓쳤는데 I8만 잡은 채널 -- overlay(top-8)에 포함됐었는지 heatmap
전용 구간(9~38위)에서 나왔는지, 신호 강도(max|z|)가 화면혼잡 회복 패턴(중앙값 54.85)에
가까운지 canmmlm_naive 패턴(중앙값 17.79)에 가까운지 비교. raw VLM 응답은 순수 JSON이라
"어느 패널을 언급했는지" 텍스트 근거는 없음 -- top-8 포함 여부로 대체(작업 지시대로).
FP 비교 샘플도 포함.
"""

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
OUT_DIR = BASE / "experiments" / "results_i8_win_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
I8_CKPT = BASE / "experiments" / "results_heatmap_overlay_74" / "checkpoint_I8.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
N_TOP = 8


def load_merged_ckpts():
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    merged = {}
    for ck in ckpts:
        merged.update(ck)
    return merged, ckpts


def find_ckpt_for(seg_id, ckpts):
    for ck in ckpts:
        if f"{seg_id}_A_vlm4ts_own" in ck:
            return ck
    return None


def get_74_segments(merged):
    return sorted({k[:-3] for k in merged if k.endswith("_gt")})


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def load_smd_train_test(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def max_abs_z(test, cs, ce, c, mu, sigma):
    if sigma < 1e-9:
        return 0.0
    seg = test[cs:ce, c]
    return float(np.max(np.abs(seg - mu)) / sigma)


def run():
    merged, ckpts = load_merged_ckpts()
    seg_ids = get_74_segments(merged)
    i8_ckpt = json.loads(I8_CKPT.read_text(encoding="utf-8"))
    score_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))

    entity_data = {}
    i8_only_rows = []
    fp_sample_rows = []

    for seg_id in seg_ids:
        ck = find_ckpt_for(seg_id, ckpts)
        if ck is None:
            continue
        gt = set(ck[f"{seg_id}_gt"]["gt_channels"])
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        i8_entry = i8_ckpt.get(seg_id, {})
        if a_entry.get("status") != "OK" or i8_entry.get("status") != "OK":
            continue
        b_pred, b_ok = set(), True
        for gi in range(4):
            ge = ck.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") == "OK" and ge.get("pred"):
                b_pred |= set(ge["pred"])
            if ge.get("status") != "OK":
                b_ok = False
        if not b_ok:
            continue

        a_pred = set(a_entry["pred"] or [])
        i8_pred = set(i8_entry["pred"] or [])
        entity, cs, ce = seg_id.rsplit("_", 2)[0], int(seg_id.rsplit("_", 2)[1]), int(seg_id.rsplit("_", 2)[2])

        ranked = get_ranked_channels(seg_id, score_cache)
        top8 = set(ranked[:N_TOP])

        # ---- 파트1.1/1.2/1.3: A/B 둘 다 놓쳤는데 I8만 잡은 GT 채널 ----
        i8_only = (gt & i8_pred) - a_pred - b_pred
        if i8_only:
            if entity not in entity_data:
                entity_data[entity] = load_smd_train_test(entity)
            train, test = entity_data[entity]
            for c in i8_only:
                mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
                mz = max_abs_z(test, cs, ce, c, mu, sigma)
                i8_only_rows.append({
                    "segment_id": seg_id, "entity": entity, "channel": c,
                    "in_top8_overlay": c in top8,
                    "step0_rank": ranked.index(c) + 1,
                    "max_abs_z": round(mz, 3),
                })

        # ---- 파트1.4: I8은 FP 안 냈는데 A/B는 낸 케이스 샘플 ----
        i8_fp = i8_pred - gt
        a_fp = a_pred - gt
        b_fp = b_pred - gt
        only_a_fp = a_fp - i8_fp - b_fp  # A만 잘못 찍은 채널(I8, B는 안 찍음)
        only_b_fp = b_fp - i8_fp - a_fp
        if only_a_fp or only_b_fp:
            if entity not in entity_data:
                entity_data[entity] = load_smd_train_test(entity)
            train, test = entity_data[entity]
            for c in only_a_fp:
                mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
                fp_sample_rows.append({"segment_id": seg_id, "channel": c, "false_positive_by": "A_only",
                                        "max_abs_z": round(max_abs_z(test, cs, ce, c, mu, sigma), 3)})
            for c in only_b_fp:
                mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
                fp_sample_rows.append({"segment_id": seg_id, "channel": c, "false_positive_by": "B_only",
                                        "max_abs_z": round(max_abs_z(test, cs, ce, c, mu, sigma), 3)})

    # ---- 저장 ----
    if i8_only_rows:
        with (OUT_DIR / "i8_only_recovered_channels.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(i8_only_rows[0].keys()))
            w.writeheader()
            w.writerows(i8_only_rows)
    if fp_sample_rows:
        with (OUT_DIR / "fp_only_by_A_or_B_sample.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fp_sample_rows[0].keys()))
            w.writeheader()
            w.writerows(fp_sample_rows)

    print("=" * 70)
    print(f"A/B 둘 다 놓쳤는데 I8만 잡은 GT 채널: {len(i8_only_rows)}개")
    print("=" * 70)
    if i8_only_rows:
        n_top8 = sum(1 for r in i8_only_rows if r["in_top8_overlay"])
        n_heatmap_only = len(i8_only_rows) - n_top8
        print(f"  overlay(top-8)에 포함됨: {n_top8}개 ({n_top8/len(i8_only_rows)*100:.1f}%)")
        print(f"  heatmap 전용 구간(9~38위)에서만 잡힘: {n_heatmap_only}개 ({n_heatmap_only/len(i8_only_rows)*100:.1f}%)")

        mzs_all = [r["max_abs_z"] for r in i8_only_rows]
        mzs_top8 = [r["max_abs_z"] for r in i8_only_rows if r["in_top8_overlay"]]
        mzs_heatmaponly = [r["max_abs_z"] for r in i8_only_rows if not r["in_top8_overlay"]]
        print(f"\n  신호 강도(max|z|): 전체 평균={np.mean(mzs_all):.2f} 중앙값={np.median(mzs_all):.2f} (n={len(mzs_all)})")
        if mzs_top8:
            print(f"    overlay(top-8) 포함분: 평균={np.mean(mzs_top8):.2f} 중앙값={np.median(mzs_top8):.2f} (n={len(mzs_top8)})")
        if mzs_heatmaponly:
            print(f"    heatmap 전용분:        평균={np.mean(mzs_heatmaponly):.2f} 중앙값={np.median(mzs_heatmaponly):.2f} (n={len(mzs_heatmaponly)})")
        print(f"\n  참고: 화면혼잡 회복 채널 중앙값=54.85, canmmlm_naive 회복 채널 중앙값=17.79")

    print("\n" + "=" * 70)
    print(f"I8은 FP 안 냈는데 A 또는 B만 낸 채널 샘플: {len(fp_sample_rows)}개")
    print("=" * 70)
    a_only = [r for r in fp_sample_rows if r["false_positive_by"] == "A_only"]
    b_only = [r for r in fp_sample_rows if r["false_positive_by"] == "B_only"]
    if a_only:
        print(f"  A만 오탐: n={len(a_only)}  max|z| 평균={np.mean([r['max_abs_z'] for r in a_only]):.2f} "
              f"중앙값={np.median([r['max_abs_z'] for r in a_only]):.2f}")
    if b_only:
        print(f"  B만 오탐: n={len(b_only)}  max|z| 평균={np.mean([r['max_abs_z'] for r in b_only]):.2f} "
              f"중앙값={np.median([r['max_abs_z'] for r in b_only]):.2f}")
    print("\n  (max|z|가 작을수록 '신호 거의 없는데 찍음'에 해당)")

    summary = {
        "n_i8_only_recovered": len(i8_only_rows),
        "n_in_top8_overlay": sum(1 for r in i8_only_rows if r["in_top8_overlay"]) if i8_only_rows else 0,
        "n_heatmap_only": sum(1 for r in i8_only_rows if not r["in_top8_overlay"]) if i8_only_rows else 0,
        "max_abs_z_all": {"mean": float(np.mean([r["max_abs_z"] for r in i8_only_rows])) if i8_only_rows else None,
                           "median": float(np.median([r["max_abs_z"] for r in i8_only_rows])) if i8_only_rows else None},
        "max_abs_z_top8_subset": {"mean": float(np.mean(mzs_top8)) if i8_only_rows and mzs_top8 else None,
                                   "median": float(np.median(mzs_top8)) if i8_only_rows and mzs_top8 else None},
        "max_abs_z_heatmaponly_subset": {"mean": float(np.mean(mzs_heatmaponly)) if i8_only_rows and mzs_heatmaponly else None,
                                          "median": float(np.median(mzs_heatmaponly)) if i8_only_rows and mzs_heatmaponly else None},
        "reference_clutter_recovered_median": 54.853732288399534,
        "reference_canmmlm_b_only_median": 17.791,
        "fp_a_only": {"n": len(a_only), "median": float(np.median([r["max_abs_z"] for r in a_only])) if a_only else None},
        "fp_b_only": {"n": len(b_only), "median": float(np.median([r["max_abs_z"] for r in b_only])) if b_only else None},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
