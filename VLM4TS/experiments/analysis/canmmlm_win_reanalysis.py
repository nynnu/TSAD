"""
canmmlm_naive 승리 원인 -- 순수 재분석, VLM 재호출 없음.

주의: VLM 클라이언트를 만드는 모듈(smd_3way_baseline_comparison 등)은 일절
import하지 않음 -- 실수로도 API 호출이 나가지 않도록, prf/naive_groups를
이 파일 안에서 독립적으로 재구현하고, 세그먼트 목록도 체크포인트 JSON에서
직접 복원한다(원본 SEGMENTS 리스트를 가진 모듈도 import하지 않음).
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
OUT_DIR = BASE / "experiments" / "results_canmmlm_reanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
CLUTTER_RECOVERY_CSV = BASE / "experiments" / "results_clutter_check" / "channel_recovery.csv"
N_CHANNELS = 38
N_GROUPS = 4
Z_THRESHOLD = 3.0

SEG_ID_RE = re.compile(r"^(machine-\d+-\d+)_(\d+)_(\d+)$")


def naive_groups(n_channels: int, n_groups: int) -> list[list[int]]:
    base = list(range(n_channels))
    size = -(-n_channels // n_groups)  # ceil
    return [base[i:i + size] for i in range(0, n_channels, size)]


def prf(pred: set, true: set) -> tuple[float, float]:
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def group_of_channel(c: int, groups: list[list[int]]) -> tuple[int, int]:
    for gi, grp in enumerate(groups):
        if c in grp:
            return gi, len(grp)
    raise ValueError(c)


def load_all_segments():
    """체크포인트들의 {seg_id}_gt 항목에서 세그먼트 목록을 직접 복원."""
    segments = {}  # seg_id -> (entity, cs, ce, gt_channels_0idx set)
    ckpts = []
    for p in CKPT_PATHS:
        ck = json.loads(p.read_text(encoding="utf-8"))
        ckpts.append(ck)
        for key, val in ck.items():
            if key.endswith("_gt") and isinstance(val, dict) and "gt_channels" in val:
                seg_id = key[:-3]
                m = SEG_ID_RE.match(seg_id)
                if not m:
                    continue
                entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
                segments[seg_id] = (entity, cs, ce, set(val["gt_channels"]))
    return segments, ckpts


def find_ckpt_for(seg_id, ckpts):
    for ck in ckpts:
        if f"{seg_id}_A_vlm4ts_own" in ck:
            return ck
    return None


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
    segments, ckpts = load_all_segments()
    print(f"복원된 세그먼트 수: {len(segments)}")

    groups_layout = naive_groups(N_CHANNELS, N_GROUPS)
    print(f"canmmlm_naive 그룹 구성: {[(gi, len(g)) for gi, g in enumerate(groups_layout)]}")

    entity_data = {}
    recovered_rows = []  # B만 잡은 채널 (A, C 둘 다 놓침)
    fp_rows = []  # 세그먼트 단위 FP vs 그룹 크기 정보

    for seg_id, (entity, cs, ce, gt_channels) in segments.items():
        ck = find_ckpt_for(seg_id, ckpts)
        if ck is None:
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        c_entry = ck.get(f"{seg_id}_C_our_design", {})
        if a_entry.get("status") != "OK" or c_entry.get("status") != "OK":
            continue
        a_pred = set(a_entry["pred"] or [])
        c_pred = set(c_entry["pred"] or [])

        b_pred_by_group, b_ok = [], True
        for gi in range(N_GROUPS):
            ge = ck.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") != "OK":
                b_ok = False
                break
            b_pred_by_group.append(set(ge["pred"] or []))
        if not b_ok:
            continue
        b_pred_union = set().union(*b_pred_by_group)

        # --- (1)(2) A/C 둘 다 놓쳤는데 B가 잡은 채널, 그룹 크기 포함 ---
        b_only = (gt_channels & b_pred_union) - a_pred - c_pred
        if b_only:
            if entity not in entity_data:
                entity_data[entity] = load_smd_train_test(entity)
            train, test = entity_data[entity]
            for c in b_only:
                gi, gsize = group_of_channel(c, groups_layout)
                mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
                mz = max_abs_z(test, cs, ce, c, mu, sigma)
                recovered_rows.append({
                    "segment_id": seg_id, "entity": entity, "channel": c,
                    "source_group": gi, "group_size": gsize, "max_abs_z": round(mz, 3),
                })

        # --- (4) FP vs 그룹 크기 ---
        fp_a = len(a_pred - gt_channels)
        fp_c = len(c_pred - gt_channels)
        # B의 FP를 그룹별로 분해 (그룹 크기별 FP 기여도)
        for gi, grp_pred in enumerate(b_pred_by_group):
            grp_fp = len(grp_pred - gt_channels)
            fp_rows.append({"segment_id": seg_id, "group": gi, "group_size": len(groups_layout[gi]), "group_fp": grp_fp})

    # ---- 저장 ----
    if recovered_rows:
        with (OUT_DIR / "b_only_recovered_channels.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recovered_rows[0].keys()))
            w.writeheader()
            w.writerows(recovered_rows)
    with (OUT_DIR / "fp_by_group_size.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
        w.writeheader()
        w.writerows(fp_rows)

    print(f"\nA/C 둘 다 놓쳤는데 canmmlm_naive(B)만 잡은 채널: {len(recovered_rows)}개")
    if recovered_rows:
        group_sizes = [r["group_size"] for r in recovered_rows]
        print(f"  그룹 크기 분포: {sorted(set(group_sizes))}  (전부 {N_CHANNELS}/{N_GROUPS}=9~10 근방, 원래 화면 38개보다 훨씬 작음)")
        mzs = [r["max_abs_z"] for r in recovered_rows]
        print(f"  신호 강도(max|z|) 평균={np.mean(mzs):.2f}  중앙값={np.median(mzs):.2f}")

    # ---- (3) 화면혼잡 실험(results_clutter_check)과 분포 비교 ----
    print("\n" + "=" * 70)
    print("(3) 화면혼잡 실험의 '회복된 채널'과 이번 'B만 잡은 채널' 특징 비교")
    print("=" * 70)
    clutter_comparison = None
    if CLUTTER_RECOVERY_CSV.exists():
        crows = list(csv.DictReader(CLUTTER_RECOVERY_CSV.open(encoding="utf-8")))
        clutter_recovered = [r for r in crows if r["recovered_in_B"] == "True"]
        print(f"화면혼잡 실험에서 회복된 채널: {len(clutter_recovered)}개 (참고: 그때는 top-k로 축소, 이번은 naive 그룹으로 축소)")
        # 화면혼잡 실험 쪽 max|z|는 results_fn_root_cause/fn_tp_detail.csv에 있음(point_ratio 계산 시 이미 구함) -- 여기선 세그먼트/채널만 재사용
        fn_detail_path = BASE / "experiments" / "results_fn_root_cause" / "fn_tp_detail.csv"
        clutter_mz = []
        if fn_detail_path.exists():
            fn_rows = {(r["segment_id"], int(r["channel"])): r for r in csv.DictReader(fn_detail_path.open(encoding="utf-8"))}
            for r in clutter_recovered:
                key = (r["segment_id"], int(r["channel"]))
                if key in fn_rows:
                    # fn_tp_detail.csv doesn't store max|z| directly (point_ratio uses same threshold) -- recompute directly
                    pass
        # 직접 재계산 (로컬, mv_data만 사용)
        for r in clutter_recovered:
            seg_id, c = r["segment_id"], int(r["channel"])
            if seg_id not in segments:
                continue
            entity, cs, ce, _ = segments[seg_id]
            if entity not in entity_data:
                entity_data[entity] = load_smd_train_test(entity)
            train, test = entity_data[entity]
            mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
            clutter_mz.append(max_abs_z(test, cs, ce, c, mu, sigma))

        if recovered_rows and clutter_mz:
            b_only_mz = [r["max_abs_z"] for r in recovered_rows]
            print(f"  화면혼잡 실험 회복 채널 max|z|: 평균={np.mean(clutter_mz):.2f} 중앙값={np.median(clutter_mz):.2f} (n={len(clutter_mz)})")
            print(f"  canmmlm_naive만 잡은 채널 max|z|: 평균={np.mean(b_only_mz):.2f} 중앙값={np.median(b_only_mz):.2f} (n={len(b_only_mz)})")
            clutter_comparison = {
                "clutter_recovered_max_abs_z_mean": float(np.mean(clutter_mz)),
                "clutter_recovered_max_abs_z_median": float(np.median(clutter_mz)),
                "clutter_recovered_n": len(clutter_mz),
                "canmmlm_b_only_max_abs_z_mean": float(np.mean(b_only_mz)),
                "canmmlm_b_only_max_abs_z_median": float(np.median(b_only_mz)),
                "canmmlm_b_only_n": len(b_only_mz),
            }

    # ---- (4) FP vs 그룹 크기 ----
    print("\n" + "=" * 70)
    print("(4) FP 개수 vs 그룹 크기")
    print("=" * 70)
    by_size = {}
    for r in fp_rows:
        by_size.setdefault(r["group_size"], []).append(r["group_fp"])
    for size, fps in sorted(by_size.items()):
        print(f"  그룹 크기 {size}: 평균 FP={np.mean(fps):.3f} (n={len(fps)})")

    mean_fp_a_all, mean_fp_c_all = [], []
    for seg_id, (entity, cs, ce, gt_channels) in segments.items():
        ck = find_ckpt_for(seg_id, ckpts)
        if ck is None:
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        c_entry = ck.get(f"{seg_id}_C_our_design", {})
        if a_entry.get("status") == "OK":
            mean_fp_a_all.append(len(set(a_entry["pred"] or []) - gt_channels))
        if c_entry.get("status") == "OK":
            mean_fp_c_all.append(len(set(c_entry["pred"] or []) - gt_channels))
    mean_fp_b_per_seg = {}
    for r in fp_rows:
        mean_fp_b_per_seg.setdefault(r["segment_id"], 0)
        mean_fp_b_per_seg[r["segment_id"]] += r["group_fp"]
    print(f"\n  세그먼트 평균 FP: vlm4ts_own(38채널,1장)={np.mean(mean_fp_a_all):.2f}  "
          f"our_design(4채널,1장)={np.mean(mean_fp_c_all):.2f}  "
          f"canmmlm_naive(9~10채널x4장 합산)={np.mean(list(mean_fp_b_per_seg.values())):.2f}")

    # ---- 최종 결론 ----
    same_mechanism = None
    if clutter_comparison:
        cz = clutter_comparison["clutter_recovered_max_abs_z_median"]
        bz = clutter_comparison["canmmlm_b_only_max_abs_z_median"]
        # 같은 메커니즘이면 두 중앙값이 크게 다르지 않아야 함 (같은 종류의 채널을 살리는 것)
        same_mechanism = abs(cz - bz) < 0.5 * max(cz, bz, 1e-9)

    summary = {
        "n_segments": len(segments),
        "n_b_only_recovered_channels": len(recovered_rows),
        "group_sizes_of_recovered": sorted(set(r["group_size"] for r in recovered_rows)) if recovered_rows else [],
        "fp_by_group_size": {str(k): float(np.mean(v)) for k, v in by_size.items()},
        "mean_fp_per_segment": {
            "vlm4ts_own_38ch_1image": float(np.mean(mean_fp_a_all)),
            "our_design_4ch_1image": float(np.mean(mean_fp_c_all)),
            "canmmlm_naive_9to10ch_4images_summed": float(np.mean(list(mean_fp_b_per_seg.values()))),
        },
        "clutter_vs_canmmlm_feature_comparison": clutter_comparison,
        "same_mechanism_as_clutter_reduction": same_mechanism,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
