"""
canmmlm_naive 승리 원인 분해 (1단계, 재호출 없음) + 콜 수 통제 재비교 (2단계, VLM 호출).

74개 세그먼트 = 기존 18(results_smd_3way_baseline 9개 + results_3way_broad 9개)
+ 신규 56(results_full_smd_3way).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\analysis")
sys.path.insert(0, str(ANALYSIS_DIR))
from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)
from smd_3way_baseline_comparison import (  # noqa: E402
    BASE, N_GROUPS, build_prompt, call_vlm, load_smd_test, naive_groups, parse_response, prf, render_overlay,
)
from reverify_3way_broad_sample import NEW_SEGMENTS as SAMPLE18_NEW, ORIG_SEGMENTS as SAMPLE18_ORIG  # noqa: E402
from full_smd_3way import select_new_56_segments  # noqa: E402

OUT_DIR = BASE / "experiments" / "results_canmmlm_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

CKPT_PATHS = {
    "orig9": BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    "new9": BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    "new56": BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
}
N_BOOT = 1000


def all_segments():
    return SAMPLE18_ORIG + SAMPLE18_NEW + select_new_56_segments()


def which_ckpt(seg_id, ckpts):
    for name, ck in ckpts.items():
        if f"{seg_id}_A_vlm4ts_own" in ck:
            return ck
    raise KeyError(seg_id)


def run_step1(segments, ckpts):
    print("=" * 70)
    print("1단계: canmmlm_naive 승리 원인 분해 (기존 예측 재사용, 재호출 없음)")
    print("=" * 70)

    rows = []
    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)
        ck = which_ckpt(seg_id, ckpts)

        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        c_entry = ck.get(f"{seg_id}_C_our_design", {})
        if a_entry.get("status") != "OK" or c_entry.get("status") != "OK":
            continue
        a_pred = set(a_entry["pred"] or [])
        c_pred = set(c_entry["pred"] or [])

        groups = naive_groups(N_CHANNELS, N_GROUPS)
        b_pred_by_group = []
        b_ok = True
        for gi in range(N_GROUPS):
            ge = ck.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") != "OK":
                b_ok = False
                break
            b_pred_by_group.append(set(ge["pred"] or []))
        if not b_ok:
            continue
        b_pred_union = set().union(*b_pred_by_group)

        # TP 채널이 어느 그룹에서 나왔는지
        tp_b = b_pred_union & gt_channels
        tp_source_groups = []
        for c in tp_b:
            srcs = [gi for gi, gp in enumerate(b_pred_by_group) if c in gp]
            tp_source_groups.append((c, srcs))

        # A/C가 놓쳤는데 B가 잡은 채널 (canmmlm_naive만의 순증분 TP)
        recovered_by_b_only = (tp_b - a_pred) & (tp_b - c_pred)

        p_a, r_a = prf(a_pred, gt_channels)
        p_b, r_b = prf(b_pred_union, gt_channels)
        p_c, r_c = prf(c_pred, gt_channels)
        fp_a, fp_b, fp_c = len(a_pred - gt_channels), len(b_pred_union - gt_channels), len(c_pred - gt_channels)

        rows.append({
            "segment_id": seg_id, "entity": entity, "k": len(gt_channels),
            "precision_A": p_a, "recall_A": r_a, "fp_A": fp_a,
            "precision_B": p_b, "recall_B": r_b, "fp_B": fp_b,
            "precision_C": p_c, "recall_C": r_c, "fp_C": fp_c,
            "n_tp_recovered_by_B_only": len(recovered_by_b_only),
            "recovered_channels": sorted(recovered_by_b_only),
        })

    with (OUT_DIR / "step1_diagnosis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_fp_a = np.mean([r["fp_A"] for r in rows])
    mean_fp_b = np.mean([r["fp_B"] for r in rows])
    mean_fp_c = np.mean([r["fp_C"] for r in rows])
    total_recovered = sum(r["n_tp_recovered_by_B_only"] for r in rows)
    n_segs_with_recovery = sum(1 for r in rows if r["n_tp_recovered_by_B_only"] > 0)

    print(f"n={n} 세그먼트")
    print(f"평균 FP: vlm4ts_own(A)={mean_fp_a:.2f}  canmmlm_naive(B)={mean_fp_b:.2f}  our_design(C)={mean_fp_c:.2f}")
    print(f"A/C 둘 다 놓쳤는데 B(어느 그룹에서든)가 잡은 채널: 총 {total_recovered}개, {n_segs_with_recovery}개 세그먼트에서 발생")

    step1_summary = {
        "n_segments": n,
        "mean_fp": {"vlm4ts_own": float(mean_fp_a), "canmmlm_naive": float(mean_fp_b), "our_design": float(mean_fp_c)},
        "total_channels_recovered_by_B_only": int(total_recovered),
        "n_segments_with_B_only_recovery": int(n_segs_with_recovery),
    }
    (OUT_DIR / "step1_summary.json").write_text(json.dumps(step1_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_compute_step0_full_rank(seg_id, entity, cs, ce, ckpts, entity_train, entity_test):
    ck = which_ckpt(seg_id, ckpts)
    key = f"{seg_id}_step0_ranked"
    if key in ck and len(ck[key]) == N_CHANNELS:
        return ck[key]
    # 원본 9개 세그먼트는 top10만 캐시돼 있어 전체 랭킹을 새로 계산 (로컬, API 비용 0)
    if entity not in entity_train:
        entity_train[entity], _ = load_smd(entity)
    if entity not in entity_test:
        entity_test[entity] = load_smd_test(entity)
    train, test = entity_train[entity], entity_test[entity]
    s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)
    scores = {}
    for c in range(N_CHANNELS):
        static_emb = embed_channel_window(train[:, c], s_static, e_static)
        dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
        scores[c] = cosine_dist(static_emb, dynamic_emb)
    return sorted(scores, key=lambda c: -scores[c])


def run_step2(segments, ckpts):
    print("\n" + "=" * 70)
    print("2단계: 콜 수 통제 재비교 (조건 D: 동일 프롬프트 4회, 조건 E: top-16을 4그룹)")
    print("=" * 70)

    checkpoint = load_checkpoint()
    entity_test, entity_train = {}, {}

    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        # --- D: vlm4ts_own 동일 프롬프트 4회 반복, 합집합 ---
        all_channels = list(range(N_CHANNELS))
        d_preds = []
        for rep in range(4):
            case_id = f"{seg_id}_D_rep{rep}"
            if checkpoint.get(case_id, {}).get("status") != "OK":
                img = render_overlay(window, all_channels)
                prompt = build_prompt(all_channels)
                raw = call_vlm(prompt, img)
                pred = parse_response(raw)
                checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
                save_checkpoint(checkpoint)
                print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)
            e_ = checkpoint[case_id]
            if e_["status"] == "OK" and e_["pred"]:
                d_preds.append(set(e_["pred"]))

        # --- E: Step0 top-16 -> 4그룹(4개씩), 각각 호출, 합집합 ---
        ranked = get_or_compute_step0_full_rank(seg_id, entity, cs, ce, ckpts, entity_train, entity_test)
        top16 = ranked[:16]
        e_groups = [top16[i:i + 4] for i in range(0, 16, 4)]
        e_preds = []
        for gi, grp in enumerate(e_groups):
            case_id = f"{seg_id}_E_group{gi}"
            if checkpoint.get(case_id, {}).get("status") != "OK":
                img = render_overlay(window, grp)
                prompt = build_prompt(grp)
                raw = call_vlm(prompt, img)
                pred = parse_response(raw)
                checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw, "group": grp}
                save_checkpoint(checkpoint)
                print(f"[{checkpoint[case_id]['status']}] {case_id}: group={grp} pred={pred}", flush=True)
            e_ = checkpoint[case_id]
            if e_["status"] == "OK" and e_["pred"]:
                e_preds.append(set(e_["pred"]))

        checkpoint[f"{seg_id}_gt"] = sorted(gt_channels)
        save_checkpoint(checkpoint)


def score_de(segments):
    checkpoint = load_checkpoint()
    rows = []
    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        d_pred = set()
        d_ok = True
        for rep in range(4):
            e_ = checkpoint.get(f"{seg_id}_D_rep{rep}", {})
            if e_.get("status") == "OK" and e_.get("pred"):
                d_pred |= set(e_["pred"])
            if e_.get("status") != "OK":
                d_ok = False

        e_pred = set()
        e_ok = True
        for gi in range(4):
            e_ = checkpoint.get(f"{seg_id}_E_group{gi}", {})
            if e_.get("status") == "OK" and e_.get("pred"):
                e_pred |= set(e_["pred"])
            if e_.get("status") != "OK":
                e_ok = False

        if not (d_ok and e_ok):
            continue

        p_d, r_d = prf(d_pred, gt_channels)
        p_e, r_e = prf(e_pred, gt_channels)
        f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
        f1_e = 2 * p_e * r_e / (p_e + r_e) if (p_e + r_e) > 0 else 0.0
        rows.append({"segment_id": seg_id, "entity": entity, "f1_D": f1_d, "f1_E": f1_e,
                      "fp_D": len(d_pred - gt_channels), "fp_E": len(e_pred - gt_channels)})
    return rows


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(1)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    obs = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs, lo, hi, (lo <= 0 <= hi)


def run():
    ckpts = {name: json.loads(p.read_text(encoding="utf-8")) for name, p in CKPT_PATHS.items()}
    segments = all_segments()
    print(f"전체 세그먼트: {len(segments)}")

    step1_rows = run_step1(segments, ckpts)
    run_step2(segments, ckpts)
    de_rows = score_de(segments)

    # A/B/C F1 재수집 (비교용)
    abc_rows = {r["segment_id"]: r for r in step1_rows}
    merged = []
    for r in de_rows:
        base = abc_rows.get(r["segment_id"])
        if not base:
            continue
        f1_a = 2 * base["precision_A"] * base["recall_A"] / (base["precision_A"] + base["recall_A"]) if (base["precision_A"] + base["recall_A"]) > 0 else 0.0
        f1_b = 2 * base["precision_B"] * base["recall_B"] / (base["precision_B"] + base["recall_B"]) if (base["precision_B"] + base["recall_B"]) > 0 else 0.0
        f1_c = 2 * base["precision_C"] * base["recall_C"] / (base["precision_C"] + base["recall_C"]) if (base["precision_C"] + base["recall_C"]) > 0 else 0.0
        merged.append({"segment_id": r["segment_id"], "f1_A": f1_a, "f1_B": f1_b, "f1_C": f1_c,
                        "f1_D": r["f1_D"], "f1_E": r["f1_E"], "fp_D": r["fp_D"], "fp_E": r["fp_E"]})

    with (OUT_DIR / "step2_all_conditions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)

    means = {k: float(np.mean([m[f"f1_{k}"] for m in merged])) for k in ["A", "B", "C", "D", "E"]}
    n_calls = {"A_vlm4ts_own": 1, "B_canmmlm_naive": 4, "C_our_design": 1, "D_repeat4x": 4, "E_step0_top16_4group": 4}
    print("\n" + "=" * 70)
    print("전체 조건 비교: mean F1, 콜 수, F1/콜")
    print("=" * 70)
    efficiency = {}
    for k, label in zip(["A", "B", "C", "D", "E"], n_calls.keys()):
        calls = n_calls[label]
        eff = means[k] / calls
        efficiency[label] = {"mean_f1": means[k], "calls_per_segment": calls, "f1_per_call": eff}
        print(f"  {label}: F1={means[k]:.4f}  콜/세그먼트={calls}  F1/콜={eff:.4f}")

    # 유의성: D vs B, E vs B
    f1_b = np.array([m["f1_B"] for m in merged])
    f1_d = np.array([m["f1_D"] for m in merged])
    f1_e = np.array([m["f1_E"] for m in merged])
    print("\n" + "=" * 70)
    print("페어드 부트스트랩: B(canmmlm_naive) vs D(반복4회) vs E(top16 4그룹)")
    print("=" * 70)
    sig_results = {}
    for name1, v1, name2, v2 in [("B", f1_b, "D", f1_d), ("B", f1_b, "E", f1_e), ("D", f1_d, "E", f1_e)]:
        obs, lo, hi, incl0 = paired_bootstrap_diff(v1, v2, rng=np.random.default_rng(99))
        key = f"{name1}_minus_{name2}"
        sig_results[key] = {"obs_diff": obs, "ci_lo": lo, "ci_hi": hi, "includes_zero": incl0}
        print(f"  {name1}-{name2}: {obs:+.3f} [{lo:+.3f},{hi:+.3f}] {'(유의X)' if incl0 else '(유의미!)'}")

    summary = {
        "n_segments": len(merged),
        "mean_f1_all_conditions": means,
        "efficiency_f1_per_call": efficiency,
        "significance_B_vs_D_vs_E": sig_results,
    }
    (OUT_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
