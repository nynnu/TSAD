"""
§29 3-way 비교(vlm4ts_own / our_design(top-4) / canmmlm_naive)를 새 9개
세그먼트(machine-2-1, machine-3-2, machine-3-4 -- 표본 밖 검증에서 쓴 것과
동일)에 그대로 재현. 파라미터 변경 없음.

- vlm4ts_own: 38채널 전체 overlay 1장, 자유형 식별
- canmmlm_naive: naive_groups(38, 4)로 4그룹 분할, 그룹별 overlay+식별 후 합산
- our_design: Step0(DINOv2 단일 static 참조, final-layer, §29 당시와 동일
  메커니즘 -- 다중참조/배경마스킹 아님) top-4 선택 후 그 4채널만 overlay+식별

Step0 랭킹은 새 세그먼트에 대해 로컬로 새로 계산해야 함(기존 캐시는
원래 9개 세그먼트용). step1v3_dino_graph_smd.py의 원본 메커니즘 그대로 재사용.
"""

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

OUT_DIR = BASE / "experiments" / "results_3way_broad"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

ORIG_3WAY_CHECKPOINT = BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json"
ORIG_SEGMENTS = [
    ("machine-1-1", (15849, 16368), [1, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (18071, 18528), [1, 2, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (24679, 24682), [9, 13, 14, 15]),
    ("machine-1-2", (4629, 4688), [9, 10, 11, 13, 15, 18]),
    ("machine-1-2", (15925, 15973), [6, 7, 10, 11, 13, 14, 20, 30]),
    ("machine-1-2", (22264, 22336), [1, 2, 3, 4]),
    ("machine-1-5", (10620, 10637), [1, 2, 3, 4, 7, 24, 26, 32]),
    ("machine-1-5", (14068, 14072), [19, 20, 21, 22, 28, 31]),
    ("machine-1-5", (21287, 21298), [1, 2, 3, 4, 6, 7, 24, 26]),
]

NEW_SEGMENTS = [
    ("machine-2-1", (6506, 6528), [23, 25, 32, 35, 36]),
    ("machine-2-1", (7907, 7961), [9, 10, 11, 13, 32]),
    ("machine-2-1", (9340, 9386), [9, 10, 11, 13, 32, 33, 34]),
    ("machine-3-2", (135, 161), [10, 12, 13, 36]),
    ("machine-3-2", (3080, 3917), [9, 13, 14, 17]),
    ("machine-3-2", (15577, 15582), [9, 13, 14, 35, 36]),
    ("machine-3-4", (2734, 3520), [1, 2, 3, 4, 6, 7, 11, 16]),
    ("machine-3-4", (6013, 6016), [10, 12, 13, 14]),
    ("machine-3-4", (10963, 10969), [30, 33, 34]),
]


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def step0_rank_original_mechanism(entity, cs, ce, train, test):
    s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)
    scores = {}
    for c in range(N_CHANNELS):
        static_emb = embed_channel_window(train[:, c], s_static, e_static)
        dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
        scores[c] = cosine_dist(static_emb, dynamic_emb)
    return sorted(scores, key=lambda c: -scores[c])


def run():
    checkpoint = load_checkpoint()
    entity_test = {}
    entity_train = {}
    groups = naive_groups(N_CHANNELS, N_GROUPS)
    print(f"canMMLM-naive groups: {groups}")

    for entity, (cs, ce), gt_dims in NEW_SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        train = entity_train[entity]

        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        # --- A: vlm4ts_own ---
        case_id = f"{seg_id}_A_vlm4ts_own"
        if checkpoint.get(case_id, {}).get("status") != "OK":
            all_channels = list(range(N_CHANNELS))
            img = render_overlay(window, all_channels)
            prompt = build_prompt(all_channels)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "A"}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)
        else:
            print(f"[SKIP] {case_id}")

        # --- B: canmmlm_naive ---
        for gi, group in enumerate(groups):
            case_id = f"{seg_id}_B_group{gi}"
            if checkpoint.get(case_id, {}).get("status") != "OK":
                img = render_overlay(window, group)
                prompt = build_prompt(group)
                raw = call_vlm(prompt, img)
                pred = parse_response(raw)
                checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                        "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "B", "group": group}
                save_checkpoint(checkpoint)
                print(f"[{checkpoint[case_id]['status']}] {case_id}: group={group} pred={pred}", flush=True)
            else:
                print(f"[SKIP] {case_id}")

        # --- C: our_design (Step0 top-4, 원본 메커니즘 새 세그먼트에 재계산) ---
        step0_key = f"{seg_id}_step0_ranked"
        if step0_key not in checkpoint:
            print(f"  [Step0] {seg_id} 랭킹 계산 중 ...", flush=True)
            ranked = step0_rank_original_mechanism(entity, cs, ce, train, test)
            checkpoint[step0_key] = ranked
            save_checkpoint(checkpoint)
        ranked = checkpoint[step0_key]
        top4 = ranked[:4]

        case_id = f"{seg_id}_C_our_design"
        if checkpoint.get(case_id, {}).get("status") != "OK":
            img = render_overlay(window, top4)
            prompt = build_prompt(top4)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "C", "top4": top4}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: top4={top4} pred={pred}", flush=True)
        else:
            print(f"[SKIP] {case_id}")

        checkpoint[f"{seg_id}_gt"] = {"gt_channels": sorted(gt_channels)}
        save_checkpoint(checkpoint)

    # ---- 채점 ----
    def score_segments(segments, ckpt):
        results = {"A": [], "B": [], "C": []}
        for entity, (cs, ce), gt_dims in segments:
            seg_id = f"{entity}_{cs}_{ce}"
            gt_channels = set(d - 1 for d in gt_dims)

            a_entry = ckpt.get(f"{seg_id}_A_vlm4ts_own", {})
            if a_entry.get("status") == "OK":
                p, r = prf(set(a_entry["pred"] or []), gt_channels)
                results["A"].append((seg_id, p, r))

            b_pred, b_ok = set(), True
            for gi in range(N_GROUPS):
                ge = ckpt.get(f"{seg_id}_B_group{gi}", {})
                if ge.get("status") == "OK" and ge.get("pred"):
                    b_pred |= set(ge["pred"])
                if ge.get("status") != "OK":
                    b_ok = False
            if b_ok:
                p, r = prf(b_pred, gt_channels)
                results["B"].append((seg_id, p, r))

            c_entry = ckpt.get(f"{seg_id}_C_our_design", {})
            if c_entry.get("status") == "OK":
                p, r = prf(set(c_entry["pred"] or []), gt_channels)
                results["C"].append((seg_id, p, r))
        return results

    def summarize(results):
        out = {}
        for cond, label in [("A", "vlm4ts_own"), ("B", "canmmlm_naive"), ("C", "our_design")]:
            vals = results[cond]
            if vals:
                precision = float(np.mean([v[1] for v in vals]))
                recall = float(np.mean([v[2] for v in vals]))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                out[label] = {"n": len(vals), "precision": precision, "recall": recall, "f1": f1}
        return out

    new_results = score_segments(NEW_SEGMENTS, checkpoint)
    new_summary = summarize(new_results)

    orig_checkpoint = json.loads(ORIG_3WAY_CHECKPOINT.read_text(encoding="utf-8"))
    orig_results = score_segments(ORIG_SEGMENTS, orig_checkpoint)
    orig_summary = summarize(orig_results)

    combined_results = {
        cond: orig_results[cond] + new_results[cond] for cond in ["A", "B", "C"]
    }
    combined_summary = summarize(combined_results)

    print("\n" + "=" * 70)
    print("기존 9개 세그먼트 (§29 원본)")
    print("=" * 70)
    print(json.dumps(orig_summary, indent=2))

    print("\n" + "=" * 70)
    print("새 9개 세그먼트")
    print("=" * 70)
    print(json.dumps(new_summary, indent=2))

    print("\n" + "=" * 70)
    print("18개 합산")
    print("=" * 70)
    print(json.dumps(combined_summary, indent=2))

    # 순위 확인
    print("\n" + "=" * 70)
    print("세그먼트별 F1 순위 (vlm4ts_own vs our_design)")
    print("=" * 70)
    a_by_seg = {seg: (p, r) for seg, p, r in new_results["A"]}
    c_by_seg = {seg: (p, r) for seg, p, r in new_results["C"]}
    flips = []
    for seg in a_by_seg:
        if seg not in c_by_seg:
            continue
        ap, ar = a_by_seg[seg]
        cp, cr = c_by_seg[seg]
        af1 = 2 * ap * ar / (ap + ar) if (ap + ar) > 0 else 0.0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        flipped = cf1 > af1
        flips.append({"segment_id": seg, "vlm4ts_own_f1": af1, "our_design_f1": cf1, "our_design_wins": flipped})
        print(f"  {seg}: vlm4ts_own F1={af1:.3f}  our_design F1={cf1:.3f}  {'** our_design 승 **' if flipped else ''}")

    n_flips = sum(1 for f in flips if f["our_design_wins"])

    summary_all = {
        "original_9seg": orig_summary,
        "new_9seg": new_summary,
        "combined_18seg": combined_summary,
        "per_segment_flips_new9": flips,
        "n_segments_our_design_beats_vlm4ts_own_in_new9": n_flips,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
