"""
SMD 확장 3-way 비교(74개 세그먼트) + entity 클러스터 부트스트랩.

기존 18개(machine-1-1/1-2/1-5, machine-2-1/3-2/3-4) + 새 56개(17개 entity,
결정론적 상한 규칙으로 배분, seed=42로 초과분 무작위 선택) = 74개.

3개 조건은 §29/§3way_broad와 완전히 동일(파라미터 변경 없음):
vlm4ts_own(38채널 전체), canmmlm_naive(naive_groups, N_GROUPS=4),
our_design(Step0 단일 static 참조 final-layer top-4, 다중참조/배경마스킹/
multi-layer 전부 미적용 -- §29 원본 메커니즘).
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

OUT_DIR = BASE / "experiments" / "results_full_smd_3way"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"
INTERP_CACHE = ANALYSIS_DIR / "_interp_cache"

ORIG_18_CKPTS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
]

N_BOOT = 1000
SEED = 42

ENTITY_CAP = {
    "machine-1-3": 1, "machine-1-4": 1, "machine-2-6": 1, "machine-3-1": 1, "machine-3-5": 1, "machine-3-8": 2,
    "machine-1-7": 5, "machine-2-9": 5, "machine-2-2": 5, "machine-2-4": 5, "machine-3-6": 5,
    "machine-1-6": 4, "machine-3-10": 4, "machine-1-8": 4, "machine-2-5": 4, "machine-2-7": 4, "machine-3-3": 4,
}


def select_new_56_segments():
    rng = np.random.default_rng(SEED)
    selected = []
    for entity, cap in sorted(ENTITY_CAP.items()):
        path = INTERP_CACHE / f"{entity}.txt"
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        candidates = []
        for l in lines:
            span, chans = l.split(":")
            s, e = span.split("-")
            gt_dims = [int(x) for x in chans.split(",")]
            if 4 <= len(gt_dims) <= 8:
                candidates.append((int(s), int(e), gt_dims))
        candidates.sort(key=lambda t: t[0])  # deterministic order before sampling
        if len(candidates) > cap:
            idx = rng.choice(len(candidates), size=cap, replace=False)
            idx = sorted(idx.tolist())
            candidates = [candidates[i] for i in idx]
        for s, e, gt_dims in candidates:
            selected.append((entity, (s, e), gt_dims))
    return selected


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def step0_rank_original_mechanism(train, test, cs, ce):
    s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
    center = (cs + ce) // 2
    s_dyn, e_dyn = _centered_window(len(test), center, WIN)
    scores = {}
    for c in range(N_CHANNELS):
        static_emb = embed_channel_window(train[:, c], s_static, e_static)
        dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
        scores[c] = cosine_dist(static_emb, dynamic_emb)
    return sorted(scores, key=lambda c: -scores[c])


def run_condition(new_segments):
    checkpoint = load_checkpoint()
    entity_test, entity_train = {}, {}
    groups = naive_groups(N_CHANNELS, N_GROUPS)

    for entity, (cs, ce), gt_dims in new_segments:
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
                print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)

        step0_key = f"{seg_id}_step0_ranked"
        if step0_key not in checkpoint:
            print(f"  [Step0] {seg_id} 랭킹 계산 중 ...", flush=True)
            ranked = step0_rank_original_mechanism(train, test, cs, ce)
            checkpoint[step0_key] = ranked
            save_checkpoint(checkpoint)
        top4 = checkpoint[step0_key][:4]

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

        checkpoint[f"{seg_id}_gt"] = {"gt_channels": sorted(gt_channels)}
        save_checkpoint(checkpoint)


def segment_f1s(segments, ckpt):
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

        rows.append({"segment_id": seg_id, "entity": entity, "f1_A": f1_a, "f1_B": f1_b, "f1_C": f1_c})
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


def cluster_bootstrap_diff(rows, method1, method2, n_boot=N_BOOT, rng=None):
    """Entity 단위 리샘플: entity를 복원추출, 뽑힌 entity의 세그먼트 전부 포함."""
    rng = rng or np.random.default_rng(2)
    entities = sorted(set(r["entity"] for r in rows))
    by_entity = {e: [r for r in rows if r["entity"] == e] for e in entities}
    n_e = len(entities)

    def mean_for_sample(sample_entities, key):
        vals = []
        for e in sample_entities:
            vals.extend(r[key] for r in by_entity[e])
        return np.mean(vals) if vals else 0.0

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_e, size=n_e)
        sample_entities = [entities[i] for i in idx]
        m1 = mean_for_sample(sample_entities, method1)
        m2 = mean_for_sample(sample_entities, method2)
        diffs.append(m1 - m2)
    diffs = np.array(diffs)
    all_a = np.array([r[method1] for r in rows])
    all_b = np.array([r[method2] for r in rows])
    obs = float(all_a.mean() - all_b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs, lo, hi, (lo <= 0 <= hi)


def run():
    new_56 = select_new_56_segments()
    print(f"새로 선정된 세그먼트 수: {len(new_56)}")
    by_entity_count = {}
    for entity, _, _ in new_56:
        by_entity_count[entity] = by_entity_count.get(entity, 0) + 1
    for e, c in sorted(by_entity_count.items()):
        print(f"  {e}: {c}개")

    print("\n" + "=" * 70)
    print(f"VLM 실행: 새 {len(new_56)}개 세그먼트 x 6콜 = {len(new_56) * 6}콜")
    print("=" * 70)
    run_condition(new_56)

    # ---- 74개 전체 F1 수집 ----
    new_ckpt = load_checkpoint()
    new_rows = segment_f1s(new_56, new_ckpt)

    orig_rows = []
    orig_ckpt_18a = json.loads(ORIG_18_CKPTS[0].read_text(encoding="utf-8"))
    orig_ckpt_18b = json.loads(ORIG_18_CKPTS[1].read_text(encoding="utf-8"))
    orig_rows += segment_f1s(SAMPLE18_ORIG, orig_ckpt_18a)
    orig_rows += segment_f1s(SAMPLE18_NEW, orig_ckpt_18b)

    all_rows = orig_rows + new_rows
    print(f"\n총 확보된 완전 채점 세그먼트 수: {len(all_rows)} (목표 74)")

    with (OUT_DIR / "segment_f1_all74.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id", "entity", "f1_A", "f1_B", "f1_C"])
        w.writeheader()
        w.writerows(all_rows)

    f1_a = np.array([r["f1_A"] for r in all_rows])
    f1_b = np.array([r["f1_B"] for r in all_rows])
    f1_c = np.array([r["f1_C"] for r in all_rows])
    n = len(all_rows)

    print("\n" + "=" * 70)
    print(f"전체 {n}개 mean F1")
    print("=" * 70)
    means = {"vlm4ts_own": float(f1_a.mean()), "canmmlm_naive": float(f1_b.mean()), "our_design": float(f1_c.mean())}
    print(json.dumps(means, indent=2))

    # ---- 세그먼트 단위 페어드 부트스트랩 ----
    print("\n" + "=" * 70)
    print("세그먼트 단위 페어드 부트스트랩 (n=%d)" % n)
    print("=" * 70)
    seg_boot = {}
    rng1 = np.random.default_rng(10)
    for name1, v1, name2, v2 in [
        ("vlm4ts_own", f1_a, "our_design", f1_c),
        ("vlm4ts_own", f1_a, "canmmlm_naive", f1_b),
        ("our_design", f1_c, "canmmlm_naive", f1_b),
    ]:
        obs, lo, hi, incl0 = paired_bootstrap_diff(v1, v2, rng=rng1)
        key = f"{name1}_minus_{name2}"
        seg_boot[key] = {"obs_diff": obs, "ci_lo": lo, "ci_hi": hi, "includes_zero": incl0}
        print(f"  {name1}-{name2}: {obs:+.3f} [{lo:+.3f},{hi:+.3f}] {'(유의X)' if incl0 else '(유의미!)'}")

    # ---- entity 단위 클러스터 부트스트랩 ----
    print("\n" + "=" * 70)
    print(f"entity 단위 클러스터 부트스트랩 (n_entity={len(set(r['entity'] for r in all_rows))})")
    print("=" * 70)
    entity_boot = {}
    rng2 = np.random.default_rng(20)
    for name1, key1, name2, key2 in [
        ("vlm4ts_own", "f1_A", "our_design", "f1_C"),
        ("vlm4ts_own", "f1_A", "canmmlm_naive", "f1_B"),
        ("our_design", "f1_C", "canmmlm_naive", "f1_B"),
    ]:
        obs, lo, hi, incl0 = cluster_bootstrap_diff(all_rows, key1, key2, rng=rng2)
        key = f"{name1}_minus_{name2}"
        entity_boot[key] = {"obs_diff": obs, "ci_lo": lo, "ci_hi": hi, "includes_zero": incl0}
        print(f"  {name1}-{name2}: {obs:+.3f} [{lo:+.3f},{hi:+.3f}] {'(유의X)' if incl0 else '(유의미!)'}")

    # ---- 두 방식 비교 ----
    print("\n" + "=" * 70)
    print("세그먼트 단위 vs entity 클러스터 단위 결론 비교")
    print("=" * 70)
    agreement = {}
    for key in seg_boot:
        seg_sig = not seg_boot[key]["includes_zero"]
        ent_sig = not entity_boot[key]["includes_zero"]
        same = seg_sig == ent_sig
        agreement[key] = {"segment_level_significant": seg_sig, "entity_cluster_significant": ent_sig, "agree": same}
        print(f"  {key}: 세그먼트단위 유의미={seg_sig}  entity클러스터 유의미={ent_sig}  {'일치' if same else '** 불일치 **'}")

    summary = {
        "n_segments_total": n,
        "n_new_segments": len(new_56),
        "entity_distribution_new": by_entity_count,
        "mean_f1": means,
        "segment_level_bootstrap": seg_boot,
        "entity_cluster_bootstrap": entity_boot,
        "agreement": agreement,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
