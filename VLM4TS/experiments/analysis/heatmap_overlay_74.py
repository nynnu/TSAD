"""
조건 I8(heatmap 38채널 전체 + overlay 상위 8채널 컴포짓, 콜 1개) 74개 확장 검증.

파일럿(18개)에서 I8이 F1 0.302로 canmmlm_naive(B, 4콜, F1 0.234)를 앞섰음(160% 회복,
hedging 0건). 74개 규모 + 통계적 유의성 검정 전까지 확정 불가 -- 이번 스크립트가 그 검증.

full_smd_3way.py와 완전히 동일한 74개 세그먼트(3개 checkpoint 병합) 재사용, A/B는
재실행 없음, step0 랭킹은 results_adaptive_vs_fixed/step0_scores_cache.json 재사용
(재계산 없음), 렌더링/프롬프트는 pilot_heatmap_overlay_nsweep.py의 N=8 버전 그대로.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

OUT_DIR = BASE / "experiments" / "results_heatmap_overlay_74"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint_I8.json"

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
GAP_DETAIL_CSV = BASE / "experiments" / "results_gap_quartile" / "gap_quartile_detail.csv"
N = 8
HEDGE_THRESHOLD = 35
N_BOOT = 1000
SEED = 42


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def get_74_segments(merged_ckpt):
    return sorted({k[:-3] for k in merged_ckpt if k.endswith("_gt")})


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


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


def paired_bootstrap_diff(diffs, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    diffs = np.array(diffs)
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def cluster_bootstrap_diff(rows, diff_key, entity_key="entity", n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    by_entity = {}
    for r in rows:
        by_entity.setdefault(r[entity_key], []).append(r[diff_key])
    entities = list(by_entity.keys())
    n = len(entities)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(entities, size=n, replace=True)
        vals = [v for e in sampled for v in by_entity[e]]
        boot_means.append(np.mean(vals))
    boot_means = np.array(boot_means)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean_diff = np.mean([v for e in entities for v in by_entity[e]])
    return float(mean_diff), float(lo), float(hi)


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
    from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response
    from step1v3_dino_graph_smd import WIN, _centered_window
    from pilot_heatmap_overlay_nsweep import render_heatmap_overlay, build_prompt_heatmap_overlay

    merged_ckpt, ckpts = load_merged_ckpts()
    seg_ids = get_74_segments(merged_ckpt)
    score_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))
    quartile_of = load_quartiles()
    checkpoint = load_checkpoint()
    entity_test = {}
    all_channels = list(range(38))

    print(f"74개 세그먼트 확인: {len(seg_ids)}개, N={N}, 예상 콜 수: {len(seg_ids)}콜\n")

    for seg_id in seg_ids:
        if checkpoint.get(seg_id, {}).get("status") == "OK":
            print(f"[SKIP] {seg_id}")
            continue

        parts = seg_id.rsplit("_", 2)
        entity, cs, ce = parts[0], int(parts[1]), int(parts[2])
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        ranked = get_ranked_channels(seg_id, score_cache)
        top_n = ranked[:N]
        img = render_heatmap_overlay(window, ranked, N)
        prompt = build_prompt_heatmap_overlay(all_channels, top_n)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)
        checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
        save_checkpoint(checkpoint)
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

    # ---- 비교 ----
    rows = []
    for seg_id in seg_ids:
        ck = find_ckpt_for(seg_id, ckpts)
        if ck is None:
            continue
        gt = set(ck[f"{seg_id}_gt"]["gt_channels"])
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        i8_entry = checkpoint.get(seg_id, {})
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
        entity = seg_id.rsplit("_", 2)[0]
        rows.append({
            "segment_id": seg_id, "entity": entity, "quartile": quartile_of.get(seg_id, "?"),
            "n_pred_I8": len(i8_pred), "hedge_I8": len(i8_pred) >= HEDGE_THRESHOLD,
            "f1_A": f1_of(a_pred, gt), "f1_B": f1_of(b_pred, gt), "f1_I8": f1_of(i8_pred, gt),
            "fp_A": len(a_pred - gt), "fp_B": len(b_pred - gt), "fp_I8": len(i8_pred - gt),
            "diff_I8_minus_A": f1_of(i8_pred, gt) - f1_of(a_pred, gt),
            "diff_I8_minus_B": f1_of(i8_pred, gt) - f1_of(b_pred, gt),
        })

    with (OUT_DIR / "comparison_I8.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a, mean_b, mean_i8 = np.mean([r["f1_A"] for r in rows]), np.mean([r["f1_B"] for r in rows]), np.mean([r["f1_I8"] for r in rows])
    mean_fp_a, mean_fp_b, mean_fp_i8 = np.mean([r["fp_A"] for r in rows]), np.mean([r["fp_B"] for r in rows]), np.mean([r["fp_I8"] for r in rows])
    n_hedge = sum(1 for r in rows if r["hedge_I8"])

    print("\n" + "=" * 70)
    print(f"조건 I8(heatmap+overlay N=8, 1콜) vs A(1콜) vs B(canmmlm_naive, 4콜)  n={n}")
    print("=" * 70)
    print(f"F1: A={mean_a:.4f}  B={mean_b:.4f}  I8={mean_i8:.4f}")
    print(f"FP: A={mean_fp_a:.2f}  B={mean_fp_b:.2f}  I8={mean_fp_i8:.2f}")
    print(f"hedging(>={HEDGE_THRESHOLD}채널) 발생: {n_hedge}/{n}")

    diffs_i8a = [r["diff_I8_minus_A"] for r in rows]
    diffs_i8b = [r["diff_I8_minus_B"] for r in rows]
    seg_i8a = paired_bootstrap_diff(diffs_i8a)
    seg_i8b = paired_bootstrap_diff(diffs_i8b)
    print(f"\n[세그먼트 단위 부트스트랩, n={n}, {N_BOOT}회]")
    print(f"I8 - A: mean={seg_i8a[0]:+.4f}  95% CI=[{seg_i8a[1]:+.4f}, {seg_i8a[2]:+.4f}]"
          f"  {'유의' if seg_i8a[1] > 0 or seg_i8a[2] < 0 else '유의하지 않음'}")
    print(f"I8 - B: mean={seg_i8b[0]:+.4f}  95% CI=[{seg_i8b[1]:+.4f}, {seg_i8b[2]:+.4f}]"
          f"  {'유의' if seg_i8b[1] > 0 or seg_i8b[2] < 0 else '유의하지 않음'}")

    ent_i8a = cluster_bootstrap_diff(rows, "diff_I8_minus_A")
    ent_i8b = cluster_bootstrap_diff(rows, "diff_I8_minus_B")
    n_entities = len(set(r["entity"] for r in rows))
    print(f"\n[entity 클러스터 부트스트랩, {n_entities}개 entity, {N_BOOT}회]")
    print(f"I8 - A: mean={ent_i8a[0]:+.4f}  95% CI=[{ent_i8a[1]:+.4f}, {ent_i8a[2]:+.4f}]"
          f"  {'유의' if ent_i8a[1] > 0 or ent_i8a[2] < 0 else '유의하지 않음'}")
    print(f"I8 - B: mean={ent_i8b[0]:+.4f}  95% CI=[{ent_i8b[1]:+.4f}, {ent_i8b[2]:+.4f}]"
          f"  {'유의' if ent_i8b[1] > 0 or ent_i8b[2] < 0 else '유의하지 않음'}")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qa, qb, qi8 = np.mean([g["f1_A"] for g in grp]), np.mean([g["f1_B"] for g in grp]), np.mean([g["f1_I8"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_A": float(qa), "f1_B": float(qb), "f1_I8": float(qi8)}
        print(f"  {q} (n={len(grp)}): A={qa:.3f}  B={qb:.3f}  I8={qi8:.3f}")

    verdict_i8a = "I8이 A보다 유의미하게 낫다" if seg_i8a[1] > 0 else ("I8이 A보다 유의미하게 못하다" if seg_i8a[2] < 0 else "I8과 A 차이 유의하지 않음")
    verdict_i8b = "I8이 B(4콜)보다 유의미하게 낫다" if seg_i8b[1] > 0 else ("I8이 B에 유의미하게 못 미친다" if seg_i8b[2] < 0 else "I8과 B 차이 유의하지 않음")
    print(f"\n핵심 판정:\n  I8 vs A: {verdict_i8a}\n  I8 vs B: {verdict_i8b}")

    summary = {
        "n": n, "n_entities": n_entities, "N": N,
        "mean_f1": {"A": float(mean_a), "B": float(mean_b), "I8": float(mean_i8)},
        "mean_fp": {"A": float(mean_fp_a), "B": float(mean_fp_b), "I8": float(mean_fp_i8)},
        "hedge_count": n_hedge,
        "segment_bootstrap": {"I8_minus_A": seg_i8a, "I8_minus_B": seg_i8b},
        "entity_cluster_bootstrap": {"I8_minus_A": ent_i8a, "I8_minus_B": ent_i8b},
        "by_quartile": q_summary,
        "verdict_I8_vs_A": verdict_i8a,
        "verdict_I8_vs_B": verdict_i8b,
    }
    (OUT_DIR / "summary_I8.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()
