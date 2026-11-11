"""
조건 H: top-4 캡 제거, Step0 점수는 정렬(front-loading)에만 사용 -- 38채널 전부를
Step0 점수 내림차순으로 그려서 vlm4ts_own(A, 컷 없음/정렬 없음, 1콜)과 직접 비교.
스코어러는 step1v3 계열(단일참조 코사인) 그대로, full_smd_3way.py의 74개 세그먼트와
동일 세그먼트 재사용, 캐시된 step0_ranked를 그대로 씀(재계산 없음).

1단계: 세그먼트/콜 수 확정 보고 후 STOP.
2단계(승인 후): 조건 H 74콜 실행 + A/B 대비 비교 + 유의성 검정 + 분위별 분석.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

OUT_DIR = BASE / "experiments" / "results_sort_no_cut_74"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint_H.json"

CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
GAP_DETAIL_CSV = BASE / "experiments" / "results_gap_quartile" / "gap_quartile_detail.csv"
N_BOOT = 1000
SEED = 42


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_merged_ckpts():
    """세 checkpoint(원본 18 + broad 9(중복) + 신규 56)를 합쳐 74개 세그먼트 전체를 커버."""
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
    seg_ids = sorted({k[:-3] for k in merged_ckpt if k.endswith("_gt")})
    return seg_ids


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def run_step1():
    merged_ckpt, _ = load_merged_ckpts()
    seg_ids = get_74_segments(merged_ckpt)
    score_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))
    missing_scores = [s for s in seg_ids if s not in score_cache]
    print(f"74개 세그먼트 재사용 확인(3개 checkpoint 병합): {len(seg_ids)}개")
    print(f"Step0 원시 점수 캐시 없는 세그먼트: {len(missing_scores)}개 (0이어야 재계산 불필요)")
    print(f"\n예상 콜 수: {len(seg_ids)}개 세그먼트 x 1콜(조건 H) = {len(seg_ids)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 H 74콜 실행)는 별도 실행 필요.")


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


def run_step2():
    from smd_3way_baseline_comparison import call_vlm, load_smd_test, build_prompt, render_overlay, parse_response

    merged_ckpt, ckpts = load_merged_ckpts()
    seg_ids = get_74_segments(merged_ckpt)
    score_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint()
    entity_test = {}

    for seg_id in seg_ids:
        if checkpoint.get(seg_id, {}).get("status") == "OK":
            print(f"[SKIP] {seg_id}")
            continue

        parts = seg_id.rsplit("_", 2)
        entity, cs, ce = parts[0], int(parts[1]), int(parts[2])
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]

        from step1v3_dino_graph_smd import WIN, _centered_window
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        ranked = get_ranked_channels(seg_id, score_cache)  # Step0 점수 내림차순 38채널 전부, 캐시 재사용
        img = render_overlay(window, ranked)
        prompt = build_prompt(ranked)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)
        checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw, "ranked": ranked}
        save_checkpoint(checkpoint)
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

    # ---- 비교 ----
    gap_rows = {r["segment_id"]: r for r in csv.DictReader(GAP_DETAIL_CSV.open(encoding="utf-8"))}
    n_all = len(gap_rows)
    q1_cut = n_all // 4
    q3_cut = n_all - n_all // 4
    sorted_by_gap = sorted(gap_rows.values(), key=lambda r: float(r["gap_normalized"]))
    quartile_of = {}
    for i, r in enumerate(sorted_by_gap):
        if i < q1_cut:
            quartile_of[r["segment_id"]] = "Q1_밀집"
        elif i < q3_cut:
            quartile_of[r["segment_id"]] = "Q2Q3_중간"
        else:
            quartile_of[r["segment_id"]] = "Q4_뚜렷"

    rows = []
    for seg_id in seg_ids:
        ck = find_ckpt_for(seg_id, ckpts)
        if ck is None:
            continue
        gt = set(ck[f"{seg_id}_gt"]["gt_channels"])
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        h_entry = checkpoint.get(seg_id, {})
        if a_entry.get("status") != "OK" or h_entry.get("status") != "OK":
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
        h_pred = set(h_entry["pred"] or [])
        entity = seg_id.rsplit("_", 2)[0]
        rows.append({
            "segment_id": seg_id, "entity": entity, "quartile": quartile_of.get(seg_id, "?"),
            "f1_A": f1_of(a_pred, gt), "f1_B": f1_of(b_pred, gt), "f1_H": f1_of(h_pred, gt),
            "p_A": prf(a_pred, gt)[0], "r_A": prf(a_pred, gt)[1],
            "p_B": prf(b_pred, gt)[0], "r_B": prf(b_pred, gt)[1],
            "p_H": prf(h_pred, gt)[0], "r_H": prf(h_pred, gt)[1],
            "diff_H_minus_A": f1_of(h_pred, gt) - f1_of(a_pred, gt),
            "diff_H_minus_B": f1_of(h_pred, gt) - f1_of(b_pred, gt),
        })

    with (OUT_DIR / "comparison_H.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a = np.mean([r["f1_A"] for r in rows])
    mean_b = np.mean([r["f1_B"] for r in rows])
    mean_h = np.mean([r["f1_H"] for r in rows])
    mean_pa, mean_ra = np.mean([r["p_A"] for r in rows]), np.mean([r["r_A"] for r in rows])
    mean_pb, mean_rb = np.mean([r["p_B"] for r in rows]), np.mean([r["r_B"] for r in rows])
    mean_ph, mean_rh = np.mean([r["p_H"] for r in rows]), np.mean([r["r_H"] for r in rows])

    print("\n" + "=" * 70)
    print(f"조건 H(정렬만, 컷없음, 1콜) vs A(정렬없음, 1콜) vs B(canmmlm_naive, 4콜)  n={n}")
    print("=" * 70)
    print(f"F1        : A={mean_a:.4f}  B={mean_b:.4f}  H={mean_h:.4f}")
    print(f"Precision : A={mean_pa:.4f}  B={mean_pb:.4f}  H={mean_ph:.4f}")
    print(f"Recall    : A={mean_ra:.4f}  B={mean_rb:.4f}  H={mean_rh:.4f}")

    # ---- 세그먼트 단위 페어드 부트스트랩 ----
    diffs_ha = [r["diff_H_minus_A"] for r in rows]
    diffs_hb = [r["diff_H_minus_B"] for r in rows]
    seg_ha = paired_bootstrap_diff(diffs_ha)
    seg_hb = paired_bootstrap_diff(diffs_hb)
    print(f"\n[세그먼트 단위 부트스트랩, n={n}, {N_BOOT}회]")
    print(f"H - A: mean={seg_ha[0]:+.4f}  95% CI=[{seg_ha[1]:+.4f}, {seg_ha[2]:+.4f}]"
          f"  {'유의(0 미포함)' if seg_ha[1] > 0 or seg_ha[2] < 0 else '유의하지 않음(0 포함)'}")
    print(f"H - B: mean={seg_hb[0]:+.4f}  95% CI=[{seg_hb[1]:+.4f}, {seg_hb[2]:+.4f}]"
          f"  {'유의(0 미포함)' if seg_hb[1] > 0 or seg_hb[2] < 0 else '유의하지 않음(0 포함)'}")

    # ---- entity 클러스터 부트스트랩 ----
    ent_ha = cluster_bootstrap_diff(rows, "diff_H_minus_A")
    ent_hb = cluster_bootstrap_diff(rows, "diff_H_minus_B")
    n_entities = len(set(r["entity"] for r in rows))
    print(f"\n[entity 클러스터 부트스트랩, {n_entities}개 entity, {N_BOOT}회]")
    print(f"H - A: mean={ent_ha[0]:+.4f}  95% CI=[{ent_ha[1]:+.4f}, {ent_ha[2]:+.4f}]"
          f"  {'유의' if ent_ha[1] > 0 or ent_ha[2] < 0 else '유의하지 않음'}")
    print(f"H - B: mean={ent_hb[0]:+.4f}  95% CI=[{ent_hb[1]:+.4f}, {ent_hb[2]:+.4f}]"
          f"  {'유의' if ent_hb[1] > 0 or ent_hb[2] < 0 else '유의하지 않음'}")

    # ---- 분위별 ----
    print("\n분위별 F1:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qa, qb, qh = np.mean([g["f1_A"] for g in grp]), np.mean([g["f1_B"] for g in grp]), np.mean([g["f1_H"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_A": float(qa), "f1_B": float(qb), "f1_H": float(qh)}
        print(f"  {q} (n={len(grp)}): A={qa:.3f}  B={qb:.3f}  H={qh:.3f}")

    verdict_ha = "H가 A보다 유의미하게 낫다" if seg_ha[1] > 0 else ("H가 A보다 유의미하게 못하다" if seg_ha[2] < 0 else "H와 A 차이 유의하지 않음")
    verdict_hb = "H가 B보다 유의미하게 낫다" if seg_hb[1] > 0 else ("H가 B에 유의미하게 못 미친다" if seg_hb[2] < 0 else "H와 B 차이 유의하지 않음(따라잡음)")
    print(f"\n핵심 판정:\n  H vs A: {verdict_ha}\n  H vs B: {verdict_hb}")

    summary = {
        "n": n, "n_entities": n_entities,
        "mean_f1": {"A": float(mean_a), "B": float(mean_b), "H": float(mean_h)},
        "mean_precision": {"A": float(mean_pa), "B": float(mean_pb), "H": float(mean_ph)},
        "mean_recall": {"A": float(mean_ra), "B": float(mean_rb), "H": float(mean_rh)},
        "segment_bootstrap": {"H_minus_A": seg_ha, "H_minus_B": seg_hb},
        "entity_cluster_bootstrap": {"H_minus_A": ent_ha, "H_minus_B": ent_hb},
        "by_quartile": q_summary,
        "verdict_H_vs_A": verdict_ha,
        "verdict_H_vs_B": verdict_hb,
    }
    (OUT_DIR / "summary_H.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
