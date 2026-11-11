"""
파일럿: I8(heatmap+overlay top-8)에 Step0 점수를 텍스트로 병기(조건 L, 콜 1개).
렌더링은 I8과 완전히 동일(render_heatmap_overlay 그대로 재사용), 프롬프트에만
top-8 각 채널의 정규화된 이상 점수 + 순위를 텍스트로 추가.

동일 18개 세그먼트(gap 3분위 층화) 재사용. K(index-aware) 조건은 이번 세션에
아직 없어서 I8 vs L만 비교.

1단계: 세그먼트/콜 수(18) 보고 후 STOP.
2단계(승인 후): 조건 L 18콜 실행 + I8 대비 비교 + 점수격차 큰/작은 세그먼트 분해.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_score_annotation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
I8_PILOT_CKPT = BASE / "experiments" / "results_pilot_heatmap_nsweep" / "checkpoint_I8.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
N = 8
HEDGE_THRESHOLD = 35


def get_ranked_with_scores(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    ranked = sorted(scores, key=lambda c: -scores[c])
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    norm = {c: ((scores[c] - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0) for c in scores}
    return ranked, norm


def build_prompt_score_annotated(all_channels: list[int], top_n: list[int], norm_scores: dict) -> str:
    lines = "; ".join(f"채널 {c}: 이상 점수 {norm_scores[c]:.2f} ({i+1}위)" for i, c in enumerate(top_n))
    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled.

The top-{len(top_n)} candidates' preliminary anomaly scores (0~1 scale, higher = more anomalous) and ranks are: {lines}.

Use both panels and the scores together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {all_channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"재사용 세그먼트(I8 파일럿과 동일): {len(selected)}개\n")
    by_q = {}
    for s in selected:
        by_q.setdefault(s["quartile"], []).append(s)
    for q, items in by_q.items():
        print(f"{q} (n={len(items)}): {[i['segment_id'] for i in items]}")
    print(f"\n예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 L) = {len(selected)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 L 실행)는 별도 실행 필요.")


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


def run_step2():
    from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response
    from step1v3_dino_graph_smd import WIN, _centered_window
    from pilot_heatmap_overlay_nsweep import render_heatmap_overlay

    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    i8_pilot_ckpt = json.loads(I8_PILOT_CKPT.read_text(encoding="utf-8"))
    score_cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))

    def find_gt(seg_id):
        for ck in ckpts:
            v = ck.get(f"{seg_id}_gt")
            if v:
                return set(v["gt_channels"])
        return None

    def find_ckpt_for(seg_id):
        for ck in ckpts:
            if f"{seg_id}_A_vlm4ts_own" in ck:
                return ck
        return None

    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    checkpoint_path = OUT_DIR / "checkpoint_L.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(38))
    entity_test = {}

    for s in selected:
        seg_id = s["segment_id"]
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        if checkpoint.get(seg_id, {}).get("status") == "OK":
            print(f"[SKIP] {seg_id}")
            continue

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        ranked, norm_scores = get_ranked_with_scores(seg_id, score_cache)
        top8 = ranked[:N]
        img = render_heatmap_overlay(window, ranked, N)
        prompt = build_prompt_score_annotated(all_channels, top8, norm_scores)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)
        checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

    # ---- 비교 ----
    rows = []
    for s in selected:
        seg_id, quartile = s["segment_id"], s["quartile"]
        gt = find_gt(seg_id)
        ck = find_ckpt_for(seg_id)
        l_entry = checkpoint.get(seg_id, {})
        i8_entry = i8_pilot_ckpt.get(seg_id, {})
        if gt is None or ck is None or l_entry.get("status") != "OK" or i8_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue

        l_pred = set(l_entry["pred"] or [])
        i8_pred = set(i8_entry["pred"] or [])

        _, norm_scores = get_ranked_with_scores(seg_id, score_cache)
        ranked, _ = get_ranked_with_scores(seg_id, score_cache)
        top8_scores = [norm_scores[c] for c in ranked[:N]]
        gap_1_8 = top8_scores[0] - top8_scores[-1]  # 1위-8위 점수 격차 (정규화 스케일)

        rows.append({
            "segment_id": seg_id, "quartile": quartile, "gap_1_8": gap_1_8,
            "f1_I8": f1_of(i8_pred, gt), "f1_L": f1_of(l_pred, gt),
            "fp_I8": len(i8_pred - gt), "fp_L": len(l_pred - gt),
            "n_pred_L": len(l_pred), "hedge_L": len(l_pred) >= HEDGE_THRESHOLD,
            "diff_L_minus_I8": f1_of(l_pred, gt) - f1_of(i8_pred, gt),
        })

    with (OUT_DIR / "pilot_comparison_L.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: I8={mean('f1_I8'):.4f}  L(점수병기)={mean('f1_L'):.4f}")
    print(f"FP: I8={mean('fp_I8'):.2f}  L={mean('fp_L'):.2f}")
    print(f"hedging(L): {sum(1 for r in rows if r['hedge_L'])}/{n}")
    print(f"\nL - I8 = {mean('diff_L_minus_I8'):+.4f}")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qi8 = np.mean([g["f1_I8"] for g in grp])
        ql = np.mean([g["f1_L"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_I8": float(qi8), "f1_L": float(ql)}
        print(f"  {q} (n={len(grp)}): I8={qi8:.3f}  L={ql:.3f}  (L-I8={ql-qi8:+.3f})")

    # ---- 점수 격차 큰/작은 세그먼트 분해 (median split) ----
    gaps = [r["gap_1_8"] for r in rows]
    med_gap = float(np.median(gaps))
    big_gap = [r for r in rows if r["gap_1_8"] >= med_gap]
    small_gap = [r for r in rows if r["gap_1_8"] < med_gap]
    print(f"\n1위-8위 점수격차 median split (median={med_gap:.3f}):")
    if big_gap:
        print(f"  격차 큰(순위 뚜렷) n={len(big_gap)}: I8={np.mean([r['f1_I8'] for r in big_gap]):.3f}  "
              f"L={np.mean([r['f1_L'] for r in big_gap]):.3f}  (L-I8={np.mean([r['diff_L_minus_I8'] for r in big_gap]):+.3f})")
    if small_gap:
        print(f"  격차 작은(순위 애매) n={len(small_gap)}: I8={np.mean([r['f1_I8'] for r in small_gap]):.3f}  "
              f"L={np.mean([r['f1_L'] for r in small_gap]):.3f}  (L-I8={np.mean([r['diff_L_minus_I8'] for r in small_gap]):+.3f})")

    helps_small_gap = small_gap and big_gap and np.mean([r["diff_L_minus_I8"] for r in small_gap]) > np.mean([r["diff_L_minus_I8"] for r in big_gap])
    overall_helps = mean("diff_L_minus_I8") > 0.01

    if overall_helps:
        verdict = "점수 병기가 I8 위에서 유의미한 추가 개선을 줌"
    else:
        verdict = "점수 병기의 전반적 효과는 미미하거나 없음"
    verdict += " / 순위 애매한 세그먼트에서 더 도움됨" if helps_small_gap else " / 순위 애매한 세그먼트에서 더 도움된다는 근거는 약함"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n,
        "mean_f1": {"I8": mean("f1_I8"), "L": mean("f1_L")},
        "mean_fp": {"I8": mean("fp_I8"), "L": mean("fp_L")},
        "hedge_L": sum(1 for r in rows if r["hedge_L"]),
        "diff_L_minus_I8": mean("diff_L_minus_I8"),
        "by_quartile": q_summary,
        "gap_median": med_gap,
        "big_gap_diff": float(np.mean([r["diff_L_minus_I8"] for r in big_gap])) if big_gap else None,
        "small_gap_diff": float(np.mean([r["diff_L_minus_I8"] for r in small_gap])) if small_gap else None,
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_L.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
