"""
파일럿: I8(heatmap+overlay top-8)에 index-aware 텍스트 이식(조건 K, 콜 1개).
렌더링은 I8과 완전히 동일(render_heatmap_overlay 그대로 재사용), 프롬프트에만
top-8 각 채널의 (index, value) 전체 리스트를 §27 원 형식(smd_index_aware_boundary.py의
build_prompt index_aware 분기) 그대로 재사용해서 채널별로 병기.

동일 18개 세그먼트(gap 3분위 층화) 재사용.

1단계: 세그먼트/콜 수(18) 보고 후 STOP.
2단계(승인 후): 조건 K 18콜 실행 + I8 대비 비교 + 낮은 std/Q2Q3 구간 분해.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_index_aware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
FEATURES_CSV = BASE / "experiments" / "results_adaptive_vs_fixed" / "segment_features.csv"
I8_PILOT_CKPT = BASE / "experiments" / "results_pilot_heatmap_nsweep" / "checkpoint_I8.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
N = 8
HEDGE_THRESHOLD = 35


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def build_prompt_index_aware(all_channels: list[int], top_n: list[int], window, norm_of) -> str:
    """§27 index_aware 형식((index, value) 텍스트) 그대로, top-8 채널 각각에 적용."""
    blocks = []
    for i, c in enumerate(top_n):
        norm = norm_of(c)
        history = "\n".join(f"({idx}, {v:.3f})" for idx, v in enumerate(norm))
        blocks.append(f"Channel {c} (rank {i+1}) index-value data:\n<history>\n{history}\n</history>")
    history_text = "\n\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled.

I will also provide you with time-series value data for each of the top-{len(top_n)} candidate channels, in (index, value) format:

{history_text}

Use the panels and the index-value data together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

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
    print(f"\n예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 K) = {len(selected)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 K 실행)는 별도 실행 필요.")


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
    features = {r["segment_id"]: r for r in csv.DictReader(FEATURES_CSV.open(encoding="utf-8"))}

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
    checkpoint_path = OUT_DIR / "checkpoint_K.json"
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

        ranked = get_ranked_channels(seg_id, score_cache)
        top8 = ranked[:N]

        def norm_of(c, window=window):
            v = window[:, c]
            lo, hi = float(v.min()), float(v.max())
            return (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)

        img = render_heatmap_overlay(window, ranked, N)
        prompt = build_prompt_index_aware(all_channels, top8, window, norm_of)
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
        k_entry = checkpoint.get(seg_id, {})
        i8_entry = i8_pilot_ckpt.get(seg_id, {})
        if gt is None or ck is None or k_entry.get("status") != "OK" or i8_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue

        k_pred = set(k_entry["pred"] or [])
        i8_pred = set(i8_entry["pred"] or [])
        std = float(features[seg_id]["step0_overall_std"]) if seg_id in features else None

        rows.append({
            "segment_id": seg_id, "quartile": quartile, "step0_overall_std": std,
            "f1_I8": f1_of(i8_pred, gt), "f1_K": f1_of(k_pred, gt),
            "fp_I8": len(i8_pred - gt), "fp_K": len(k_pred - gt),
            "n_pred_K": len(k_pred), "hedge_K": len(k_pred) >= HEDGE_THRESHOLD,
            "diff_K_minus_I8": f1_of(k_pred, gt) - f1_of(i8_pred, gt),
        })

    with (OUT_DIR / "pilot_comparison_K.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: I8={mean('f1_I8'):.4f}  K(index-aware)={mean('f1_K'):.4f}")
    print(f"FP: I8={mean('fp_I8'):.2f}  K={mean('fp_K'):.2f}")
    print(f"hedging(K): {sum(1 for r in rows if r['hedge_K'])}/{n}")
    print(f"\nK - I8 = {mean('diff_K_minus_I8'):+.4f}")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qi8 = np.mean([g["f1_I8"] for g in grp])
        qk = np.mean([g["f1_K"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_I8": float(qi8), "f1_K": float(qk)}
        print(f"  {q} (n={len(grp)}): I8={qi8:.3f}  K={qk:.3f}  (K-I8={qk-qi8:+.3f})")

    # ---- 낮은 std 하위군 ----
    stds = [r["step0_overall_std"] for r in rows if r["step0_overall_std"] is not None]
    med_std = float(np.median(stds)) if stds else None
    low_std_rows = [r for r in rows if r["step0_overall_std"] is not None and r["step0_overall_std"] < med_std]
    print(f"\n낮은 std(중앙값 {med_std:.4f} 이하) 하위군 (n={len(low_std_rows)}):")
    if low_std_rows:
        qi8_low = np.mean([r["f1_I8"] for r in low_std_rows])
        qk_low = np.mean([r["f1_K"] for r in low_std_rows])
        print(f"  I8={qi8_low:.3f}  K={qk_low:.3f}  (K-I8={qk_low-qi8_low:+.3f})")

    helps_low_std = low_std_rows and (qk_low - qi8_low) > 0.02
    q23_grp = by_q.get("Q2Q3_중간", [])
    helps_q23 = q23_grp and (np.mean([g["f1_K"] for g in q23_grp]) - np.mean([g["f1_I8"] for g in q23_grp])) > 0.02

    if mean("diff_K_minus_I8") > 0.02:
        verdict = "index-aware 텍스트가 I8 위에서 유의미한 추가 개선을 줌"
    else:
        verdict = "index-aware 텍스트의 전반적 효과는 미미하거나 없음"
    verdict += " / 낮은 std 구간을 메꿈" if helps_low_std else " / 낮은 std 구간을 메꾼다는 근거는 약함"
    verdict += " / Q2Q3(heatmap이 방해였던 구간) 보완함" if helps_q23 else " / Q2Q3 보완 근거는 약함"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n,
        "mean_f1": {"I8": mean("f1_I8"), "K": mean("f1_K")},
        "mean_fp": {"I8": mean("fp_I8"), "K": mean("fp_K")},
        "hedge_K": sum(1 for r in rows if r["hedge_K"]),
        "diff_K_minus_I8": mean("diff_K_minus_I8"),
        "by_quartile": q_summary,
        "low_std_median": med_std,
        "low_std_f1_I8": float(qi8_low) if low_std_rows else None,
        "low_std_f1_K": float(qk_low) if low_std_rows else None,
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_K.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
