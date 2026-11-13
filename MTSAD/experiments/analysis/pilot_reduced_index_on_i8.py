"""
파일럿: I8(heatmap+overlay top-8)에 축소된 (index,value) 정보 추가(조건 K2, 콜 1개).
채널당 224개 전체가 아니라, train 대비 |z-score| 상위 20~30개 지점만 텍스트로 추가
(8채널 합쳐도 ~200개 안팎, L/K보다 훨씬 적은 텍스트량) -- "정보 과잉이 원인이었나"
검증용 대조 조건.

동일 18개 세그먼트(gap 3분위 층화) 재사용.

1단계: 세그먼트/콜 수(18) 보고 후 STOP.
2단계(승인 후): 조건 K2 18콜 실행 + I8 대비 비교, 특히 Q4에서 L과 다른지 확인.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_reduced_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
FEATURES_CSV = BASE / "experiments" / "results_adaptive_vs_fixed" / "segment_features.csv"
I8_PILOT_CKPT = BASE / "experiments" / "results_pilot_heatmap_nsweep" / "checkpoint_I8.json"
L_SUMMARY = BASE / "experiments" / "results_pilot_score_annotation" / "pilot_summary_L.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
N = 8
N_POINTS_PER_CHANNEL = 25
HEDGE_THRESHOLD = 35


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def build_prompt_reduced_index(all_channels: list[int], top_n: list[int], window, train, n_points: int) -> str:
    blocks = []
    for i, c in enumerate(top_n):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.argsort(-z)[:n_points]
        top_idx = np.sort(top_idx)  # 시간순 정렬

        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{n_points} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled.

For each of the top-{len(top_n)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels and this point data together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

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
    print(f"\n채널당 {N_POINTS_PER_CHANNEL}포인트 x 8채널 = 약 {N_POINTS_PER_CHANNEL*N}쌍/세그먼트")
    print(f"예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 K2) = {len(selected)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 K2 실행)는 별도 실행 필요.")


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
    from step1v3_dino_graph_smd import WIN, _centered_window, load_smd
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
    checkpoint_path = OUT_DIR / "checkpoint_K2.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(38))
    entity_test, entity_train = {}, {}

    for s in selected:
        seg_id = s["segment_id"]
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        if checkpoint.get(seg_id, {}).get("status") == "OK":
            print(f"[SKIP] {seg_id}")
            continue

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        test = entity_test[entity]
        train = entity_train[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        ranked = get_ranked_channels(seg_id, score_cache)
        top8 = ranked[:N]

        img = render_heatmap_overlay(window, ranked, N)
        prompt = build_prompt_reduced_index(all_channels, top8, window, train, N_POINTS_PER_CHANNEL)
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
        k2_entry = checkpoint.get(seg_id, {})
        i8_entry = i8_pilot_ckpt.get(seg_id, {})
        if gt is None or ck is None or k2_entry.get("status") != "OK" or i8_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue

        k2_pred = set(k2_entry["pred"] or [])
        i8_pred = set(i8_entry["pred"] or [])
        std = float(features[seg_id]["step0_overall_std"]) if seg_id in features else None

        rows.append({
            "segment_id": seg_id, "quartile": quartile, "step0_overall_std": std,
            "f1_I8": f1_of(i8_pred, gt), "f1_K2": f1_of(k2_pred, gt),
            "fp_I8": len(i8_pred - gt), "fp_K2": len(k2_pred - gt),
            "n_pred_K2": len(k2_pred), "hedge_K2": len(k2_pred) >= HEDGE_THRESHOLD,
            "diff_K2_minus_I8": f1_of(k2_pred, gt) - f1_of(i8_pred, gt),
        })

    with (OUT_DIR / "pilot_comparison_K2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: I8={mean('f1_I8'):.4f}  K2(축소 index)={mean('f1_K2'):.4f}")
    print(f"FP: I8={mean('fp_I8'):.2f}  K2={mean('fp_K2'):.2f}")
    print(f"hedging(K2): {sum(1 for r in rows if r['hedge_K2'])}/{n}")
    print(f"\nK2 - I8 = {mean('diff_K2_minus_I8'):+.4f}")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qi8 = np.mean([g["f1_I8"] for g in grp])
        qk2 = np.mean([g["f1_K2"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_I8": float(qi8), "f1_K2": float(qk2)}
        print(f"  {q} (n={len(grp)}): I8={qi8:.3f}  K2={qk2:.3f}  (K2-I8={qk2-qi8:+.3f})")

    # ---- L(점수 병기)과 Q4 직접 대조 ----
    l_summary = json.loads(L_SUMMARY.read_text(encoding="utf-8")) if L_SUMMARY.exists() else None
    if l_summary:
        l_q4 = l_summary["by_quartile"]["Q4_뚜렷"]
        l_q4_diff = l_q4["f1_L"] - l_q4["f1_I8"]
        k2_q4_diff = q_summary.get("Q4_뚜렷", {}).get("f1_K2", 0) - q_summary.get("Q4_뚜렷", {}).get("f1_I8", 0)
        print(f"\nQ4 대조: L(224pt/ch)의 I8 대비 변화={l_q4_diff:+.3f}  vs  K2(25pt/ch)의 I8 대비 변화={k2_q4_diff:+.3f}")
        same_direction = (l_q4_diff < 0) == (k2_q4_diff < 0)
        print(f"같은 방향(둘 다 악화 또는 둘 다 개선): {same_direction}")

    if mean("diff_K2_minus_I8") > 0.02:
        verdict = "축소된 정보량에서는 L과 달리 개선됨 -- 정보 과잉이 L 실패의 원인이었다는 가설 지지"
    elif mean("diff_K2_minus_I8") < -0.02:
        verdict = "K2도 실패 -- 정보량을 줄여도 안 됨. 텍스트로 값 정보를 추가하는 것 자체가 이 구조(heatmap+overlay)와 안 맞는다는 더 강한 결론"
    else:
        verdict = "K2는 I8과 거의 동일 -- 텍스트 값 정보 추가가 이 구조에서는 득도 실도 없음"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n,
        "mean_f1": {"I8": mean("f1_I8"), "K2": mean("f1_K2")},
        "mean_fp": {"I8": mean("fp_I8"), "K2": mean("fp_K2")},
        "hedge_K2": sum(1 for r in rows if r["hedge_K2"]),
        "diff_K2_minus_I8": mean("diff_K2_minus_I8"),
        "by_quartile": q_summary,
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_K2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
