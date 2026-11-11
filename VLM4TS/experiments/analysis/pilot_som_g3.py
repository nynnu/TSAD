"""
파일럿: SoM 마킹 v3(조건 G3, 콜 1개) -- 라벨을 y값 순서로 왼쪽 여백에 균등 간격 배치 +
leader line 연결 (겹침 구조적으로 불가능, pilot_som_v3_render.py에서 육안 확인 완료).
G/F/A와 동일 18개 세그먼트 재사용, VLM 실제 호출.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_som_g3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
HEDGE_THRESHOLD = 35


def build_prompt_g3(channels: list[int]) -> str:
    return f"""You are shown an overlay plot of {len(channels)} time-series channels from a multivariate industrial system, labeled by channel number: {channels}. Each channel's number is listed in the left margin, connected to its corresponding line's starting point by a thin leader line of the same color. Use these leader lines to identify each channel. No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern).

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step2():
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response, prf
    from step1v3_dino_graph_smd import WIN, _centered_window
    from pilot_som_v3_render import render_condition_g3

    CKPT_PATHS = [
        BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
        BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
        BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
    ]
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    f_ckpt = json.loads((BASE / "experiments" / "results_pilot_layout" / "checkpoint_F.json").read_text(encoding="utf-8"))
    g_ckpt = json.loads((BASE / "experiments" / "results_pilot_som" / "checkpoint_G.json").read_text(encoding="utf-8"))

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
    checkpoint_path = OUT_DIR / "checkpoint_G3.json"
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

        img = render_condition_g3(window, all_channels)
        prompt = build_prompt_g3(all_channels)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)
        checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

    def f1_fp(pred, gt):
        p, r = prf(pred, gt)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1, len(pred - gt)

    rows = []
    for s in selected:
        seg_id, quartile = s["segment_id"], s["quartile"]
        gt = find_gt(seg_id)
        ck = find_ckpt_for(seg_id)
        g3_entry = checkpoint.get(seg_id, {})
        g_entry = g_ckpt.get(seg_id, {})
        f_entry = f_ckpt.get(seg_id, {})
        if gt is None or ck is None or g3_entry.get("status") != "OK" or g_entry.get("status") != "OK" or f_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue

        g3_pred = set(g3_entry["pred"] or [])
        g_pred = set(g_entry["pred"] or [])
        f_pred = set(f_entry["pred"] or [])

        f1_a, fp_a = f1_fp(a_pred, gt)
        f1_f, fp_f = f1_fp(f_pred, gt)
        f1_g, fp_g = f1_fp(g_pred, gt)
        f1_g3, fp_g3 = f1_fp(g3_pred, gt)
        rows.append({
            "segment_id": seg_id, "quartile": quartile,
            "n_pred_A": len(a_pred), "n_pred_G": len(g_pred), "n_pred_G3": len(g3_pred),
            "hedge_A": len(a_pred) >= HEDGE_THRESHOLD,
            "hedge_G": len(g_pred) >= HEDGE_THRESHOLD,
            "hedge_G3": len(g3_pred) >= HEDGE_THRESHOLD,
            "f1_A": f1_a, "f1_F": f1_f, "f1_G": f1_g, "f1_G3": f1_g3,
            "fp_A": fp_a, "fp_F": fp_f, "fp_G": fp_g, "fp_G3": fp_g3,
        })

    with (OUT_DIR / "pilot_comparison_g3.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a = np.mean([r["f1_A"] for r in rows])
    mean_f = np.mean([r["f1_F"] for r in rows])
    mean_g = np.mean([r["f1_G"] for r in rows])
    mean_g3 = np.mean([r["f1_G3"] for r in rows])
    mean_fp_a = np.mean([r["fp_A"] for r in rows])
    mean_fp_g = np.mean([r["fp_G"] for r in rows])
    mean_fp_g3 = np.mean([r["fp_G3"] for r in rows])
    n_hedge_g = sum(1 for r in rows if r["hedge_G"])
    n_hedge_g3 = sum(1 for r in rows if r["hedge_G3"])

    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: A={mean_a:.4f}  F={mean_f:.4f}  G(v1)={mean_g:.4f}  G3={mean_g3:.4f}")
    print(f"mean FP: A={mean_fp_a:.2f}  G(v1)={mean_fp_g:.2f}  G3={mean_fp_g3:.2f}")
    print(f"hedging(>={HEDGE_THRESHOLD}채널) 발생 수: G(v1)={n_hedge_g}/{n}  G3={n_hedge_g3}/{n}")

    prev_new_hedge = ["machine-1-8_2105_2171", "machine-1-6_5167_5172", "machine-3-2_3080_3917", "machine-1-2_4629_4688"]
    print("\n이전 NEW-hedge 4개 세그먼트가 G3에서 정상으로 돌아왔는지:")
    recovery = {}
    for seg_id in prev_new_hedge:
        r = next((r for r in rows if r["segment_id"] == seg_id), None)
        if r is None:
            continue
        recovered = not r["hedge_G3"]
        recovery[seg_id] = {"n_pred_G": r["n_pred_G"], "n_pred_G3": r["n_pred_G3"], "recovered": recovered}
        print(f"  {seg_id}: G n_pred={r['n_pred_G']}  G3 n_pred={r['n_pred_G3']}  -> {'회복' if recovered else '여전히 hedging'}")

    clean_g3 = [r for r in rows if not r["hedge_G3"]]
    mean_a_clean = np.mean([r["f1_A"] for r in clean_g3]) if clean_g3 else None
    mean_g3_clean = np.mean([r["f1_G3"] for r in clean_g3]) if clean_g3 else None
    diff_clean = (mean_g3_clean - mean_a_clean) if clean_g3 else None
    print(f"\nG3 hedging 제외(n={len(clean_g3)}): A={mean_a_clean:.4f}  G3={mean_g3_clean:.4f}  차이={diff_clean:+.4f}" if clean_g3 else "")

    gap_a_to_b_ref = 0.10113287749293615
    pct_g3 = ((mean_g3 - mean_a) / gap_a_to_b_ref * 100)
    print(f"\nG3가 B(4콜) 개선폭의 {pct_g3:.1f}%를 콜 1개로 회복 (참고: G(v1)=52.3%, F=23.8%)")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qa = np.mean([g["f1_A"] for g in grp])
        qf = np.mean([g["f1_F"] for g in grp])
        qg = np.mean([g["f1_G"] for g in grp])
        qg3 = np.mean([g["f1_G3"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_A": float(qa), "f1_F": float(qf), "f1_G": float(qg), "f1_G3": float(qg3)}
        print(f"  {q} (n={len(grp)}): A={qa:.3f}  F={qf:.3f}  G={qg:.3f}  G3={qg3:.3f}")

    if n_hedge_g3 <= 1 and diff_clean is not None and diff_clean > 0.01:
        verdict = "hedging 대부분 해소, hedging 제외 후에도 binding 개선 유의미하게 남음"
    elif diff_clean is not None and diff_clean <= 0.01:
        verdict = "hedging은 줄었으나(또는 안 줄었으나) hedging 제외 시 binding 개선 효과 거의 없음 -- SoM 방향 보류 권고"
    else:
        verdict = "hedging 여전히 다발 -- SoM 방향 보류 권고"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n,
        "mean_f1": {"A": float(mean_a), "F": float(mean_f), "G_v1": float(mean_g), "G3": float(mean_g3)},
        "mean_fp": {"A": float(mean_fp_a), "G_v1": float(mean_fp_g), "G3": float(mean_fp_g3)},
        "hedge_count": {"G_v1": n_hedge_g, "G3": n_hedge_g3},
        "prev_new_hedge_recovery": recovery,
        "g3_clean_n": len(clean_g3),
        "g3_clean_mean_f1": {"A": mean_a_clean, "G3": mean_g3_clean} if clean_g3 else None,
        "g3_clean_diff": diff_clean,
        "pct_of_B_improvement_captured_by_G3": float(pct_g3),
        "by_quartile": q_summary,
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_g3.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run_step2()
