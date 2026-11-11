"""
파일럿: heatmap 제거, top-8 overlay만(조건 J, 콜 1개) -- heatmap이 top-8 컷의
"안전망" 역할을 하는지, top-8 자체가 성능을 다 만드는지 분리.

I8과 동일 18개 세그먼트, I8의 overlay 렌더링 로직 그대로 재사용(heatmap 패널만 제거).
I8 파일럿 값은 results_pilot_heatmap_nsweep/checkpoint_I8.json에서 그대로 재사용(재실행 없음).

1단계: 세그먼트/콜 수(18) 보고 후 STOP.
2단계(승인 후): 조건 J 18콜 실행 + A/B/I8 대비 비교.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_top8_only"
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


def render_top8_only(window: np.ndarray, top_n: list[int]) -> str:
    """I8의 overlay 패널과 완전히 동일한 렌더링, heatmap 없이 단독 1장."""
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    cmap = plt.get_cmap("tab10" if len(top_n) <= 10 else "tab20")
    for i, c in enumerate(top_n):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(len(norm)), norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax.legend(fontsize=6, ncol=min(6, len(top_n)), loc="upper right")
    ax.set_title(f"Overlay: top-{len(top_n)} candidate channels", fontsize=8)
    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_top8_only(top_n: list[int]) -> str:
    return f"""You are shown an overlay plot of the top-{len(top_n)} candidate channels ({top_n}) from a multivariate industrial system, labeled by channel number. No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern).

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {top_n} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"재사용 세그먼트(I8 파일럿과 동일): {len(selected)}개\n")
    by_q = {}
    for s in selected:
        by_q.setdefault(s["quartile"], []).append(s)
    for q, items in by_q.items():
        print(f"{q} (n={len(items)}): {[i['segment_id'] for i in items]}")
    print(f"\n예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 J) = {len(selected)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 J 실행)는 별도 실행 필요.")


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
    checkpoint_path = OUT_DIR / "checkpoint_J.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
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
        img = render_top8_only(window, top8)
        prompt = build_prompt_top8_only(top8)
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
        j_entry = checkpoint.get(seg_id, {})
        i8_entry = i8_pilot_ckpt.get(seg_id, {})
        if gt is None or ck is None or j_entry.get("status") != "OK" or i8_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
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

        j_pred = set(j_entry["pred"] or [])
        i8_pred = set(i8_entry["pred"] or [])
        std = float(features[seg_id]["step0_overall_std"]) if seg_id in features else None

        rows.append({
            "segment_id": seg_id, "quartile": quartile, "step0_overall_std": std,
            "f1_A": f1_of(a_pred, gt), "f1_B": f1_of(b_pred, gt),
            "f1_I8": f1_of(i8_pred, gt), "f1_J": f1_of(j_pred, gt),
            "fp_A": len(a_pred - gt), "fp_B": len(b_pred - gt),
            "fp_I8": len(i8_pred - gt), "fp_J": len(j_pred - gt),
            "n_pred_J": len(j_pred), "hedge_J": len(j_pred) >= HEDGE_THRESHOLD,
        })

    with (OUT_DIR / "pilot_comparison_J.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: A={mean('f1_A'):.4f}  B={mean('f1_B'):.4f}  I8={mean('f1_I8'):.4f}  J(top8만)={mean('f1_J'):.4f}")
    print(f"FP: A={mean('fp_A'):.2f}  B={mean('fp_B'):.2f}  I8={mean('fp_I8'):.2f}  J={mean('fp_J'):.2f}")
    print(f"hedging(J): {sum(1 for r in rows if r['hedge_J'])}/{n}")

    delta_j_i8 = mean("f1_J") - mean("f1_I8")
    print(f"\nJ - I8 = {delta_j_i8:+.4f} (heatmap 제거로 인한 손실)")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qi8 = np.mean([g["f1_I8"] for g in grp])
        qj = np.mean([g["f1_J"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_I8": float(qi8), "f1_J": float(qj)}
        print(f"  {q} (n={len(grp)}): I8={qi8:.3f}  J={qj:.3f}  (J-I8={qj-qi8:+.3f})")

    # ---- 낮은 std(std 중앙값 이하) 하위군에서 J vs I8 ----
    stds = [r["step0_overall_std"] for r in rows if r["step0_overall_std"] is not None]
    med_std = np.median(stds) if stds else None
    low_std_rows = [r for r in rows if r["step0_overall_std"] is not None and r["step0_overall_std"] < med_std]
    print(f"\n낮은 std(중앙값 {med_std:.4f} 이하) 하위군 (n={len(low_std_rows)}):")
    if low_std_rows:
        qi8_low = np.mean([r["f1_I8"] for r in low_std_rows])
        qj_low = np.mean([r["f1_J"] for r in low_std_rows])
        print(f"  I8={qi8_low:.3f}  J={qj_low:.3f}  (J-I8={qj_low-qi8_low:+.3f})")

    verdict = "heatmap이 실제 안전망 역할을 함" if delta_j_i8 < -0.02 else "top-8 자체가 성능 대부분을 만듦, heatmap 기여 미미"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n,
        "mean_f1": {"A": mean("f1_A"), "B": mean("f1_B"), "I8": mean("f1_I8"), "J": mean("f1_J")},
        "mean_fp": {"A": mean("fp_A"), "B": mean("fp_B"), "I8": mean("fp_I8"), "J": mean("fp_J")},
        "hedge_J": sum(1 for r in rows if r["hedge_J"]),
        "delta_J_minus_I8": delta_j_i8,
        "by_quartile": q_summary,
        "low_std_median": med_std,
        "low_std_f1_I8": float(qi8_low) if low_std_rows else None,
        "low_std_f1_J": float(qj_low) if low_std_rows else None,
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_J.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
