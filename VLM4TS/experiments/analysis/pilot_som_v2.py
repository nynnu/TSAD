"""
파일럿: SoM 마킹 렌더링 수정(조건 G2, 콜 1개) -- G와 동일하되 라벨 위치/크기/가독성만 변경.

직전 G에서 라벨이 선 끝점(밀집 구간)에 4.5pt로 몰려 판독 불가 -> hedging 유발이 의심됨.
G2: 라벨을 선 시작점(왼쪽, 덜 밀집)에 8pt+ 흰 배경 박스로 배치. 그 외(overlay, 색상,
프롬프트)는 G와 완전히 동일. 세그먼트도 동일 18개 재사용.

1단계: hedging 발생했던 4개 세그먼트 렌더링 육안 확인 (VLM 호출 전).
2단계(확인 후): 18콜 실행 + G/F/A 대비 비교.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_som_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
HEDGE_THRESHOLD = 35
PREV_NEW_HEDGE_SEGMENTS = [
    "machine-1-8_2105_2171", "machine-1-6_5167_5172",
    "machine-3-2_3080_3917", "machine-1-2_4629_4688",
]


def render_condition_g2(window: np.ndarray, channels: list[int]) -> str:
    """G와 동일한 38채널 1장 overlay, 라벨만 선 시작점(왼쪽)에 8pt+흰 배경 박스로 배치."""
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    cmap = plt.get_cmap("tab10" if len(channels) <= 10 else "tab20")
    for i, c in enumerate(channels):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        color = cmap(i % cmap.N)
        ax.plot(np.arange(len(norm)), norm, linewidth=0.8, color=color)
        ax.text(0, norm[0], f"{c}", fontsize=8, color=color, fontweight="bold",
                 va="center", ha="right", clip_on=False,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.13)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_g2(channels: list[int]) -> str:
    return f"""You are shown an overlay plot of {len(channels)} time-series channels from a multivariate industrial system, labeled by channel number: {channels}. Each line is marked with its channel number directly next to its starting point on the left side -- use these markers to identify each channel. No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern).

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def _load_window(seg_id):
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import load_smd_test
    from step1v3_dino_graph_smd import WIN, _centered_window
    m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
    entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
    test = load_smd_test(entity)
    center = (cs + ce) // 2
    s_, e_ = _centered_window(len(test), center, WIN)
    return test[s_:e_]


def run_render_check():
    scratch = Path(r"C:\Users\김나영\AppData\Local\Temp\claude\C--Windows-System32\f3e23650-e2c0-4e74-815d-1f8a0efaa543\scratchpad")
    scratch.mkdir(parents=True, exist_ok=True)
    import base64
    print("[렌더링 확인] hedging 발생했던 4개 세그먼트, G2 렌더링 저장 중...")
    for seg_id in PREV_NEW_HEDGE_SEGMENTS:
        window = _load_window(seg_id)
        img_b64 = render_condition_g2(window, list(range(38)))
        out_path = scratch / f"som_v2_{seg_id}.png"
        out_path.write_bytes(base64.b64decode(img_b64))
        print(f"  saved: {out_path}")
    print("\n[STOP] 렌더링 이미지 저장 완료 -- 육안 확인 후 보고, 승인되면 run_step1/step2 진행.")


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"재사용 세그먼트(G/F 파일럿과 동일): {len(selected)}개")
    print(f"예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 G2) = {len(selected)}콜")
    print("\n[STOP] 승인 대기. 2단계(조건 G2 실행)는 별도 실행 필요.")


def run_step2():
    import re as re_
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response, prf
    from step1v3_dino_graph_smd import WIN, _centered_window

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
    checkpoint_path = OUT_DIR / "checkpoint_G2.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(38))
    entity_test = {}

    for s in selected:
        seg_id = s["segment_id"]
        m = re_.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
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

        img = render_condition_g2(window, all_channels)
        prompt = build_prompt_g2(all_channels)
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
        g2_entry = checkpoint.get(seg_id, {})
        g_entry = g_ckpt.get(seg_id, {})
        f_entry = f_ckpt.get(seg_id, {})
        if gt is None or ck is None or g2_entry.get("status") != "OK" or g_entry.get("status") != "OK" or f_entry.get("status") != "OK":
            continue
        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue

        g2_pred = set(g2_entry["pred"] or [])
        g_pred = set(g_entry["pred"] or [])
        f_pred = set(f_entry["pred"] or [])

        f1_a, fp_a = f1_fp(a_pred, gt)
        f1_f, fp_f = f1_fp(f_pred, gt)
        f1_g, fp_g = f1_fp(g_pred, gt)
        f1_g2, fp_g2 = f1_fp(g2_pred, gt)
        rows.append({
            "segment_id": seg_id, "quartile": quartile,
            "n_pred_A": len(a_pred), "n_pred_G": len(g_pred), "n_pred_G2": len(g2_pred),
            "hedge_A": len(a_pred) >= HEDGE_THRESHOLD,
            "hedge_G": len(g_pred) >= HEDGE_THRESHOLD,
            "hedge_G2": len(g2_pred) >= HEDGE_THRESHOLD,
            "f1_A": f1_a, "f1_F": f1_f, "f1_G": f1_g, "f1_G2": f1_g2,
            "fp_A": fp_a, "fp_F": fp_f, "fp_G": fp_g, "fp_G2": fp_g2,
        })

    with (OUT_DIR / "pilot_comparison_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a = np.mean([r["f1_A"] for r in rows])
    mean_f = np.mean([r["f1_F"] for r in rows])
    mean_g = np.mean([r["f1_G"] for r in rows])
    mean_g2 = np.mean([r["f1_G2"] for r in rows])
    n_hedge_g = sum(1 for r in rows if r["hedge_G"])
    n_hedge_g2 = sum(1 for r in rows if r["hedge_G2"])

    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: A={mean_a:.4f}  F={mean_f:.4f}  G(v1)={mean_g:.4f}  G2(v2)={mean_g2:.4f}")
    print(f"hedging(>={HEDGE_THRESHOLD}채널) 발생 수: G(v1)={n_hedge_g}/{n}  G2(v2)={n_hedge_g2}/{n}")

    print("\n이전 NEW-hedge 4개 세그먼트가 G2에서 정상으로 돌아왔는지:")
    prev_recovery = {}
    for seg_id in PREV_NEW_HEDGE_SEGMENTS:
        r = next((r for r in rows if r["segment_id"] == seg_id), None)
        if r is None:
            continue
        recovered = not r["hedge_G2"]
        prev_recovery[seg_id] = {"n_pred_G": r["n_pred_G"], "n_pred_G2": r["n_pred_G2"], "recovered": recovered}
        print(f"  {seg_id}: G n_pred={r['n_pred_G']}  G2 n_pred={r['n_pred_G2']}  -> {'회복' if recovered else '여전히 hedging'}")

    # hedging 제외 clean 비교 (v1과 동일 방식으로 v2도 재확인)
    clean_g2 = [r for r in rows if not r["hedge_G2"]]
    mean_a_clean = np.mean([r["f1_A"] for r in clean_g2]) if clean_g2 else None
    mean_g2_clean = np.mean([r["f1_G2"] for r in clean_g2]) if clean_g2 else None
    diff_clean = (mean_g2_clean - mean_a_clean) if clean_g2 else None
    print(f"\nG2 hedging 제외(n={len(clean_g2)}): A={mean_a_clean:.4f}  G2={mean_g2_clean:.4f}  차이={diff_clean:+.4f}" if clean_g2 else "")

    gap_a_to_b_ref = 0.10113287749293615  # B(canmmlm_naive) 개선폭, 이전 파일럿 값
    pct_g2 = ((mean_g2 - mean_a) / gap_a_to_b_ref * 100)
    print(f"\nG2가 B(4콜) 개선폭의 {pct_g2:.1f}%를 콜 1개로 회복 (참고: G(v1)=52.3%, F=23.8%)")

    if n_hedge_g2 <= 1 and diff_clean is not None and diff_clean > 0.01:
        verdict = "렌더링 수정 후 hedging 대부분 해소, binding 개선 효과 유의미하게 남음"
    elif n_hedge_g2 >= n_hedge_g - 1:
        verdict = "렌더링을 고쳐도 hedging이 여전히 다발 -- SoM 방향 보류 (라벨 밀집이 근본 원인이 아니었거나, 다른 요인 존재)"
    else:
        verdict = "hedging은 일부 줄었으나 binding 개선 효과는 미미 -- SoM 방향 보류 권고"
    print(f"\n판정: {verdict}")

    summary = {
        "n": n, "mean_f1": {"A": float(mean_a), "F": float(mean_f), "G_v1": float(mean_g), "G2_v2": float(mean_g2)},
        "hedge_count": {"G_v1": n_hedge_g, "G2_v2": n_hedge_g2},
        "prev_new_hedge_recovery": prev_recovery,
        "g2_clean_n": len(clean_g2),
        "g2_clean_mean_f1": {"A": mean_a_clean, "G2": mean_g2_clean} if clean_g2 else None,
        "g2_clean_diff": diff_clean,
        "pct_of_B_improvement_captured_by_G2": float(pct_g2),
        "verdict": verdict,
    }
    (OUT_DIR / "pilot_summary_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "render_check":
        run_render_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
