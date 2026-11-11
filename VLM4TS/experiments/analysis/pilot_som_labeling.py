"""
파일럿: SoM(Set-of-Mark) 스타일 채널 라벨링(조건 G, 콜 1개) vs 레이아웃 분리(F, 콜 1개)
vs 그룹 분할(canmmlm_naive, 콜 4개).

화면을 나누지 않고 vlm4ts_own과 동일하게 38채널 1장 overlay를 쓰되, 각 채널 라인 끝에
채널 번호를 직접 마킹(SoM)해서 binding을 돕는지 확인. 세그먼트는 pilot_layout_vs_grouping.py
의 18개(gap 3분위 층화)를 그대로 재사용.

1단계: 세그먼트 목록 + 예상 콜 수 보고 후 STOP.
2단계(승인 후): 조건 G 실행.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_som"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
LAYOUT_SUMMARY = BASE / "experiments" / "results_pilot_layout" / "pilot_summary.json"


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"재사용 세그먼트(레이아웃 파일럿과 동일): {len(selected)}개\n")
    by_q = {}
    for s in selected:
        by_q.setdefault(s["quartile"], []).append(s)
    for q, items in by_q.items():
        entities = sorted(set(re.match(r"^(machine-\d+-\d+)_", i["segment_id"]).group(1) for i in items))
        print(f"{q} (n={len(items)}, entity {len(entities)}개: {entities}):")
        for i in items:
            print(f"  {i['segment_id']}  gap={i['gap_normalized']:.4f}")
    print(f"\n예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 G) = {len(selected)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 G 실행)는 별도 실행 필요.")


def render_condition_g(window: np.ndarray, channels: list[int]) -> str:
    """vlm4ts_own과 동일한 38채널 1장 overlay, 라인 끝에 채널 번호 직접 마킹(SoM)."""
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
        ax.text(len(norm) - 1, norm[-1], f"{c}", fontsize=4.5, color=color,
                 va="center", ha="left", clip_on=False)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(right=0.92)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_g(channels: list[int]) -> str:
    return f"""You are shown an overlay plot of {len(channels)} time-series channels from a multivariate industrial system, labeled by channel number: {channels}. Each line is marked with its channel number directly next to its endpoint on the right side -- use these markers to identify each channel. No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern).

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step2():
    import re as re_
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import (  # noqa: E402
        call_vlm, load_smd_test, parse_response, prf,
    )
    from step1v3_dino_graph_smd import WIN, _centered_window  # noqa: E402

    CKPT_PATHS = [
        BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
        BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
        BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
    ]
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    layout_ckpt = json.loads((BASE / "experiments" / "results_pilot_layout" / "checkpoint_F.json").read_text(encoding="utf-8"))

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
    checkpoint_path = OUT_DIR / "checkpoint_G.json"
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

        img = render_condition_g(window, all_channels)
        prompt = build_prompt_g(all_channels)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)
        checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

    # ---- 비교 (A/B/F/G) ----
    rows = []
    for s in selected:
        seg_id, quartile = s["segment_id"], s["quartile"]
        gt = find_gt(seg_id)
        ck = find_ckpt_for(seg_id)
        g_entry = checkpoint.get(seg_id, {})
        f_entry = layout_ckpt.get(seg_id, {})
        if gt is None or ck is None or g_entry.get("status") != "OK" or f_entry.get("status") != "OK":
            continue
        g_pred = set(g_entry["pred"] or [])
        f_pred = set(f_entry["pred"] or [])

        a_entry = ck.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        b_pred, b_ok = set(), True
        for gi in range(4):
            ge = ck.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") == "OK" and ge.get("pred"):
                b_pred |= set(ge["pred"])
            if ge.get("status") != "OK":
                b_ok = False
        if a_pred is None or not b_ok:
            continue

        def f1_fp(pred):
            p, r = prf(pred, gt)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return f1, len(pred - gt)

        f1_a, fp_a = f1_fp(a_pred)
        f1_b, fp_b = f1_fp(b_pred)
        f1_f, fp_f = f1_fp(f_pred)
        f1_g, fp_g = f1_fp(g_pred)
        rows.append({"segment_id": seg_id, "quartile": quartile,
                      "f1_A_vlm4ts_own": f1_a, "f1_B_canmmlm": f1_b, "f1_F_layout": f1_f, "f1_G_som": f1_g,
                      "fp_A": fp_a, "fp_B": fp_b, "fp_F": fp_f, "fp_G": fp_g})

    with (OUT_DIR / "pilot_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a = np.mean([r["f1_A_vlm4ts_own"] for r in rows])
    mean_b = np.mean([r["f1_B_canmmlm"] for r in rows])
    mean_f = np.mean([r["f1_F_layout"] for r in rows])
    mean_g = np.mean([r["f1_G_som"] for r in rows])
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: A(vlm4ts_own,1콜)={mean_a:.4f}  B(canmmlm_naive,4콜)={mean_b:.4f}  "
          f"F(layout,1콜)={mean_f:.4f}  G(SoM,1콜)={mean_g:.4f}")

    gap_a_to_b = mean_b - mean_a
    gap_a_to_g = mean_g - mean_a
    pct_captured_g = (gap_a_to_g / gap_a_to_b * 100) if gap_a_to_b != 0 else None
    print(f"\nA->B 개선폭(4콜): {gap_a_to_b:+.4f}")
    print(f"A->G 개선폭(1콜): {gap_a_to_g:+.4f}")
    print(f"G가 B 개선폭의 {pct_captured_g:.1f}%를 콜 1개로 따라잡음 (F는 23.8%)" if pct_captured_g is not None else "N/A")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qa = np.mean([g["f1_A_vlm4ts_own"] for g in grp])
        qb = np.mean([g["f1_B_canmmlm"] for g in grp])
        qf = np.mean([g["f1_F_layout"] for g in grp])
        qg = np.mean([g["f1_G_som"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_A": float(qa), "f1_B": float(qb), "f1_F": float(qf), "f1_G": float(qg)}
        print(f"  {q} (n={len(grp)}): A={qa:.3f}  B={qb:.3f}  F={qf:.3f}  G={qg:.3f}")

    summary = {
        "n": n,
        "mean_f1": {"A_vlm4ts_own": float(mean_a), "B_canmmlm_naive": float(mean_b),
                    "F_layout": float(mean_f), "G_som": float(mean_g)},
        "calls_per_segment": {"A": 1, "B": 4, "F": 1, "G": 1},
        "improvement_A_to_B": float(gap_a_to_b),
        "improvement_A_to_G": float(gap_a_to_g),
        "pct_of_B_improvement_captured_by_G": pct_captured_g,
        "pct_of_B_improvement_captured_by_F_reference": 23.84125022532845,
        "by_quartile": q_summary,
    }
    (OUT_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
