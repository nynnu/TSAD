"""
파일럿: heatmap(38채널 전체, Step0 점수 순 front-loading) + overlay(상위 N채널) 컴포짓,
콜 1개 x 3버전(N=4/8/12). 레이아웃분리(F)/SoM(G3)와 동일 18개 세그먼트 재사용.

heatmap은 imshow라 행이 자동으로 균등 간격 -- G3에서 겪은 라벨 겹침 문제가 구조적으로
없음. overlay 패널은 기존 검증된 render_overlay 방식 그대로 재사용.

1단계: 세그먼트/콜 수(18x3=54) 보고 후 STOP.
2단계(승인 후): I4/I8/I12 각 18콜씩 실행 + A/B/F/G3 대비 비교.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_heatmap_nsweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
F_CKPT = BASE / "experiments" / "results_pilot_layout" / "checkpoint_F.json"
G3_CKPT = BASE / "experiments" / "results_pilot_som_g3" / "checkpoint_G3.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]
N_VALUES = [4, 8, 12]
HEDGE_THRESHOLD = 35


def get_ranked_channels(seg_id, score_cache):
    scores = {int(c): v for c, v in score_cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def render_heatmap_overlay(window: np.ndarray, ranked: list[int], n: int) -> str:
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100, gridspec_kw={"height_ratios": [1.4, 1]})

    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title(f"Heatmap: 38 channels, sorted by Step0 score (top row = highest score)", fontsize=7)

    top_n = ranked[:n]
    cmap = plt.get_cmap("tab10" if len(top_n) <= 10 else "tab20")
    for i, c in enumerate(top_n):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax2.plot(np.arange(len(norm)), norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax2.legend(fontsize=5, ncol=min(6, len(top_n)), loc="upper right")
    ax2.set_xticks([])
    ax2.set_title(f"Overlay: top-{n} candidate channels (detail)", fontsize=7)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_heatmap_overlay(all_channels: list[int], top_n: list[int]) -> str:
    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled.

Use both panels together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {all_channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step1():
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"재사용 세그먼트(F/G3 파일럿과 동일): {len(selected)}개\n")
    by_q = {}
    for s in selected:
        by_q.setdefault(s["quartile"], []).append(s)
    for q, items in by_q.items():
        print(f"{q} (n={len(items)}): {[i['segment_id'] for i in items]}")
    print(f"\nN 스윕: {N_VALUES}")
    print(f"예상 콜 수: {len(selected)}개 세그먼트 x {len(N_VALUES)}버전(I4/I8/I12) = {len(selected)*len(N_VALUES)}콜")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 I4/I8/I12 실행)는 별도 실행 필요.")


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
    f_ckpt = json.loads(F_CKPT.read_text(encoding="utf-8"))
    g3_ckpt = json.loads(G3_CKPT.read_text(encoding="utf-8"))
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
    all_channels = list(range(38))
    entity_test = {}

    checkpoints = {}
    for n in N_VALUES:
        cpath = OUT_DIR / f"checkpoint_I{n}.json"
        checkpoints[n] = {"path": cpath, "data": json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}}

    for s in selected:
        seg_id = s["segment_id"]
        m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
        entity, cs, ce = m.group(1), int(m.group(2)), int(m.group(3))
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]
        ranked = get_ranked_channels(seg_id, score_cache)

        for n in N_VALUES:
            ck = checkpoints[n]["data"]
            if ck.get(seg_id, {}).get("status") == "OK":
                print(f"[SKIP] I{n} {seg_id}")
                continue
            top_n = ranked[:n]
            img = render_heatmap_overlay(window, ranked, n)
            prompt = build_prompt_heatmap_overlay(all_channels, top_n)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            ck[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw, "n": n}
            checkpoints[n]["path"].write_text(json.dumps(ck, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{ck[seg_id]['status']}] I{n} {seg_id}: pred={pred}", flush=True)

    # ---- 비교 ----
    rows = []
    for s in selected:
        seg_id, quartile = s["segment_id"], s["quartile"]
        gt = find_gt(seg_id)
        ckt = find_ckpt_for(seg_id)
        f_entry = f_ckpt.get(seg_id, {})
        g3_entry = g3_ckpt.get(seg_id, {})
        if gt is None or ckt is None or f_entry.get("status") != "OK" or g3_entry.get("status") != "OK":
            continue
        a_entry = ckt.get(f"{seg_id}_A_vlm4ts_own", {})
        a_pred = set(a_entry["pred"] or []) if a_entry.get("status") == "OK" else None
        if a_pred is None:
            continue
        b_pred, b_ok = set(), True
        for gi in range(4):
            ge = ckt.get(f"{seg_id}_B_group{gi}", {})
            if ge.get("status") == "OK" and ge.get("pred"):
                b_pred |= set(ge["pred"])
            if ge.get("status") != "OK":
                b_ok = False
        if not b_ok:
            continue

        f_pred = set(f_entry["pred"] or [])
        g3_pred = set(g3_entry["pred"] or [])

        row = {
            "segment_id": seg_id, "quartile": quartile,
            "f1_A": f1_of(a_pred, gt), "f1_B": f1_of(b_pred, gt),
            "f1_F": f1_of(f_pred, gt), "f1_G3": f1_of(g3_pred, gt),
            "fp_A": len(a_pred - gt), "fp_B": len(b_pred - gt),
        }
        ok_all_n = True
        for n in N_VALUES:
            entry = checkpoints[n]["data"].get(seg_id, {})
            if entry.get("status") != "OK":
                ok_all_n = False
                break
            pred = set(entry["pred"] or [])
            row[f"f1_I{n}"] = f1_of(pred, gt)
            row[f"fp_I{n}"] = len(pred - gt)
            row[f"n_pred_I{n}"] = len(pred)
            row[f"hedge_I{n}"] = len(pred) >= HEDGE_THRESHOLD
        if not ok_all_n:
            continue
        rows.append(row)

    with (OUT_DIR / "pilot_comparison_I.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_seg = len(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n_seg})")
    print("=" * 70)
    print(f"F1: A={mean('f1_A'):.4f}  B={mean('f1_B'):.4f}  F={mean('f1_F'):.4f}  G3={mean('f1_G3'):.4f}  "
          + "  ".join(f"I{n}={mean(f'f1_I{n}'):.4f}" for n in N_VALUES))
    print(f"FP: A={mean('fp_A'):.2f}  B={mean('fp_B'):.2f}  " + "  ".join(f"I{n}={mean(f'fp_I{n}'):.2f}" for n in N_VALUES))
    print("hedging(>=35채널) 발생 수: " + "  ".join(f"I{n}={sum(1 for r in rows if r[f'hedge_I{n}'])}/{n_seg}" for n in N_VALUES))

    gap_a_to_b = mean("f1_B") - mean("f1_A")
    print(f"\nA->B 개선폭(4콜): {gap_a_to_b:+.4f}")
    best_n, best_pct, best_f1 = None, -1e9, -1e9
    pct_by_n = {}
    for n in N_VALUES:
        gap = mean(f"f1_I{n}") - mean("f1_A")
        pct = gap / gap_a_to_b * 100 if gap_a_to_b != 0 else None
        pct_by_n[n] = pct
        print(f"I{n} 개선폭: {gap:+.4f}  B 개선폭의 {pct:.1f}% 회복 (참고: F=23.8%, G3=-52.2%)")
        if mean(f"f1_I{n}") > best_f1:
            best_f1, best_n, best_pct = mean(f"f1_I{n}"), n, pct

    print(f"\nN 간 편차(민감도): I4~I12 F1 범위 = {max(mean(f'f1_I{n}') for n in N_VALUES) - min(mean(f'f1_I{n}') for n in N_VALUES):.4f}")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        line = f"  {q} (n={len(grp)}): A={np.mean([g['f1_A'] for g in grp]):.3f}  B={np.mean([g['f1_B'] for g in grp]):.3f}"
        qs = {"n": len(grp), "f1_A": float(np.mean([g["f1_A"] for g in grp])), "f1_B": float(np.mean([g["f1_B"] for g in grp]))}
        for n in N_VALUES:
            v = float(np.mean([g[f"f1_I{n}"] for g in grp]))
            qs[f"f1_I{n}"] = v
            line += f"  I{n}={v:.3f}"
        q_summary[q] = qs
        print(line)

    summary = {
        "n": n_seg,
        "mean_f1": {"A": mean("f1_A"), "B": mean("f1_B"), "F": mean("f1_F"), "G3": mean("f1_G3"),
                    **{f"I{n}": mean(f"f1_I{n}") for n in N_VALUES}},
        "mean_fp": {"A": mean("fp_A"), "B": mean("fp_B"), **{f"I{n}": mean(f"fp_I{n}") for n in N_VALUES}},
        "hedge_count": {f"I{n}": sum(1 for r in rows if r[f"hedge_I{n}"]) for n in N_VALUES},
        "pct_of_B_improvement_captured": pct_by_n,
        "best_n": best_n, "best_n_pct_captured": best_pct,
        "n_sensitivity_f1_range": max(mean(f"f1_I{n}") for n in N_VALUES) - min(mean(f"f1_I{n}") for n in N_VALUES),
        "by_quartile": q_summary,
    }
    (OUT_DIR / "pilot_summary_I.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        run_step2()
    else:
        run_step1()
