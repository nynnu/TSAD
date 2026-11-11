"""
파일럿: 레이아웃 분리(조건 F, 콜 1개) vs 그룹 분할(canmmlm_naive, 콜 4개).

1단계: gap 3분위 층화추출로 15~18개 세그먼트 확정, 예상 콜 수 보고 후 STOP.
2단계(승인 후): 조건 F 실행.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
OUT_DIR = BASE / "experiments" / "results_pilot_layout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAP_DETAIL_CSV = BASE / "experiments" / "results_gap_quartile" / "gap_quartile_detail.csv"
N_PER_QUARTILE = 6
SEED = 7


def select_pilot_segments():
    rows = list(csv.DictReader(GAP_DETAIL_CSV.open(encoding="utf-8")))
    for r in rows:
        r["gap_normalized"] = float(r["gap_normalized"])
    rows.sort(key=lambda r: r["gap_normalized"])
    n = len(rows)
    q1_cut = n // 4
    q3_cut = n - n // 4
    quartiles = {
        "Q1_밀집": rows[:q1_cut],
        "Q2Q3_중간": rows[q1_cut:q3_cut],
        "Q4_뚜렷": rows[q3_cut:],
    }

    rng = np.random.default_rng(SEED)
    selected = []
    for name, grp in quartiles.items():
        # entity 추출 (segment_id에서)
        entity_of = {r["segment_id"]: re.match(r"^(machine-\d+-\d+)_", r["segment_id"]).group(1) for r in grp}
        by_entity = {}
        for r in grp:
            by_entity.setdefault(entity_of[r["segment_id"]], []).append(r)
        entities = sorted(by_entity.keys())
        rng.shuffle(entities)
        picked = []
        ei = 0
        # 라운드로빈으로 entity 다양성 확보하며 N_PER_QUARTILE개 채움
        while len(picked) < min(N_PER_QUARTILE, len(grp)) and entities:
            e = entities[ei % len(entities)]
            pool = by_entity[e]
            if pool:
                idx = rng.integers(0, len(pool))
                picked.append(pool.pop(idx))
            if not pool:
                entities.remove(e)
                if not entities:
                    break
                ei -= 1  # stay at same index after removal shift
            ei += 1
        for p in picked:
            selected.append({"segment_id": p["segment_id"], "gap_normalized": p["gap_normalized"], "quartile": name})
    return selected


def run_step1():
    selected = select_pilot_segments()
    print(f"파일럿 선정 세그먼트: {len(selected)}개\n")
    by_q = {}
    for s in selected:
        by_q.setdefault(s["quartile"], []).append(s)
    for q, items in by_q.items():
        entities = sorted(set(re.match(r"^(machine-\d+-\d+)_", i["segment_id"]).group(1) for i in items))
        print(f"{q} (n={len(items)}, entity {len(entities)}개: {entities}):")
        for i in items:
            print(f"  {i['segment_id']}  gap={i['gap_normalized']:.4f}")
    print(f"\n예상 콜 수: {len(selected)}개 세그먼트 x 1콜(조건 F) = {len(selected)}콜")

    (OUT_DIR / "pilot_selection.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved selection: {OUT_DIR / 'pilot_selection.json'}")
    print("\n[STOP] 1단계 완료 -- 승인 대기. 2단계(조건 F 실행)는 별도 실행 필요.")


def render_condition_f(window: np.ndarray, groups: list[list[int]]) -> str:
    """38채널 전부, 2x2 서브플롯 그리드로 배치 -- 각 사분면이 canmmlm_naive의
    한 그룹(9~10채널)을 overlay로 보여줌. 채널을 자르지 않고 레이아웃만 분리."""
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=100)
    cmap = plt.get_cmap("tab10")
    for gi, group in enumerate(groups):
        ax = axes[gi // 2, gi % 2]
        for i, c in enumerate(group):
            v = window[:, c]
            lo, hi = float(v.min()), float(v.max())
            norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
            ax.plot(norm, linewidth=0.7, color=cmap(i % cmap.N), label=f"Ch{c}")
        ax.legend(fontsize=4, ncol=min(6, len(group)), loc="upper right")
        ax.set_title(f"구역 {gi}: 채널 {group[0]}-{group[-1]}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_f(groups: list[list[int]]) -> str:
    all_channels = sorted(c for g in groups for c in g)
    layout_desc = "; ".join(f"구역{gi}={g}" for gi, g in enumerate(groups))
    return f"""You are shown an overlay plot of {len(all_channels)} time-series channels from a multivariate industrial system, labeled by channel number: {all_channels}. The image is divided into 4 quadrants; each quadrant shows a different, non-overlapping subset of channels ({layout_desc}). No ground truth or hints are given.

Identify which of these channels show anomalous or deviating behavior in this window (compared to the other channels / the general pattern), across all 4 quadrants.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {all_channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run_step2():
    import re as re_
    import sys
    sys.path.insert(0, str(ANALYSIS_DIR))
    from smd_3way_baseline_comparison import (  # noqa: E402
        N_GROUPS, call_vlm, load_smd_test, naive_groups, parse_response, prf,
    )
    from step1v3_dino_graph_smd import WIN, _centered_window  # noqa: E402

    CKPT_PATHS = [
        BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
        BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
        BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
    ]
    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]

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

    selected = json.loads((OUT_DIR / "pilot_selection.json").read_text(encoding="utf-8"))
    checkpoint_path = OUT_DIR / "checkpoint_F.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    groups_layout = naive_groups(38, N_GROUPS)
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

        img = render_condition_f(window, groups_layout)
        prompt = build_prompt_f(groups_layout)
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
        f_entry = checkpoint.get(seg_id, {})
        if gt is None or ck is None or f_entry.get("status") != "OK":
            continue
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
        rows.append({"segment_id": seg_id, "quartile": quartile,
                      "f1_A_vlm4ts_own": f1_a, "f1_B_canmmlm": f1_b, "f1_F_layout": f1_f,
                      "fp_A": fp_a, "fp_B": fp_b, "fp_F": fp_f})

    with (OUT_DIR / "pilot_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    mean_a = np.mean([r["f1_A_vlm4ts_own"] for r in rows])
    mean_b = np.mean([r["f1_B_canmmlm"] for r in rows])
    mean_f = np.mean([r["f1_F_layout"] for r in rows])
    print("\n" + "=" * 70)
    print(f"파일럿 결과 (n={n})")
    print("=" * 70)
    print(f"F1: vlm4ts_own(A)={mean_a:.4f}  canmmlm_naive(B, 4콜)={mean_b:.4f}  layout(F, 1콜)={mean_f:.4f}")

    gap_a_to_b = mean_b - mean_a
    gap_a_to_f = mean_f - mean_a
    pct_captured = (gap_a_to_f / gap_a_to_b * 100) if gap_a_to_b != 0 else None
    print(f"\nA->B 개선폭(4콜): {gap_a_to_b:+.4f}")
    print(f"A->F 개선폭(1콜): {gap_a_to_f:+.4f}")
    print(f"F가 B 개선폭의 {pct_captured:.1f}%를 콜 1개로 따라잡음" if pct_captured is not None else "N/A")

    print("\n분위별:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quartile"], []).append(r)
    q_summary = {}
    for q, grp in by_q.items():
        qa = np.mean([g["f1_A_vlm4ts_own"] for g in grp])
        qb = np.mean([g["f1_B_canmmlm"] for g in grp])
        qf = np.mean([g["f1_F_layout"] for g in grp])
        q_summary[q] = {"n": len(grp), "f1_A": float(qa), "f1_B": float(qb), "f1_F": float(qf)}
        print(f"  {q} (n={len(grp)}): A={qa:.3f}  B={qb:.3f}  F={qf:.3f}")

    summary = {
        "n": n, "mean_f1": {"A_vlm4ts_own": float(mean_a), "B_canmmlm_naive": float(mean_b), "F_layout": float(mean_f)},
        "calls_per_segment": {"A": 1, "B": 4, "F": 1},
        "improvement_A_to_B": float(gap_a_to_b), "improvement_A_to_F": float(gap_a_to_f),
        "pct_of_B_improvement_captured_by_F": pct_captured,
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
