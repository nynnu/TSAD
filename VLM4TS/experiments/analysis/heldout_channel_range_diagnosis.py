"""
Held-out 5개 entity(machine-2-3/2-8/3-7/3-9/3-11) 진단: 지금까지 단 한 번도 안 쓰인
진짜 held-out 데이터에서, top-8 고정 설계(I8: heatmap+overlay)가 GT 채널 수가
4~8개 범위를 벗어나는 지점(1~3개, 9개+)에서 어떻게 무너지는지 확인.

필터링 없이 23개 세그먼트 전부 사용(GT 채널 수 1~34개 널뛰는 그대로).
조건 A = 기존 I8 (render_heatmap_overlay + build_prompt_heatmap_overlay, 변경 없음).
조건 B = 신규: heatmap(A와 동일) + top-8 subplot grid(overlay 대신) + z-score index-aware
텍스트(top-25pt/채널, pilot_reduced_index_on_i8.py 패턴 재사용).

Step0 랭킹은 full_smd_3way.py의 step0_rank_original_mechanism 그대로(로컬, API 비용 없음).

23개 세그먼트 x 2조건 = 46콜.
"""

import base64
import csv
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from step1v3_dino_graph_smd import (  # noqa: E402
    N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window, load_smd,
)
from smd_3way_baseline_comparison import call_vlm, load_smd_test, parse_response, prf  # noqa: E402
from full_smd_3way import step0_rank_original_mechanism  # noqa: E402
from pilot_heatmap_overlay_nsweep import render_heatmap_overlay, build_prompt_heatmap_overlay  # noqa: E402

INTERP_CACHE = ANALYSIS_DIR / "_interp_cache"
OUT_DIR = BASE / "experiments" / "results_heldout_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "checkpoint.json"

ENTITIES = ["machine-2-3", "machine-2-8", "machine-3-7", "machine-3-9", "machine-3-11"]
N = 8
N_POINTS_PER_CHANNEL = 25
HEDGE_THRESHOLD = 35


def load_all_segments():
    """필터링 없이 5개 entity의 _interp_cache 세그먼트 전부."""
    segments = []
    for entity in ENTITIES:
        path = INTERP_CACHE / f"{entity}.txt"
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for l in lines:
            span, chans = l.split(":")
            s, e = span.split("-")
            gt_dims = [int(x) for x in chans.split(",")]
            segments.append((entity, (int(s), int(e)), gt_dims))
    return segments


# ---------------------------------------------------------------------------
# 조건 B: heatmap(A와 동일) + top-8 subplot grid + z-score index-aware 텍스트
# ---------------------------------------------------------------------------

def render_heatmap_subplot(window: np.ndarray, ranked: list[int], n: int = N) -> str:
    fig = plt.figure(figsize=(6, 8.5), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1])

    ax1 = fig.add_subplot(gs[0])
    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: 38 channels, sorted by Step0 score (top row = highest score)", fontsize=7)

    top_n = ranked[:n]
    n_cols = 4
    n_rows = -(-len(top_n) // n_cols)
    gs_bottom = gs[1].subgridspec(n_rows, n_cols, wspace=0.35, hspace=0.6)
    for i, c in enumerate(top_n):
        ax = fig.add_subplot(gs_bottom[i // n_cols, i % n_cols])
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(norm, linewidth=0.7, color="black")
        ax.set_title(f"Ch{c}", fontsize=6, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Bottom: top-{len(top_n)} candidate channels, one small subplot each", fontsize=7, y=0.36)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_heatmap_subplot_index(all_channels: list[int], top_n: list[int], window, train,
                                        n_points: int = N_POINTS_PER_CHANNEL) -> str:
    blocks = []
    for i, c in enumerate(top_n):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.argsort(-z)[:n_points]
        top_idx = np.sort(top_idx)

        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{n_points} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: a grid of small subplots, one per channel, for the top-{len(top_n)} candidate channels ({top_n}) from the heatmap -- each subplot shows that channel's own waveform in this window, with the channel number as its title.

For each of the top-{len(top_n)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels and this point data together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {all_channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


# ---------------------------------------------------------------------------

def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def f1_of(pred, gt):
    p, r = prf(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run(dry_run=False):
    segments = load_all_segments()
    by_entity = {}
    for entity, _, _ in segments:
        by_entity[entity] = by_entity.get(entity, 0) + 1
    print(f"총 세그먼트: {len(segments)}개 (기대: 23)")
    for e, c in sorted(by_entity.items()):
        print(f"  {e}: {c}개")
    gt_sizes = [len(g) for _, _, g in segments]
    print(f"GT 채널 수 범위: {min(gt_sizes)} ~ {max(gt_sizes)}")
    print(f"예상 콜 수: {len(segments)} x 2 = {len(segments) * 2}")
    assert len(segments) == 23, f"세그먼트 수 불일치: {len(segments)}"
    assert len(segments) * 2 == 46
    if dry_run:
        print("[DRY RUN] STOP.")
        return

    checkpoint = load_checkpoint()
    entity_test, entity_train = {}, {}
    all_channels = list(range(N_CHANNELS))

    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)

        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        test = entity_test[entity]
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        train = entity_train[entity]

        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        window = test[s:e]

        step0_key = f"{seg_id}_step0_ranked"
        if step0_key not in checkpoint:
            print(f"  [Step0] {seg_id} 랭킹 계산 중 ...", flush=True)
            ranked = step0_rank_original_mechanism(train, test, cs, ce)
            checkpoint[step0_key] = ranked
            save_checkpoint(checkpoint)
        ranked = checkpoint[step0_key]
        top8 = ranked[:N]

        # --- 조건 A: I8 (기존 그대로) ---
        case_id = f"{seg_id}_A_I8"
        if checkpoint.get(case_id, {}).get("status") != "OK":
            img = render_heatmap_overlay(window, ranked, N)
            prompt = build_prompt_heatmap_overlay(all_channels, top8)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "A"}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)

        # --- 조건 B: heatmap + subplot grid + z-score index-aware ---
        case_id = f"{seg_id}_B_subplot_index"
        if checkpoint.get(case_id, {}).get("status") != "OK":
            img = render_heatmap_subplot(window, ranked, N)
            prompt = build_prompt_heatmap_subplot_index(all_channels, top8, window, train, N_POINTS_PER_CHANNEL)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "condition": "B"}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: pred={pred}", flush=True)

        checkpoint[f"{seg_id}_gt"] = {"gt_channels": sorted(gt_channels), "n_gt": len(gt_channels)}
        save_checkpoint(checkpoint)

    # ---- 세그먼트별 CSV + 구간별 요약 ----
    def bucket(n_gt):
        if n_gt <= 3:
            return "저(1-3)"
        elif n_gt <= 8:
            return "중(4-8)"
        else:
            return "고(9+)"

    rows = []
    for entity, (cs, ce), gt_dims in segments:
        seg_id = f"{entity}_{cs}_{ce}"
        gt_channels = set(d - 1 for d in gt_dims)
        n_gt = len(gt_channels)

        a_entry = checkpoint.get(f"{seg_id}_A_I8", {})
        b_entry = checkpoint.get(f"{seg_id}_B_subplot_index", {})
        if a_entry.get("status") != "OK" or b_entry.get("status") != "OK":
            continue
        a_pred = set(a_entry["pred"] or [])
        b_pred = set(b_entry["pred"] or [])
        pa, ra = prf(a_pred, gt_channels)
        pb, rb = prf(b_pred, gt_channels)

        rows.append({
            "segment_id": seg_id, "entity": entity, "n_gt": n_gt, "bucket": bucket(n_gt),
            "precision_A": pa, "recall_A": ra, "f1_A": f1_of(a_pred, gt_channels),
            "fp_A": len(a_pred - gt_channels), "n_pred_A": len(a_pred), "hedge_A": len(a_pred) >= HEDGE_THRESHOLD,
            "precision_B": pb, "recall_B": rb, "f1_B": f1_of(b_pred, gt_channels),
            "fp_B": len(b_pred - gt_channels), "n_pred_B": len(b_pred), "hedge_B": len(b_pred) >= HEDGE_THRESHOLD,
        })

    with (OUT_DIR / "segment_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"세그먼트별 결과 저장: {OUT_DIR / 'segment_results.csv'} (n={len(rows)})")
    print("=" * 70)

    bucket_summary = {}
    for bname in ["저(1-3)", "중(4-8)", "고(9+)"]:
        grp = [r for r in rows if r["bucket"] == bname]
        if not grp:
            bucket_summary[bname] = {"n": 0}
            continue
        mean = lambda key: float(np.mean([g[key] for g in grp]))
        bucket_summary[bname] = {
            "n": len(grp),
            "f1_A": mean("f1_A"), "precision_A": mean("precision_A"), "recall_A": mean("recall_A"),
            "fp_A": mean("fp_A"), "hedge_A": sum(1 for g in grp if g["hedge_A"]),
            "f1_B": mean("f1_B"), "precision_B": mean("precision_B"), "recall_B": mean("recall_B"),
            "fp_B": mean("fp_B"), "hedge_B": sum(1 for g in grp if g["hedge_B"]),
        }
        print(f"\n{bname} (n={len(grp)}):")
        print(f"  A(I8):      F1={mean('f1_A'):.3f}  P={mean('precision_A'):.3f}  R={mean('recall_A'):.3f}  "
              f"FP={mean('fp_A'):.2f}  hedge={bucket_summary[bname]['hedge_A']}/{len(grp)}")
        print(f"  B(subplot): F1={mean('f1_B'):.3f}  P={mean('precision_B'):.3f}  R={mean('recall_B'):.3f}  "
              f"FP={mean('fp_B'):.2f}  hedge={bucket_summary[bname]['hedge_B']}/{len(grp)}")

    (OUT_DIR / "bucket_summary.json").write_text(json.dumps(bucket_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    dry = len(sys.argv) > 1 and sys.argv[1] == "dry"
    run(dry_run=dry)
