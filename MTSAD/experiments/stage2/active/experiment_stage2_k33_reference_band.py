"""K33: K18 selection with a large, aligned normal-reference-band visual interface.

The Stage-1 hysteresis selector, DINO focus localization, VLM model, and decision
target are held constant relative to K18.  The intervention is the Stage-2 evidence
display: a full candidate trace plus an aligned normal top-k reference band.

Usage:
  python experiment_stage2_k33_reference_band.py --smoke  # first 12 paired segments
  python experiment_stage2_k33_reference_band.py --run    # all 48 paired segments
"""
import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))

from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, OUT_DIR as K4_OUT_DIR, constant_channels, compute_zscores, f1_of,
)
from experiment_stage2_k18_zoomed_subplot import OUT_DIR as K18_OUT_DIR
from experiment_stage2_k26_reference_zoomed import ALPHA_LOOSE, ALPHA_STRICT, CORR_THR, N_CHANNELS, select_channels_hysteresis
import experiment_stage2_v16 as v16
from proto_render_k33_reference_band import render
from smd_3way_baseline_comparison import call_vlm, parse_response
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN

OUT_DIR = BASE / "experiments" / "results_stage2_k33_reference_band"
OUT_DIR.mkdir(parents=True, exist_ok=True)
K4_CHECKPOINT = K4_OUT_DIR / "checkpoint_hyst_s0.01_l0.1_c0.5.json"


def encode_figure(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt(selected, scores):
    score_text = ", ".join(
        f"ch {channel} (selection z {'>=20' if scores[channel] >= 20 else f'{scores[channel]:.2f}'})"
        for channel in selected
    )
    return f"""You are diagnosing a multivariate industrial system. Stage 1 selected these candidate channels: {score_text}.

The image contains one row for each selected channel. The LEFT plot shows the full 224-tick candidate window. The amber band marks the particular 60-tick evidence interval selected by a separate DINOv2 local-pattern detector. The green background marks the training-calibrated normal value range; red dashed lines are its bounds.

The RIGHT plot magnifies that same evidence interval. Black is the candidate. Blue is the median of eight normal training windows retrieved using ONLY the candidate's pre-interval context; the pale-blue band is their 10th–90th percentile range. Judge a channel anomalous when its candidate pattern meaningfully departs from both its own temporal context and this normal-reference band. A brief isolated spike may be benign; do not mark a channel solely because it was selected in Stage 1.

Return ONLY valid JSON, with no Markdown or explanation:
{{"anomalous_channels": [a subset of {selected}], "confidence": "low" or "medium" or "high"}}"""


def run(n_limit=None):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if n_limit:
        segments = segments[:n_limit]
    k4 = json.loads(K4_CHECKPOINT.read_text(encoding="utf-8"))
    k18_rows = json.loads((K18_OUT_DIR / "summary.json").read_text(encoding="utf-8"))["rows"]
    k18_f1 = {row["seg_id"]: row["f1_zoomed"] for row in k18_rows}
    ck_path = OUT_DIR / "checkpoint_k33.json"
    checkpoint = json.loads(ck_path.read_text(encoding="utf-8")) if ck_path.exists() else {}

    data, degenerate, correlations, calibration = {}, {}, {}, {}
    rows = []
    for segment in segments:
        entity, cs, ce = segment["entity"], segment["start"], segment["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if k4.get(seg_id, {}).get("status") != "OK":
            continue
        if entity not in data:
            data[entity] = load_smd(entity)
            degenerate[entity] = constant_channels(data[entity][0])
            correlations[entity] = np.nan_to_num(np.corrcoef(data[entity][0].T), nan=0.0)
        train, test = data[entity]
        start, end = _centered_window(len(test), (cs + ce) // 2, WIN)
        window = test[start:end]
        scores = compute_zscores(entity, train, window, calibration, degenerate[entity])
        ranked, selected = select_channels_hysteresis(scores, correlations[entity], ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
        cmin, cmax = v16.gn_train(train, range(N_CHANNELS))

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            prediction = checkpoint[seg_id]["pred"]
        else:
            fig = render(entity, window, train, ranked, selected, scores, calibration, cmin, cmax)
            raw = call_vlm(build_prompt(selected, scores), encode_figure(fig))
            prediction = parse_response(raw)
            checkpoint[seg_id] = {
                "status": "OK" if prediction is not None else "PARSE_ERROR",
                "pred": prediction,
                "selected": selected,
                "raw": raw,
            }
            ck_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {seg_id}: selected={selected}, pred={prediction}", flush=True)

        if prediction is None or seg_id not in k18_f1:
            continue
        truth = set(dimension - 1 for dimension in segment["dims"])
        rows.append({"seg_id": seg_id, "k": len(truth), "n_selected": len(selected),
                     "f1_k18": k18_f1[seg_id], "f1_k33": f1_of(prediction, truth)})

    result = {"n": len(rows), "rows": rows}
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rows:
        old = np.array([row["f1_k18"] for row in rows])
        new = np.array([row["f1_k33"] for row in rows])
        p = wilcoxon(new, old).pvalue if np.any(new != old) else 1.0
        print(f"n={len(rows)} K18={old.mean():.4f} K33={new.mean():.4f} diff={new.mean()-old.mean():+.4f} p={p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run(n_limit=12)
    elif args.run:
        run()
    else:
        parser.error("choose --smoke or --run")
