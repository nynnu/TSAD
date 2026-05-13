"""
Stage-1 Only: backbone comparison on MSL P-11 + 10 channels.

Supported backbones: clip | dino | convnext | mae | all
Checkpoint system ensures already-computed results are never re-run.
Summary table always shows every backbone that has complete cached results.

Outputs:
  results/clip_vs_dino/
    results_all.json   ← per-channel metrics for all completed backbones
    summary_all.txt    ← comparison table

Usage:
    cd VLM4TS
    python experiments/compare_all_backbones.py --backbone mae
    python experiments/compare_all_backbones.py --backbone all
    python experiments/compare_all_backbones.py --backbone clip --backbone dino
"""

import argparse
import ast
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.vit4ts import ViT4TS
from models.vit4ts_dino import ViT4TS_DINO
from models.vit4ts_convnext import ViT4TS_ConvNeXt
from models.vit4ts_mae import ViT4TS_MAE
from preprocessing.data_utils import orion_to_internal
from evaluation.evaluate import evaluate_intervals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = ROOT / "results" / "clip_vs_dino"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
MSL_DIR    = ROOT / "data" / "MSL"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"

ALPHA        = 0.01
RANDOM_SEED  = 42
N_EXTRA_CHANNELS = 10

ALL_BACKBONES = ["clip", "dino", "convnext", "mae"]

BACKBONE_PARAMS = {
    "clip": dict(
        cls=ViT4TS,
        params=dict(window_size=240, window_step_ratio=4.0,
                    model_name="ViT-B-16", patch_size=16,
                    image_size=(224, 224), alpha=ALPHA, verbose=False),
    ),
    "dino": dict(
        cls=ViT4TS_DINO,
        params=dict(window_size=240, window_step_ratio=4.0,
                    model_name="dinov2_vitb14", patch_size=14,
                    image_size=(224, 224), alpha=ALPHA, verbose=False),
    ),
    "convnext": dict(
        cls=ViT4TS_ConvNeXt,
        params=dict(window_size=240, window_step_ratio=4.0,
                    model_name="convnextv2_base.fcmae_ft_in22k_in1k", patch_size=16,
                    image_size=(224, 224), alpha=ALPHA, verbose=False),
    ),
    "mae": dict(
        cls=ViT4TS_MAE,
        params=dict(window_size=240, window_step_ratio=4.0,
                    model_name="vit_base_patch16_224.mae", patch_size=16,
                    image_size=(224, 224), alpha=ALPHA, verbose=False),
    ),
}

BACKBONE_LABEL = {
    "clip":     "CLIP ViT-B/16",
    "dino":     "DINOv2 ViT-B/14",
    "convnext": "ConvNeXt V2 Base",
    "mae":      "MAE ViT-B/16",
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt(name: str) -> Path:
    return CKPT_DIR / f"{name}.pkl"

def load_ckpt(name: str):
    p = _ckpt(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def save_ckpt(name: str, value) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ckpt(name), "wb") as f:
        pickle.dump(value, f)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_anomalies() -> dict:
    anomalies = {}
    with open(ANOMALY_CSV, "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        try:
            anomalies[parts[0]] = ast.literal_eval(parts[1].strip('"'))
        except Exception:
            pass
    return anomalies

def select_channels(all_gt: dict) -> list:
    all_names = [f.stem for f in sorted(MSL_DIR.glob("*.csv"))]
    candidates = [n for n in all_names if n != "P-11" and n in all_gt]
    rng = random.Random(RANDOM_SEED)
    extra = rng.sample(candidates, min(N_EXTRA_CHANNELS, len(candidates)))
    return ["P-11"] + extra

def _intervals_to_list(df: pd.DataFrame) -> list:
    return df[["start", "end"]].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------
# Run one channel / backbone
# ---------------------------------------------------------------------------

def run_channel(channel: str, gt: list, detector, tag: str) -> dict:
    key = f"{channel}_{tag}"
    cached = load_ckpt(key)
    if cached is not None:
        print(f"  [{tag:8s}] {channel}: checkpoint  F1={cached['f1']:.4f}")
        return cached

    data = pd.read_csv(MSL_DIR / f"{channel}.csv")
    print(f"  [{tag:8s}] {channel}: running Stage-1 ...")
    intervals = detector.detect(data)
    metrics   = evaluate_intervals(gt, _intervals_to_list(intervals))

    result = {
        "channel":   channel,
        "backbone":  tag,
        "n_gt":      len(gt),
        "n_detected": len(intervals),
        "precision": round(metrics["precision"], 4),
        "recall":    round(metrics["recall"],    4),
        "f1":        round(metrics["F1"],        4),
    }
    save_ckpt(key, result)
    print(f"           P={result['precision']:.4f}  R={result['recall']:.4f}"
          f"  F1={result['f1']:.4f}  ({result['n_detected']} detected)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 backbone comparison on MSL")
    parser.add_argument(
        "--backbone", action="append", default=None,
        choices=ALL_BACKBONES + ["all"],
        help="Backbone(s) to run. Repeat for multiple. Default: all",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    # Resolve which backbones to actively run
    requested = args.backbone or ["all"]
    if "all" in requested:
        to_run = ALL_BACKBONES
    else:
        to_run = list(dict.fromkeys(requested))   # deduplicate, preserve order

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Stage-1 Backbone Comparison — MSL P-11 + {N_EXTRA_CHANNELS} channels")
    print(f"Running: {', '.join(to_run)}")
    print("=" * 70)

    all_gt   = load_anomalies()
    channels = select_channels(all_gt)

    print(f"\nChannels ({len(channels)}):")
    for c in channels:
        print(f"  {c:8s}  GT: {len(all_gt.get(c, []))} intervals")

    # Initialise only the detectors we need
    detectors = {}
    for tag in to_run:
        cfg = BACKBONE_PARAMS[tag]
        # Skip if all channels already cached
        if all(_ckpt(f"{c}_{tag}").exists() for c in channels if all_gt.get(c)):
            print(f"\n[{tag}] All results cached — skipping model init.")
        else:
            print(f"\nInitialising {BACKBONE_LABEL[tag]} ...")
            detectors[tag] = cfg["cls"](**cfg["params"])

    # Collect results for every backbone (run or load from cache)
    all_results: dict[str, list] = {tag: [] for tag in ALL_BACKBONES}

    for channel in channels:
        gt = all_gt.get(channel, [])
        if not gt:
            print(f"\n  SKIP {channel}: no GT")
            continue
        print(f"\n--- {channel} ---")
        for tag in ALL_BACKBONES:
            if tag in to_run:
                det = detectors.get(tag)
                if det is not None:
                    all_results[tag].append(run_channel(channel, gt, det, tag))
                else:
                    # All cached
                    r = load_ckpt(f"{channel}_{tag}")
                    if r:
                        all_results[tag].append(r)
                        print(f"  [{tag:8s}] {channel}: checkpoint  F1={r['f1']:.4f}")
            else:
                # Not requested to run, but load cache for display
                r = load_ckpt(f"{channel}_{tag}")
                if r:
                    all_results[tag].append(r)

    # Only show backbones that have full results
    complete = [tag for tag in ALL_BACKBONES
                if len(all_results[tag]) == len([c for c in channels if all_gt.get(c)])]

    def avg(rows, key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avgs = {
        tag: {"channel": "AVERAGE", "backbone": tag, "n_gt": "-", "n_detected": "-",
              "precision": avg(all_results[tag], "precision"),
              "recall":    avg(all_results[tag], "recall"),
              "f1":        avg(all_results[tag], "f1")}
        for tag in complete
    }

    # Save JSON
    json_out = {
        "config": {"alpha": ALPHA, "n_channels": len(channels), "seed": RANDOM_SEED,
                   "ran": to_run},
    }
    for tag in complete:
        json_out[tag] = all_results[tag] + [avgs[tag]]
    with open(OUTPUT_DIR / "results_all.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # Build comparison table
    maps = {tag: {r["channel"]: r for r in all_results[tag] + [avgs[tag]]}
            for tag in complete}

    col = 8
    short = {"clip": "CL", "dino": "DN", "convnext": "CN", "mae": "MA"}

    def hdr_block(tag):
        s = short[tag]
        return f" {s+'-P':>{col}} {s+'-R':>{col}} {s+'-F1':>{col+1}}"

    header = f"\n{'Channel':<10}" + "".join(hdr_block(t) for t in complete) + f"  {'Best':>8}"
    sep    = "-" * (10 + len(complete) * (col*3 + 5) + 10)
    rows   = [header, sep]

    ch_order = [r["channel"] for r in all_results[complete[0]]] + ["AVERAGE"]
    for ch in ch_order:
        vals = {tag: maps[tag].get(ch, {}) for tag in complete}
        if not all(vals.values()):
            continue
        f1s  = {tag: vals[tag]["f1"] for tag in complete}
        best = BACKBONE_LABEL[max(f1s, key=f1s.get)]

        row = f"{ch:<10}"
        for tag in complete:
            v = vals[tag]
            row += f" {v['precision']:>{col}.4f} {v['recall']:>{col}.4f} {v['f1']:>{col+1}.4f}"
        row += f"  {best:>8}"
        rows.append(row)

    rows.append(sep)

    avg_line = "  ".join(
        f"{BACKBONE_LABEL[t]}={avgs[t]['f1']:.4f}" for t in complete
    )
    labels_str = " | ".join(f"{short[t]}={BACKBONE_LABEL[t]}" for t in complete)

    summary = "\n".join([
        "=" * 70,
        "Stage-1 Comparison — MSL P-11 + 10 channels",
        f"alpha={ALPHA}   seed={RANDOM_SEED}",
        f"({labels_str})",
        "=" * 70,
    ] + rows + [
        "",
        f"Avg F1  {avg_line}",
        f"\nResults: {OUTPUT_DIR / 'results_all.json'}",
    ])

    print("\n" + summary)
    txt_path = OUTPUT_DIR / "summary_all.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Summary: {txt_path}")


if __name__ == "__main__":
    run()
