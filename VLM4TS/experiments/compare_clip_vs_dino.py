"""
Stage-1 Only: CLIP vs DINOv2 comparison on MSL P-11 + 10 channels.

Runs only Stage 1 (vision screening) — no VLM — for each backbone.
Channels: P-11 (fixed) + 10 randomly sampled MSL channels (seed=42, excl. P-11).

Outputs:
  results/clip_vs_dino/
    results.json       ← per-channel metrics for CLIP and DINOv2
    summary.txt        ← side-by-side comparison table
    checkpoints/       ← cached per-channel scores for each backbone

Usage:
    cd VLM4TS
    python experiments/compare_clip_vs_dino.py
"""

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
from preprocessing.data_utils import orion_to_internal
from evaluation.evaluate import evaluate_intervals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = ROOT / "results" / "clip_vs_dino"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
MSL_DIR = ROOT / "data" / "MSL"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"

ALPHA = 0.01
RANDOM_SEED = 42
N_EXTRA_CHANNELS = 10  # additional channels beyond P-11

CLIP_PARAMS = {
    "window_size": 240,
    "window_step_ratio": 4.0,
    "model_name": "ViT-B-16",
    "patch_size": 16,
    "image_size": (224, 224),
    "alpha": ALPHA,
    "verbose": False,
}

DINO_PARAMS = {
    "window_size": 240,
    "window_step_ratio": 4.0,
    "model_name": "dinov2_vitb14",
    "patch_size": 14,
    "image_size": (224, 224),
    "alpha": ALPHA,
    "verbose": False,
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
    """Parse anomalies.csv → {signal_name: [[start, end], ...]}"""
    anomalies = {}
    with open(ANOMALY_CSV, "r") as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        signal = parts[0]
        try:
            events = ast.literal_eval(parts[1].strip('"'))
            anomalies[signal] = events
        except Exception:
            pass
    return anomalies


def select_channels(all_gt: dict) -> list:
    """P-11 first, then 10 random MSL channels (seed=42, excl. P-11)."""
    msl_files = sorted(MSL_DIR.glob("*.csv"))
    all_names = [f.stem for f in msl_files]  # e.g. ['C-1', 'P-11', ...]

    candidates = [n for n in all_names if n != "P-11" and n in all_gt]
    rng = random.Random(RANDOM_SEED)
    extra = rng.sample(candidates, min(N_EXTRA_CHANNELS, len(candidates)))
    return ["P-11"] + extra


def _intervals_to_list(df: pd.DataFrame) -> list:
    return df[["start", "end"]].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------
# Run one channel with one backbone (with caching)
# ---------------------------------------------------------------------------

def run_channel(channel: str, gt_events: list, detector, backbone_tag: str) -> dict:
    ckpt_key = f"{channel}_{backbone_tag}"
    cached = load_ckpt(ckpt_key)
    if cached is not None:
        print(f"  [{backbone_tag}] {channel}: loaded from checkpoint")
        return cached

    data_path = MSL_DIR / f"{channel}.csv"
    data = pd.read_csv(data_path)

    print(f"  [{backbone_tag}] {channel}: running Stage-1 ...")
    intervals = detector.detect(data)
    metrics = evaluate_intervals(gt_events, _intervals_to_list(intervals))

    result = {
        "channel": channel,
        "backbone": backbone_tag,
        "n_gt": len(gt_events),
        "n_detected": len(intervals),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["F1"], 4),
    }
    save_ckpt(ckpt_key, result)
    print(
        f"           P={result['precision']:.4f}  R={result['recall']:.4f}  F1={result['f1']:.4f}"
        f"  ({result['n_detected']} detected)"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Stage-1 Only: CLIP vs DINOv2 — MSL P-11 + 10 channels")
    print("=" * 65)

    all_gt = load_anomalies()
    channels = select_channels(all_gt)

    print(f"\nChannels ({len(channels)} total):")
    for c in channels:
        print(f"  {c:8s}  GT intervals: {len(all_gt.get(c, []))}")

    # Initialize detectors (once each; DINOv2 downloads on first use)
    print("\nInitializing CLIP ViT-B-16 ...")
    clip_detector = ViT4TS(**CLIP_PARAMS)

    print("\nInitializing DINOv2 ViT-B-14 ...")
    dino_detector = ViT4TS_DINO(**DINO_PARAMS)

    # Run
    clip_results = []
    dino_results = []

    for channel in channels:
        gt = all_gt.get(channel, [])
        if not gt:
            print(f"  SKIP {channel}: no ground truth in anomalies.csv")
            continue

        print(f"\n--- {channel} ---")
        clip_results.append(run_channel(channel, gt, clip_detector, "clip"))
        dino_results.append(run_channel(channel, gt, dino_detector, "dino"))

    # Compute averages
    def avg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    clip_avg = {
        "channel": "AVERAGE", "backbone": "clip",
        "n_gt": "-", "n_detected": "-",
        "precision": avg(clip_results, "precision"),
        "recall":    avg(clip_results, "recall"),
        "f1":        avg(clip_results, "f1"),
    }
    dino_avg = {
        "channel": "AVERAGE", "backbone": "dino",
        "n_gt": "-", "n_detected": "-",
        "precision": avg(dino_results, "precision"),
        "recall":    avg(dino_results, "recall"),
        "f1":        avg(dino_results, "f1"),
    }

    # Save JSON
    all_results = {
        "config": {"alpha": ALPHA, "n_channels": len(channels), "seed": RANDOM_SEED},
        "clip": clip_results + [clip_avg],
        "dino": dino_results + [dino_avg],
    }
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Build comparison table
    header = f"\n{'Channel':<10} {'CLIP-P':>8} {'CLIP-R':>8} {'CLIP-F1':>9} {'DINO-P':>8} {'DINO-R':>8} {'DINO-F1':>9} {'ΔF1':>8}"
    sep = "-" * 75
    rows = [header, sep]

    paired = {r["channel"]: r for r in clip_results}
    paired_dino = {r["channel"]: r for r in dino_results}

    for ch in ([r["channel"] for r in clip_results] + ["AVERAGE"]):
        if ch == "AVERAGE":
            cr = clip_avg
            dr = dino_avg
        else:
            cr = paired.get(ch, {})
            dr = paired_dino.get(ch, {})
        if not cr or not dr:
            continue
        delta = round(dr["f1"] - cr["f1"], 4)
        marker = " ▲" if delta > 0 else (" ▼" if delta < 0 else "  ")
        rows.append(
            f"{ch:<10} {cr['precision']:>8.4f} {cr['recall']:>8.4f} {cr['f1']:>9.4f}"
            f" {dr['precision']:>8.4f} {dr['recall']:>8.4f} {dr['f1']:>9.4f}"
            f" {delta:>+8.4f}{marker}"
        )

    rows.append(sep)
    summary = "\n".join([
        "=" * 65,
        "Stage-1 Comparison: CLIP ViT-B-16  vs  DINOv2 ViT-B-14",
        f"Dataset : MSL ({len(channels)} channels)   alpha={ALPHA}",
        "=" * 65,
    ] + rows + [
        "",
        f"▲ = DINOv2 better   ▼ = CLIP better",
        f"ΔF1 = DINOv2 F1 - CLIP F1",
        "",
        f"Results saved: {json_path}",
    ])

    print("\n" + summary)

    txt_path = OUTPUT_DIR / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    run()
