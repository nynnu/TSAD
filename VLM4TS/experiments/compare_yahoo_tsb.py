"""
Stage-1 Only: CLIP vs DINOv2 vs ConvNeXt V2 on TSB-AD-U Yahoo data.

Uses TSB-AD-U Yahoo files (Data,Label format) — 259 files total.
Randomly samples 10 files with seed=42.
Ground truth is extracted from the Label column (1=anomaly).

Outputs:
  results/yahoo_tsb/
    results.json     ← per-file metrics for all 3 backbones
    summary.txt      ← 3-way comparison table
    checkpoints/     ← cached per-file results

Usage:
    cd VLM4TS
    pip install timm          # if not installed
    python experiments/compare_yahoo_tsb.py
"""

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
from evaluation.evaluate import evaluate_intervals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TSB_DIR = ROOT.parent / "TSB-AD-U"
OUTPUT_DIR = ROOT / "results" / "yahoo_tsb"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

ALPHA = 0.01
RANDOM_SEED = 42
N_SAMPLES = 10

CLIP_PARAMS = {
    "window_size": 224,
    "window_step_ratio": 4.0,
    "model_name": "ViT-B-16",
    "patch_size": 16,
    "image_size": (224, 224),
    "alpha": ALPHA,
    "verbose": False,
}

DINO_PARAMS = {
    "window_size": 224,
    "window_step_ratio": 4.0,
    "model_name": "dinov2_vitb14",
    "patch_size": 14,
    "image_size": (224, 224),
    "alpha": ALPHA,
    "verbose": False,
}

CONVNEXT_PARAMS = {
    "window_size": 224,
    "window_step_ratio": 4.0,
    "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
    "patch_size": 16,
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
# TSB-AD-U data helpers
# ---------------------------------------------------------------------------

def labels_to_intervals(labels: np.ndarray) -> list:
    """Convert binary label array → list of [start, end] index intervals."""
    intervals = []
    in_anom = False
    start = None
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            start = i
            in_anom = True
        elif v == 0 and in_anom:
            intervals.append([start, i - 1])
            in_anom = False
    if in_anom:
        intervals.append([start, len(labels) - 1])
    return intervals


def load_tsb_file(file_path: Path):
    """Load a TSB-AD-U file and return (orion_df, gt_intervals).

    orion_df: DataFrame with 'timestamp' (row index) and 'value' columns.
    gt_intervals: list of [start, end] index pairs.
    """
    df = pd.read_csv(file_path)
    values = df["Data"].values.astype(float)
    labels = df["Label"].values.astype(int)

    orion_df = pd.DataFrame({
        "timestamp": np.arange(len(values), dtype=float),
        "value": values,
    })
    gt_intervals = labels_to_intervals(labels)
    return orion_df, gt_intervals


def select_files() -> list:
    """Sample N_SAMPLES Yahoo files from TSB-AD-U with seed=42."""
    all_files = sorted(TSB_DIR.glob("*YAHOO*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No YAHOO files found in {TSB_DIR}")

    # All 259 Yahoo files have at least one anomaly — no filtering needed
    rng = random.Random(RANDOM_SEED)
    return rng.sample(all_files, min(N_SAMPLES, len(all_files)))


def _intervals_to_list(df: pd.DataFrame) -> list:
    return df[["start", "end"]].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------
# Run one file with one backbone (with caching)
# ---------------------------------------------------------------------------

def run_file(file_path: Path, gt_intervals: list, orion_df: pd.DataFrame,
             detector, backbone_tag: str) -> dict:
    key = f"{file_path.stem}_{backbone_tag}"
    cached = load_ckpt(key)
    if cached is not None:
        print(f"  [{backbone_tag}] {file_path.stem}: loaded from checkpoint")
        return cached

    print(f"  [{backbone_tag}] {file_path.stem}: running Stage-1 ...")
    intervals = detector.detect(orion_df)
    metrics = evaluate_intervals(gt_intervals, _intervals_to_list(intervals))

    result = {
        "file": file_path.name,
        "backbone": backbone_tag,
        "n_gt": len(gt_intervals),
        "n_detected": len(intervals),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["F1"], 4),
    }
    save_ckpt(key, result)
    print(
        f"           P={result['precision']:.4f}  R={result['recall']:.4f}"
        f"  F1={result['f1']:.4f}  ({result['n_detected']} detected)"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage-1: CLIP vs DINOv2 vs ConvNeXt V2 — TSB-AD-U Yahoo (10 files)")
    print("=" * 70)

    files = select_files()
    print(f"\nSampled {len(files)} files (seed={RANDOM_SEED}):")
    for i, f in enumerate(files):
        print(f"  {i+1:2d}. {f.name}")

    # Initialize detectors
    print("\nInitializing CLIP ViT-B-16 ...")
    clip_det = ViT4TS(**CLIP_PARAMS)

    print("\nInitializing DINOv2 ViT-B-14 ...")
    dino_det = ViT4TS_DINO(**DINO_PARAMS)

    print("\nInitializing ConvNeXt V2 Base ...")
    cnxt_det = ViT4TS_ConvNeXt(**CONVNEXT_PARAMS)

    clip_results = []
    dino_results = []
    cnxt_results = []

    for file_path in files:
        orion_df, gt_intervals = load_tsb_file(file_path)
        n_pts = len(orion_df)
        print(f"\n--- {file_path.stem}  ({n_pts} pts, {len(gt_intervals)} GT intervals) ---")

        clip_results.append(run_file(file_path, gt_intervals, orion_df, clip_det, "clip"))
        dino_results.append(run_file(file_path, gt_intervals, orion_df, dino_det, "dino"))
        cnxt_results.append(run_file(file_path, gt_intervals, orion_df, cnxt_det, "convnext"))

    # Averages
    def avg(rows, key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def make_avg(rows, tag):
        return {
            "file": "AVERAGE", "backbone": tag, "n_gt": "-", "n_detected": "-",
            "precision": avg(rows, "precision"),
            "recall":    avg(rows, "recall"),
            "f1":        avg(rows, "f1"),
        }

    clip_avg = make_avg(clip_results, "clip")
    dino_avg = make_avg(dino_results, "dino")
    cnxt_avg = make_avg(cnxt_results, "convnext")

    # Save JSON
    all_results = {
        "config": {"alpha": ALPHA, "n_files": len(files), "seed": RANDOM_SEED, "source": "TSB-AD-U Yahoo"},
        "clip":     clip_results + [clip_avg],
        "dino":     dino_results + [dino_avg],
        "convnext": cnxt_results + [cnxt_avg],
    }
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Build comparison table
    clip_map = {r["file"]: r for r in clip_results + [clip_avg]}
    dino_map = {r["file"]: r for r in dino_results + [dino_avg]}
    cnxt_map = {r["file"]: r for r in cnxt_results + [cnxt_avg]}

    col = 7
    header = (
        f"\n{'File':<14}"
        f" {'CL-P':>{col}} {'CL-R':>{col}} {'CL-F1':>{col+1}}"
        f" {'DN-P':>{col}} {'DN-R':>{col}} {'DN-F1':>{col+1}}"
        f" {'CN-P':>{col}} {'CN-R':>{col}} {'CN-F1':>{col+1}}"
        f" {'Best':>8}"
    )
    sep = "-" * 90

    rows = [header, sep]
    file_keys = [r["file"] for r in clip_results] + ["AVERAGE"]
    for fk in file_keys:
        # short display name
        label = fk if fk == "AVERAGE" else f"id_{fk.split('_id_')[1].split('_')[0]}" if "_id_" in fk else fk[:14]
        cr = clip_map.get(fk, {})
        dr = dino_map.get(fk, {})
        nr = cnxt_map.get(fk, {})
        if not (cr and dr and nr):
            continue
        f1s = {"CLIP": cr["f1"], "DINO": dr["f1"], "CNX": nr["f1"]}
        best = max(f1s, key=f1s.get)
        rows.append(
            f"{label:<14}"
            f" {cr['precision']:>{col}.4f} {cr['recall']:>{col}.4f} {cr['f1']:>{col+1}.4f}"
            f" {dr['precision']:>{col}.4f} {dr['recall']:>{col}.4f} {dr['f1']:>{col+1}.4f}"
            f" {nr['precision']:>{col}.4f} {nr['recall']:>{col}.4f} {nr['f1']:>{col+1}.4f}"
            f" {best:>8}"
        )
    rows.append(sep)

    summary = "\n".join([
        "=" * 70,
        "Stage-1: CLIP ViT-B-16 | DINOv2 ViT-B-14 | ConvNeXt V2 Base",
        f"Dataset: TSB-AD-U Yahoo ({len(files)} files, seed={RANDOM_SEED})   alpha={ALPHA}",
        "(CL=CLIP, DN=DINOv2, CN=ConvNeXt V2)",
        "=" * 70,
    ] + rows + [
        "",
        f"Avg F1  CLIP={clip_avg['f1']:.4f}  DINOv2={dino_avg['f1']:.4f}  ConvNeXt={cnxt_avg['f1']:.4f}",
        "",
        f"Results: {json_path}",
    ])

    print("\n" + summary)

    txt_path = OUTPUT_DIR / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    run()
