"""
Stage-1: CLIP fixed vs DINOv2 fixed vs DINOv2 adaptive vs MAE — NAB-AWS dataset.

Four conditions:
  clip_fixed   : CLIP ViT-B/16,  window_size=224 (fixed)
  dino_fixed   : DINOv2 ViT-B/14, window_size=224 (fixed)
  dino_adaptive: DINOv2 ViT-B/14, window_size=FFT-estimated ∈ {56,112,224}
  mae_fixed    : MAE ViT-B/16 (pretrain-only), window_size=224 (fixed)

Dataset: data/realAWSCloudwatch/ (17 files, 13 with GT in anomalies.csv)
Metric : window-overlap F1 (same as all other experiments)

Outputs:
  results/nab_aws/
    results.json    ← per-signal metrics for all 3 conditions
    summary.txt     ← comparison table
    checkpoints/    ← cached per-signal results

Usage:
    cd VLM4TS
    python experiments/compare_nab_aws.py
"""

import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.vit4ts import ViT4TS
from models.vit4ts_dino import ViT4TS_DINO
from models.vit4ts_mae import ViT4TS_MAE
from evaluation.evaluate import evaluate_intervals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_aws"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA = 0.01

BASE_PARAMS = dict(
    window_step_ratio=4.0,
    image_size=(224, 224),
    alpha=ALPHA,
    verbose=True,
)

CONFIGS = {
    "clip_fixed": dict(
        cls=ViT4TS,
        params=dict(**BASE_PARAMS,
                    window_size=224,
                    model_name="ViT-B-16",
                    patch_size=16),
    ),
    "dino_fixed": dict(
        cls=ViT4TS_DINO,
        params=dict(**BASE_PARAMS,
                    window_size=224,
                    model_name="dinov2_vitb14",
                    patch_size=14,
                    adaptive_window=False),
    ),
    "dino_adaptive": dict(
        cls=ViT4TS_DINO,
        params=dict(**BASE_PARAMS,
                    window_size=224,
                    model_name="dinov2_vitb14",
                    patch_size=14,
                    adaptive_window=True),
    ),
    "mae_fixed": dict(
        cls=ViT4TS_MAE,
        params=dict(**BASE_PARAMS,
                    window_size=224,
                    model_name="vit_base_patch16_224.mae",
                    patch_size=16),
    ),
}

LABELS = {
    "clip_fixed":    "CLIP fixed",
    "dino_fixed":    "DINO fixed",
    "dino_adaptive": "DINO adaptive",
    "mae_fixed":     "MAE fixed",
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

def _intervals_to_list(df: pd.DataFrame) -> list:
    return df[["start", "end"]].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------
# Run one signal with one condition
# ---------------------------------------------------------------------------

def run_signal(signal: str, gt: list, detector, tag: str) -> dict:
    key = f"{signal}_{tag}"
    cached = load_ckpt(key)
    if cached is not None:
        print(f"  [{tag}] {signal}: checkpoint  F1={cached['f1']:.4f}")
        return cached

    data = pd.read_csv(DATA_DIR / f"{signal}.csv")
    print(f"\n  [{tag}] {signal}: running Stage-1 ...")
    intervals = detector.detect(data)
    metrics   = evaluate_intervals(gt, _intervals_to_list(intervals))

    result = {
        "signal":    signal,
        "condition": tag,
        "n_gt":      len(gt),
        "n_detected": len(intervals),
        "precision": round(metrics["precision"], 4),
        "recall":    round(metrics["recall"],    4),
        "f1":        round(metrics["F1"],        4),
    }
    save_ckpt(key, result)
    print(f"     P={result['precision']:.4f}  R={result['recall']:.4f}"
          f"  F1={result['f1']:.4f}  ({result['n_detected']} detected)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Stage-1: CLIP fixed / DINO fixed / DINO adaptive — NAB-AWS")
    print("=" * 65)

    all_gt  = load_anomalies()
    signals = sorted(
        f.stem for f in DATA_DIR.glob("*.csv") if f.stem in all_gt
    )

    print(f"\nSignals with GT: {len(signals)}")
    for s in signals:
        print(f"  {s}  ({len(all_gt[s])} GT intervals)")

    # --- Initialise detectors (skip if all cached) ---
    detectors = {}
    for tag, cfg in CONFIGS.items():
        all_cached = all(_ckpt(f"{s}_{tag}").exists() for s in signals)
        if all_cached:
            print(f"\n[{tag}] All results cached.")
        else:
            print(f"\nInitialising {LABELS[tag]} ...")
            detectors[tag] = cfg["cls"](**cfg["params"])

    # --- Run ---
    results = {tag: [] for tag in CONFIGS}

    for signal in signals:
        gt = all_gt[signal]
        print(f"\n{'='*50}")
        print(f"Signal: {signal}  ({len(gt)} GT intervals)")

        for tag in CONFIGS:
            det = detectors.get(tag)
            if det is not None:
                results[tag].append(run_signal(signal, gt, det, tag))
            else:
                r = load_ckpt(f"{signal}_{tag}")
                if r:
                    results[tag].append(r)
                    print(f"  [{tag}] {signal}: checkpoint  F1={r['f1']:.4f}")

    # --- Averages ---
    def avg(rows, key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avgs = {
        tag: {
            "signal": "AVERAGE", "condition": tag,
            "precision": avg(results[tag], "precision"),
            "recall":    avg(results[tag], "recall"),
            "f1":        avg(results[tag], "f1"),
        }
        for tag in CONFIGS
    }

    # --- Save JSON ---
    json_out = {
        "config": {"alpha": ALPHA, "n_signals": len(signals)},
    }
    for tag in CONFIGS:
        json_out[tag] = results[tag] + [avgs[tag]]
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # --- Build table ---
    tags = list(CONFIGS.keys())
    maps = {tag: {r["signal"]: r for r in results[tag] + [avgs[tag]]} for tag in tags}

    col = 7
    header = (
        f"\n{'Signal':<38}"
        + "".join(f" {LABELS[t]+'-F1':>{col+6}}" for t in tags)
        + "  Best"
    )
    sep = "-" * (38 + len(tags) * (col + 7) + 6)
    rows = [header, sep]

    sig_order = [r["signal"] for r in results[tags[0]]] + ["AVERAGE"]
    for sig in sig_order:
        vals = {t: maps[t].get(sig, {}) for t in tags}
        if not all(vals.values()):
            continue
        f1s  = {t: vals[t]["f1"] for t in tags}
        best = LABELS[max(f1s, key=f1s.get)]
        row  = f"{sig:<38}"
        for t in tags:
            row += f" {vals[t]['f1']:>{col+6}.4f}"
        row += f"  {best}"
        rows.append(row)

    rows.append(sep)

    avg_line = "   ".join(f"{LABELS[t]}={avgs[t]['f1']:.4f}" for t in tags)
    summary = "\n".join([
        "=" * 65,
        "Stage-1: CLIP fixed | DINOv2 fixed | DINOv2 adaptive (FFT window)",
        f"Dataset: NAB-AWS ({len(signals)} signals)   alpha={ALPHA}",
        "=" * 65,
    ] + rows + [
        "",
        f"Avg F1  {avg_line}",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ])

    print("\n" + summary)

    txt_path = OUTPUT_DIR / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Summary: {txt_path}")


if __name__ == "__main__":
    run()
