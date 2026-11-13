"""
Stage-1: CLIP fixed / DINO fixed / DINO adaptive / MAE — SMAP 14 channels.

Channels (user-specified, P-2 excluded — no data file):
  P: P-1, P-3, P-4, P-7
  D: D-1, D-2, D-3
  F: F-1, F-2, F-3
  T: T-1, T-2, T-3
  R: R-1

Outputs:
  results/smap/
    results.json
    summary.txt
    checkpoints/

Usage:
    cd VLM4TS
    python experiments/compare_smap.py
"""

import ast, json, pickle, sys
from pathlib import Path

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

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA = 0.01

CHANNELS = [
    "P-1", "P-3", "P-4", "P-7",
    "D-1", "D-2", "D-3",
    "F-1", "F-2", "F-3",
    "T-1", "T-2", "T-3",
    "R-1",
]

BASE = dict(window_step_ratio=4.0, image_size=(224, 224), alpha=ALPHA, verbose=True)

CONFIGS = {
    "clip_fixed": dict(
        cls=ViT4TS,
        params=dict(**BASE, window_size=224, model_name="ViT-B-16", patch_size=16),
    ),
    "dino_fixed": dict(
        cls=ViT4TS_DINO,
        params=dict(**BASE, window_size=224, model_name="dinov2_vitb14",
                    patch_size=14, adaptive_window=False),
    ),
    "dino_adaptive": dict(
        cls=ViT4TS_DINO,
        params=dict(**BASE, window_size=224, model_name="dinov2_vitb14",
                    patch_size=14, adaptive_window=True),
    ),
    "mae_fixed": dict(
        cls=ViT4TS_MAE,
        params=dict(**BASE, window_size=224, model_name="vit_base_patch16_224.mae",
                    patch_size=16),
    ),
}

LABELS = {
    "clip_fixed":    "CLIP",
    "dino_fixed":    "DINO",
    "dino_adaptive": "DINO-adp",
    "mae_fixed":     "MAE",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt(name): return CKPT_DIR / f"{name}.pkl"

def load_ckpt(name):
    p = _ckpt(name)
    return pickle.load(open(p, "rb")) if p.exists() else None

def save_ckpt(name, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(name), "wb"))

def load_gt():
    gt = {}
    with open(ANOMALY_CSV) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _to_list(df):
    return df[["start", "end"]].values.tolist() if len(df) > 0 else []

def run_channel(ch, gt, detector, tag):
    key = f"{ch}_{tag}"
    cached = load_ckpt(key)
    if cached:
        print(f"  [{tag:10s}] {ch}: checkpoint  F1={cached['f1']:.4f}")
        return cached
    data = pd.read_csv(DATA_DIR / f"{ch}.csv")
    print(f"\n  [{tag:10s}] {ch}: running Stage-1 ...")
    intervals = detector.detect(data)
    metrics   = evaluate_intervals(gt, _to_list(intervals))
    result = {
        "channel": ch, "condition": tag,
        "n_gt": len(gt), "n_detected": len(intervals),
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
    print("Stage-1: CLIP / DINO / DINO-adp / MAE — SMAP 14 channels")
    print("=" * 65)

    all_gt = load_gt()

    # Init only detectors with uncached channels
    detectors = {}
    for tag, cfg in CONFIGS.items():
        if all(_ckpt(f"{ch}_{tag}").exists() for ch in CHANNELS):
            print(f"\n[{tag}] 전부 캐시됨 — 로드만 합니다.")
        else:
            print(f"\nInitialising {LABELS[tag]} ...")
            detectors[tag] = cfg["cls"](**cfg["params"])

    results = {tag: [] for tag in CONFIGS}

    for ch in CHANNELS:
        gt = all_gt.get(ch, [])
        if not gt:
            print(f"\nSKIP {ch}: GT 없음"); continue
        print(f"\n{'='*50}\nChannel: {ch}  ({len(gt)} GT intervals)")
        for tag in CONFIGS:
            det = detectors.get(tag)
            if det:
                results[tag].append(run_channel(ch, gt, det, tag))
            else:
                r = load_ckpt(f"{ch}_{tag}")
                if r:
                    results[tag].append(r)
                    print(f"  [{tag:10s}] {ch}: checkpoint  F1={r['f1']:.4f}")

    # Averages
    def avg(rows, k): return round(sum(r[k] for r in rows)/len(rows), 4) if rows else 0.0
    avgs = {tag: {"channel":"AVERAGE","condition":tag,
                  "precision":avg(results[tag],"precision"),
                  "recall":avg(results[tag],"recall"),
                  "f1":avg(results[tag],"f1")} for tag in CONFIGS}

    # Save JSON
    json_out = {"config": {"alpha": ALPHA, "channels": CHANNELS}}
    for tag in CONFIGS:
        json_out[tag] = results[tag] + [avgs[tag]]
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # Table
    tags = list(CONFIGS.keys())
    maps = {tag: {r["channel"]: r for r in results[tag]+[avgs[tag]]} for tag in tags}

    col = 7
    header = f"\n{'Channel':<10}" + "".join(f" {LABELS[t]:>{col+1}}" for t in tags) + "  Best"
    sep = "-" * (10 + len(tags)*(col+2) + 6)
    rows = [header, sep]

    for ch in CHANNELS + ["AVERAGE"]:
        vals = {t: maps[t].get(ch, {}) for t in tags}
        if not all(vals.values()): continue
        f1s  = {t: vals[t]["f1"] for t in tags}
        best = LABELS[max(f1s, key=f1s.get)]
        row  = f"{ch:<10}" + "".join(f" {vals[t]['f1']:>{col+1}.4f}" for t in tags) + f"  {best}"
        rows.append(row)
    rows.append(sep)

    avg_line = "  ".join(f"{LABELS[t]}={avgs[t]['f1']:.4f}" for t in tags)
    summary = "\n".join([
        "="*65,
        "Stage-1: CLIP | DINO | DINO-adaptive | MAE — SMAP 14 channels",
        f"alpha={ALPHA}",
        "="*65,
    ] + rows + ["", f"Avg F1  {avg_line}", f"\nResults: {OUTPUT_DIR/'results.json'}"])

    print("\n" + summary)
    txt = OUTPUT_DIR / "summary.txt"
    open(txt, "w", encoding="utf-8").write(summary + "\n")
    print(f"Summary: {txt}")

if __name__ == "__main__":
    run()
