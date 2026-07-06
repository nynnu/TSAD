"""
MAE + Local Temporal Reference (k=5) on SMAP 14 channels.
Compares: MAE global (기존) vs MAE local k=5

No existing files modified.
Uses ViT4TS_Local with MAE_AD backbone.

Outputs:
  results/smap_local_ref/
    results.json
    summary.txt
    checkpoints/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_local import ViT4TS_Local
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA    = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']
K_VALUES = [5, 0]   # 0 = global
K_LABELS = {5: "local k=5", 0: "global"}

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

# ---------------------------------------------------------------------------
def _ckpt(ch, k): return CKPT_DIR / f"{ch}__mae__k{k}.pkl"
def load_ckpt(ch, k):
    p = _ckpt(ch, k)
    return pickle.load(open(p,"rb")) if p.exists() else None
def save_ckpt(ch, k, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(ch, k),"wb"))

def load_gt():
    gt = {}
    with open(ANOMALY_CSV) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts)==2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _to_list(df):
    return df[["start","end"]].values.tolist() if len(df)>0 else []

# ---------------------------------------------------------------------------
def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*60)
    print("MAE Local Ref (k=5) vs Global — SMAP 14 channels")
    print("="*60)

    results = {k: [] for k in K_VALUES}

    for k in K_VALUES:
        klbl = K_LABELS[k]
        all_cached = all(_ckpt(ch, k).exists() for ch in CHANNELS)

        if all_cached:
            det = None
            print(f"\n[{klbl}] 전부 캐시됨")
        else:
            print(f"\n[{klbl}] 초기화 중...")
            backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
            det = ViT4TS_Local(backbone=backbone, patch_size=16,
                               local_k=k, device=str(device), **DETECTOR_PARAMS)

        for ch in CHANNELS:
            gt = all_gt.get(ch, [])
            if not gt:
                print(f"  SKIP {ch}")
                continue

            cached = load_ckpt(ch, k)
            if cached:
                f1 = cached["f1"]
                print(f"  {ch}: ckpt  F1={f1:.4f}")
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n  {ch}: running ...")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt(ch, k, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
                print(f"    F1={f1:.4f}")
            results[k].append(f1)

    # Table
    print("\n" + "="*60)
    print(f"{'Channel':<10} {'global':>8} {'local k=5':>10}  Delta")
    print("-"*40)

    for i, ch in enumerate(CHANNELS):
        g = results[0][i] if i < len(results[0]) else 0
        l = results[5][i] if i < len(results[5]) else 0
        delta = l - g
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"{ch:<10} {g:>8.4f} {l:>10.4f}  {delta:+.4f} {marker}")

    print("-"*40)
    avg_g = sum(results[0])/len(results[0])
    avg_l = sum(results[5])/len(results[5])
    print(f"{'AVERAGE':<10} {avg_g:>8.4f} {avg_l:>10.4f}  {avg_l-avg_g:+.4f}")

    # Save
    json_out = {"config": {"alpha": ALPHA, "channels": CHANNELS},
                "global": results[0], "local_k5": results[5]}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")

if __name__ == "__main__":
    run()
