"""
MAE + EWMA smoothing α=0.3 + Local Temporal Reference k=5 on NAB-AWS.

Goal: check if combining smoothing + local ref beats MAE local k=5 alone (0.6272).

Baseline references:
  MAE local k=5  (no smoothing) : 0.6272  [results/nab_aws_local_ref/]
  MAE smoothing α=0.3 (global)  : 0.5331  [results/nab_aws_smoothing/]

No existing files modified.
Outputs: results/nab_aws_smoothing_local_ref/
           results.json, summary.txt, checkpoints/
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

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_aws_smoothing_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA          = 0.01
SMOOTHING_ALPHA = 0.3
K              = 5

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=SMOOTHING_ALPHA, verbose=True,
)

# ---------------------------------------------------------------------------
# Checkpoint helpers — fallback to existing results where possible
# ---------------------------------------------------------------------------

def _ckpt_fallbacks(sig):
    """탐색 순서: 이 실험 전용 폴더 → 없으면 새로 계산."""
    return [CKPT_DIR / f"{sig}__mae__sa0p3__k5.pkl"]

def load_ckpt(sig):
    for path in _ckpt_fallbacks(sig):
        if path.exists():
            d = pickle.load(open(path, "rb"))
            return d
    return None

def save_ckpt(sig, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{sig}__mae__sa0p3__k5.pkl", "wb"))

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
    return df[["start","end"]].values.tolist() if len(df) > 0 else []

# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()
    signals = sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.stem in all_gt)

    print("="*65)
    print(f"MAE  smoothing α={SMOOTHING_ALPHA}  local k={K}  —  NAB-AWS ({len(signals)} signals)")
    print(f"Baseline: MAE local k=5 (no smoothing) = 0.6272")
    print("="*65)

    all_cached = all(load_ckpt(s) is not None for s in signals)
    if all_cached:
        det = None
        print("전부 캐시됨")
    else:
        print("모델 초기화 중...")
        backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        det = ViT4TS_Local(
            backbone=backbone, patch_size=16,
            local_k=K, device=str(device), **DETECTOR_PARAMS,
        )

    f1_list = []
    for sig in signals:
        gt = all_gt[sig]
        cached = load_ckpt(sig)
        if cached:
            f1 = cached["f1"]
            print(f"  {sig}: ckpt  F1={f1:.4f}")
        else:
            data = pd.read_csv(DATA_DIR / f"{sig}.csv")
            print(f"  {sig}: running ...")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(sig, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
            print(f"    F1={f1:.4f}")
        f1_list.append(f1)

    avg = sum(f1_list) / len(f1_list)

    # Summary
    print("\n" + "="*65)
    print(f"{'Signal':<45} {'F1':>6}")
    print("-"*52)
    for sig, f1 in zip(signals, f1_list):
        print(f"  {sig:<43} {f1:.4f}")
    print("-"*52)
    print(f"  {'AVERAGE':<43} {avg:.4f}")
    print()
    print(f"  MAE local k=5  (no smoothing)  : 0.6272")
    print(f"  MAE smoothing α=0.3 (global)   : 0.5331")
    print(f"  MAE α=0.3 + local k=5 [THIS]   : {avg:.4f}  {'✓ BETTER' if avg > 0.6272 else '✗ worse'}")

    json_out = {
        "config": {"smoothing_alpha": SMOOTHING_ALPHA, "local_k": K, "alpha": ALPHA, "signals": signals},
        "f1_per_signal": dict(zip(signals, f1_list)),
        "avg_f1": round(avg, 4),
        "baseline_local_k5_no_smoothing": 0.6272,
        "baseline_smoothing_03_global": 0.5331,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    summary_lines = [
        "="*65,
        f"MAE  smoothing α={SMOOTHING_ALPHA}  local k={K}  —  NAB-AWS",
        f"alpha(detection)={ALPHA}",
        "="*65,
        f"{'Signal':<45} {'F1':>6}",
        "-"*52,
    ] + [f"  {s:<43} {f:.4f}" for s, f in zip(signals, f1_list)] + [
        "-"*52,
        f"  {'AVERAGE':<43} {avg:.4f}",
        "",
        f"  MAE local k=5  (no smoothing)  : 0.6272",
        f"  MAE smoothing α=0.3 (global)   : 0.5331",
        f"  MAE α=0.3 + local k=5 [THIS]   : {avg:.4f}  {'BETTER' if avg > 0.6272 else 'worse'}",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    txt = OUTPUT_DIR / "summary.txt"
    open(txt, "w", encoding="utf-8").write("\n".join(summary_lines) + "\n")
    print(f"\nSummary: {txt}")

if __name__ == "__main__":
    run()
