"""
EWMA smoothing alpha sweep on NAB-AWS — Stage-1 only.

Sweeps smoothing_alpha ∈ {0.1, 0.3, 0.5, 0.7, 1.0} for each backbone.
alpha=1.0 is identical to no smoothing (EWMA identity).

Backbones: clip_fixed / dino_fixed / dino_adaptive / mae_fixed

Usage:
    cd VLM4TS
    python experiments/nab_aws_smoothing.py [--backbone clip dino_fixed dino_adaptive mae]

Outputs:
  results/nab_aws_smoothing/
    results.json
    summary.txt
    checkpoints/
"""

import argparse, ast, json, pickle, sys
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

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_aws_smoothing"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA           = 0.01
SMOOTHING_ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

BASE = dict(window_step_ratio=4.0, image_size=(224, 224), alpha=ALPHA, verbose=False)

BACKBONE_CFGS = {
    "clip":          dict(cls=ViT4TS,      params=dict(**BASE, window_size=224, model_name="ViT-B-16",                   patch_size=16)),
    "dino_fixed":    dict(cls=ViT4TS_DINO, params=dict(**BASE, window_size=224, model_name="dinov2_vitb14",              patch_size=14, adaptive_window=False)),
    "dino_adaptive": dict(cls=ViT4TS_DINO, params=dict(**BASE, window_size=224, model_name="dinov2_vitb14",              patch_size=14, adaptive_window=True)),
    "mae":           dict(cls=ViT4TS_MAE,  params=dict(**BASE, window_size=224, model_name="vit_base_patch16_224.mae",   patch_size=16)),
}

LABELS = {"clip": "CLIP", "dino_fixed": "DINO", "dino_adaptive": "DINO-adp", "mae": "MAE"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt(sig, backbone, sa):
    sa_str = f"{sa:.1f}".replace(".","p")
    return CKPT_DIR / f"{sig}__{backbone}__sa{sa_str}.pkl"

def load_ckpt(sig, backbone, sa):
    p = _ckpt(sig, backbone, sa)
    return pickle.load(open(p,"rb")) if p.exists() else None

def save_ckpt(sig, backbone, sa, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(sig, backbone, sa),"wb"))

def load_gt():
    gt = {}
    with open(ANOMALY_CSV) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",",1)
            if len(parts)==2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _to_list(df):
    return df[["start","end"]].values.tolist() if len(df)>0 else []

def run_signal(sig, gt, backbone_tag, sa, det):
    cached = load_ckpt(sig, backbone_tag, sa)
    if cached:
        return cached["f1"]
    data = pd.read_csv(DATA_DIR / f"{sig}.csv")
    intervals = det.detect(data)
    metrics   = evaluate_intervals(gt, _to_list(intervals))
    f1 = round(metrics["F1"], 4)
    save_ckpt(sig, backbone_tag, sa, {"f1": f1, "p": round(metrics["precision"],4), "r": round(metrics["recall"],4)})
    return f1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", nargs="+", default=["mae"],
                   choices=list(BACKBONE_CFGS.keys()) + ["all"])
    return p.parse_args()

def run():
    args = parse_args()
    backbones = list(BACKBONE_CFGS.keys()) if "all" in args.backbone else args.backbone
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_gt  = load_gt()
    signals = sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.stem in all_gt)

    print("="*70)
    print(f"EWMA Smoothing Sweep — NAB-AWS  backbones={backbones}")
    print(f"smoothing_alpha: {SMOOTHING_ALPHAS}")
    print(f"signals: {len(signals)}")
    print("="*70)

    # results[backbone][sa] = list of per-signal F1
    results = {b: {sa: [] for sa in SMOOTHING_ALPHAS} for b in backbones}

    for backbone_tag in backbones:
        cfg = BACKBONE_CFGS[backbone_tag]
        print(f"\n{'='*50}")
        print(f"Backbone: {LABELS[backbone_tag]}")

        for sa in SMOOTHING_ALPHAS:
            sa_str = f"{sa:.1f}".replace(".","p")
            # check if all cached
            all_cached = all(_ckpt(s, backbone_tag, sa).exists() for s in signals)

            if all_cached:
                det = None
                print(f"  smoothing_alpha={sa:.1f} — 전부 캐시됨")
            else:
                print(f"  smoothing_alpha={sa:.1f} — 모델 초기화 중...")
                det = cfg["cls"](**cfg["params"], smoothing_alpha=sa)

            for sig in signals:
                gt = all_gt[sig]
                if det is not None:
                    f1 = run_signal(sig, gt, backbone_tag, sa, det)
                else:
                    cached = load_ckpt(sig, backbone_tag, sa)
                    f1 = cached["f1"] if cached else 0.0
                results[backbone_tag][sa].append(f1)

            avg = sum(results[backbone_tag][sa]) / len(results[backbone_tag][sa])
            print(f"    avg F1={avg:.4f}")

    # Summary table
    print("\n" + "="*70)
    print("EWMA Smoothing Alpha Sweep — Avg F1 per backbone")
    print("="*70)

    col = 8
    sa_headers = [f"α={sa:.1f}" for sa in SMOOTHING_ALPHAS]
    header = f"{'Backbone':<14}" + "".join(f" {h:>{col}}" for h in sa_headers) + "  Best α"
    sep = "-" * (14 + len(SMOOTHING_ALPHAS)*(col+1) + 8)
    rows = [header, sep]

    all_avgs = {}
    for backbone_tag in backbones:
        avgs = [sum(results[backbone_tag][sa])/len(results[backbone_tag][sa]) for sa in SMOOTHING_ALPHAS]
        all_avgs[backbone_tag] = avgs
        best_sa = SMOOTHING_ALPHAS[avgs.index(max(avgs))]
        row = f"{LABELS[backbone_tag]:<14}" + "".join(f" {v:>{col}.4f}" for v in avgs) + f"  α={best_sa:.1f}"
        rows.append(row)
    rows.append(sep)

    # Per-signal detail for each backbone
    detail_rows = []
    for backbone_tag in backbones:
        detail_rows.append(f"\n[{LABELS[backbone_tag]}] per-signal F1:")
        sig_header = f"  {'Signal':<35}" + "".join(f" {h:>{col}}" for h in sa_headers)
        detail_rows.append("  " + "-"*(35 + len(SMOOTHING_ALPHAS)*(col+1)))
        detail_rows.append(sig_header)
        for i, sig in enumerate(signals):
            row = f"  {sig:<35}"
            for sa in SMOOTHING_ALPHAS:
                row += f" {results[backbone_tag][sa][i]:>{col}.4f}"
            detail_rows.append(row)

    summary = "\n".join([
        "="*70,
        "EWMA Smoothing Sweep — NAB-AWS",
        f"backbones: {', '.join(LABELS[b] for b in backbones)}",
        f"alpha(detection)={ALPHA}",
        "="*70,
    ] + rows + detail_rows + [f"\nResults: {OUTPUT_DIR/'results.json'}"])

    print("\n".join(rows))
    print("\n".join(detail_rows))

    # Save JSON
    json_out = {"config": {"smoothing_alphas": SMOOTHING_ALPHAS, "backbones": backbones, "signals": signals}}
    for b in backbones:
        json_out[b] = {f"sa_{sa}": results[b][sa] for sa in SMOOTHING_ALPHAS}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    txt = OUTPUT_DIR / "summary.txt"
    open(txt, "w", encoding="utf-8").write(summary + "\n")
    print(f"\nSummary: {txt}")

if __name__ == "__main__":
    run()
