"""
Local Temporal Reference sweep on NAB-AWS — Stage-1 only.

Compares k ∈ {5, 10, 20, global} × {CLIP, DINO, MAE} backbones.
global = original whole-sequence median (k=0 in ViT4TS_Local).

No existing source files are modified.
New classes used: ViT4TS_Local (vit4ts_local.py)
                  CLIP_AD / DINO_AD / MAE_AD as injected backbones.

Outputs:
  results/nab_aws_local_ref/
    results.json
    summary.txt
    checkpoints/

Usage:
    cd VLM4TS
    python experiments/compare_nab_aws_local_ref.py [--backbone clip dino mae]
"""

import argparse, ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.clip_vision import CLIP_AD
from models.dino_vision import DINO_AD
from models.mae_vision import MAE_AD
from models.vit4ts_local import ViT4TS_Local
from evaluation.evaluate import evaluate_intervals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_aws_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA = 0.01
K_VALUES = [5, 10, 20, 0]    # 0 = global (original behaviour)
K_LABELS = {5: "k=5", 10: "k=10", 20: "k=20", 0: "global"}

BACKBONE_CFG = {
    "clip": dict(
        cls=CLIP_AD, kwargs=dict(model_name="ViT-B-16"),
        patch_size=16, label="CLIP",
    ),
    "dino": dict(
        cls=DINO_AD, kwargs=dict(model_name="dinov2_vitb14"),
        patch_size=14, label="DINO",
    ),
    "mae": dict(
        cls=MAE_AD, kwargs=dict(model_name="vit_base_patch16_224.mae"),
        patch_size=16, label="MAE",
    ),
}

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt(sig, bb, k):
    return CKPT_DIR / f"{sig}__{bb}__k{k}.pkl"

def load_ckpt(sig, bb, k):
    p = _ckpt(sig, bb, k)
    return pickle.load(open(p,"rb")) if p.exists() else None

def save_ckpt(sig, bb, k, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(sig, bb, k),"wb"))

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

def make_detector(bb_tag, k, device):
    cfg = BACKBONE_CFG[bb_tag]
    backbone = cfg["cls"](device=device, **cfg["kwargs"])
    return ViT4TS_Local(
        backbone=backbone, patch_size=cfg["patch_size"],
        local_k=k, device=str(device), **DETECTOR_PARAMS,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", nargs="+", default=["mae"],
                   choices=list(BACKBONE_CFG) + ["all"])
    return p.parse_args()


def run():
    args = parse_args()
    backbones = list(BACKBONE_CFG) if "all" in args.backbone else args.backbone
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()
    signals = sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.stem in all_gt)

    print("="*70)
    print(f"Local Temporal Reference sweep — NAB-AWS")
    print(f"backbones={backbones}  k-values={K_VALUES}  signals={len(signals)}")
    print("="*70)

    # results[bb][k] = list of per-signal F1
    results = {bb: {k: [] for k in K_VALUES} for bb in backbones}

    for bb in backbones:
        lbl = BACKBONE_CFG[bb]["label"]
        print(f"\n{'='*55}\nBackbone: {lbl}")

        for k in K_VALUES:
            klbl = K_LABELS[k]
            all_cached = all(_ckpt(s, bb, k).exists() for s in signals)

            if all_cached:
                det = None
                print(f"\n  [{klbl}] 전부 캐시됨")
            else:
                print(f"\n  [{klbl}] 초기화 중...")
                det = make_detector(bb, k, device)

            for sig in signals:
                gt = all_gt[sig]
                cached = load_ckpt(sig, bb, k)
                if cached is not None:
                    f1 = cached["f1"]
                    print(f"    {sig}: ckpt F1={f1:.4f}")
                else:
                    data = pd.read_csv(DATA_DIR / f"{sig}.csv")
                    print(f"    {sig}: running ...")
                    ivs   = det.detect(data)
                    m     = evaluate_intervals(gt, _to_list(ivs))
                    f1    = round(m["F1"], 4)
                    save_ckpt(sig, bb, k, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
                    print(f"      F1={f1:.4f}")
                results[bb][k].append(f1)

            avg = sum(results[bb][k]) / len(results[bb][k])
            print(f"  [{klbl}] avg F1 = {avg:.4f}")

    # Save JSON
    json_out = {"config": {"k_values": K_VALUES, "backbones": backbones, "signals": signals}}
    for bb in backbones:
        json_out[bb] = {f"k{k}": results[bb][k] for k in K_VALUES}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # Summary table
    col = 8
    k_hdrs = [K_LABELS[k] for k in K_VALUES]
    header = f"{'Backbone':<8}" + "".join(f" {h:>{col}}" for h in k_hdrs) + "  Best k"
    sep = "-" * (8 + len(K_VALUES)*(col+1) + 8)
    rows = [header, sep]

    for bb in backbones:
        avgs = [sum(results[bb][k])/len(results[bb][k]) for k in K_VALUES]
        best_k = K_VALUES[avgs.index(max(avgs))]
        row = f"{BACKBONE_CFG[bb]['label']:<8}" + "".join(f" {v:>{col}.4f}" for v in avgs)
        row += f"  {K_LABELS[best_k]}"
        rows.append(row)
    rows.append(sep)

    # Per-signal detail
    detail = []
    for bb in backbones:
        detail.append(f"\n[{BACKBONE_CFG[bb]['label']}] per-signal:")
        detail.append(f"  {'Signal':<38}" + "".join(f" {h:>{col}}" for h in k_hdrs))
        detail.append("  " + "-"*(38 + len(K_VALUES)*(col+1)))
        for j, sig in enumerate(signals):
            row = f"  {sig:<38}" + "".join(f" {results[bb][k][j]:>{col}.4f}" for k in K_VALUES)
            detail.append(row)

    summary = "\n".join([
        "="*70,
        "Local Temporal Reference — NAB-AWS",
        f"backbones: {', '.join(BACKBONE_CFG[b]['label'] for b in backbones)}",
        f"alpha(detection)={ALPHA}",
        "="*70,
    ] + rows + detail + [f"\nResults: {OUTPUT_DIR/'results.json'}"])

    print("\n" + summary)
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write(summary + "\n")
    print(f"Summary: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    run()
