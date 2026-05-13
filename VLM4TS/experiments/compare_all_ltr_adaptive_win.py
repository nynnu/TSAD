"""
MAE + LTR k=5 + Adaptive Window — SMAP / NAB-AWS / MSL 통합 실험

실험 설계 (변수 하나씩 격리):
  Cond 1: Global ref  + Fixed 224     (baseline)
  Cond 2: LTR k=5    + Fixed 224     (LTR 효과)
  Cond 3: LTR k=5    + Adaptive Win  (adaptive window 효과만 추가)

MAE 모델 한 번만 로드하고 3개 데이터셋 순차 실행.

기존 체크포인트 재활용:
  SMAP   global  : results/smap/checkpoints/{ch}_mae_fixed.pkl
  SMAP   LTR k=5 : results/smap_local_ref/checkpoints/{ch}__mae__k5.pkl
  SMAP   AdpWin  : results/smap_ltr_adaptive_win/checkpoints/{ch}__ltr_k5_adaptive.pkl
  NAB    global  : results/nab_aws_local_ref/checkpoints/{sig}__mae__k0.pkl
  NAB    LTR k=5 : results/nab_aws_local_ref/checkpoints/{sig}__mae__k5.pkl
  MSL    global  : results/clip_vs_dino/checkpoints/{ch}_mae.pkl
  MSL    LTR k=5 : results/msl_local_ref/checkpoints/{ch}__mae__k5.pkl

No existing files modified.
Outputs: results/all_ltr_adaptive_win/
           results.json, summary.txt, checkpoints/

Usage:
    cd VLM4TS
    python experiments/compare_all_ltr_adaptive_win.py [--dataset smap nab msl]
"""

import argparse, ast, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_local import ViT4TS_Local
from models.vit4ts_ltr_adaptive_win_v2 import ViT4TS_LTR_AdaptiveWin_V2 as ViT4TS_LTR_AdaptiveWin
from evaluation.evaluate import evaluate_intervals
from preprocessing.preprocess import preprocess_time_series
from preprocessing.data_utils import orion_to_internal

OUTPUT_DIR = ROOT / "results" / "all_ltr_adaptive_win"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
ALPHA      = 0.01

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CFG = {
    "smap": {
        "data_dir":   ROOT / "data" / "SMAP",
        "channels":   ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
                       'F-1','F-2','F-3','T-1','T-2','T-3','R-1'],
        "ckpt_global": lambda ch: ROOT / "results" / "smap" / "checkpoints" / f"{ch}_mae_fixed.pkl",
        "ckpt_ltr":    lambda ch: ROOT / "results" / "smap_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl",
        "ckpt_adp":    lambda ch: ROOT / "results" / "smap_ltr_adaptive_win" / "checkpoints" / f"{ch}__ltr_k5_adaptive.pkl",
    },
    "nab": {
        "data_dir":   ROOT / "data" / "realAWSCloudwatch",
        "channels":   None,   # loaded from data dir
        "ckpt_global": lambda sig: ROOT / "results" / "nab_aws_local_ref" / "checkpoints" / f"{sig}__mae__k0.pkl",
        "ckpt_ltr":    lambda sig: ROOT / "results" / "nab_aws_local_ref" / "checkpoints" / f"{sig}__mae__k5.pkl",
        "ckpt_adp":    lambda sig: None,  # no existing
    },
    "msl": {
        "data_dir":   ROOT / "data" / "MSL",
        "channels":   ['P-11','T-12','D-15','C-1','F-8','F-7','T-13','D-16','T-8','P-14','D-14'],
        "ckpt_global": lambda ch: ROOT / "results" / "clip_vs_dino" / "checkpoints" / f"{ch}_mae.pkl",
        "ckpt_ltr":    lambda ch: ROOT / "results" / "msl_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl",
        "ckpt_adp":    lambda ch: None,  # no existing
    },
}

CONDITIONS  = ["global", "ltr_k5", "ltr_k5_adp"]
COND_LABELS = {
    "global":     "Global+Fixed224",
    "ltr_k5":     "LTR k=5+Fixed224",
    "ltr_k5_adp": "LTR k=5+AdpWin",
}

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

# ---------------------------------------------------------------------------
# Anomaly GT loader
# ---------------------------------------------------------------------------

def load_gt(anomaly_csv):
    gt = {}
    with open(anomaly_csv) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _to_list(df):
    return df[["start","end"]].values.tolist() if len(df) > 0 else []

# ---------------------------------------------------------------------------
# Checkpoint helpers (per dataset)
# ---------------------------------------------------------------------------

def load_ckpt_for(ch, cond, ds_name):
    cfg = DATASET_CFG[ds_name]
    paths = []
    if cond == "global":
        p = cfg["ckpt_global"](ch)
        if p: paths.append(p)
    elif cond == "ltr_k5":
        p = cfg["ckpt_ltr"](ch)
        if p: paths.append(p)
    elif cond == "ltr_k5_adp":
        p = cfg["ckpt_adp"](ch)
        if p: paths.append(p)
    # always check this experiment's own ckpt last
    paths.append(CKPT_DIR / f"{ds_name}__{ch}__{cond}.pkl")

    for path in paths:
        if path and Path(path).exists():
            d = pickle.load(open(path, "rb"))
            return {"f1": d.get("f1", d.get("F1", 0)),
                    "p":  d.get("p", d.get("precision", 0)),
                    "r":  d.get("r", d.get("recall", 0)),
                    "_from": str(path)}
    return None

def save_ckpt_for(ch, cond, ds_name, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{ds_name}__{ch}__{cond}.pkl", "wb"))

# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------

def run_dataset(ds_name, device, all_gt, detectors):
    cfg      = DATASET_CFG[ds_name]
    data_dir = cfg["data_dir"]

    if cfg["channels"] is None:
        channels = sorted(f.stem for f in data_dir.glob("*.csv") if f.stem in all_gt)
    else:
        channels = [ch for ch in cfg["channels"] if ch in all_gt]

    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name.upper()}  ({len(channels)} channels/signals)")
    print(f"{'='*65}")

    results = {c: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        lbl = COND_LABELS[cond]
        all_cached = all(load_ckpt_for(ch, cond, ds_name) is not None for ch in channels)

        if all_cached:
            print(f"\n[{lbl}] 전부 캐시됨")
            for ch in channels:
                results[cond].append(load_ckpt_for(ch, cond, ds_name)["f1"])
            continue

        print(f"\n[{lbl}] 실행 중...")
        det = detectors[cond]

        for ch in channels:
            gt = all_gt.get(ch, [])
            if not gt:
                results[cond].append(0.0)
                continue

            cached = load_ckpt_for(ch, cond, ds_name)
            if cached:
                f1  = cached["f1"]
                src = Path(cached["_from"]).parent.parent.name
                print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
            else:
                data = pd.read_csv(data_dir / f"{ch}.csv")
                print(f"\n  {ch}: running [{lbl}]...")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt_for(ch, cond, ds_name,
                              {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
                print(f"    F1={f1:.4f}")
            results[cond].append(f1)

    return channels, results

# ---------------------------------------------------------------------------
# Print summary table for one dataset
# ---------------------------------------------------------------------------

def print_summary(ds_name, channels, results):
    w = 17
    avgs = {c: sum(results[c])/len(results[c]) if results[c] else 0 for c in CONDITIONS}

    print(f"\n[{ds_name.upper()} Summary]")
    hdr = f"  {'Channel':<38}" + "".join(f" {COND_LABELS[c]:>{w}}" for c in CONDITIONS)
    print(hdr)
    print("  " + "-"*(38 + len(CONDITIONS)*(w+1)))

    for i, ch in enumerate(channels):
        vals = [results[c][i] if i < len(results[c]) else 0 for c in CONDITIONS]
        best = max(vals)
        row  = f"  {ch:<38}"
        for v in vals:
            marker = "*" if (v == best and best > 0) else " "
            row += f" {v:>{w-1}.4f}{marker}"
        print(row)

    print("  " + "-"*(38 + len(CONDITIONS)*(w+1)))
    avg_row = f"  {'AVERAGE':<38}" + "".join(f" {avgs[c]:>{w}.4f}" for c in CONDITIONS)
    print(avg_row)

    delta = avgs["ltr_k5_adp"] - avgs["ltr_k5"]
    print(f"\n  Adaptive window effect (Cond3-Cond2): {delta:+.4f}")
    return avgs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", default=["smap","nab","msl"],
                   choices=["smap","nab","msl","all"])
    return p.parse_args()

def run():
    args     = parse_args()
    datasets = ["smap","nab","msl"] if "all" in args.dataset else args.dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt(ROOT / "data" / "anomalies.csv")

    print("="*65)
    print(f"MAE + LTR k=5 + Adaptive Window — {', '.join(d.upper() for d in datasets)}")
    print(f"device={device}")
    print("="*65)

    # 모델 한 번만 로드
    print("\n모델 초기화 중 (MAE × 3 conditions)...")
    backbone_g   = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
    backbone_ltr = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
    backbone_adp = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)

    detectors = {
        "global":     ViT4TS_Local(backbone=backbone_g,   patch_size=16,
                                   local_k=0, device=str(device), **BASE_PARAMS),
        "ltr_k5":     ViT4TS_Local(backbone=backbone_ltr, patch_size=16,
                                   local_k=5, device=str(device), **BASE_PARAMS),
        "ltr_k5_adp": ViT4TS_LTR_AdaptiveWin(
                          backbone=backbone_adp, patch_size=16, local_k=5,
                          device=str(device), window_step_ratio=4.0,
                          image_size=(224,224), alpha=ALPHA,
                          batch_size=32,
                          smoothing_alpha=1.0, verbose=True),
    }

    all_results = {}
    all_avgs    = {}

    for ds in datasets:
        channels, results       = run_dataset(ds, device, all_gt, detectors)
        avgs                    = print_summary(ds, channels, results)
        all_results[ds]         = {"channels": channels, "results": results}
        all_avgs[ds]            = avgs

    # Cross-dataset summary
    print("\n" + "="*65)
    print("Cross-dataset Summary — avg F1")
    print("="*65)
    w = 17
    hdr = f"{'Dataset':<10}" + "".join(f" {COND_LABELS[c]:>{w}}" for c in CONDITIONS)
    print(hdr)
    print("-"*(10 + len(CONDITIONS)*(w+1)))
    for ds in datasets:
        row = f"{ds.upper():<10}"
        for c in CONDITIONS:
            row += f" {all_avgs[ds][c]:>{w}.4f}"
        delta = all_avgs[ds]["ltr_k5_adp"] - all_avgs[ds]["ltr_k5"]
        row += f"   adp_effect={delta:+.4f}"
        print(row)

    # Save JSON
    json_out = {"config": {"alpha": ALPHA, "local_k": 5, "datasets": datasets}}
    for ds in datasets:
        ch_list = all_results[ds]["channels"]
        res     = all_results[ds]["results"]
        json_out[ds] = {
            c: {"f1_per_channel": dict(zip(ch_list, res[c])),
                "avg_f1": round(all_avgs[ds][c], 4)}
            for c in CONDITIONS
        }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # Save summary txt
    lines = [
        "="*65,
        "MAE + LTR k=5 + Adaptive Window --- SMAP / NAB-AWS / MSL",
        f"alpha={ALPHA}  local_k=5",
        "="*65, "", hdr,
        "-"*(10 + len(CONDITIONS)*(w+1)),
    ] + [
        f"{ds.upper():<10}" + "".join(f" {all_avgs[ds][c]:>{w}.4f}" for c in CONDITIONS)
        for ds in datasets
    ] + [f"\nResults: {OUTPUT_DIR / 'results.json'}"]

    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
