"""
CLIP + local k=5  vs  DINO + local k=5  vs  MAE + local k=5  on NAB-AWS.

Goal: check if CLIP or DINO beats MAE local k=5 (baseline = 0.6272).

Existing checkpoints reused:
  MAE k=5  : results/nab_aws_local_ref/checkpoints/{sig}__mae__k5.pkl

No existing files modified.
Outputs: results/nab_aws_clip_dino_local_ref/
           results.json, summary.txt, checkpoints/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.clip_vision import CLIP_AD
from models.dino_vision import DINO_AD
from models.mae_vision  import MAE_AD
from models.vit4ts_local import ViT4TS_Local
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_aws_clip_dino_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA = 0.01
K     = 5

BACKBONE_CFG = {
    "clip": dict(cls=CLIP_AD, kwargs=dict(model_name="ViT-B-16"),  patch_size=16, label="CLIP"),
    "dino": dict(cls=DINO_AD, kwargs=dict(model_name="dinov2_vitb14"), patch_size=14, label="DINO"),
}

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

MAE_BASELINE = 0.6272

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_fallbacks(sig, bb):
    return [
        # 기존 nab_aws_local_ref 결과 재활용 (있으면)
        ROOT / "results" / "nab_aws_local_ref" / "checkpoints" / f"{sig}__{bb}__k{K}.pkl",
        # 이 실험 전용 저장소
        CKPT_DIR / f"{sig}__{bb}__k{K}.pkl",
    ]

def load_ckpt(sig, bb):
    for path in _ckpt_fallbacks(sig, bb):
        if path.exists():
            d = pickle.load(open(path, "rb"))
            return {"f1": d["f1"], "p": d.get("p", 0), "r": d.get("r", 0)}
    return None

def save_ckpt(sig, bb, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{sig}__{bb}__k{K}.pkl", "wb"))

def load_mae_ckpt(sig):
    p = ROOT / "results" / "nab_aws_local_ref" / "checkpoints" / f"{sig}__mae__k{K}.pkl"
    if p.exists():
        d = pickle.load(open(p, "rb"))
        return d["f1"]
    return None

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

    print("="*70)
    print(f"CLIP / DINO  local k={K}  —  NAB-AWS ({len(signals)} signals)")
    print(f"Baseline: MAE local k={K} = {MAE_BASELINE}")
    print("="*70)

    results = {bb: [] for bb in BACKBONE_CFG}

    for bb, cfg in BACKBONE_CFG.items():
        lbl = cfg["label"]
        all_cached = all(load_ckpt(s, bb) is not None for s in signals)

        if all_cached:
            det = None
            print(f"\n[{lbl}] 전부 캐시됨")
        else:
            print(f"\n[{lbl}] 모델 초기화 중...")
            backbone = cfg["cls"](device=device, **cfg["kwargs"])
            det = ViT4TS_Local(
                backbone=backbone, patch_size=cfg["patch_size"],
                local_k=K, device=str(device), **DETECTOR_PARAMS,
            )

        for sig in signals:
            gt = all_gt[sig]
            cached = load_ckpt(sig, bb)
            if cached:
                f1 = cached["f1"]
                print(f"  {sig}: ckpt  F1={f1:.4f}")
            else:
                data = pd.read_csv(DATA_DIR / f"{sig}.csv")
                print(f"  {sig}: running ...")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt(sig, bb, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
                print(f"    F1={f1:.4f}")
            results[bb].append(f1)

        avg = sum(results[bb]) / len(results[bb])
        print(f"  [{lbl}] avg F1 = {avg:.4f}  ({'BETTER' if avg > MAE_BASELINE else 'worse'} than MAE {avg-MAE_BASELINE:+.4f})")

    # MAE 기존 결과 로드
    mae_f1s = [load_mae_ckpt(s) for s in signals]
    mae_avg = sum(f for f in mae_f1s if f is not None) / len([f for f in mae_f1s if f is not None])

    # Summary table
    avgs = {bb: sum(results[bb])/len(results[bb]) for bb in BACKBONE_CFG}

    print("\n" + "="*70)
    print(f"{'Signal':<45} {'CLIP':>7} {'DINO':>7} {'MAE':>7}")
    print("-"*66)
    for i, sig in enumerate(signals):
        mae_v = f"{mae_f1s[i]:.4f}" if mae_f1s[i] is not None else "  -- "
        print(f"  {sig:<43} {results['clip'][i]:>7.4f} {results['dino'][i]:>7.4f} {mae_v:>7}")
    print("-"*66)
    print(f"  {'AVERAGE':<43} {avgs['clip']:>7.4f} {avgs['dino']:>7.4f} {mae_avg:>7.4f}")
    print()
    print(f"  Baseline MAE local k={K}: {MAE_BASELINE}")
    for bb in BACKBONE_CFG:
        lbl = BACKBONE_CFG[bb]["label"]
        diff = avgs[bb] - MAE_BASELINE
        print(f"  {lbl:<6} local k={K}    : {avgs[bb]:.4f}  ({'BETTER' if diff>0 else 'worse'}, {diff:+.4f})")

    # Save
    json_out = {
        "config": {"local_k": K, "alpha": ALPHA, "signals": signals},
        "clip":  {"f1_per_signal": dict(zip(signals, results["clip"])),  "avg_f1": round(avgs["clip"],4)},
        "dino":  {"f1_per_signal": dict(zip(signals, results["dino"])),  "avg_f1": round(avgs["dino"],4)},
        "mae_k5_baseline": {"f1_per_signal": dict(zip(signals, mae_f1s)), "avg_f1": round(mae_avg,4)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*70,
        f"CLIP / DINO  local k={K}  ---  NAB-AWS",
        f"alpha(detection)={ALPHA}",
        "="*70,
        f"{'Signal':<45} {'CLIP':>7} {'DINO':>7} {'MAE':>7}",
        "-"*66,
    ] + [
        f"  {s:<43} {results['clip'][i]:>7.4f} {results['dino'][i]:>7.4f} {(str(round(mae_f1s[i],4)) if mae_f1s[i] is not None else '  --'):>7}"
        for i, s in enumerate(signals)
    ] + [
        "-"*66,
        f"  {'AVERAGE':<43} {avgs['clip']:>7.4f} {avgs['dino']:>7.4f} {mae_avg:>7.4f}",
        "",
        f"  Baseline MAE local k={K}: {MAE_BASELINE}",
    ] + [
        f"  {BACKBONE_CFG[bb]['label']:<6} local k={K}    : {avgs[bb]:.4f}  ({'BETTER' if avgs[bb]-MAE_BASELINE>0 else 'worse'}, {avgs[bb]-MAE_BASELINE:+.4f})"
        for bb in BACKBONE_CFG
    ] + [f"\nResults: {OUTPUT_DIR / 'results.json'}"]

    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
