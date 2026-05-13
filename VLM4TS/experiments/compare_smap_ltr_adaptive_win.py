"""
MAE + LTR k=5 + Adaptive Window — SMAP 14채널

실험 설계 (변수 하나씩 추가):
  Cond 1: Global ref  + Fixed 224      (baseline)
  Cond 2: LTR k=5    + Fixed 224      (LTR 효과)
  Cond 3: LTR k=5    + Adaptive Win   (adaptive window 효과만 추가)

Cond 2 vs Cond 3 비교 → adaptive window가 LTR k=5 위에 얼마나 기여하는가

기존 체크포인트 재활용:
  Cond 1: results/smap/checkpoints/{ch}_mae_fixed.pkl
  Cond 2: results/smap_local_ref/checkpoints/{ch}__mae__k5.pkl

No existing files modified.
Outputs: results/smap_ltr_adaptive_win/
"""

import ast, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_local import ViT4TS_Local
from models.vit4ts_ltr_adaptive_win import ViT4TS_LTR_AdaptiveWin, determine_window_size
from evaluation.evaluate import evaluate_intervals
from preprocessing.preprocess import preprocess_time_series
from preprocessing.data_utils import orion_to_internal

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_ltr_adaptive_win"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA    = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

CONDITIONS  = ["global_fixed", "ltr_k5_fixed", "ltr_k5_adaptive"]
COND_LABELS = {
    "global_fixed":   "Global+Fixed224",
    "ltr_k5_fixed":   "LTR k=5+Fixed224",
    "ltr_k5_adaptive":"LTR k=5+AdpWin",
}

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_fallbacks(ch, cond):
    if cond == "global_fixed":
        return [ROOT / "results" / "smap" / "checkpoints" / f"{ch}_mae_fixed.pkl",
                CKPT_DIR / f"{ch}__global_fixed.pkl"]
    if cond == "ltr_k5_fixed":
        return [ROOT / "results" / "smap_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl",
                CKPT_DIR / f"{ch}__ltr_k5_fixed.pkl"]
    return [CKPT_DIR / f"{ch}__ltr_k5_adaptive.pkl"]

def load_ckpt(ch, cond):
    for path in _ckpt_fallbacks(ch, cond):
        if path.exists():
            d = pickle.load(open(path, "rb"))
            return {"f1": d.get("f1", 0), "p": d.get("p", d.get("precision", 0)),
                    "r": d.get("r", d.get("recall", 0)), "_from": str(path)}
    return None

def save_ckpt(ch, cond, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{ch}__{cond}.pkl", "wb"))

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
# Preview
# ---------------------------------------------------------------------------

def preview_window_sizes():
    print("\n[Adaptive Window Preview — Cond 3]")
    print(f"  {'Ch':<8} {'win':>5}  {'L':>5}  {'k':>5}")
    print("  " + "-"*28)
    for ch in CHANNELS:
        df = pd.read_csv(DATA_DIR / f"{ch}.csv")
        vals, _ = orion_to_internal(df)
        vp  = preprocess_time_series(vals)
        win = determine_window_size(vp)
        L   = max(1, (len(vals) - win) // max(1, win // 4) + 1)
        print(f"  {ch:<8} {win:>5}  {L:>5}  {'5':>5}")
    print()

# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*65)
    print("MAE: Global+Fixed vs LTR k=5+Fixed vs LTR k=5+AdpWin — SMAP")
    print("변수 격리: Cond2 vs Cond3 = adaptive window 효과만")
    print("="*65)

    preview_window_sizes()

    results = {c: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        lbl = COND_LABELS[cond]
        all_cached = all(load_ckpt(ch, cond) is not None for ch in CHANNELS)

        if all_cached:
            det = None
            print(f"\n[{lbl}] 전부 캐시됨")
        else:
            print(f"\n[{lbl}] 모델 초기화 중...")
            backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)

            if cond == "global_fixed":
                det = ViT4TS_Local(backbone=backbone, patch_size=16,
                                   local_k=0, device=str(device), **BASE_PARAMS)
            elif cond == "ltr_k5_fixed":
                det = ViT4TS_Local(backbone=backbone, patch_size=16,
                                   local_k=5, device=str(device), **BASE_PARAMS)
            else:  # ltr_k5_adaptive
                det = ViT4TS_LTR_AdaptiveWin(
                    backbone=backbone, patch_size=16,
                    local_k=5, device=str(device),
                    window_step_ratio=4.0, image_size=(224, 224),
                    alpha=ALPHA, smoothing_alpha=1.0, verbose=True,
                )

        for ch in CHANNELS:
            gt = all_gt.get(ch, [])
            if not gt:
                print(f"  SKIP {ch}")
                results[cond].append(0.0)
                continue

            cached = load_ckpt(ch, cond)
            if cached:
                f1  = cached["f1"]
                src = Path(cached["_from"]).parent.parent.name
                print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n  {ch}: running [{lbl}]...")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt(ch, cond, {"f1": f1, "p": round(m["precision"],4),
                                     "r": round(m["recall"],4)})
                print(f"    F1={f1:.4f}")
            results[cond].append(f1)

    # Summary table
    w = 17
    print("\n" + "="*68)
    hdr = f"{'Channel':<10}" + "".join(f" {COND_LABELS[c]:>{w}}" for c in CONDITIONS)
    print(hdr)
    print("-"*68)

    for i, ch in enumerate(CHANNELS):
        vals = [results[c][i] for c in CONDITIONS]
        best = max(vals)
        row  = f"{ch:<10}"
        for v in vals:
            marker = "*" if (v == best and best > 0) else " "
            row += f" {v:>{w-1}.4f}{marker}"
        note = " ← short" if ch in ("F-1","F-3") else \
               " ← drift" if ch in ("T-1","T-2") else \
               " ← non-stat" if ch in ("D-1","D-2") else ""
        print(row + note)

    print("-"*68)
    avgs = {c: sum(results[c])/len(results[c]) for c in CONDITIONS}
    print(f"{'AVERAGE':<10}" + "".join(f" {avgs[c]:>{w}.4f}" for c in CONDITIONS))
    print()

    base = avgs["global_fixed"]
    for c in CONDITIONS:
        diff = avgs[c] - base
        tag  = "BETTER" if diff > 0 else ("same" if diff == 0 else "worse")
        print(f"  {COND_LABELS[c]:<22}: {avgs[c]:.4f}  ({tag}, {diff:+.4f})")

    delta_adp = avgs["ltr_k5_adaptive"] - avgs["ltr_k5_fixed"]
    print(f"\n  Adaptive window effect (Cond3 - Cond2): {delta_adp:+.4f}")

    # Save
    json_out = {
        "config": {"alpha": ALPHA, "channels": CHANNELS, "local_k": 5},
        **{c: {"f1_per_channel": dict(zip(CHANNELS, results[c])),
               "avg_f1": round(avgs[c], 4)} for c in CONDITIONS},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*68,
        "MAE: Global+Fixed vs LTR k=5+Fixed vs LTR k=5+AdpWin --- SMAP",
        f"alpha={ALPHA}  local_k=5  (variable isolated: adaptive window only)",
        "="*68, hdr, "-"*68,
    ] + [
        f"{ch:<10}" + "".join(f" {results[c][i]:>{w}.4f}" for c in CONDITIONS)
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*68,
        f"{'AVERAGE':<10}" + "".join(f" {avgs[c]:>{w}.4f}" for c in CONDITIONS),
        "",
        f"  Adaptive window effect (Cond3 - Cond2): {delta_adp:+.4f}",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
