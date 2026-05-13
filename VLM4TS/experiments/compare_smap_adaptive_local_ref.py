"""
MAE + Adaptive Window + LTR on SMAP 14 channels.

3가지 조건 비교:
  Cond 1: MAE + global ref  + fixed window 224  (기존)
  Cond 2: MAE + LTR k=5    + fixed window 224  (실험 8)
  Cond 3: MAE + LTR adp-k  + adaptive window   (새로운 조합)

기존 체크포인트 재활용:
  Cond 1: results/smap/checkpoints/{ch}_mae_fixed.pkl
  Cond 2: results/smap_local_ref/checkpoints/{ch}__mae__k5.pkl

특히 확인: F-1, F-3 채널 (기존 MAE 0.0, high_freq_ratio 기반으로 개선 기대)

No existing files modified.
Outputs: results/smap_adaptive_local_ref/
           results.json, summary.txt, checkpoints/

Usage:
    cd VLM4TS
    python experiments/compare_smap_adaptive_local_ref.py
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
from models.vit4ts_adaptive_local import ViT4TS_AdaptiveLocal, determine_window_size, adaptive_k
from evaluation.evaluate import evaluate_intervals
from preprocessing.preprocess import preprocess_time_series
from preprocessing.data_utils import orion_to_internal

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_adaptive_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA    = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

CONDITIONS = ["global_fixed", "ltr_k5_fixed", "ltr_adp_adaptive"]
COND_LABELS = {
    "global_fixed":    "Global+Fixed224",
    "ltr_k5_fixed":    "LTR k=5+Fixed224",
    "ltr_adp_adaptive":"LTR adp-k+AdpWin",
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
    fallbacks = []
    if cond == "global_fixed":
        fallbacks.append(ROOT / "results" / "smap" / "checkpoints" / f"{ch}_mae_fixed.pkl")
    if cond == "ltr_k5_fixed":
        fallbacks.append(ROOT / "results" / "smap_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl")
    fallbacks.append(CKPT_DIR / f"{ch}__{cond}.pkl")
    return fallbacks

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
# Adaptive window preview (for logging)
# ---------------------------------------------------------------------------

def preview_window_sizes():
    print("\n[Adaptive Window Preview]")
    print(f"  {'Channel':<8} {'n_pts':>7} {'win':>5} {'step':>5} {'L':>5} {'adp_k':>6}")
    print("  " + "-"*40)
    for ch in CHANNELS:
        df = pd.read_csv(DATA_DIR / f"{ch}.csv")
        vals, _ = orion_to_internal(df)
        vals_proc = preprocess_time_series(vals)
        win  = determine_window_size(vals_proc)
        step = max(1, win // 4)
        L    = max(1, (len(vals) - win) // step + 1)
        k    = adaptive_k(L)
        print(f"  {ch:<8} {len(vals):>7} {win:>5} {step:>5} {L:>5} {k if k>0 else 'global':>6}")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*65)
    print("MAE: Global+Fixed vs LTR k=5+Fixed vs LTR adp-k+AdpWin — SMAP")
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
            else:  # ltr_adp_adaptive
                det = ViT4TS_AdaptiveLocal(backbone=backbone, patch_size=16,
                                           device=str(device),
                                           window_step_ratio=4.0,
                                           image_size=(224, 224), alpha=ALPHA,
                                           smoothing_alpha=1.0, verbose=True)

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
    w = 16
    print("\n" + "="*70)
    hdr = f"{'Channel':<10}" + "".join(f" {COND_LABELS[c]:>{w}}" for c in CONDITIONS)
    print(hdr)
    print("-"*70)

    for i, ch in enumerate(CHANNELS):
        row = f"{ch:<10}"
        vals = [results[c][i] for c in CONDITIONS]
        best = max(vals)
        for v in vals:
            marker = "*" if v == best and best > 0 else " "
            row += f" {v:>{w-1}.4f}{marker}"

        # Highlight F-1 and F-3
        tag = " ← 주목" if ch in ("F-1","F-3") else ""
        print(row + tag)

    print("-"*70)
    avgs = {c: sum(results[c])/len(results[c]) for c in CONDITIONS}
    avg_row = f"{'AVERAGE':<10}" + "".join(f" {avgs[c]:>{w}.4f}" for c in CONDITIONS)
    print(avg_row)
    print()
    for c in CONDITIONS:
        base = avgs["global_fixed"]
        diff = avgs[c] - base
        marker = "BETTER" if diff > 0 else ("same" if diff == 0 else "worse")
        print(f"  {COND_LABELS[c]:<22}: {avgs[c]:.4f}  ({marker}, {diff:+.4f})")

    # Save
    json_out = {
        "config": {"alpha": ALPHA, "channels": CHANNELS, "conditions": CONDITIONS},
        **{c: {"f1_per_channel": dict(zip(CHANNELS, results[c])),
               "avg_f1": round(avgs[c], 4)} for c in CONDITIONS},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*70,
        "MAE: Global+Fixed vs LTR k=5+Fixed vs LTR adp-k+AdpWin --- SMAP",
        f"alpha(detection)={ALPHA}",
        "="*70,
        hdr, "-"*70,
    ] + [
        f"{ch:<10}" + "".join(f" {results[c][i]:>{w}.4f}" for c in CONDITIONS)
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*70, avg_row, "",
    ] + [
        f"  {COND_LABELS[c]:<22}: {avgs[c]:.4f}  ({('BETTER' if avgs[c]-avgs['global_fixed']>0 else 'worse')}, {avgs[c]-avgs['global_fixed']:+.4f})"
        for c in CONDITIONS
    ] + [f"\nResults: {OUTPUT_DIR / 'results.json'}"]

    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
