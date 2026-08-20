"""
MAE Dual-Anchor GAP+RMS — MSL 11채널

비교 조건:
  1. LTR k=5 only  (캐시 재활용, baseline 0.6344)
  2. GAP+RMS α=0.7
  3. GAP+RMS α=0.5  ← SMAP에서 best였던 config

기존 체크포인트 재활용:
  LTR k=5: results/msl_local_ref/checkpoints/{ch}__mae__k5.pkl

No existing files modified.
Outputs: results/msl_gap_rms/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_dual_anchor_v2 import ViT4TS_DualAnchor_V2
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "MSL"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "msl_gap_rms"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-11','T-12','D-15','C-1','F-8','F-7',
            'T-13','D-16','T-8','P-14','D-14']

ALPHA_SWEEP = [0.7, 0.5]
ALPHA_TAGS  = {0.7: "grms_a07", 0.5: "grms_a05"}

LTR_BASELINE = 0.6344


# ---------------------------------------------------------------------------

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def load_ckpt(ch, tag):
    # LTR k=5 baseline: look in msl_local_ref first
    paths = [_ckpt(ch, tag)]
    if tag == "ltr_k5":
        paths.insert(0, ROOT/"results"/"msl_local_ref"/"checkpoints"/f"{ch}__mae__k5.pkl")
    for p in paths:
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", 0), "_from": str(p)}
    return None

def save_ckpt(ch, tag, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(ch, tag), "wb"))

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*65)
    print("MSL: MAE + GAP+RMS α sweep (LTR k=5 baseline 재활용)")
    print(f"device={device}  channels={len(CHANNELS)}")
    print("="*65)

    all_tags = ["ltr_k5"] + [ALPHA_TAGS[a] for a in ALPHA_SWEEP]
    results  = {tag: [] for tag in all_tags}

    # ---- Baseline: LTR k=5 캐시 ----
    print("\n[LTR k=5] 캐시 로드 중...")
    for ch in CHANNELS:
        cached = load_ckpt(ch, "ltr_k5")
        f1 = cached["f1"] if cached else 0.0
        src = f"  ({Path(cached['_from']).parent.name})" if cached else " ← 캐시 없음"
        print(f"  {ch}: F1={f1:.4f}{src}")
        results["ltr_k5"].append(f1)

    # ---- GAP+RMS α sweep ----
    for alpha in ALPHA_SWEEP:
        tag = ALPHA_TAGS[alpha]
        all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)

        if not all_cached:
            print(f"\n[GAP+RMS α={alpha}] 모델 초기화 중...")
            backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
            det = ViT4TS_DualAnchor_V2(
                backbone=backbone, patch_size=16,
                local_k=5, min_ref=5, alpha=alpha,
                window_size=224, window_step_ratio=4.0,
                device=str(device),
                image_size=(224, 224), alpha_detect=ALPHA_DETECT,
                smoothing_alpha=1.0, batch_size=32, verbose=True,
            )
        else:
            det = None
            print(f"\n[GAP+RMS α={alpha}] 전부 캐시됨")

        for ch in CHANNELS:
            gt = all_gt.get(ch, [])
            cached = load_ckpt(ch, tag)
            if cached:
                f1 = cached["f1"]
                print(f"  {ch}: ckpt  F1={f1:.4f}")
            elif not gt:
                f1 = 0.0
                print(f"  {ch}: GT 없음 → 0.0")
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n{'='*55}")
                print(f"Channel: {ch}  (α={alpha})")
                print(f"{'='*55}")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt(ch, tag, {"f1": f1, "p": round(m["precision"], 4),
                                    "r": round(m["recall"], 4)})
                print(f"  → F1={f1:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
            results[tag].append(f1)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    w = 12
    hdr = (f"{'Channel':<10} {'LTR k=5':>{w}}"
           + "".join(f" {'GRMS α='+str(a):>{w}}" for a in ALPHA_SWEEP)
           + "   vs LTR")
    print("\n" + "="*65)
    print("MSL — GAP+RMS 결과")
    print(hdr)
    print("-"*65)

    best_tag = ALPHA_TAGS[0.5]
    for i, ch in enumerate(CHANNELS):
        base  = results["ltr_k5"][i]
        best  = results[best_tag][i]
        delta = best - base
        mark  = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else " ")
        row   = f"{ch:<10} {base:>{w}.4f}"
        for a in ALPHA_SWEEP:
            row += f" {results[ALPHA_TAGS[a]][i]:>{w}.4f}"
        row += f"  {delta:+.4f} {mark}"
        print(row)

    print("-"*65)
    avg_ltr = sum(results["ltr_k5"]) / len(CHANNELS)
    row_avg = f"{'AVERAGE':<10} {avg_ltr:>{w}.4f}"
    for a in ALPHA_SWEEP:
        avg_a = sum(results[ALPHA_TAGS[a]]) / len(CHANNELS)
        row_avg += f" {avg_a:>{w}.4f}"
    best_avg = sum(results[best_tag]) / len(CHANNELS)
    row_avg += f"  {best_avg - avg_ltr:+.4f}"
    print(row_avg)
    print()
    print(f"  LTR k=5 baseline : {avg_ltr:.4f}")
    for a in ALPHA_SWEEP:
        avg_a = sum(results[ALPHA_TAGS[a]]) / len(CHANNELS)
        marker = "BETTER" if avg_a > avg_ltr else "worse"
        print(f"  GAP+RMS α={a}    : {avg_a:.4f}  ({marker}, {avg_a - avg_ltr:+.4f})")

    # Save JSON
    json_out = {
        "config": {
            "alpha_detect": ALPHA_DETECT,
            "channels": CHANNELS,
            "alpha_sweep": ALPHA_SWEEP,
            "ltr_baseline": LTR_BASELINE,
        },
        "ltr_k5": {
            "f1_per_channel": dict(zip(CHANNELS, results["ltr_k5"])),
            "avg_f1": round(avg_ltr, 4),
        },
    }
    for a in ALPHA_SWEEP:
        tag = ALPHA_TAGS[a]
        avg = sum(results[tag]) / len(CHANNELS)
        json_out[tag] = {
            "f1_per_channel": dict(zip(CHANNELS, results[tag])),
            "avg_f1": round(avg, 4),
        }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*65,
        "MSL — MAE GAP+RMS α sweep",
        f"LTR k=5 baseline: {avg_ltr:.4f}",
        "="*65, hdr, "-"*65,
    ] + [
        f"{ch:<10} {results['ltr_k5'][i]:>{w}.4f}"
        + "".join(f" {results[ALPHA_TAGS[a]][i]:>{w}.4f}" for a in ALPHA_SWEEP)
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*65, row_avg,
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    run()
