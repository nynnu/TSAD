"""
MAE + Local Temporal Reference (k=5) on MSL 11 channels.
Compares: MAE global (기존) vs MAE local k=5

기존 체크포인트 재활용:
  global(k=0) : results/clip_vs_dino/checkpoints/{ch}_mae.pkl

No existing files modified.
Outputs: results/msl_local_ref/
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

DATA_DIR    = ROOT / "data" / "MSL"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "msl_local_ref"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA    = 0.01
CHANNELS = ['P-11','T-12','D-15','C-1','F-8','F-7','T-13','D-16','T-8','P-14','D-14']
K_VALUES = [5, 0]
K_LABELS = {5: "local k=5", 0: "global"}

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

MAE_GLOBAL_BASELINE = 0.6208

# ---------------------------------------------------------------------------
# Checkpoint helpers — 기존 결과 재활용
# ---------------------------------------------------------------------------

def _ckpt_fallbacks(ch, k):
    fallbacks = []
    if k == 0:
        # 기존 compare_all_backbones 결과 재활용
        fallbacks.append(ROOT / "results" / "clip_vs_dino" / "checkpoints" / f"{ch}_mae.pkl")
    if k == 5:
        # 이전 local ref 실험 결과 (있으면)
        fallbacks.append(CKPT_DIR / f"{ch}__mae__k5.pkl")
    # 이 실험 전용 저장소
    fallbacks.append(CKPT_DIR / f"{ch}__mae__k{k}.pkl")
    return fallbacks

def load_ckpt(ch, k):
    for path in _ckpt_fallbacks(ch, k):
        if path.exists():
            d = pickle.load(open(path, "rb"))
            return {
                "f1": d.get("f1", d.get("F1", 0)),
                "p":  d.get("p", d.get("precision", 0)),
                "r":  d.get("r", d.get("recall", 0)),
                "_from": str(path),
            }
    return None

def save_ckpt(ch, k, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{ch}__mae__k{k}.pkl", "wb"))

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

    print("="*60)
    print("MAE Local Ref (k=5) vs Global — MSL 11 channels")
    print(f"Baseline: MAE global = {MAE_GLOBAL_BASELINE}")
    print("="*60)

    results = {k: [] for k in K_VALUES}

    for k in K_VALUES:
        klbl = K_LABELS[k]
        all_cached = all(load_ckpt(ch, k) is not None for ch in CHANNELS)

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
                print(f"  SKIP {ch} (no GT)")
                results[k].append(0.0)
                continue

            cached = load_ckpt(ch, k)
            if cached:
                f1 = cached["f1"]
                src = Path(cached["_from"]).parent.parent.name
                print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
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
    print("-"*42)
    for i, ch in enumerate(CHANNELS):
        g = results[0][i]
        l = results[5][i]
        delta = l - g
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"{ch:<10} {g:>8.4f} {l:>10.4f}  {delta:+.4f} {marker}")
    print("-"*42)
    avg_g = sum(results[0]) / len(results[0])
    avg_l = sum(results[5]) / len(results[5])
    print(f"{'AVERAGE':<10} {avg_g:>8.4f} {avg_l:>10.4f}  {avg_l-avg_g:+.4f}")
    print()
    print(f"  MAE global  : {avg_g:.4f}")
    print(f"  MAE LTR k=5 : {avg_l:.4f}  ({'BETTER' if avg_l > avg_g else 'worse'}, {avg_l-avg_g:+.4f})")

    # Save
    json_out = {
        "config": {"alpha": ALPHA, "channels": CHANNELS, "k_values": K_VALUES},
        "global":   results[0],
        "local_k5": results[5],
        "avg_global":   round(avg_g, 4),
        "avg_local_k5": round(avg_l, 4),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*60,
        "MAE Local Ref (k=5) vs Global --- MSL 11 channels",
        f"alpha(detection)={ALPHA}",
        "="*60,
        f"{'Channel':<10} {'global':>8} {'local k=5':>10}  Delta",
        "-"*42,
    ] + [
        f"{ch:<10} {results[0][i]:>8.4f} {results[5][i]:>10.4f}  {results[5][i]-results[0][i]:+.4f} {'UP' if results[5][i]>results[0][i] else ('DN' if results[5][i]<results[0][i] else '')}"
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*42,
        f"{'AVERAGE':<10} {avg_g:>8.4f} {avg_l:>10.4f}  {avg_l-avg_g:+.4f}",
        "",
        f"  MAE global  : {avg_g:.4f}",
        f"  MAE LTR k=5 : {avg_l:.4f}  ({'BETTER' if avg_l > avg_g else 'worse'}, {avg_l-avg_g:+.4f})",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
