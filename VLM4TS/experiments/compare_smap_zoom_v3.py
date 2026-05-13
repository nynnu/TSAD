"""
MAE Multi-Resolution Zoom V3 vs Dual-Anchor α=0.9 — SMAP 14채널

v3 구조:
  Coarse: Dual-Anchor α_ltr=0.9 (LTR 90% + GAP 10%)
  Fine:   LTR only (GAP 제거, localization 집중)
  Fusion: max(coarse, fine)
  Threshold: mean + 1.5*std

비교:
  1. DA α=0.9  (기존 최강)
  2. Zoom V3

특히 확인:
  F-3 (spike, 기존 0.0)
  T-1 (amplitude decrease, 기존 0.0)
  P-3, P-4, T-3, R-1 (기존 잘 되는 채널 유지 확인)
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_zoom_v3 import ViT4TS_Zoom_V3
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_zoom_v3"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

WATCH  = {"F-3": "spike (기존 0.0)", "T-1": "amplitude dec (기존 0.0)"}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def load_ckpt(ch, tag):
    paths = []
    if tag == "da09":
        paths.append(ROOT/"results"/"smap_dual_anchor"/"checkpoints"/f"{ch}__dual_a09.pkl")
    paths.append(_ckpt(ch, tag))
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

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*68)
    print("MAE Zoom V3 (coarse:DA α=0.9 / fine:LTR only) — SMAP")
    print(f"device={device}")
    print("="*68)

    results = {"da09": [], "zoom_v3": []}

    # --- Cond 1: DA α=0.9 캐시 ---
    print("\n[DA α=0.9] 로드 중...")
    for ch in CHANNELS:
        cached = load_ckpt(ch, "da09")
        f1 = cached["f1"] if cached else 0.0
        print(f"  {ch}: F1={f1:.4f}" + (" (캐시없음)" if not cached else ""))
        results["da09"].append(f1)

    # --- Cond 2: Zoom V3 ---
    tag = "zoom_v3"
    all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)

    if not all_cached:
        print("\n[Zoom V3] 모델 초기화 중...")
        bb  = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        det = ViT4TS_Zoom_V3(
            backbone=bb, patch_size=16,
            alpha_ltr_coarse=0.9, k_sigma=1.5,
            window_coarse=224, window_fine=56,
            device=str(device),
            image_size=(224, 224), alpha_detect=ALPHA_DETECT,
            smoothing_alpha=1.0, batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[Zoom V3] 전부 캐시됨")

    for ch in CHANNELS:
        gt = all_gt.get(ch, [])
        if not gt:
            results[tag].append(0.0)
            continue
        cached = load_ckpt(ch, tag)
        if cached:
            f1 = cached["f1"]
            print(f"  {ch}: ckpt  F1={f1:.4f}")
        else:
            data = pd.read_csv(DATA_DIR / f"{ch}.csv")
            print(f"\n{'='*55}")
            print(f"Channel: {ch}  {'← ' + WATCH[ch] if ch in WATCH else ''}")
            print(f"{'='*55}")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(ch, tag, {"f1": f1, "p": round(m["precision"],4),
                                "r": round(m["recall"],4)})
            print(f"  → F1={f1:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        results[tag].append(f1)

    # Summary
    w = 12
    print("\n" + "="*58)
    print("SMAP — Zoom V3 결과")
    hdr = f"{'Channel':<10} {'DA α=0.9':>{w}} {'Zoom V3':>{w}}   Delta"
    print(hdr)
    print("-"*58)

    for i, ch in enumerate(CHANNELS):
        da_v  = results["da09"][i]
        zv    = results["zoom_v3"][i]
        delta = zv - da_v
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        note   = f"  ← {WATCH[ch]}" if ch in WATCH else \
                 "  (stable)" if ch in STABLE else ""
        print(f"{ch:<10} {da_v:>{w}.4f} {zv:>{w}.4f}  {delta:+.4f} {marker}{note}")

    print("-"*58)
    avg_da = sum(results["da09"])    / len(results["da09"])
    avg_zv = sum(results["zoom_v3"]) / len(results["zoom_v3"])
    delta  = avg_zv - avg_da
    print(f"{'AVERAGE':<10} {avg_da:>{w}.4f} {avg_zv:>{w}.4f}  {delta:+.4f}")
    print()
    print(f"  Zoom V3 vs DA α=0.9: {'BETTER' if delta>0 else 'worse'} ({delta:+.4f})")

    json_out = {
        "config": {"alpha_detect": ALPHA_DETECT, "channels": CHANNELS,
                   "alpha_ltr_coarse": 0.9, "k_sigma": 1.5,
                   "window_coarse": 224, "window_fine": 56,
                   "fine_mode": "LTR_only", "fusion": "max"},
        "da09":    {"f1_per_channel": dict(zip(CHANNELS, results["da09"])),
                    "avg_f1": round(avg_da, 4)},
        "zoom_v3": {"f1_per_channel": dict(zip(CHANNELS, results["zoom_v3"])),
                    "avg_f1": round(avg_zv, 4)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*58,
        "MAE Zoom V3 (coarse:DA a=0.9 / fine:LTR only) --- SMAP",
        f"k_sigma=1.5  window_coarse=224  window_fine=56",
        "="*58, hdr, "-"*58,
    ] + [
        f"{ch:<10} {results['da09'][i]:>{w}.4f} {results['zoom_v3'][i]:>{w}.4f}"
        f"  {results['zoom_v3'][i]-results['da09'][i]:+.4f}"
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*58,
        f"{'AVERAGE':<10} {avg_da:>{w}.4f} {avg_zv:>{w}.4f}  {delta:+.4f}",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines)+"\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
