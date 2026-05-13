"""
MAE Multi-Resolution Zoom V2 vs Dual-Anchor α=0.7 — SMAP 14채널

v2 변경사항:
  - τ_coarse: 고정 30% → adaptive mean + 1.0*std
  - Score fusion: weighted sum → max(coarse, fine)

비교 조건:
  1. DA α=0.7  (기존, 캐시 재활용)
  2. Zoom V2   (k_sigma=1.0, max fusion)

특히 확인:
  F-3 (spike, 기존 0.0) — fine zoom + max fusion으로 damping 없이 잡히는지
  T-1 (amplitude decrease, 기존 0.0)
  P-3, P-4, T-3, R-1 (기존 잘 되는 채널 — candidate에 안 잡혀서 coarse 그대로)
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
from models.vit4ts_zoom_v2 import ViT4TS_Zoom_V2
from evaluation.evaluate import evaluate_intervals
from preprocessing.preprocess import preprocess_time_series
from preprocessing.data_utils import orion_to_internal
from models.vit4ts_zoom_v2 import _extract_candidates_adaptive

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_zoom_v2"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

WATCH  = {"F-3": "spike (기존 0.0)", "T-1": "amplitude dec (기존 0.0)"}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]

# ---------------------------------------------------------------------------

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def load_ckpt(ch, tag):
    paths = []
    if tag == "da07":
        paths.append(ROOT/"results"/"smap_dual_anchor"/"checkpoints"/f"{ch}__dual_a07.pkl")
    paths.append(_ckpt(ch, tag))
    for p in paths:
        if p.exists():
            d = pickle.load(open(p,"rb"))
            return {"f1": d.get("f1",0), "_from": str(p)}
    return None

def save_ckpt(ch, tag, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(ch, tag), "wb"))

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

# ---------------------------------------------------------------------------
# Threshold preview (채널별 adaptive threshold 미리 보기)
# ---------------------------------------------------------------------------

def preview_thresholds(all_gt, ltr_k5_results=None):
    """coarse score를 실제로 돌려보기 전 간단히 signal 통계만 보기."""
    print("\n[Signal Statistics (preprocessing 기준)]")
    print(f"  {'Channel':<8} {'n_pts':>7} {'mean':>8} {'std':>8}")
    print("  " + "-"*35)
    for ch in CHANNELS:
        df = pd.read_csv(DATA_DIR / f"{ch}.csv")
        vals, _ = orion_to_internal(df)
        vp = preprocess_time_series(vals)
        print(f"  {ch:<8} {len(vals):>7} {vp.mean():>8.4f} {vp.std():>8.4f}")
    print()

# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*68)
    print("MAE Multi-Resolution Zoom V2 (max fusion + adaptive τ) — SMAP")
    print(f"device={device}  k_sigma=1.5  fusion=max")
    print("="*68)

    preview_thresholds(all_gt)

    results = {"da07": [], "zoom_v2": []}

    # --- Cond 1: DA α=0.7 캐시 ---
    print("[DA α=0.7] 캐시 로드 중...")
    for ch in CHANNELS:
        cached = load_ckpt(ch, "da07")
        f1 = cached["f1"] if cached else 0.0
        if cached:
            print(f"  {ch}: F1={f1:.4f}")
        else:
            print(f"  {ch}: 캐시 없음 (0.0으로 처리 — compare_smap_dual_anchor.py 먼저 실행)")
        results["da07"].append(f1)

    # --- Cond 2: Zoom V2 ---
    tag = "zoom_v2"
    all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)

    if not all_cached:
        print("\n[Zoom V2] 모델 초기화 중...")
        bb = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        det = ViT4TS_Zoom_V2(
            backbone=bb, patch_size=16,
            alpha_ltr=0.7, k_sigma=1.5,
            window_coarse=224, window_fine=56,
            device=str(device),
            image_size=(224,224), alpha_detect=ALPHA_DETECT,
            smoothing_alpha=1.0, batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[Zoom V2] 전부 캐시됨")

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

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    w = 12
    print("\n" + "="*58)
    print("SMAP — Zoom V2 (max fusion + adaptive τ)")
    hdr = f"{'Channel':<10} {'DA α=0.7':>{w}} {'Zoom V2':>{w}}   Delta"
    print(hdr)
    print("-"*58)

    for i, ch in enumerate(CHANNELS):
        da_v   = results["da07"][i]
        zv     = results["zoom_v2"][i]
        delta  = zv - da_v
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        row    = f"{ch:<10} {da_v:>{w}.4f} {zv:>{w}.4f}  {delta:+.4f} {marker}"
        note   = f"  ← {WATCH[ch]}" if ch in WATCH else \
                 "  (stable)" if ch in STABLE else ""
        print(row + note)

    print("-"*58)
    avg_da = sum(results["da07"])  / len(results["da07"])
    avg_zv = sum(results["zoom_v2"]) / len(results["zoom_v2"])
    delta  = avg_zv - avg_da
    print(f"{'AVERAGE':<10} {avg_da:>{w}.4f} {avg_zv:>{w}.4f}  {delta:+.4f}")
    print()
    print(f"  Zoom V2 vs DA α=0.7: {'BETTER' if delta>0 else 'worse'} ({delta:+.4f})")

    # 핵심 채널 상세
    print("\n[핵심 채널]")
    for ch in list(WATCH.keys()) + STABLE[:3]:
        i = CHANNELS.index(ch)
        print(f"  {ch:<8}  DA={results['da07'][i]:.4f}  ZoomV2={results['zoom_v2'][i]:.4f}"
              f"  {WATCH.get(ch,'')}")

    # Save
    json_out = {
        "config": {"alpha_detect": ALPHA_DETECT, "channels": CHANNELS,
                   "k_sigma": 1.5, "window_coarse": 224, "window_fine": 56,
                   "alpha_ltr": 0.7, "fusion": "max"},
        "da07":    {"f1_per_channel": dict(zip(CHANNELS, results["da07"])),
                    "avg_f1": round(avg_da, 4)},
        "zoom_v2": {"f1_per_channel": dict(zip(CHANNELS, results["zoom_v2"])),
                    "avg_f1": round(avg_zv, 4)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*58,
        "MAE Zoom V2 (max fusion + adaptive tau) --- SMAP",
        f"k_sigma=1.0  window_coarse=224  window_fine=56  alpha_ltr=0.7",
        "="*58, hdr, "-"*58,
    ] + [
        f"{ch:<10} {results['da07'][i]:>{w}.4f} {results['zoom_v2'][i]:>{w}.4f}"
        f"  {results['zoom_v2'][i]-results['da07'][i]:+.4f}"
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
