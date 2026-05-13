"""
MAE Multi-Resolution Zoom vs Dual-Anchor α=0.7 — SMAP 14채널

비교 조건:
  1. MAE + LTR k=5 + GAP  Dual-Anchor α=0.7  (기존 캐시 재활용)
  2. MAE + Zoom  (Multi-Resolution)

특히 확인:
  F-3 (spike, 기존 0.0)   — fine zoom으로 잡히는지
  T-1 (amplitude decrease) — 기존 0.0
  기존 잘 되는 채널 유지 여부: P-3, P-4, P-7, T-3, R-1

No existing files modified.
Outputs: results/smap_zoom/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_zoom import ViT4TS_Zoom
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_zoom"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

WATCH  = {"F-3": "spike (기존 0.0)", "T-1": "amplitude dec (기존 0.0)"}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]

# ---------------------------------------------------------------------------

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def _fallbacks_da(ch):
    return [
        ROOT / "results" / "smap_dual_anchor" / "checkpoints" / f"{ch}__dual_a07.pkl",
        _ckpt(ch, "da07"),
    ]

def load_ckpt(ch, tag):
    paths = _fallbacks_da(ch) if tag == "da07" else [_ckpt(ch, tag)]
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
    print("MAE Multi-Resolution Zoom vs Dual-Anchor α=0.7 — SMAP")
    print(f"device={device}")
    print("="*65)

    results = {"da07": [], "zoom": []}

    # --- Cond 1: Dual-Anchor α=0.7 (캐시 재활용) ---
    tag = "da07"
    all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)
    print(f"\n[DA α=0.7] {'전부 캐시됨' if all_cached else '캐시 없음 — compare_smap_dual_anchor.py 먼저 실행 필요'}")
    for ch in CHANNELS:
        cached = load_ckpt(ch, tag)
        f1 = cached["f1"] if cached else 0.0
        if cached:
            print(f"  {ch}: ckpt  F1={f1:.4f}")
        results[tag].append(f1)

    # --- Cond 2: Zoom ---
    tag = "zoom"
    all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)

    if not all_cached:
        print("\n[Zoom] 모델 초기화 중...")
        bb = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        det = ViT4TS_Zoom(
            backbone=bb, patch_size=16,
            alpha_ltr=0.7, alpha_coarse=0.3,
            window_coarse=224, window_fine=56,
            tau_coarse=0.30,
            device=str(device),
            image_size=(224, 224), alpha_detect=ALPHA_DETECT,
            smoothing_alpha=1.0, batch_size=32, verbose=True,
        )
    else:
        print("\n[Zoom] 전부 캐시됨")

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
            print(f"\n{'='*50}")
            print(f"Channel: {ch}")
            print(f"{'='*50}")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(ch, tag, {"f1": f1, "p": round(m["precision"],4),
                                "r": round(m["recall"],4)})
            print(f"  → F1={f1:.4f}")
        results[tag].append(f1)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    all_tags = ["da07", "zoom"]
    labels   = {"da07": "DA α=0.7", "zoom": "Zoom"}
    w = 12

    print("\n" + "="*55)
    print("SMAP — Multi-Resolution Zoom 결과")
    hdr = f"{'Channel':<10}" + "".join(f" {labels[t]:>{w}}" for t in all_tags) + "  Delta"
    print(hdr)
    print("-"*55)

    for i, ch in enumerate(CHANNELS):
        da_v   = results["da07"][i]
        zoom_v = results["zoom"][i]
        delta  = zoom_v - da_v
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        row    = f"{ch:<10} {da_v:>{w}.4f} {zoom_v:>{w}.4f}  {delta:+.4f} {marker}"
        note   = ""
        if ch in WATCH:
            note = f"  ← {WATCH[ch]}"
        elif ch in STABLE:
            note = "  (stable)"
        print(row + note)

    print("-"*55)
    avg_da   = sum(results["da07"]) / len(results["da07"])
    avg_zoom = sum(results["zoom"]) / len(results["zoom"])
    delta_avg = avg_zoom - avg_da
    print(f"{'AVERAGE':<10} {avg_da:>{w}.4f} {avg_zoom:>{w}.4f}  {delta_avg:+.4f}")
    print()
    flag = "BETTER" if delta_avg > 0 else "worse"
    print(f"  Zoom vs DA α=0.7: {avg_zoom:.4f}  ({flag}, {delta_avg:+.4f})")

    # 핵심 채널 상세
    print("\n[핵심 채널 상세]")
    for ch in list(WATCH.keys()) + STABLE[:3]:
        i = CHANNELS.index(ch)
        print(f"  {ch:<8}  DA={results['da07'][i]:.4f}  Zoom={results['zoom'][i]:.4f}"
              f"  {WATCH.get(ch, '')}")

    # Save
    json_out = {
        "config": {"alpha_detect": ALPHA_DETECT, "channels": CHANNELS,
                   "window_coarse": 224, "window_fine": 56,
                   "alpha_ltr": 0.7, "alpha_coarse": 0.3, "tau_coarse": 0.30},
        "da07": {"f1_per_channel": dict(zip(CHANNELS, results["da07"])),
                 "avg_f1": round(avg_da, 4)},
        "zoom": {"f1_per_channel": dict(zip(CHANNELS, results["zoom"])),
                 "avg_f1": round(avg_zoom, 4)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*55,
        "MAE Multi-Resolution Zoom vs Dual-Anchor α=0.7 --- SMAP",
        f"window_coarse=224 window_fine=56 tau=0.30 alpha_ltr=0.7",
        "="*55, hdr, "-"*55,
    ] + [
        f"{ch:<10} {results['da07'][i]:>{w}.4f} {results['zoom'][i]:>{w}.4f}"
        f"  {results['zoom'][i]-results['da07'][i]:+.4f}"
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*55,
        f"{'AVERAGE':<10} {avg_da:>{w}.4f} {avg_zoom:>{w}.4f}  {delta_avg:+.4f}",
        "",
        f"  Zoom ({flag}, {delta_avg:+.4f})",
        f"\nResults: {OUTPUT_DIR / 'results.json'}",
    ]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
