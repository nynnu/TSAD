"""
MAE Dual-Anchor GAP+RMS vs GAP only — SMAP 14채널

비교 조건:
  1. DA α=0.7 GAP only    (기존, 캐시 재활용)
  2. DA α=0.7 GAP+RMS     (새로운)
  3. DA α=0.9 GAP+RMS
  4. DA α=0.5 GAP+RMS

특히 확인:
  T-1 (amplitude decrease, 기존 0.0)
  T-2 (amplitude increase, 기존 α=0.9에서 1.0 — 유지되는지)
  F-3 (spike, 기존 0.0)
  P-3, P-4, P-7, D-3, T-3, R-1 (잘 되던 채널 — 망가지지 않는지)

기존 체크포인트 재활용:
  DA α=0.7 GAP: results/smap_dual_anchor/checkpoints/{ch}__dual_a07.pkl

No existing files modified.
Outputs: results/smap_gap_rms/
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

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_gap_rms"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

# GAP+RMS α sweep
ALPHA_SWEEP = [0.9, 0.7, 0.5]

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha_detect=ALPHA_DETECT,
    smoothing_alpha=1.0, batch_size=32, verbose=True,
)

# 특별히 주목할 채널
WATCH = {
    "T-1": "amplitude decrease (기존 0.0)",
    "T-2": "amplitude increase (α=0.9 기존 1.0?)",
    "F-3": "spike (기존 0.0)",
}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def _fallbacks(ch, tag):
    paths = []
    # 기존 GAP-only 결과 재활용
    if tag == "gap07":
        paths.append(ROOT / "results" / "smap_dual_anchor" / "checkpoints" / f"{ch}__dual_a07.pkl")
    paths.append(_ckpt(ch, tag))
    return paths

def load_ckpt(ch, tag):
    for p in _fallbacks(ch, tag):
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

def run_cond(det, channels, all_gt, tag):
    f1s = []
    for ch in channels:
        gt = all_gt.get(ch, [])
        if not gt:
            f1s.append(0.0); continue
        cached = load_ckpt(ch, tag)
        if cached:
            f1  = cached["f1"]
            src = Path(cached["_from"]).parent.parent.name
            print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
        else:
            data = pd.read_csv(DATA_DIR / f"{ch}.csv")
            print(f"\n  {ch}: running [{tag}]...")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(ch, tag, {"f1": f1, "p": round(m["precision"],4),
                                "r": round(m["recall"],4)})
            print(f"    F1={f1:.4f}")
        f1s.append(f1)
    return f1s

# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*68)
    print("MAE Dual-Anchor: GAP+RMS vs GAP only — SMAP")
    print(f"device={device}")
    print("="*68)

    results  = {}
    all_tags = ["gap07"] + [f"grms_a{int(a*10):02d}" for a in ALPHA_SWEEP]
    labels   = {
        "gap07": "DA α=0.7 GAP",
        **{f"grms_a{int(a*10):02d}": f"DA α={a:.1f} GAP+RMS" for a in ALPHA_SWEEP},
    }

    # --- Cond 1: GAP only α=0.7 (캐시 재활용) ---
    tag = "gap07"
    all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)
    print(f"\n[{labels[tag]}] {'전부 캐시됨' if all_cached else '실행 필요'}")
    if not all_cached:
        print("  ※ smap_dual_anchor 체크포인트 없음 — compare_smap_dual_anchor.py 먼저 실행 필요")
    results[tag] = [load_ckpt(ch, tag)["f1"] if load_ckpt(ch, tag) else 0.0
                    for ch in CHANNELS]

    # --- GAP+RMS α sweep ---
    print("\n[GAP+RMS] 모델 초기화 중...")
    bb = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)

    for alpha_ltr in ALPHA_SWEEP:
        tag   = f"grms_a{int(alpha_ltr*10):02d}"
        label = labels[tag]
        all_cached = all(load_ckpt(ch, tag) for ch in CHANNELS)

        if all_cached:
            print(f"\n[{label}] 전부 캐시됨")
            results[tag] = [load_ckpt(ch, tag)["f1"] for ch in CHANNELS]
        else:
            print(f"\n[{label}] 실행 중...")
            det = ViT4TS_DualAnchor_V2(
                backbone=bb, patch_size=16, local_k=5,
                alpha=alpha_ltr, device=str(device), **BASE_PARAMS,
            )
            results[tag] = run_cond(det, CHANNELS, all_gt, tag)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    w = 16
    print("\n" + "="*72)
    print("SMAP — GAP+RMS vs GAP only")
    hdr = f"{'Channel':<10}" + "".join(f" {labels[t]:>{w}}" for t in all_tags)
    print(hdr)
    print("-"*72)

    for i, ch in enumerate(CHANNELS):
        vals = [results[t][i] for t in all_tags]
        best = max(vals)
        row  = f"{ch:<10}"
        for v in vals:
            marker = "*" if (v == best and best > 0) else " "
            row += f" {v:>{w-1}.4f}{marker}"

        note = ""
        if ch in WATCH:
            note = f"  ← {WATCH[ch]}"
        elif ch in STABLE:
            note = "  (stable check)"
        print(row + note)

    print("-"*72)
    avgs = {t: sum(results[t])/len(results[t]) for t in all_tags}
    print(f"{'AVERAGE':<10}" + "".join(f" {avgs[t]:>{w}.4f}" for t in all_tags))

    # delta vs GAP only
    base = avgs["gap07"]
    print()
    print(f"  Baseline DA α=0.7 GAP: {base:.4f}")
    for t in all_tags[1:]:
        diff = avgs[t] - base
        flag = "BETTER" if diff > 0 else ("same" if abs(diff)<0.0001 else "worse")
        print(f"  {labels[t]:<22}: {avgs[t]:.4f}  ({flag}, {diff:+.4f})")

    # 주목 채널 비교
    print("\n[주목 채널 상세]")
    for ch in list(WATCH.keys()) + STABLE[:3]:
        i   = CHANNELS.index(ch)
        row = f"  {ch:<8}" + "".join(f"  {labels[t]}={results[t][i]:.4f}" for t in all_tags)
        print(row)

    # Save
    json_out = {
        "config": {"alpha_sweep": ALPHA_SWEEP, "channels": CHANNELS,
                   "alpha_detect": ALPHA_DETECT},
        **{t: {"f1_per_channel": dict(zip(CHANNELS, results[t])),
               "avg_f1": round(avgs[t], 4)} for t in all_tags},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*72,
        "MAE Dual-Anchor: GAP+RMS vs GAP only --- SMAP",
        f"alpha_detect={ALPHA_DETECT}  local_k=5",
        "="*72, hdr, "-"*72,
    ] + [
        f"{ch:<10}" + "".join(f" {results[t][i]:>{w}.4f}" for t in all_tags)
        for i, ch in enumerate(CHANNELS)
    ] + [
        "-"*72,
        f"{'AVERAGE':<10}" + "".join(f" {avgs[t]:>{w}.4f}" for t in all_tags),
        "",
    ] + [
        f"  {labels[t]:<22}: {avgs[t]:.4f}  ({'BETTER' if avgs[t]-base>0 else 'worse'}, {avgs[t]-base:+.4f})"
        for t in all_tags[1:]
    ] + [f"\nResults: {OUTPUT_DIR / 'results.json'}"]

    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
