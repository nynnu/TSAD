"""
MAE Dual-Anchor (LTR + GAP) — SMAP 14채널 + NAB-AWS 검증

비교 조건:
  Cond 1: LTR k=5 only   (α=1.0, Local만)     ← 기존 최강
  Cond 2: GAP only       (α=0.0, Global만)
  Cond 3: Dual α=0.7     (LTR 70% + GAP 30%)
  Cond 4: Dual α=0.5     (LTR 50% + GAP 50%)
  + α sweep: 0.3, 0.5, 0.7, 0.9

특히 확인: SMAP T-1, T-2 (drift anomaly) — GAP이 잡는지
          NAB-AWS LTR k=5 (0.6272) 성능 유지되는지

기존 체크포인트 재활용:
  LTR k=5: results/smap_local_ref/checkpoints/{ch}__mae__k5.pkl

No existing files modified.
Outputs: results/smap_dual_anchor/
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
from models.vit4ts_dual_anchor import ViT4TS_DualAnchor
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
NAB_DIR     = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_dual_anchor"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
SMAP_CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
                 'F-1','F-2','F-3','T-1','T-2','T-3','R-1']
ALPHA_SWEEP   = [0.9, 0.7, 0.5, 0.3]   # LTR weight

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha_detect=ALPHA_DETECT,
    smoothing_alpha=1.0, batch_size=32, verbose=True,
)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def _fallbacks(ch, tag):
    paths = []
    if tag == "ltr_k5":
        paths.append(ROOT / "results" / "smap_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl")
    paths.append(_ckpt_path(ch, tag))
    return paths

def load_ckpt(ch, tag):
    for p in _fallbacks(ch, tag):
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", 0), "p": d.get("p", d.get("precision", 0)),
                    "r": d.get("r", d.get("recall", 0)), "_from": str(p)}
    return None

def save_ckpt(ch, tag, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt_path(ch, tag), "wb"))

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
# Run one condition on SMAP
# ---------------------------------------------------------------------------

def run_condition(det, channels, data_dir, all_gt, tag):
    f1s = []
    for ch in channels:
        gt = all_gt.get(ch, [])
        if not gt:
            f1s.append(0.0)
            continue
        cached = load_ckpt(ch, tag)
        if cached:
            f1 = cached["f1"]
            src = Path(cached["_from"]).parent.parent.name
            print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
        else:
            data = pd.read_csv(data_dir / f"{ch}.csv")
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
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()

    print("="*65)
    print("MAE Dual-Anchor (LTR + GAP) — SMAP")
    print(f"device={device}  alpha_sweep={ALPHA_SWEEP}")
    print("="*65)

    # --- 결과 저장 ---
    all_results = {}   # tag → [f1 per channel]

    # ===== Cond 1: LTR k=5 (α=1.0) — 기존 체크포인트 재활용 =====
    print("\n[LTR k=5 only (α=1.0)] 모델 초기화 중...")
    bb_ltr = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
    det_ltr = ViT4TS_Local(backbone=bb_ltr, patch_size=16, local_k=5,
                           device=str(device), window_size=224,
                           window_step_ratio=4.0, image_size=(224,224),
                           alpha=ALPHA_DETECT, smoothing_alpha=1.0, verbose=True)
    all_results["ltr_k5"] = run_condition(det_ltr, SMAP_CHANNELS, DATA_DIR, all_gt, "ltr_k5")

    # ===== α sweep: Dual-Anchor =====
    print("\n[Dual-Anchor] 모델 초기화 중...")
    bb_da = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)

    for alpha_ltr in ALPHA_SWEEP:
        tag = f"dual_a{int(alpha_ltr*10):02d}"   # e.g. dual_a07
        label = f"Dual α={alpha_ltr:.1f}"

        all_cached = all(load_ckpt(ch, tag) is not None for ch in SMAP_CHANNELS)
        if all_cached:
            print(f"\n[{label}] 전부 캐시됨")
            f1s = [load_ckpt(ch, tag)["f1"] for ch in SMAP_CHANNELS]
        else:
            print(f"\n[{label}] 실행 중...")
            det = ViT4TS_DualAnchor(
                backbone=bb_da, patch_size=16, local_k=5,
                alpha=alpha_ltr, device=str(device), **BASE_PARAMS,
            )
            f1s = run_condition(det, SMAP_CHANNELS, DATA_DIR, all_gt, tag)

        all_results[tag] = f1s

    # ===== GAP only (α=0.0) =====
    tag_gap = "gap_only"
    all_cached = all(load_ckpt(ch, tag_gap) is not None for ch in SMAP_CHANNELS)
    if all_cached:
        print("\n[GAP only (α=0.0)] 전부 캐시됨")
        all_results[tag_gap] = [load_ckpt(ch, tag_gap)["f1"] for ch in SMAP_CHANNELS]
    else:
        print("\n[GAP only (α=0.0)] 실행 중...")
        det_gap = ViT4TS_DualAnchor(
            backbone=bb_da, patch_size=16, local_k=5,
            alpha=0.0, device=str(device), **BASE_PARAMS,
        )
        all_results[tag_gap] = run_condition(det_gap, SMAP_CHANNELS, DATA_DIR, all_gt, tag_gap)

    # ===== Summary table =====
    tags_ordered = ["ltr_k5"] + [f"dual_a{int(a*10):02d}" for a in ALPHA_SWEEP] + ["gap_only"]
    labels = {
        "ltr_k5":    "LTR k=5 (α=1.0)",
        "gap_only":  "GAP only (α=0.0)",
        **{f"dual_a{int(a*10):02d}": f"Dual α={a:.1f}" for a in ALPHA_SWEEP},
    }

    w = 17
    print("\n" + "="*75)
    print("SMAP — MAE Dual-Anchor α sweep")
    hdr = f"{'Channel':<10}" + "".join(f" {labels[t]:>{w}}" for t in tags_ordered)
    print(hdr)
    print("-"*75)

    drift_channels = ("T-1","T-2")
    for i, ch in enumerate(SMAP_CHANNELS):
        vals = [all_results[t][i] for t in tags_ordered]
        best = max(vals)
        row  = f"{ch:<10}"
        for v in vals:
            marker = "*" if (v == best and best > 0) else " "
            row += f" {v:>{w-1}.4f}{marker}"
        note = " ← drift!" if ch in drift_channels else ""
        print(row + note)

    print("-"*75)
    avgs = {t: sum(all_results[t])/len(all_results[t]) for t in tags_ordered}
    avg_row = f"{'AVERAGE':<10}" + "".join(f" {avgs[t]:>{w}.4f}" for t in tags_ordered)
    print(avg_row)

    base = avgs["ltr_k5"]
    print()
    for t in tags_ordered:
        diff = avgs[t] - base
        tag_str = "same" if abs(diff)<0.0001 else ("BETTER" if diff>0 else "worse")
        print(f"  {labels[t]:<22}: {avgs[t]:.4f}  ({tag_str}, {diff:+.4f})")

    # Save
    json_out = {
        "config": {"alpha_sweep": ALPHA_SWEEP, "channels": SMAP_CHANNELS,
                   "alpha_detect": ALPHA_DETECT},
        **{t: {"f1_per_channel": dict(zip(SMAP_CHANNELS, all_results[t])),
               "avg_f1": round(avgs[t], 4)} for t in tags_ordered},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    lines = [
        "="*75,
        "MAE Dual-Anchor (LTR + GAP) --- SMAP 14channels",
        f"alpha_detect={ALPHA_DETECT}  local_k=5",
        "="*75, hdr, "-"*75,
    ] + [
        f"{ch:<10}" + "".join(f" {all_results[t][i]:>{w}.4f}" for t in tags_ordered)
        for i, ch in enumerate(SMAP_CHANNELS)
    ] + [
        "-"*75, avg_row, "",
    ] + [
        f"  {labels[t]:<22}: {avgs[t]:.4f}  ({('BETTER' if avgs[t]-base>0 else 'worse' if avgs[t]-base<0 else 'same')}, {avgs[t]-base:+.4f})"
        for t in tags_ordered
    ] + [f"\nResults: {OUTPUT_DIR / 'results.json'}"]

    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nSummary: {OUTPUT_DIR / 'summary.txt'}")

if __name__ == "__main__":
    run()
