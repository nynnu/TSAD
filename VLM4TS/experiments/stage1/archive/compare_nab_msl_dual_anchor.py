"""
MAE Dual-Anchor α sweep — NAB-AWS + MSL (+ SMAP 요약 포함)

비교 조건:
  1. Global ref   (캐시 재활용)
  2. LTR k=5      (캐시 재활용)
  3. Dual α=0.9
  4. Dual α=0.7
  5. Dual α=0.5
  6. Dual α=0.3

기존 체크포인트 재활용:
  NAB Global  : results/nab_aws_local_ref/checkpoints/{sig}__mae__k0.pkl
  NAB LTR k=5 : results/nab_aws_local_ref/checkpoints/{sig}__mae__k5.pkl
  MSL Global  : results/clip_vs_dino/checkpoints/{ch}_mae.pkl
  MSL LTR k=5 : results/msl_local_ref/checkpoints/{ch}__mae__k5.pkl

No existing files modified.
Outputs: results/nab_msl_dual_anchor/
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

ANOMALY_CSV  = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR   = ROOT / "results" / "nab_msl_dual_anchor"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
ALPHA_SWEEP  = [0.9, 0.7, 0.5, 0.3]

DATASET_CFG = {
    "nab": {
        "data_dir":    ROOT / "data" / "realAWSCloudwatch",
        "channels":    None,   # from glob
        "ckpt_global": lambda s: ROOT/"results"/"nab_aws_local_ref"/"checkpoints"/f"{s}__mae__k0.pkl",
        "ckpt_ltr":    lambda s: ROOT/"results"/"nab_aws_local_ref"/"checkpoints"/f"{s}__mae__k5.pkl",
        "baseline_ltr": 0.6272,
    },
    "msl": {
        "data_dir":    ROOT / "data" / "MSL",
        "channels":    ['P-11','T-12','D-15','C-1','F-8','F-7',
                        'T-13','D-16','T-8','P-14','D-14'],
        "ckpt_global": lambda c: ROOT/"results"/"clip_vs_dino"/"checkpoints"/f"{c}_mae.pkl",
        "ckpt_ltr":    lambda c: ROOT/"results"/"msl_local_ref"/"checkpoints"/f"{c}__mae__k5.pkl",
        "baseline_ltr": 0.6344,
    },
}

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha_detect=ALPHA_DETECT,
    smoothing_alpha=1.0, batch_size=32, verbose=True,
)

ALPHA_TAGS   = [f"dual_a{int(a*10):02d}" for a in ALPHA_SWEEP]
ALPHA_LABELS = {f"dual_a{int(a*10):02d}": f"DA α={a:.1f}" for a in ALPHA_SWEEP}

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt(ds, ch, tag):
    return CKPT_DIR / f"{ds}__{ch}__{tag}.pkl"

def load_ckpt(ds, ch, tag, cfg):
    # check fallback paths first
    fallbacks = []
    if tag == "global":
        fallbacks.append(cfg["ckpt_global"](ch))
    elif tag == "ltr_k5":
        fallbacks.append(cfg["ckpt_ltr"](ch))
    fallbacks.append(_ckpt(ds, ch, tag))

    for p in fallbacks:
        if p and Path(p).exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", d.get("F1", 0)),
                    "p":  d.get("p", d.get("precision", 0)),
                    "r":  d.get("r", d.get("recall", 0)),
                    "_from": str(p)}
    return None

def save_ckpt(ds, ch, tag, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(ds, ch, tag), "wb"))

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
# Run one condition on one dataset
# ---------------------------------------------------------------------------

def run_condition(det, ds, channels, cfg, all_gt, tag):
    f1s = []
    for ch in channels:
        gt = all_gt.get(ch, [])
        if not gt:
            f1s.append(0.0)
            continue
        cached = load_ckpt(ds, ch, tag, cfg)
        if cached:
            f1  = cached["f1"]
            src = Path(cached["_from"]).parent.parent.name
            print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
        else:
            data = pd.read_csv(cfg["data_dir"] / f"{ch}.csv")
            print(f"\n  {ch}: running [{tag}]...")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(ds, ch, tag, {"f1": f1, "p": round(m["precision"],4),
                                    "r": round(m["recall"],4)})
            print(f"    F1={f1:.4f}")
        f1s.append(f1)
    return f1s

# ---------------------------------------------------------------------------
# Print per-dataset summary
# ---------------------------------------------------------------------------

def print_ds_summary(ds, channels, results, baseline_ltr):
    all_tags  = ["global", "ltr_k5"] + ALPHA_TAGS
    all_labels = {"global": "Global", "ltr_k5": "LTR k=5",
                  **ALPHA_LABELS}
    w = 10

    print(f"\n{'='*75}")
    print(f"[{ds.upper()} Summary]")
    hdr = f"  {'Signal/Channel':<40}" + "".join(f" {all_labels[t]:>{w}}" for t in all_tags) + "  Best"
    print(hdr)
    print("  " + "-"*(40 + len(all_tags)*(w+1) + 6))

    for i, ch in enumerate(channels):
        vals = [results[t][i] for t in all_tags]
        best = max(vals)
        best_tag = all_labels[all_tags[vals.index(best)]]
        row = f"  {ch:<40}"
        for v in vals:
            marker = "*" if (v == best and best > 0) else " "
            row += f" {v:>{w-1}.4f}{marker}"
        print(row + f"  {best_tag}")

    print("  " + "-"*(40 + len(all_tags)*(w+1) + 6))
    avgs = {t: sum(results[t])/len(results[t]) for t in all_tags}
    avg_row = f"  {'AVERAGE':<40}" + "".join(f" {avgs[t]:>{w}.4f}" for t in all_tags)
    print(avg_row)

    print()
    base = avgs["ltr_k5"]
    print(f"  Baseline LTR k=5: {baseline_ltr} (reference)")
    for t in ALPHA_TAGS:
        diff = avgs[t] - base
        flag = "✓ BETTER" if diff > 0 else ("✗ worse")
        print(f"  {all_labels[t]:<12}: {avgs[t]:.4f}  ({flag}, {diff:+.4f} vs LTR k=5)")

    return avgs

# ---------------------------------------------------------------------------
# Load SMAP dual-anchor results (already computed)
# ---------------------------------------------------------------------------

def load_smap_results():
    p = ROOT / "results" / "smap_dual_anchor" / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    out = {}
    out["global"]  = d.get("global",  {}).get("avg_f1", None) or \
                     round(sum(d.get("global",{}).get("f1_per_channel",{}).values() or [0])/
                           max(1, len(d.get("global",{}).get("f1_per_channel",{}))), 4) \
                     if "global" in d else None
    out["ltr_k5"]  = d.get("ltr_k5",  {}).get("avg_f1", None)
    for a in ALPHA_SWEEP:
        tag = f"dual_a{int(a*10):02d}"
        out[tag] = d.get(tag, {}).get("avg_f1", None)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()

    print("="*75)
    print(f"MAE Dual-Anchor α sweep — NAB-AWS + MSL")
    print(f"device={device}  α sweep={ALPHA_SWEEP}")
    print("="*75)

    # Single MAE backbone shared across all dual-anchor conditions
    print("\n모델 초기화 중...")
    bb_base = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
    bb_dual = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)

    det_global = ViT4TS_Local(backbone=bb_base, patch_size=16, local_k=0,
                              device=str(device), window_size=224,
                              window_step_ratio=4.0, image_size=(224,224),
                              alpha=ALPHA_DETECT, smoothing_alpha=1.0, verbose=True)
    det_ltr    = ViT4TS_Local(backbone=bb_base, patch_size=16, local_k=5,
                              device=str(device), window_size=224,
                              window_step_ratio=4.0, image_size=(224,224),
                              alpha=ALPHA_DETECT, smoothing_alpha=1.0, verbose=True)

    all_ds_results = {}   # ds → tag → [f1]
    all_ds_avgs    = {}   # ds → tag → avg_f1

    for ds, cfg in DATASET_CFG.items():
        print(f"\n{'='*75}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*75}")

        # resolve channel list
        if cfg["channels"] is None:
            channels = sorted(f.stem for f in cfg["data_dir"].glob("*.csv") if f.stem in all_gt)
        else:
            channels = [c for c in cfg["channels"] if c in all_gt]

        results = {}

        # --- Global ---
        all_cached = all(load_ckpt(ds, ch, "global", cfg) for ch in channels)
        print(f"\n[Global] {'전부 캐시됨' if all_cached else '실행 중...'}")
        results["global"] = run_condition(
            det_global if not all_cached else None,
            ds, channels, cfg, all_gt, "global"
        )

        # --- LTR k=5 ---
        all_cached = all(load_ckpt(ds, ch, "ltr_k5", cfg) for ch in channels)
        print(f"\n[LTR k=5] {'전부 캐시됨' if all_cached else '실행 중...'}")
        results["ltr_k5"] = run_condition(
            det_ltr if not all_cached else None,
            ds, channels, cfg, all_gt, "ltr_k5"
        )

        # --- Dual-Anchor α sweep ---
        for alpha_ltr in ALPHA_SWEEP:
            tag   = f"dual_a{int(alpha_ltr*10):02d}"
            label = f"Dual α={alpha_ltr:.1f}"
            all_cached = all(load_ckpt(ds, ch, tag, cfg) for ch in channels)

            if all_cached:
                print(f"\n[{label}] 전부 캐시됨")
                results[tag] = [load_ckpt(ds, ch, tag, cfg)["f1"] for ch in channels]
            else:
                print(f"\n[{label}] 실행 중...")
                det_da = ViT4TS_DualAnchor(
                    backbone=bb_dual, patch_size=16, local_k=5,
                    alpha=alpha_ltr, device=str(device), **BASE_PARAMS,
                )
                results[tag] = run_condition(det_da, ds, channels, cfg, all_gt, tag)

        all_ds_results[ds] = results
        all_ds_avgs[ds]    = print_ds_summary(ds, channels, results, cfg["baseline_ltr"])

    # ---------------------------------------------------------------------------
    # Cross-dataset summary
    # ---------------------------------------------------------------------------
    all_tags   = ["global", "ltr_k5"] + ALPHA_TAGS
    all_labels = {"global": "Global", "ltr_k5": "LTR k=5", **ALPHA_LABELS}
    w = 10

    smap_avgs = load_smap_results()

    print("\n" + "="*75)
    print("Cross-dataset Summary — avg F1")
    print("="*75)
    hdr = f"{'Dataset':<10}" + "".join(f" {all_labels[t]:>{w}}" for t in all_tags)
    print(hdr)
    print("-"*75)

    # SMAP row
    if smap_avgs:
        row = f"{'SMAP':<10}"
        for t in all_tags:
            v = smap_avgs.get(t)
            row += f" {v:>{w}.4f}" if v is not None else f" {'?':>{w}}"
        print(row)

    for ds in ["nab", "msl"]:
        avgs = all_ds_avgs[ds]
        row  = f"{ds.upper():<10}" + "".join(f" {avgs[t]:>{w}.4f}" for t in all_tags)
        print(row)
    print("-"*75)

    # Best α per dataset
    print()
    ds_names = (["smap"] if smap_avgs else []) + list(DATASET_CFG.keys())
    for ds in ds_names:
        avgs = smap_avgs if ds == "smap" else all_ds_avgs[ds]
        if avgs is None: continue
        da_avgs  = {t: avgs[t] for t in ALPHA_TAGS if avgs.get(t) is not None}
        if da_avgs:
            best_t   = max(da_avgs, key=da_avgs.get)
            best_v   = da_avgs[best_t]
            base_ltr = avgs.get("ltr_k5", 0)
            print(f"  {ds.upper():<8} best α: {all_labels[best_t]}  "
                  f"F1={best_v:.4f}  vs LTR k=5={base_ltr:.4f}  "
                  f"({'+' if best_v-base_ltr>=0 else ''}{best_v-base_ltr:.4f})")

    # Save
    json_out = {
        "config": {"alpha_sweep": ALPHA_SWEEP, "alpha_detect": ALPHA_DETECT},
        **{ds: {t: {"f1_per_channel": dict(zip(
                        all_ds_results[ds].get("channels", []),
                        all_ds_results[ds][t])),
                    "avg_f1": round(all_ds_avgs[ds][t], 4)}
               for t in all_tags if t in all_ds_results[ds]}
           for ds in DATASET_CFG},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")

    # summary.txt
    summary_lines = [
        "="*75,
        "MAE Dual-Anchor alpha sweep --- NAB-AWS + MSL",
        f"alpha_detect={ALPHA_DETECT}  local_k=5",
        "="*75, "", hdr, "-"*75,
    ]
    if smap_avgs:
        r = f"{'SMAP':<10}"
        for t in all_tags:
            v = smap_avgs.get(t)
            r += f" {v:>{w}.4f}" if v is not None else f" {'?':>{w}}"
        summary_lines.append(r)
    for ds in DATASET_CFG:
        summary_lines.append(
            f"{ds.upper():<10}" + "".join(f" {all_ds_avgs[ds][t]:>{w}.4f}" for t in all_tags)
        )
    summary_lines += ["-"*75, f"\nResults: {OUTPUT_DIR / 'results.json'}"]
    open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8").write(
        "\n".join(summary_lines) + "\n"
    )
    print(f"Summary: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    run()
