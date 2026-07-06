"""NAB-AWS: LTR × MAE Recon (product fusion) vs 기존 방법들

비교 조건:
  1. LTR k=5       (baseline, 캐시 재활용)
  2. GRMS α=0.5    (캐시 재활용)
  3. β=0.9         (캐시 재활용)
  4. LTR × Recon   (새 방법)

No existing files modified.
Outputs: results/nab_ltr_recon/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.mae_recon_ad import MAE_Recon
from models.vit4ts_ltr_recon import ViT4TS_LTR_Recon
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_ltr_recon"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01


def _ckpt(sig, tag): return CKPT_DIR / f"{sig}__{tag}.pkl"

def load_ckpt(sig, tag):
    fallbacks = {
        "ltr_k5":    ROOT/"results"/"nab_aws_local_ref"/"checkpoints"/f"{sig}__mae__k5.pkl",
        "grms_a05":  ROOT/"results"/"nab_mae_recon"/"checkpoints"/f"{sig}__grms_a05.pkl",
        "recon_b09": ROOT/"results"/"nab_mae_recon"/"checkpoints"/f"{sig}__recon_b09.pkl",
    }
    paths = [fallbacks[tag]] if tag in fallbacks else []
    paths.append(_ckpt(sig, tag))
    for p in paths:
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", 0)}
    return None

def save_ckpt(sig, tag, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(_ckpt(sig, tag), "wb"))

def load_gt():
    gt = {}
    with open(ANOMALY_CSV) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def get_signals():
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))

def _to_list(df):
    return df[["start","end"]].values.tolist() if len(df) > 0 else []


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()
    signals = get_signals()

    print("="*65)
    print("NAB-AWS: LTR × MAE Recon (product fusion)")
    print(f"device={device}  signals={len(signals)}")
    print("="*65)

    ref_tags = ["ltr_k5", "grms_a05", "recon_b09"]
    new_tag  = "ltr_x_recon"
    all_tags = ref_tags + [new_tag]
    results  = {tag: [] for tag in all_tags}

    for tag in ref_tags:
        label = {"ltr_k5": "LTR k=5", "grms_a05": "GRMS α=0.5", "recon_b09": "β=0.9"}[tag]
        print(f"\n[{label}] 캐시 로드...")
        for sig in signals:
            cached = load_ckpt(sig, tag)
            results[tag].append(cached["f1"] if cached else 0.0)
        print(f"  avg={sum(results[tag])/len(signals):.4f}")

    needs_run = [s for s in signals if not _ckpt(s, new_tag).exists()]

    if needs_run:
        print(f"\n[LTR × Recon] 모델 초기화 ({len(needs_run)}시그널 미완료)...")
        backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        recon    = MAE_Recon(device=device, n_iter=5, mask_ratio=0.5, seed=42)
        det      = ViT4TS_LTR_Recon(
            backbone=backbone, recon=recon,
            local_k=5, min_ref=5, patch_size=16,
            window_size=224, window_step_ratio=4.0,
            device=str(device), image_size=(224, 224),
            alpha_detect=ALPHA_DETECT, smoothing_alpha=1.0,
            batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[LTR × Recon] 전부 캐시됨")

    for sig in signals:
        gt     = all_gt.get(sig, [])
        cached = load_ckpt(sig, new_tag)
        if cached:
            f1 = cached["f1"]
            print(f"  {sig}: ckpt F1={f1:.4f}")
        elif not gt:
            f1 = 0.0
        else:
            data = pd.read_csv(DATA_DIR / f"{sig}.csv")
            print(f"\n{'='*55}\nSignal: {sig}\n{'='*55}")
            ivs = det.detect(data)
            m   = evaluate_intervals(gt, _to_list(ivs))
            f1  = round(m["F1"], 4)
            save_ckpt(sig, new_tag, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
            print(f"  → F1={f1:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        results[new_tag].append(f1)

    # ---- Summary ----
    w = 11
    labels = {"ltr_k5":"LTR k=5", "grms_a05":"GRMS α=0.5", "recon_b09":"β=0.9", "ltr_x_recon":"LTR×Recon"}
    hdr = f"{'Signal':<38}" + "".join(f" {labels[t]:>{w}}" for t in all_tags)
    print("\n" + "="*85)
    print("NAB-AWS — LTR × Recon 결과")
    print(hdr); print("-"*85)

    for i, sig in enumerate(signals):
        row  = f"{sig:<38}"
        base = results["ltr_k5"][i]
        for tag in all_tags:
            v    = results[tag][i]
            mark = "▲" if (tag==new_tag and v-base>0.001) else ("▼" if (tag==new_tag and v-base<-0.001) else " ")
            row += f" {v:>{w}.4f}{mark}"
        print(row)

    print("-"*85)
    avgs = {tag: sum(results[tag])/len(signals) for tag in all_tags}
    print(f"{'AVERAGE':<38}" + "".join(f" {avgs[t]:>{w}.4f} " for t in all_tags))

    json_out = {"config": {"alpha_detect": ALPHA_DETECT, "signals": signals}}
    for tag in all_tags:
        json_out[tag] = {"f1_per_signal": dict(zip(signals, results[tag])),
                         "avg_f1": round(avgs[tag], 4)}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    run()
