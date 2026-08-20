"""
NAB-AWS: DA(GAP+RMS α=0.5) + MAE Reconstruction Error  β=0.9

비교 조건:
  1. LTR k=5        (캐시 재활용, baseline 0.6272)
  2. GRMS α=0.5     (β=1.0, DA only)
  3. β=0.9          (DA 90% + Recon 10%)  ← SMAP best config

기존 체크포인트 재활용:
  LTR k=5: results/nab_aws_local_ref/checkpoints/{sig}__mae__k5.pkl

No existing files modified.
Outputs: results/nab_mae_recon/
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
from models.vit4ts_mae_recon import ViT4TS_MAE_Recon
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "realAWSCloudwatch"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "nab_mae_recon"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
BETAS        = [1.0, 0.9]
BETA_TAGS    = {1.0: "grms_a05", 0.9: "recon_b09"}


# ---------------------------------------------------------------------------

def _ckpt(sig, tag):
    return CKPT_DIR / f"{sig}__{tag}.pkl"

def load_ckpt(sig, tag):
    paths = [_ckpt(sig, tag)]
    if tag == "ltr_k5":
        paths.insert(0, ROOT/"results"/"nab_aws_local_ref"/"checkpoints"/f"{sig}__mae__k5.pkl")
    for p in paths:
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", 0), "_from": str(p)}
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
    csvs = sorted(DATA_DIR.glob("*.csv"))
    return [p.stem for p in csvs]

def _to_list(df):
    return df[["start","end"]].values.tolist() if len(df) > 0 else []


# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()
    signals = get_signals()

    print("="*65)
    print("NAB-AWS: DA(GAP+RMS α=0.5) + MAE Recon β=0.9")
    print(f"device={device}  signals={len(signals)}")
    print("="*65)

    all_tags = ["ltr_k5"] + [BETA_TAGS[b] for b in BETAS]
    results  = {tag: [] for tag in all_tags}

    # ---- Baseline: LTR k=5 캐시 ----
    print("\n[LTR k=5] 캐시 로드 중...")
    for sig in signals:
        cached = load_ckpt(sig, "ltr_k5")
        f1 = cached["f1"] if cached else 0.0
        src = f"  ({Path(cached['_from']).parent.name})" if cached else " ← 캐시 없음"
        print(f"  {sig}: F1={f1:.4f}{src}")
        results["ltr_k5"].append(f1)

    # ---- GRMS α=0.5 + β=0.9 ----
    def all_cached_for_sig(sig):
        return all(load_ckpt(sig, BETA_TAGS[b]) for b in BETAS)

    needs_run = [s for s in signals if not all_cached_for_sig(s)]

    if needs_run:
        print(f"\n[MAE Recon] 모델 초기화 중... ({len(needs_run)}시그널 미완료)")
        backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        recon    = MAE_Recon(device=device, n_iter=5, mask_ratio=0.5, seed=42)
        det      = ViT4TS_MAE_Recon(
            backbone=backbone, recon=recon,
            alpha_da=0.5, local_k=5, min_ref=5, patch_size=16,
            window_size=224, window_step_ratio=4.0,
            device=str(device),
            image_size=(224, 224), alpha_detect=ALPHA_DETECT,
            smoothing_alpha=1.0, batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[MAE Recon] 전부 캐시됨")

    for sig in signals:
        gt = all_gt.get(sig, [])

        sig_cached = {}
        for b in BETAS:
            tag    = BETA_TAGS[b]
            cached = load_ckpt(sig, tag)
            if cached:
                sig_cached[b] = cached["f1"]

        if len(sig_cached) < len(BETAS) and det is not None:
            if not gt:
                for b in BETAS:
                    sig_cached[b] = 0.0
            else:
                data = pd.read_csv(DATA_DIR / f"{sig}.csv")
                print(f"\n{'='*55}")
                print(f"Signal: {sig}")
                print(f"{'='*55}")

                beta_scores = det.predict_scores_all_betas(data, betas=BETAS)

                for b in BETAS:
                    tag = BETA_TAGS[b]
                    if b in sig_cached:
                        print(f"  β={b}: ckpt  F1={sig_cached[b]:.4f}")
                        continue

                    scores, timestamps = beta_scores[b]
                    from preprocessing.data_utils import intervals_from_indices
                    from models.model_utils import compute_detection_intervals
                    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=ALPHA_DETECT)
                    ivs = intervals_from_indices(idx, timestamps, scores)
                    m   = evaluate_intervals(gt, _to_list(ivs))
                    f1  = round(m["F1"], 4)
                    save_ckpt(sig, tag, {"f1": f1, "p": round(m["precision"], 4),
                                         "r": round(m["recall"], 4)})
                    sig_cached[b] = f1
                    print(f"  β={b}: F1={f1:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        for b in BETAS:
            results[BETA_TAGS[b]].append(sig_cached.get(b, 0.0))

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    w = 12
    hdr = f"{'Signal':<40} {'LTR k=5':>{w}} {'GRMS α=0.5':>{w}} {'β=0.9':>{w}}"
    print("\n" + "="*80)
    print("NAB-AWS — DA(GAP+RMS α=0.5) + MAE Recon β=0.9")
    print(hdr)
    print("-"*80)

    for i, sig in enumerate(signals):
        ltr  = results["ltr_k5"][i]
        grms = results["grms_a05"][i]
        b09  = results["recon_b09"][i]
        delta = b09 - ltr
        mark  = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else " ")
        print(f"{sig:<40} {ltr:>{w}.4f} {grms:>{w}.4f} {b09:>{w}.4f} {mark}")

    print("-"*80)
    avg_ltr  = sum(results["ltr_k5"])    / len(signals)
    avg_grms = sum(results["grms_a05"])  / len(signals)
    avg_b09  = sum(results["recon_b09"]) / len(signals)
    print(f"{'AVERAGE':<40} {avg_ltr:>{w}.4f} {avg_grms:>{w}.4f} {avg_b09:>{w}.4f}")
    print()
    print(f"  LTR k=5    baseline : {avg_ltr:.4f}")
    print(f"  GRMS α=0.5          : {avg_grms:.4f}  ({'BETTER' if avg_grms > avg_ltr else 'worse'}, {avg_grms-avg_ltr:+.4f})")
    print(f"  β=0.9 (DA+Recon)    : {avg_b09:.4f}  ({'BETTER' if avg_b09 > avg_ltr else 'worse'}, {avg_b09-avg_ltr:+.4f})")

    json_out = {
        "config": {"alpha_detect": ALPHA_DETECT, "signals": signals,
                   "alpha_da": 0.5, "beta": 0.9, "n_iter_recon": 5},
        "ltr_k5":    {"f1_per_signal": dict(zip(signals, results["ltr_k5"])),
                      "avg_f1": round(avg_ltr, 4)},
        "grms_a05":  {"f1_per_signal": dict(zip(signals, results["grms_a05"])),
                      "avg_f1": round(avg_grms, 4)},
        "recon_b09": {"f1_per_signal": dict(zip(signals, results["recon_b09"])),
                      "avg_f1": round(avg_b09, 4)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    run()
