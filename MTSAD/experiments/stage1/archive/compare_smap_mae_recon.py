"""SMAP: DA(GAP+RMS α=0.5) + MAE Reconstruction Error — β sweep

비교 조건:
  grms_a05    : DA α=0.5 GAP+RMS (현재 최강, 0.7762) — 캐시 재활용
  recon_b09   : β=0.9  (DA 90% + Recon 10%)
  recon_b07   : β=0.7  (DA 70% + Recon 30%)
  recon_b05   : β=0.5  (DA 50% + Recon 50%)
  recon_only  : β=0.0  (Recon 100%)

특히 확인:
  F-3 (spike, 기존 0.0) → 올라오는지
  T-1 (amplitude dec, α=0.5에서 0.4) → 유지/향상
  P-3, P-4, T-3, R-1 → 유지되는지
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

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_mae_recon"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

BETAS     = [0.9, 0.7, 0.5, 0.0]
BETA_TAGS = {0.9: "recon_b09", 0.7: "recon_b07", 0.5: "recon_b05", 0.0: "recon_only"}

WATCH  = {"F-3": "spike (기존 0.0)", "T-1": "amp dec (기존 0.4)"}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]


# ---------------------------------------------------------------------------

def _ckpt(ch, tag):
    return CKPT_DIR / f"{ch}__{tag}.pkl"

def load_ckpt(ch, tag):
    # grms_a05 baseline: look in smap_gap_rms checkpoints
    paths = [_ckpt(ch, tag)]
    if tag == "grms_a05":
        paths.insert(0, ROOT/"results"/"smap_gap_rms"/"checkpoints"/f"{ch}__grms_a05.pkl")
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

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt  = load_gt()

    print("="*70)
    print("SMAP: DA(GAP+RMS α=0.5) + MAE Recon — β sweep")
    print(f"device={device}  betas={BETAS}")
    print("="*70)

    # Conditions: baseline + 4 beta variants
    all_tags = ["grms_a05"] + list(BETA_TAGS.values())
    results  = {tag: [] for tag in all_tags}

    # ---- Baseline: grms_a05 (캐시 로드) ----
    print("\n[grms_a05] 캐시 로드 중...")
    for ch in CHANNELS:
        cached = load_ckpt(ch, "grms_a05")
        f1 = cached["f1"] if cached else 0.0
        src = f"  ({Path(cached['_from']).parent.name})" if cached else " ← 캐시 없음"
        print(f"  {ch}: F1={f1:.4f}{src}")
        results["grms_a05"].append(f1)

    # ---- Beta sweep: 채널별 새 실험 ----
    # 모든 beta tag가 캐시됐는지 확인
    def all_cached_for_ch(ch):
        return all(load_ckpt(ch, BETA_TAGS[b]) for b in BETAS)

    needs_run = [ch for ch in CHANNELS if not all_cached_for_ch(ch)]

    if needs_run:
        print(f"\n[Recon β sweep] 모델 초기화 중... ({len(needs_run)}채널 미완료)")
        backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        recon    = MAE_Recon(device=device, n_iter=5, mask_ratio=0.5, seed=42)
        det      = ViT4TS_MAE_Recon(
            backbone=backbone, recon=recon,
            alpha_da=0.5, local_k=5, min_ref=5, patch_size=16,
            window_size=224, window_step_ratio=4.0,
            device=str(device),
            image_size=(224,224), alpha_detect=ALPHA_DETECT,
            smoothing_alpha=1.0, batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[Recon β sweep] 전부 캐시됨")

    for ch in CHANNELS:
        gt = all_gt.get(ch, [])

        # Load cached betas
        ch_cached = {}
        for b in BETAS:
            tag    = BETA_TAGS[b]
            cached = load_ckpt(ch, tag)
            if cached:
                ch_cached[b] = cached["f1"]

        # Run if any beta is missing
        if len(ch_cached) < len(BETAS) and det is not None:
            if not gt:
                for b in BETAS:
                    ch_cached[b] = 0.0
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n{'='*60}")
                print(f"Channel: {ch}  {'← ' + WATCH[ch] if ch in WATCH else ''}")
                print(f"{'='*60}")

                beta_scores = det.predict_scores_all_betas(data, betas=BETAS)

                for b in BETAS:
                    tag = BETA_TAGS[b]
                    if b in ch_cached:
                        print(f"  β={b}: ckpt  F1={ch_cached[b]:.4f}")
                        continue
                    if b not in beta_scores:
                        ch_cached[b] = 0.0
                        continue

                    scores, timestamps = beta_scores[b]
                    from preprocessing.data_utils import intervals_from_indices
                    from models.model_utils import compute_detection_intervals
                    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=ALPHA_DETECT)
                    ivs = intervals_from_indices(idx, timestamps, scores)
                    m   = evaluate_intervals(gt, _to_list(ivs))
                    f1  = round(m["F1"], 4)
                    save_ckpt(ch, tag, {"f1": f1, "p": round(m["precision"],4),
                                        "r": round(m["recall"],4)})
                    ch_cached[b] = f1
                    print(f"  β={b}: F1={f1:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        for b in BETAS:
            results[BETA_TAGS[b]].append(ch_cached.get(b, 0.0))

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    w = 10
    header_parts = [f"{'Channel':<10}"] + [f"{'grms_a05':>{w}}"] + \
                   [f"β={b:>3}".rjust(w) for b in BETAS]
    hdr = " ".join(header_parts)

    print("\n" + "="*80)
    print("SMAP — DA(GAP+RMS α=0.5) + MAE Recon  β sweep")
    print(hdr)
    print("-"*80)

    for i, ch in enumerate(CHANNELS):
        base   = results["grms_a05"][i]
        row    = f"{ch:<10} {base:>{w}.4f}"
        best_b = None
        best_f1 = base
        for b in BETAS:
            f1    = results[BETA_TAGS[b]][i]
            delta = f1 - base
            mark  = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else " ")
            row  += f" {f1:>{w}.4f}{mark}"
            if f1 > best_f1:
                best_f1, best_b = f1, b
        note = f"  ← {WATCH[ch]}" if ch in WATCH else \
               "  (stable)" if ch in STABLE else ""
        print(row + note)

    print("-"*80)
    avg_base = sum(results["grms_a05"]) / len(CHANNELS)
    row_avg  = f"{'AVERAGE':<10} {avg_base:>{w}.4f}"
    for b in BETAS:
        avg_b = sum(results[BETA_TAGS[b]]) / len(CHANNELS)
        row_avg += f" {avg_b:>{w}.4f}"
    print(row_avg)

    print("\n[핵심 채널 상세]")
    for ch in list(WATCH.keys()) + STABLE[:3]:
        i = CHANNELS.index(ch)
        base = results["grms_a05"][i]
        parts = [f"grms_a05={base:.4f}"]
        for b in BETAS:
            parts.append(f"β={b}:{results[BETA_TAGS[b]][i]:.4f}")
        print(f"  {ch:<6}  " + "  ".join(parts))

    # Save JSON
    json_out = {
        "config": {
            "alpha_detect": ALPHA_DETECT,
            "channels": CHANNELS,
            "alpha_da": 0.5,
            "betas": BETAS,
            "n_iter_recon": 5,
            "mask_ratio": 0.5,
        },
        "grms_a05": {
            "f1_per_channel": dict(zip(CHANNELS, results["grms_a05"])),
            "avg_f1": round(avg_base, 4),
        },
    }
    for b in BETAS:
        tag = BETA_TAGS[b]
        avg = sum(results[tag]) / len(CHANNELS)
        json_out[tag] = {
            "f1_per_channel": dict(zip(CHANNELS, results[tag])),
            "avg_f1": round(avg, 4),
        }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    run()
