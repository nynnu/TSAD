"""SMAP: LTR + Sub-adjacent LTR + Intra-window Spatial, max fusion

비교 조건:
  1. LTR k=5            (캐시 재활용)
  2. GRMS α=0.5         (캐시 재활용, 0.7762)
  3. β=0.9              (캐시 재활용, 0.7952)
  4. Spatial only
  5. SubLTR only        (skip i±9, ref i±[10,20])
  6. max(LTR, Spatial)
  7. max(SubLTR, Spatial)
  8. max(LTR, SubLTR, Spatial)

특히 확인:
  F-3 (spike) → SubLTR or Spatial이 잡는지
  T-1 (amp dec) → 유지되는지
  D-1, D-2 (level shift) → LTR이 지키는지

No existing files modified.
Outputs: results/smap_mae_spatial/
"""

import ast, json, pickle, sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.mae_vision import MAE_AD
from models.vit4ts_mae_spatial import ViT4TS_MAE_Spatial
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_mae_spatial"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA_DETECT = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']

WATCH  = {"F-3": "spike", "T-1": "amp dec"}
STABLE = ["P-3","P-4","P-7","D-3","T-3","R-1"]

# 새로 계산할 태그 ↔ predict_scores의 key 매핑
NEW_MODE_TAG = {
    "spatial":       "spatial_only",
    "sub_ltr":       "sub_ltr_only",
    "max_ltr_sp":    "max_ltr_sp",
    "max_subltr_sp": "max_subltr_sp",
    "max_all":       "max_all",
}
NEW_TAGS = list(NEW_MODE_TAG.values())


def _ckpt(ch, tag): return CKPT_DIR / f"{ch}__{tag}.pkl"

def load_ckpt(ch, tag):
    fallbacks = {
        "ltr_k5":    ROOT/"results"/"smap_local_ref"/"checkpoints"/f"{ch}__mae__k5.pkl",
        "grms_a05":  ROOT/"results"/"smap_gap_rms"/"checkpoints"/f"{ch}__grms_a05.pkl",
        "recon_b09": ROOT/"results"/"smap_mae_recon"/"checkpoints"/f"{ch}__recon_b09.pkl",
    }
    paths = [fallbacks[tag]] if tag in fallbacks else []
    paths.append(_ckpt(ch, tag))
    for p in paths:
        if p.exists():
            d = pickle.load(open(p, "rb"))
            return {"f1": d.get("f1", 0)}
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

def _eval(scores, timestamps, gt):
    from preprocessing.data_utils import intervals_from_indices
    from models.model_utils import compute_detection_intervals
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=ALPHA_DETECT)
    ivs = intervals_from_indices(idx, timestamps, scores)
    m   = evaluate_intervals(gt, _to_list(ivs))
    return round(m["F1"], 4), round(m["precision"], 4), round(m["recall"], 4)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_gt = load_gt()

    print("="*75)
    print("SMAP: LTR + SubLTR + Spatial (max fusion)")
    print(f"device={device}")
    print("="*75)

    ref_tags = ["ltr_k5", "grms_a05", "recon_b09"]
    all_tags = ref_tags + NEW_TAGS
    results  = {tag: [] for tag in all_tags}

    # ---- 기존 결과 캐시 ----
    for tag in ref_tags:
        label = {"ltr_k5":"LTR k=5","grms_a05":"GRMS α=0.5","recon_b09":"β=0.9"}[tag]
        print(f"\n[{label}] 캐시 로드...")
        for ch in CHANNELS:
            cached = load_ckpt(ch, tag)
            results[tag].append(cached["f1"] if cached else 0.0)
        print(f"  avg={sum(results[tag])/len(CHANNELS):.4f}")

    # ---- 새 방법들 ----
    needs_run = [ch for ch in CHANNELS
                 if not all(_ckpt(ch, t).exists() for t in NEW_TAGS)]

    if needs_run:
        print(f"\n[Spatial+SubLTR] 모델 초기화 ({len(needs_run)}채널 미완료)...")
        backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
        det = ViT4TS_MAE_Spatial(
            backbone=backbone,
            local_k=5, sub_k_min=10, sub_k_max=20,
            top_k_ratio=0.1, min_ref=5, patch_size=16,
            window_size=224, window_step_ratio=4.0,
            device=str(device), image_size=(224, 224),
            alpha_detect=ALPHA_DETECT, smoothing_alpha=1.0,
            batch_size=32, verbose=True,
        )
    else:
        det = None
        print("\n[Spatial+SubLTR] 전부 캐시됨")

    for ch in CHANNELS:
        gt = all_gt.get(ch, [])

        ch_new = {}
        for t in NEW_TAGS:
            cached = load_ckpt(ch, t)
            if cached:
                ch_new[t] = cached["f1"]

        if len(ch_new) < len(NEW_TAGS) and det is not None:
            if not gt:
                for t in NEW_TAGS:
                    ch_new[t] = 0.0
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n{'='*55}")
                print(f"Channel: {ch}  {'← '+WATCH[ch] if ch in WATCH else ''}")
                print(f"{'='*55}")

                all_scores = det.predict_scores(data)

                for mode, tag in NEW_MODE_TAG.items():
                    if tag in ch_new:
                        print(f"  {tag}: ckpt F1={ch_new[tag]:.4f}")
                        continue
                    scores, timestamps = all_scores[mode]
                    f1, p, r = _eval(scores, timestamps, gt)
                    save_ckpt(ch, tag, {"f1": f1, "p": p, "r": r})
                    ch_new[tag] = f1
                    print(f"  {tag}: F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        for t in NEW_TAGS:
            results[t].append(ch_new.get(t, 0.0))

    # ---- Summary ----
    w = 11
    col_labels = {
        "ltr_k5": "LTR k=5", "grms_a05": "GRMS α=0.5", "recon_b09": "β=0.9",
        "spatial_only": "Spatial", "sub_ltr_only": "SubLTR",
        "max_ltr_sp": "max(L,Sp)", "max_subltr_sp": "max(sL,Sp)", "max_all": "max(ALL)",
    }
    hdr = f"{'Ch':<6}" + "".join(f" {col_labels[t]:>{w}}" for t in all_tags)
    print("\n" + "="*95)
    print("SMAP — LTR + SubLTR + Spatial (max fusion)")
    print(hdr); print("-"*95)

    best_prev = "recon_b09"
    for i, ch in enumerate(CHANNELS):
        row  = f"{ch:<6}"
        base = results[best_prev][i]
        for tag in all_tags:
            v    = results[tag][i]
            mark = ""
            if tag == "max_all":
                mark = "▲" if v-base > 0.001 else ("▼" if v-base < -0.001 else " ")
            row += f" {v:>{w}.4f}{mark}"
        note = f" ← {WATCH[ch]}" if ch in WATCH else (" (stable)" if ch in STABLE else "")
        print(row + note)

    print("-"*95)
    avgs = {tag: sum(results[tag])/len(CHANNELS) for tag in all_tags}
    print(f"{'AVG':<6}" + "".join(f" {avgs[t]:>{w}.4f} " for t in all_tags))

    # Best new method
    best_new = max(NEW_TAGS, key=lambda t: avgs[t])
    print(f"\n  현재 BEST (β=0.9): {avgs['recon_b09']:.4f}")
    for t in NEW_TAGS:
        diff = avgs[t] - avgs["recon_b09"]
        mark = "✓ NEW BEST" if avgs[t] > avgs["recon_b09"] else ""
        print(f"  {col_labels[t]:<14}: {avgs[t]:.4f}  ({diff:+.4f}) {mark}")

    json_out = {"config": {"alpha_detect": ALPHA_DETECT, "channels": CHANNELS,
                            "sub_k_min": 10, "sub_k_max": 20, "top_k_ratio": 0.1}}
    for tag in all_tags:
        json_out[tag] = {"f1_per_channel": dict(zip(CHANNELS, results[tag])),
                         "avg_f1": round(avgs[tag], 4)}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    run()
