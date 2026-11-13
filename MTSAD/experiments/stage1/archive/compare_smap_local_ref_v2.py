"""
MAE + Local Temporal Reference (k=5) on SMAP 14 channels.
Compares: MAE global (기존) vs MAE local k=5

v2 변경점: 기존 체크포인트 재활용
  - global(k=0) 결과는 results/smap/checkpoints/{ch}_mae_fixed.pkl 에서 먼저 탐색
  - 없을 때만 results/smap_local_ref_v2/checkpoints/{ch}__mae__k0.pkl 에 새로 저장
  - local k=5 결과는 results/smap_local_ref/checkpoints/{ch}__mae__k5.pkl 에서 먼저 탐색

No existing files modified.
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
from evaluation.evaluate import evaluate_intervals

DATA_DIR    = ROOT / "data" / "SMAP"
ANOMALY_CSV = ROOT / "data" / "anomalies.csv"
OUTPUT_DIR  = ROOT / "results" / "smap_local_ref_v2"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

ALPHA    = 0.01
CHANNELS = ['P-1','P-3','P-4','P-7','D-1','D-2','D-3',
            'F-1','F-2','F-3','T-1','T-2','T-3','R-1']
K_VALUES = [5, 0]
K_LABELS = {5: "local k=5", 0: "global"}

DETECTOR_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    image_size=(224, 224), alpha=ALPHA,
    smoothing_alpha=1.0, verbose=True,
)

# ---------------------------------------------------------------------------
# 체크포인트 탐색 우선순위 정의
# fallback_paths[k] = [(path, f1_key), ...]  순서대로 탐색, 첫 번째 존재하는 파일 사용
# ---------------------------------------------------------------------------
def _ckpt_fallbacks(ch, k):
    """기존 결과 → 새 저장소 순으로 탐색할 경로 목록 반환."""
    fallbacks = []
    if k == 0:
        # 원래 SMAP 실험 결과 (mae_fixed)
        fallbacks.append((ROOT / "results" / "smap" / "checkpoints" / f"{ch}_mae_fixed.pkl", "f1"))
    if k == 5:
        # 이전 local ref 실험 결과
        fallbacks.append((ROOT / "results" / "smap_local_ref" / "checkpoints" / f"{ch}__mae__k5.pkl", "f1"))
    # 이 실험 자체의 저장소 (최후 fallback)
    fallbacks.append((CKPT_DIR / f"{ch}__mae__k{k}.pkl", "f1"))
    return fallbacks

def load_ckpt(ch, k):
    for path, key in _ckpt_fallbacks(ch, k):
        if path.exists():
            data = pickle.load(open(path, "rb"))
            return {"f1": data[key], "p": data.get("p", data.get("precision", 0)),
                    "r": data.get("r", data.get("recall", 0)),
                    "_from": str(path)}
    return None

def save_ckpt(ch, k, val):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(val, open(CKPT_DIR / f"{ch}__mae__k{k}.pkl", "wb"))

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

    print("="*60)
    print("MAE Local Ref (k=5) vs Global — SMAP 14 channels [v2]")
    print("="*60)

    results = {k: [] for k in K_VALUES}

    for k in K_VALUES:
        klbl = K_LABELS[k]
        all_cached = all(load_ckpt(ch, k) is not None for ch in CHANNELS)

        if all_cached:
            det = None
            print(f"\n[{klbl}] 전부 캐시됨 (기존 체크포인트 포함)")
        else:
            print(f"\n[{klbl}] 초기화 중...")
            backbone = MAE_AD(model_name="vit_base_patch16_224.mae", device=device)
            det = ViT4TS_Local(backbone=backbone, patch_size=16,
                               local_k=k, device=str(device), **DETECTOR_PARAMS)

        for ch in CHANNELS:
            gt = all_gt.get(ch, [])
            if not gt:
                print(f"  SKIP {ch}")
                continue

            cached = load_ckpt(ch, k)
            if cached:
                f1 = cached["f1"]
                src = Path(cached["_from"]).parent.parent.name  # results 폴더명
                print(f"  {ch}: ckpt({src})  F1={f1:.4f}")
            else:
                data = pd.read_csv(DATA_DIR / f"{ch}.csv")
                print(f"\n  {ch}: running ...")
                ivs = det.detect(data)
                m   = evaluate_intervals(gt, _to_list(ivs))
                f1  = round(m["F1"], 4)
                save_ckpt(ch, k, {"f1": f1, "p": round(m["precision"],4), "r": round(m["recall"],4)})
                print(f"    F1={f1:.4f}")
            results[k].append(f1)

    # Table
    print("\n" + "="*60)
    print(f"{'Channel':<10} {'global':>8} {'local k=5':>10}  Delta")
    print("-"*40)

    for i, ch in enumerate(CHANNELS):
        g = results[0][i] if i < len(results[0]) else 0
        l = results[5][i] if i < len(results[5]) else 0
        delta = l - g
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"{ch:<10} {g:>8.4f} {l:>10.4f}  {delta:+.4f} {marker}")

    print("-"*40)
    avg_g = sum(results[0]) / len(results[0])
    avg_l = sum(results[5]) / len(results[5])
    print(f"{'AVERAGE':<10} {avg_g:>8.4f} {avg_l:>10.4f}  {avg_l-avg_g:+.4f}")

    json_out = {"config": {"alpha": ALPHA, "channels": CHANNELS},
                "global": results[0], "local_k5": results[5]}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'results.json'}")

if __name__ == "__main__":
    run()
