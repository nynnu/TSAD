"""
Line Plot 이미지 기반 Patch 토큰 실험

CLS-G (line plot → DINOv2 CLS) 가 wavelet CLS보다 SMAP에서 훨씬 좋음(0.9762 vs 0.8355).
→ Line plot에서 patch 토큰을 뽑아도 동일한 이점이 있는지 확인.

윈도우: WINDOW_SIZE=224, STEP=56 (CLS-G와 동일, overlapping)
이미지: time_series_to_image (꺾은선 그래프, 224×224)
CLS 캐시: {channel}_f{feat_idx}_{split}_cls.npy 재사용
Patch 캐시: lineplt_{channel}_f{feat_idx}_{split}_patches.npy (신규)

비교 방식:
  knn_max    : max(KNN_dist)              기존 patch memory bank 방식
  gate_sum   : Σ (1-max(0,CosSim)) * KNN 이전 best gate 방식
  resid_sum  : Residual KNN sum           Strategy B

데이터셋: MSL / SMAP / NAB
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from data_loader import load_dataset
from time2image import get_windows, time_series_to_image, image_to_dinov2_tensor
from time2image import WINDOW_SIZE, STEP
from feature_extractor import _get_model, DEVICE
from experiment_clsg import build_channel_map as build_msl_channel_map, _to_1d
from run_smap_nab import build_smap_channel_map, build_nab_channel_map
from eval_vit4ts import f1max_vit4ts, get_intervals

K           = 5
BATCH_SIZE  = 32
PATCH_BATCH = 512


# ════════════════════════════════════════════════════════
# Patch 토큰 추출 (line plot 이미지)
# ════════════════════════════════════════════════════════

def _extract_lineplt_patches(
    data:    np.ndarray,
    feat_idx: int,
    save_dir: Path,
    channel:  str,
    split:    str,
) -> np.ndarray:
    """line plot 이미지 → DINOv2 patch 토큰 (N_windows, 256, 768)."""
    cache = save_dir / f"lineplt_{channel}_f{feat_idx}_{split}_patches.npy"
    if cache.exists():
        arr = np.load(cache)
        print(f"    캐시: {cache.name}  {arr.shape}")
        return arr

    ts   = _to_1d(data, feat_idx).astype(float)
    lo, hi = ts.min(), ts.max()
    ts   = (ts - lo) / (hi - lo + 1e-8)
    wins = get_windows(ts, window_size=WINDOW_SIZE, step=STEP)
    imgs = [time_series_to_image(w) for w in wins]

    model = _get_model()
    all_patches = []
    with torch.no_grad():
        for s in range(0, len(imgs), BATCH_SIZE):
            batch = torch.stack(
                [image_to_dinov2_tensor(img) for img in imgs[s:s + BATCH_SIZE]]
            ).to(DEVICE)
            patches = model.forward_features(batch)["x_norm_patchtokens"].cpu().numpy()
            all_patches.append(patches)

    arr = np.concatenate(all_patches, axis=0)   # (N, 256, 768)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    print(f"    추출: {cache.name}  {arr.shape}")
    return arr


def _load_lineplt_cls(
    save_dir: Path, channel: str, feat_idx: int, split: str,
) -> np.ndarray:
    """CLS-G 실험과 동일한 캐시 재사용."""
    path = save_dir / f"{channel}_f{feat_idx}_{split}_cls.npy"
    if not path.exists():
        raise FileNotFoundError(f"CLS 캐시 없음: {path.name}\n"
                                "먼저 experiment_clsg.py 를 실행하세요.")
    arr = np.load(path)
    print(f"    CLS 캐시: {path.name}  {arr.shape}")
    return arr


# ════════════════════════════════════════════════════════
# 이상 점수 계산 (overlapping window → ts 매핑)
# ════════════════════════════════════════════════════════

def _win_to_ts(win_scores: np.ndarray, n_ts: int) -> np.ndarray:
    """overlapping 윈도우 점수 → 시계열 길이 (평균)."""
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STEP
        en = min(st + WINDOW_SIZE, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def _all_scores(
    tr_patches: np.ndarray,   # (N_tr, 256, 768)
    te_patches: np.ndarray,   # (N_te, 256, 768)
    tr_cls:     np.ndarray,   # (N_tr, 768)
    te_cls:     np.ndarray,   # (N_te, 768)
) -> dict[str, np.ndarray]:
    N_te, P, D = te_patches.shape

    # 정규화
    tr_flat = tr_patches.reshape(-1, D).astype(np.float32)
    te_flat = te_patches.reshape(-1, D).astype(np.float32)
    tr_n    = tr_flat / (np.linalg.norm(tr_flat, axis=1, keepdims=True) + 1e-8)
    te_n    = te_flat / (np.linalg.norm(te_flat, axis=1, keepdims=True) + 1e-8)

    # gate (linear): 1 - max(0, CosSim(P_j, CLS_test))
    cls_n   = te_cls / (np.linalg.norm(te_cls, axis=1, keepdims=True) + 1e-8)
    cls_rep = np.repeat(cls_n, P, axis=0)
    w       = (te_n * cls_rep).sum(axis=1)
    gate    = 1.0 - np.maximum(0.0, w)          # (N_te*256,)

    # residual: P - (P·CLS)*CLS
    cls_exp  = cls_n[:, None, :]                              # (N_te,1,768)
    dot_te   = (te_patches * cls_exp).sum(axis=-1, keepdims=True)
    te_resid = te_patches - dot_te * cls_exp                  # (N_te,256,768)

    cls_tr_n  = tr_cls / (np.linalg.norm(tr_cls, axis=1, keepdims=True) + 1e-8)
    cls_tr_exp = cls_tr_n[:, None, :]
    dot_tr    = (tr_patches * cls_tr_exp).sum(axis=-1, keepdims=True)
    tr_resid  = tr_patches - dot_tr * cls_tr_exp              # (N_tr,256,768)

    tr_r_flat = tr_resid.reshape(-1, D).astype(np.float32)
    te_r_flat = te_resid.reshape(-1, D).astype(np.float32)
    tr_r_n    = tr_r_flat / (np.linalg.norm(tr_r_flat, axis=1, keepdims=True) + 1e-8)
    te_r_n    = te_r_flat / (np.linalg.norm(te_r_flat, axis=1, keepdims=True) + 1e-8)

    # KNN: raw patch
    tr_t      = torch.tensor(tr_n,   dtype=torch.float32).to(DEVICE)
    tr_r_t    = torch.tensor(tr_r_n, dtype=torch.float32).to(DEVICE)
    knn_raw, knn_res = [], []
    for i in range(0, len(te_n), PATCH_BATCH):
        te_b  = torch.tensor(te_n[i:i + PATCH_BATCH],   dtype=torch.float32).to(DEVICE)
        te_rb = torch.tensor(te_r_n[i:i + PATCH_BATCH], dtype=torch.float32).to(DEVICE)
        dist_r = 1.0 - te_b  @ tr_t.T
        dist_res = 1.0 - te_rb @ tr_r_t.T
        knn_raw.append(torch.topk(dist_r,   K, dim=1, largest=False).values.mean(1).cpu().numpy())
        knn_res.append(torch.topk(dist_res, K, dim=1, largest=False).values.mean(1).cpu().numpy())

    knn_flat  = np.concatenate(knn_raw)    # (N_te*256,)
    knn_r_flat= np.concatenate(knn_res)   # (N_te*256,)

    knn_win   = knn_flat.reshape(N_te, P)
    gate_knn  = (gate * knn_flat).reshape(N_te, P)
    knn_r_win = knn_r_flat.reshape(N_te, P)

    return {
        "knn_max":   knn_win.max(axis=1),
        "knn_sum":   knn_win.sum(axis=1),
        "gate_sum":  gate_knn.sum(axis=1),
        "gate_max":  gate_knn.max(axis=1),
        "resid_sum": knn_r_win.sum(axis=1),
        "resid_max": knn_r_win.max(axis=1),
    }


# ════════════════════════════════════════════════════════
# 채널 단위 실험
# ════════════════════════════════════════════════════════

def run_channel(
    dataset_name: str,
    channel_name: str,
    feat_idx:     int,
    verbose:      bool = True,
) -> dict:
    save_dir = Path(f"results/{dataset_name}")
    t0 = time.time()

    train_data, test_data, labels = load_dataset(
        dataset_name, channel_name, verbose=False
    )
    n_ts = len(test_data)
    n_gt = len(get_intervals(labels.astype(int)))

    tr_patches = _extract_lineplt_patches(train_data, feat_idx, save_dir, channel_name, "train")
    te_patches = _extract_lineplt_patches(test_data,  feat_idx, save_dir, channel_name, "test")
    tr_cls     = _load_lineplt_cls(save_dir, channel_name, feat_idx, "train")
    te_cls     = _load_lineplt_cls(save_dir, channel_name, feat_idx, "test")

    assert tr_patches.shape[0] == tr_cls.shape[0], "train 패치·CLS 윈도우 수 불일치"
    assert te_patches.shape[0] == te_cls.shape[0], "test 패치·CLS 윈도우 수 불일치"

    sc = _all_scores(tr_patches, te_patches, tr_cls, te_cls)

    results = {}
    for key, win_scores in sc.items():
        ts = _win_to_ts(win_scores, n_ts)
        f1, p, r, a = f1max_vit4ts(ts, labels)
        results[key] = dict(f1max=f1, p=p, r=r, alpha=a)

    if verbose:
        print(f"  [{dataset_name}/{channel_name}]  GT={n_gt}  ({time.time()-t0:.1f}s)")
        for key in ("knn_max", "gate_sum", "resid_sum"):
            print(f"    {key:<12}  F1={results[key]['f1max']:.4f}")

    return {"channel": channel_name, "n_gt": n_gt, **results}


# ════════════════════════════════════════════════════════
# 데이터셋 단위 실행
# ════════════════════════════════════════════════════════

def run_dataset(dataset_name: str, channel_map: dict) -> list:
    print(f"\n{'='*70}")
    print(f"Line Plot Patch Memory Bank  —  {dataset_name}  ({len(channel_map)}채널)")
    print(f"  window={WINDOW_SIZE}  step={STEP}  K={K}")
    print(f"{'='*70}")
    results = []
    for chan_id, info in channel_map.items():
        try:
            r = run_channel(dataset_name, chan_id, info["feat_idx"])
            results.append(r)
        except Exception as e:
            print(f"  ERROR {chan_id}: {e}")
    return results


# ════════════════════════════════════════════════════════
# 결과 출력
# ════════════════════════════════════════════════════════

MSL_BASELINES = {
    "CLS Max Fusion":          0.7093,
    "Wavelet w=28 KNN":        0.6894,
    "CLS-G line plot":         0.5455,
    "Wavelet patch gate_sum":  0.5114,
    "Wavelet patch knn_max":   0.4664,
}
SMAP_BASELINES = {
    "CLS-G line plot":         0.9762,
    "Wavelet w=224 KNN":       0.8355,
    "Wavelet patch resid_sum": 0.7880,
    "Cross-Scale Recon":       0.7819,
    "Wavelet patch gate_sum":  0.5980,
    "Wavelet patch knn_max":   0.6356,
}
NAB_BASELINES = {
    "Wavelet w=28 KNN":        0.4845,
    "CLS Max Fusion":          0.4670,
    "Wavelet patch resid_max": 0.4260,
    "Wavelet patch knn_max":   0.3720,
}

_KEYS   = ["knn_max", "knn_sum", "gate_sum", "gate_max", "resid_sum", "resid_max"]
_LABELS = ["LP knn_max", "LP knn_sum", "LP gate_sum", "LP gate_max", "LP resid_sum", "LP resid_max"]


def print_table(dataset_name: str, results: list) -> dict:
    W = 14
    print(f"\n{'='*76}")
    print(f"Line Plot Patch  —  {dataset_name}")
    print(f"{'='*76}")
    print(f"  {'채널':<{W}} {'GT':>3}  "
          f"{'kmax':>7} {'ksum':>7} {'gsum':>7} {'gmax':>7} {'rsum':>7} {'rmax':>7}")
    print(f"  {'─'*70}")
    avgs = {k: [] for k in _KEYS}
    for r in results:
        vals = " ".join(f"{r[k]['f1max']:>7.4f}" for k in _KEYS)
        print(f"  {r['channel']:<{W}} {r['n_gt']:>3}  {vals}")
        for k in _KEYS:
            avgs[k].append(r[k]["f1max"])
    print(f"  {'─'*70}")
    avg_vals = {k: float(np.mean(v)) for k, v in avgs.items()}
    avg_str  = " ".join(f"{avg_vals[k]:>7.4f}" for k in _KEYS)
    print(f"  {'avg':<{W}} {'':3}  {avg_str}")
    return avg_vals


def print_summary(dataset_name: str, avg_vals: dict, baselines: dict) -> None:
    new_items = list(zip(_LABELS, [avg_vals[k] for k in _KEYS]))
    all_items = list(baselines.items()) + new_items
    all_items.sort(key=lambda x: x[1], reverse=True)

    new_set = set(_LABELS)
    print(f"\n{'─'*60}")
    print(f"  {dataset_name}  방식 비교  (ViT4TS F1max 기준)")
    print(f"{'─'*60}")
    print(f"  {'방식':<44} {'avg F1max':>8}")
    print(f"{'─'*60}")
    for name, val in all_items:
        marker = "  ◀" if name in new_set else ""
        print(f"  {name:<44} {val:>8.4f}{marker}")
    print(f"{'─'*60}")


# ════════════════════════════════════════════════════════
# 진입점
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["MSL", "SMAP", "NAB"]

    if "MSL" in datasets:
        res  = run_dataset("MSL", build_msl_channel_map())
        avgs = print_table("MSL", res)
        print_summary("MSL", avgs, MSL_BASELINES)

    if "SMAP" in datasets:
        res  = run_dataset("SMAP", build_smap_channel_map())
        avgs = print_table("SMAP", res)
        print_summary("SMAP", avgs, SMAP_BASELINES)

    if "NAB" in datasets:
        res  = run_dataset("NAB", build_nab_channel_map())
        avgs = print_table("NAB", res)
        print_summary("NAB", avgs, NAB_BASELINES)
