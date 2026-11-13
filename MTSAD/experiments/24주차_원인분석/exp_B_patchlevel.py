"""
방향 B: patch 단위(14틱)로 세밀하게 점수 내기. 224틱 창 전체를 하나로 합치지 않고,
시간축 기준 16칸(칸당 14틱)으로 나눠서 각각 점수를 냄 (세로/값축 16개 patch는 max로 압축).
train 뱅크 비교는 기존과 동일 (patch-residual KNN), 집계 방식만 다름.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/experiments/20주차실험")
import colab_multivariate_v2 as cm  # noqa: E402

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
OUT_DIR = Path(__file__).resolve().parent
ENTITY = "machine-1-1"
WIN = 224
STEP = 224
N_BANK = 60
N_CALIB = 30
GRID = 16       # DINOv2 ViT-B/14, 224/14=16
SUBTICK = WIN // GRID  # 14틱씩

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE:", cm.DEVICE, flush=True)


def patch_col_scores(tr_patches, te_patches, tr_cls, te_cls):
    """반환: (N_te, GRID) -- 시간축 16칸 각각의 점수 (값축 16개 patch는 max)."""
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)
    knn_win = sc["knn_win"]  # (N_te, 256)
    N_te = knn_win.shape[0]
    grid2d = knn_win.reshape(N_te, GRID, GRID)  # (N, row=값, col=시간) 가정
    return grid2d.max(axis=1)  # (N, GRID) -- 시간축 칸별 최댓값


def win_to_ts_subtick(col_scores, n_ts, win, step, subtick):
    """col_scores: (N_win, GRID). 각 창의 각 칸(subtick틱)을 해당 위치에 배치."""
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i in range(col_scores.shape[0]):
        base = i * step
        for g in range(col_scores.shape[1]):
            st = base + g * subtick
            en = min(st + subtick, n_ts)
            if st >= n_ts:
                break
            scores[st:en] += col_scores[i, g]
            counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def build_channel_stats(train, c):
    windows = cm.get_windows(train[:, c], window_size=WIN, step=STEP)
    n = len(windows)
    bank_idx = sorted(set(np.linspace(0, n - 1, min(N_BANK, n)).astype(int).tolist()))
    fine = np.linspace(0, n - 1, min(N_BANK + N_CALIB + 20, n)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]

    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    col_sc = patch_col_scores(tr_patches, ca_patches, tr_cls, ca_cls)  # (N_calib, GRID)
    pooled = col_sc.reshape(-1)  # 16칸 다 합쳐서 하나의 분포로 (단순화)
    mu, sigma = float(pooled.mean()), float(pooled.std())
    if sigma < 1e-3:
        return tr_cls, tr_patches, mu, None
    return tr_cls, tr_patches, mu, sigma


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1))); fp = int(np.sum((pred == 1) & (labels == 0))); fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)


def best_prf(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0)
    for thr in np.linspace(lo, hi, n):
        p, r, f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def main():
    train = np.loadtxt(SMD_DIR / "train" / f"{ENTITY}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
    n_ch = test.shape[1]

    t0 = time.time()
    per_ch_scores = np.zeros((n_ch, len(test)))
    for c in range(n_ch):
        tr_cls, tr_patches, mu, sigma = build_channel_stats(train, c)
        if sigma is None:
            continue
        windows = cm.get_windows(test[:, c], window_size=WIN, step=STEP)
        imgs = [cm.ts_to_image_fast(w) for w in windows]
        te_cls, te_patches = cm.extract_dinov2(imgs, multilayer=False)
        col_sc = patch_col_scores(tr_patches, te_patches, tr_cls, te_cls)  # (N_test, GRID)
        z_col = (col_sc - mu) / sigma
        per_ch_scores[c] = win_to_ts_subtick(z_col, len(test), WIN, STEP, SUBTICK)
        print(f"  ch{c} done ({time.time()-t0:.1f}s elapsed)", flush=True)

    combined = per_ch_scores.max(axis=0)
    np.savez(OUT_DIR / f"{ENTITY}_expB_patchlevel.npz", scores=per_ch_scores, combined=combined, labels=labels)
    p, r, f1 = best_prf(combined, labels)
    print(f"\n[방향B: patch단위(14틱)] {ENTITY} Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}  (총 {time.time()-t0:.1f}s)")
    print("(비교: 기존 개별+비중첩(창단위) P=0.2149 R=0.7506 F1=0.3342)")


if __name__ == "__main__":
    main()
