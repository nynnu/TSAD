"""
exp_C_globalnorm.py를 machine-1-2,1-3,1-4,1-5,3-2 5개 entity에 대해 순서대로 반복.
machine-1-1은 이미 완료 (machine-1-1_expC_globalnorm.npz).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/experiments/20주차실험")
import colab_multivariate_v2 as cm  # noqa: E402

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
OUT_DIR = Path(__file__).resolve().parent
ENTITIES = ["machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-3-2"]
WIN = 224
STEP = 224
N_BANK = 60
N_CALIB = 30
IMAGE_SIZE = cm.IMAGE_SIZE

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE:", cm.DEVICE, flush=True)


def ts_to_image_global(window, g_min, g_max):
    rng = g_max - g_min + 1e-8
    normed = np.clip((window - g_min) / rng, 0.0, 1.0)
    n = len(normed)
    xs = (np.arange(n) * (IMAGE_SIZE - 1) / (n - 1)).astype(int)
    ys = IMAGE_SIZE - 1 - (normed * (IMAGE_SIZE - 5) + 2).astype(int)
    ys = np.clip(ys, 0, IMAGE_SIZE - 1)
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(img)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=2)
    return img


def build_channel_stats(train, c):
    g_min, g_max = float(train[:, c].min()), float(train[:, c].max())
    windows = cm.get_windows(train[:, c], window_size=WIN, step=STEP)
    n = len(windows)
    bank_idx = sorted(set(np.linspace(0, n - 1, min(N_BANK, n)).astype(int).tolist()))
    fine = np.linspace(0, n - 1, min(N_BANK + N_CALIB + 20, n)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]

    bank_imgs = [ts_to_image_global(windows[i], g_min, g_max) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [ts_to_image_global(windows[i], g_min, g_max) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(sc["sum"].mean()), float(sc["sum"].std())
    if sigma < 1e-3:
        return tr_cls, tr_patches, mu, None, g_min, g_max
    return tr_cls, tr_patches, mu, sigma, g_min, g_max


def win_to_ts(win_scores, n_ts, win, step):
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * step
        en = min(st + win, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def score_channel(test, c, tr_cls, tr_patches, mu, sigma, g_min, g_max):
    if sigma is None:
        return np.zeros(len(test))
    windows = cm.get_windows(test[:, c], window_size=WIN, step=STEP)
    imgs = [ts_to_image_global(w, g_min, g_max) for w in windows]
    te_cls, te_patches = cm.extract_dinov2(imgs, multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    win_scores = (sc["sum"] - mu) / sigma
    return win_to_ts(win_scores, len(test), WIN, STEP)


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


def run_entity(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    n_ch = test.shape[1]

    t0 = time.time()
    per_ch_scores = np.zeros((n_ch, len(test)))
    for c in range(n_ch):
        tr_cls, tr_patches, mu, sigma, g_min, g_max = build_channel_stats(train, c)
        per_ch_scores[c] = score_channel(test, c, tr_cls, tr_patches, mu, sigma, g_min, g_max)
    combined = per_ch_scores.max(axis=0)
    np.savez(OUT_DIR / f"{entity}_expC_globalnorm.npz", scores=per_ch_scores, combined=combined, labels=labels)
    p, r, f1 = best_prf(combined, labels)
    dt = time.time() - t0
    print(f"[{entity}] Precision={p:.4f} Recall={r:.4f} F1={f1:.4f} ({dt:.1f}s)", flush=True)
    return p, r, f1


def main():
    results = {}
    for ent in ENTITIES:
        results[ent] = run_entity(ent)
    print("\n=== 요약 (방향C: 전역정규화) ===")
    for ent, (p, r, f1) in results.items():
        print(f"{ent}: P={p:.4f} R={r:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
