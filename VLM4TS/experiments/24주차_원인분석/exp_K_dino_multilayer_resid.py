"""
방향 K: 탈주기(phase-deseason) 처리한 잔차를 이미지로 그려서 DINOv2에 넣되,
마지막 레이어 대신 8/11번째 레이어 patch 임베딩 합(ML_LAYERS)으로 KNN 점수 계산.
주기성 없는 채널은 원본 신호(전역정규화)를 그대로 씀.
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
ENTITIES = ["machine-2-2", "machine-2-9", "machine-3-1", "machine-3-2", "machine-3-8"]
WIN = 224
STEP = 224
N_BANK = 60
N_CALIB = 30
IMAGE_SIZE = cm.IMAGE_SIZE

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE:", cm.DEVICE, flush=True)


def estimate_period(x, lo_bound=5, hi_bound=1000):
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return None
    fft = np.abs(np.fft.rfft(xc))
    freqs = np.fft.rfftfreq(len(xc))
    mask = (freqs > 1 / hi_bound) & (freqs < 1 / lo_bound)
    if not mask.any():
        return None
    peak_freq = freqs[mask][np.argmax(fft[mask])]
    power_at_peak = fft[mask].max()
    if power_at_peak < 3 * np.median(fft[mask]):
        return None
    return int(np.clip(round(1 / peak_freq), lo_bound, hi_bound))


def deseasonalize(train_c, test_c):
    """주기 있으면 (train_resid, test_resid, True), 없으면 (train_c, test_c, False)."""
    P = estimate_period(train_c)
    if P is None:
        return train_c, test_c, False
    phase_profile = np.array([train_c[ph::P].mean() for ph in range(P)])
    resid_train = train_c - phase_profile[np.arange(len(train_c)) % P]
    resid_test = test_c - phase_profile[np.arange(len(test_c)) % P]
    return resid_train, resid_test, True


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


def build_channel_stats(train_c):
    g_min, g_max = float(train_c.min()), float(train_c.max())
    windows = cm.get_windows(train_c, window_size=WIN, step=STEP)
    n = len(windows)
    bank_idx = sorted(set(np.linspace(0, n - 1, min(N_BANK, n)).astype(int).tolist()))
    fine = np.linspace(0, n - 1, min(N_BANK + N_CALIB + 20, n)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]

    bank_imgs = [ts_to_image_global(windows[i], g_min, g_max) for i in bank_idx]
    tr_cls, tr_patches, tr_ml = cm.extract_dinov2(bank_imgs, multilayer=True)

    calib_imgs = [ts_to_image_global(windows[i], g_min, g_max) for i in calib_idx]
    ca_cls, ca_patches, ca_ml = cm.extract_dinov2(calib_imgs, multilayer=True)
    sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls, use_ml_tr=tr_ml, use_ml_te=ca_ml)
    mu, sigma = float(sc["sum"].mean()), float(sc["sum"].std())
    if sigma < 1e-3:
        return tr_cls, tr_patches, tr_ml, mu, None, g_min, g_max
    return tr_cls, tr_patches, tr_ml, mu, sigma, g_min, g_max


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


def score_channel(test_c, tr_cls, tr_patches, tr_ml, mu, sigma, g_min, g_max):
    if sigma is None:
        return np.zeros(len(test_c))
    windows = cm.get_windows(test_c, window_size=WIN, step=STEP)
    imgs = [ts_to_image_global(w, g_min, g_max) for w in windows]
    te_cls, te_patches, te_ml = cm.extract_dinov2(imgs, multilayer=True)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=tr_ml, use_ml_te=te_ml)
    win_scores = (sc["sum"] - mu) / sigma
    return win_to_ts(win_scores, len(test_c), WIN, STEP)


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
    n_periodic = 0
    for c in range(n_ch):
        tr_c, te_c, is_periodic = deseasonalize(train[:, c], test[:, c])
        n_periodic += int(is_periodic)
        tr_cls, tr_patches, tr_ml, mu, sigma, g_min, g_max = build_channel_stats(tr_c)
        per_ch_scores[c] = score_channel(te_c, tr_cls, tr_patches, tr_ml, mu, sigma, g_min, g_max)
    combined = per_ch_scores.max(axis=0)
    np.savez(OUT_DIR / f"{entity}_dino_ml_resid.npz", scores=per_ch_scores, combined=combined, labels=labels)
    p, r, f1 = best_prf(combined, labels)
    dt = time.time() - t0
    print(f"[{entity}] 주기채널={n_periodic}/{n_ch} P={p:.4f} R={r:.4f} F1={f1:.4f} ({dt:.1f}s)", flush=True)
    return p, r, f1


def main():
    results = {}
    for ent in ENTITIES:
        results[ent] = run_entity(ent)
    print("\n=== 요약 (방향K: 탈주기잔차 + DINOv2 L8+L11) ===")
    for ent, (p, r, f1) in results.items():
        print(f"{ent}: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    print(f"평균 F1: {np.mean([f1 for _,_,f1 in results.values()]):.4f}")


if __name__ == "__main__":
    main()
