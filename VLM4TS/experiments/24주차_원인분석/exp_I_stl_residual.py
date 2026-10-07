"""
방향 I: STL 분해 후 잔차(residual)의 이동표준편차를 채널별 점수로,
38채널 max로 combined. 채널마다 FFT로 주기 추정 후 STL(period=그 값).
"""
import time
from pathlib import Path

import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.ndimage import uniform_filter1d

SMD_DIR = Path(__file__).resolve().parents[2] / "mv_data" / "SMD"
OUT_DIR = Path(__file__).resolve().parent
ENTITIES = ["machine-2-2", "machine-2-9", "machine-3-1", "machine-3-2", "machine-3-8"]
ROLL_WIN = 60


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0)


def best_prf(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0)
    for thr in np.linspace(lo, hi, n):
        p, r, f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def estimate_period(x, lo_bound=5, hi_bound=1000):
    xc = x - x.mean()
    fft = np.abs(np.fft.rfft(xc))
    freqs = np.fft.rfftfreq(len(xc))
    mask = (freqs > 1 / hi_bound) & (freqs < 1 / lo_bound)
    if not mask.any():
        return 100
    peak_freq = freqs[mask][np.argmax(fft[mask])]
    return int(np.clip(round(1 / peak_freq), lo_bound, hi_bound))


def channel_resid_score(train_c, test_c):
    if train_c.std() < 1e-6:
        return np.zeros(len(test_c))
    period = estimate_period(train_c)
    full = np.concatenate([train_c, test_c])
    try:
        res = STL(full, period=period, robust=True).fit()
    except Exception:
        return np.zeros(len(test_c))
    resid = res.resid[len(train_c):]
    roll_std = uniform_filter1d(np.abs(resid), size=ROLL_WIN)
    train_roll_std = uniform_filter1d(np.abs(res.resid[:len(train_c)]), size=ROLL_WIN)
    mu, sigma = train_roll_std.mean(), train_roll_std.std()
    sigma = max(sigma, 1e-6)
    return (roll_std - mu) / sigma


def main():
    results = {}
    for e in ENTITIES:
        t0 = time.time()
        train = np.loadtxt(SMD_DIR / "train" / f"{e}.txt", delimiter=",")
        test = np.loadtxt(SMD_DIR / "test" / f"{e}.txt", delimiter=",")
        labels = np.loadtxt(SMD_DIR / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        n_ch = test.shape[1]

        per_ch = np.zeros((n_ch, len(test)))
        for c in range(n_ch):
            per_ch[c] = channel_resid_score(train[:, c], test[:, c])
        combined = per_ch.max(axis=0)
        np.savez(OUT_DIR / f"{e}_stl_resid.npz", scores=per_ch, combined=combined, labels=labels)

        p, r, f1 = best_prf(combined, labels)
        results[e] = f1
        print(f"[{e}] STL잔차 P={p:.4f} R={r:.4f} F1={f1:.4f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n평균 F1 (STL 잔차 기반): {np.mean(list(results.values())):.4f}")
    print("(참고) 원래 z-score=0.4149, DINOv2=0.3991, z+dino 앙상블=0.4540")


if __name__ == "__main__":
    main()
