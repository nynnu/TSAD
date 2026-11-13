"""
방향 J: STL 대신, 채널별 FFT로 주기(P) 추정 -> phase(=t%P)별 train 평균 프로파일을 빼는
간단한 탈주기(deseasonalize) -> 잔차의 이동표준편차를 채널 점수로, 38채널 max.
STL과 같은 아이디어지만 반복적 스무딩 없이 평균만 내서 훨씬 빠름(entity당 1초 미만).
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

SMD_DIR = Path(__file__).resolve().parents[2] / "mv_data" / "SMD"
OUT_DIR = Path(__file__).resolve().parent
ENTITIES = [
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5",
    "machine-1-6", "machine-1-7", "machine-1-8",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5",
    "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", "machine-3-5",
    "machine-3-6", "machine-3-7", "machine-3-8", "machine-3-9", "machine-3-10", "machine-3-11",
]
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
        return None  # 뚜렷한 주기성이 없으면 스킵
    return int(np.clip(round(1 / peak_freq), lo_bound, hi_bound))


def channel_score(train_c, test_c):
    if train_c.std() < 1e-6:
        return np.zeros(len(test_c))
    P = estimate_period(train_c)
    if P is None:
        return np.zeros(len(test_c))  # 주기성 없는 채널은 이 방법으로 점수 안 냄(0)

    phase_profile = np.array([train_c[ph::P].mean() for ph in range(P)])
    resid_train = train_c - phase_profile[np.arange(len(train_c)) % P]
    resid_test = test_c - phase_profile[np.arange(len(test_c)) % P]

    roll_train = uniform_filter1d(np.abs(resid_train), size=ROLL_WIN)
    roll_test = uniform_filter1d(np.abs(resid_test), size=ROLL_WIN)
    mu, sigma = roll_train.mean(), roll_train.std()
    sigma = max(sigma, 1e-6)
    return (roll_test - mu) / sigma


def main():
    results = {}
    for e in ENTITIES:
        train = np.loadtxt(SMD_DIR / "train" / f"{e}.txt", delimiter=",")
        test = np.loadtxt(SMD_DIR / "test" / f"{e}.txt", delimiter=",")
        labels = np.loadtxt(SMD_DIR / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        n_ch = test.shape[1]

        per_ch = np.zeros((n_ch, len(test)))
        for c in range(n_ch):
            per_ch[c] = channel_score(train[:, c], test[:, c])
        combined = per_ch.max(axis=0)
        np.savez(OUT_DIR / f"{e}_phase_deseason.npz", scores=per_ch, combined=combined, labels=labels)

        p, r, f1 = best_prf(combined, labels)
        results[e] = f1
        print(f"[{e}] 탈주기잔차 P={p:.4f} R={r:.4f} F1={f1:.4f}", flush=True)

    print(f"\n평균 F1 (28개 entity, 탈주기 잔차): {np.mean(list(results.values())):.4f}")


if __name__ == "__main__":
    main()
