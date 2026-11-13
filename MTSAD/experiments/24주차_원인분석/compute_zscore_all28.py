"""
순수 z-score(모델 없음) 기반 combined score를 SMD 28개 entity 전체에 대해 계산해서 캐시.
combined_z[t] = max_c( |test[t,c] - train_mean[c]| / train_std[c] )
"""
from pathlib import Path

import numpy as np

SMD_DIR = Path(__file__).resolve().parents[2] / "mv_data" / "SMD"
OUT_DIR = Path(__file__).resolve().parent / "zscore_scores"
OUT_DIR.mkdir(exist_ok=True)

ENTITIES = [
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5",
    "machine-1-6", "machine-1-7", "machine-1-8",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5",
    "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", "machine-3-5",
    "machine-3-6", "machine-3-7", "machine-3-8", "machine-3-9", "machine-3-10", "machine-3-11",
]


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


def main():
    results = {}
    for e in ENTITIES:
        train = np.loadtxt(SMD_DIR / "train" / f"{e}.txt", delimiter=",")
        test = np.loadtxt(SMD_DIR / "test" / f"{e}.txt", delimiter=",")
        labels = np.loadtxt(SMD_DIR / "test_label" / f"{e}.txt", delimiter=",").astype(int)

        mu, sigma = train.mean(axis=0), train.std(axis=0)
        sigma[sigma < 1e-3] = 1e-3
        z_per_ch = np.abs((test - mu) / sigma)  # (T, 38)
        combined_z = z_per_ch.max(axis=1)

        np.savez(OUT_DIR / f"{e}.npz", combined_z=combined_z, labels=labels)
        p, r, f1 = best_prf(combined_z, labels)
        results[e] = f1
        print(f"[{e}] T={len(test)} z-score(max) P={p:.4f} R={r:.4f} F1={f1:.4f}", flush=True)

    print(f"\n평균 F1 (28개 entity, 순수 z-score): {np.mean(list(results.values())):.4f}")


if __name__ == "__main__":
    main()
