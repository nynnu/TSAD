"""
방향 A: train 뱅크 대신 "같은 test 신호 안의 다른 창들"을 참조로 쓰는 self-referential
스코어링 (아카이브 zs_resid_testbank 방식). GT 라벨 미사용 확인됨.

224틱 비중첩 창 106개(machine-1-1)를 전부 DINOv2에 넣고, 각 창을 "자기 자신만 빼고
나머지 창들 전체"와 KNN 비교. train 데이터는 전혀 안 씀.
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
K = 5

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE:", cm.DEVICE, flush=True)


def testbank_scores(resid_n):
    """resid_n: (N, P, D). 반환: (N,) sum-score, 자기 자신 창만 제외하고 나머지 전부와 비교."""
    N, P, D = resid_n.shape
    all_r = resid_n.reshape(-1, D).astype(np.float32)
    all_t = torch.tensor(all_r, dtype=torch.float32).to(cm.DEVICE)
    sum_scores = np.zeros(N)
    for i in range(N):
        r_i = torch.tensor(resid_n[i].astype(np.float32)).to(cm.DEVICE)
        dist = 1.0 - r_i @ all_t.T
        dist[:, i * P:(i + 1) * P] = float("inf")
        knn = torch.topk(dist, K, dim=1, largest=False).values.mean(dim=1)
        sum_scores[i] = knn.cpu().numpy().sum()
    return sum_scores


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
    test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
    n_ch = test.shape[1]

    t0 = time.time()
    per_ch_scores = np.zeros((n_ch, len(test)))
    for c in range(n_ch):
        if test[:, c].std() < 1e-6:
            continue
        windows = cm.get_windows(test[:, c], window_size=WIN, step=STEP)
        imgs = [cm.ts_to_image_fast(w) for w in windows]
        cls, patches = cm.extract_dinov2(imgs, multilayer=False)
        resid = cm.compute_residuals(patches, cls)
        win_scores = testbank_scores(resid)
        per_ch_scores[c] = win_to_ts(win_scores, len(test), WIN, STEP)
        print(f"  ch{c} done ({time.time()-t0:.1f}s elapsed)", flush=True)

    combined = per_ch_scores.max(axis=0)
    np.savez(OUT_DIR / f"{ENTITY}_expA_testbank.npz", scores=per_ch_scores, combined=combined, labels=labels)
    p, r, f1 = best_prf(combined, labels)
    print(f"\n[방향A: testbank] {ENTITY} Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}  (총 {time.time()-t0:.1f}s)")
    print("(비교: 기존 개별+비중첩(train뱅크) P=0.2149 R=0.7506 F1=0.3342)")


if __name__ == "__main__":
    main()
