"""
Stage1을 통해서 GT 위치 찾기 -- 재설계판:
  1. window을 겹치지 않게 이동 (step=224, 기존 56에서 변경) -> test 창 개수 421->106개, 4배 감소
  2. 38채널 전부 "개별로" DINOv2에 넘김 (그룹으로 묶지 않음) -> group7류 오귀속 문제 제거
  3. multilayer=False (기존 "20주차_나연_stage1_1"/"24주차_나연_PatchKNN-SubplotHeatmap_1"과
     동일한 단일 last-layer residual-KNN-sum 방식)

순비용: 38채널 x 106창 = 4,028  vs  기존 18스코어러 x 421창 = 7,578 (약 47% 감소 예상)

DINOv2 재연산 필요(기존 그룹 캐시 재사용 불가) -- machine-1-1 하나만 우선 실행.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/experiments/20주차실험")
import colab_multivariate_v2 as cm  # noqa: E402

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
OUT_DIR = Path(__file__).resolve().parent / "results_stage1_individual_nonoverlap"
OUT_DIR.mkdir(exist_ok=True)

WIN = 224
STEP = 224  # 겹치지 않게
N_BANK = 60
N_CALIB = 30
ENTITY = "machine-1-1"

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE:", cm.DEVICE)


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
    sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(sc["sum"].mean()), float(sc["sum"].std())
    if sigma < 1e-3:
        # 채널이 거의 항상 상수라 정상분포 표준편차가 0에 가까움 (예: ch26,28) --
        # 이대로 나누면 미세한 부동소수점 차이가 수십억짜리 z로 폭발함.
        # 이런 채널은 이 스코어 방식으로는 판단 불가 -> 항상 0(정보 없음)으로 처리.
        return tr_cls, tr_patches, mu, None
    return tr_cls, tr_patches, mu, sigma


def win_to_ts(win_scores, n_ts):
    """cm.win_to_ts는 자기 모듈의 WINDOW_SIZE=224/STEP=56 상수를 그대로 써서
    이 스크립트의 STEP=224(비중첩)와 안 맞음 -- 로컬 버전으로 직접 구현."""
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STEP
        en = min(st + WIN, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def score_channel(test, c, tr_cls, tr_patches, mu, sigma):
    if sigma is None:
        return np.zeros(len(test))
    windows = cm.get_windows(test[:, c], window_size=WIN, step=STEP)
    imgs = [cm.ts_to_image_fast(w) for w in windows]
    te_cls, te_patches = cm.extract_dinov2(imgs, multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    win_scores = (sc["sum"] - mu) / sigma
    return win_to_ts(win_scores, len(test))


def main():
    train = np.loadtxt(SMD_DIR / "train" / f"{ENTITY}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
    n_ch = test.shape[1]

    t0 = time.time()
    per_ch_scores = np.zeros((n_ch, len(test)))
    for c in range(n_ch):
        tc = time.time()
        tr_cls, tr_patches, mu, sigma = build_channel_stats(train, c)
        per_ch_scores[c] = score_channel(test, c, tr_cls, tr_patches, mu, sigma)
        print(f"  ch{c} done ({time.time()-tc:.1f}s)", flush=True)

    combined = per_ch_scores.max(axis=0)
    np.savez(OUT_DIR / f"{ENTITY}_per_channel.npz", scores=per_ch_scores, combined=combined, labels=labels)
    print(f"\nTotal time: {time.time()-t0:.1f}s")

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

    p, r, f1 = best_prf(combined, labels)
    print(f"\n[{ENTITY}] 개별+비중첩 Stage1: Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print("(비교: 기존 그룹방식 Precision=0.2464 Recall=0.5891 F1=0.3475, TimeRCD F1=0.2194)")


if __name__ == "__main__":
    main()
