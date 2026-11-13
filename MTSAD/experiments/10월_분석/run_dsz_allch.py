"""
가정 9: Dinov2_deseson_col_z를 GT 없이(38채널 전부) 재현
  원래(26주차): GT spike의 원인 채널만 후보 → 채널마다 GT로 임계값 → F1 오르면 채택 (채널 선택에 GT 사용)
  여기: 모든 채널(train에서 상수 아닌 것)에 2차를 돌림. 임계값만 다른 방법들과 같은 오라클.

채널마다 (26주차 설정 그대로):
  - 탈주기 잔차: FFT 주기 P(train) → 위상별 train 평균 프로파일을 뺌. 주기 없는 채널은 잔차 = 원본
  - 잔차를 train 잔차 min/max로 정규화해 224틱 비중첩 그래프로 렌더링
  - DINOv2: train 뱅크 60장 / 캘리브레이션 25장, CLS 방향 제거 후 patch-kNN(k=5, 코사인)
            → 열(14틱)마다 16행 중 최댓값 → 캘리브 열점수로 (x-mu)/sigma, sigma<1e-3이면 채널 제외
  - 숫자 버전(B, 비교용): 같은 잔차·같은 뱅크/캘리브, 14틱 잔차 조각(정규화 값 14개)을 뱅크 조각과 kNN(k=5, 유클리드)
결과: dsz_scores/{entity}.npz — D (C,T) DINOv2 열점수, B (C,T) 숫자 버전 열점수 (NaN = 점수 없음)
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20주차실험"))
sys.path.insert(0, str(HERE.parent / "24주차_원인분석"))
import colab_multivariate_v2 as cm  # noqa: E402
from exp_C_globalnorm import ts_to_image_global  # noqa: E402
from exp_J_phase_deseason import estimate_period, ENTITIES  # noqa: E402

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SMD = HERE.parents[1] / "mv_data" / "SMD"
OUT = HERE / "dsz_scores"
OUT.mkdir(exist_ok=True)
WIN, GRID, P, K, N_BANK, N_CALIB = 224, 16, 14, 5, 60, 25


def residual(train_c, test_c):
    P_ = estimate_period(train_c) if train_c.std() >= 1e-6 else None
    if P_ is None:
        return train_c.copy(), test_c.copy()
    prof = np.array([train_c[ph::P_].mean() for ph in range(P_)])
    return train_c - prof[np.arange(len(train_c)) % P_], test_c - prof[np.arange(len(test_c)) % P_]


def windows(x):
    return [x[i:i + WIN] for i in range(0, len(x) - WIN + 1, WIN)]


def dino_col_knn(tr_cls, tr_p, cls, pt):
    """patch-kNN (CLS 방향 제거, 코사인 k=5) → (N, 16) 열 최댓값"""
    tr = torch.tensor(cm.compute_residuals(tr_p, tr_cls).reshape(-1, tr_p.shape[-1]), dtype=torch.float32, device=cm.DEVICE)
    te = cm.compute_residuals(pt, cls).reshape(-1, pt.shape[-1]).astype(np.float32)
    out = []
    for i in range(0, len(te), 4096):
        b = torch.tensor(te[i:i + 4096], device=cm.DEVICE)
        out.append(torch.topk(1 - b @ tr.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(out).reshape(len(pt), GRID, GRID).max(1)


def num_col_knn(bank_seg, segs):
    """14틱 조각 kNN (유클리드 k=5) → (N, 16)"""
    d = np.sqrt(((segs.reshape(-1, 1, P) - bank_seg[None]) ** 2).sum(-1))
    return np.sort(d, axis=1)[:, :K].mean(1).reshape(-1, GRID)


def to_segs(ws, lo, hi):
    a = np.clip((np.stack(ws) - lo) / (hi - lo + 1e-8), 0, 1)
    return a.reshape(len(ws), GRID, P)


def main():
    for e in ENTITIES:
        if (OUT / f"{e}.npz").exists():
            continue
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T, C = te.shape
        D = np.full((C, T), np.nan, dtype=np.float32); B = np.full((C, T), np.nan, dtype=np.float32)
        for c in range(C):
            if tr[:, c].max() - tr[:, c].min() < 1e-6:
                continue
            rtr, rte = residual(tr[:, c], te[:, c])
            lo, hi = float(rtr.min()), float(rtr.max())
            if hi - lo < 1e-6:
                continue
            tw = windows(rtr); n = len(tw)
            bank_idx = sorted(set(np.linspace(0, n - 1, min(N_BANK, n)).astype(int).tolist()))
            fine = np.linspace(0, n - 1, min(N_BANK + N_CALIB + 20, n)).astype(int)
            calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
            test_w = windows(rte)
            imgs = [ts_to_image_global(w, lo, hi) for w in [tw[i] for i in bank_idx] + [tw[i] for i in calib_idx] + test_w]
            cls, pt = [], []
            for s0 in range(0, len(imgs), 64):
                c_, p_ = cm.extract_dinov2(imgs[s0:s0 + 64], multilayer=False)
                cls.append(c_); pt.append(p_)
            cls, pt = np.concatenate(cls), np.concatenate(pt)
            nb, nc = len(bank_idx), len(calib_idx)
            # DINOv2 열점수
            cal_d = dino_col_knn(cls[:nb], pt[:nb], cls[nb:nb + nc], pt[nb:nb + nc])
            te_d = dino_col_knn(cls[:nb], pt[:nb], cls[nb + nc:], pt[nb + nc:])
            mu, sd = cal_d.mean(), cal_d.std()
            if sd >= 1e-3:
                D[c, :len(test_w) * WIN] = np.repeat((te_d - mu) / sd, P, axis=1).reshape(-1)
            # 숫자 버전 열점수 (같은 뱅크/캘리브)
            bank_seg = to_segs([tw[i] for i in bank_idx], lo, hi).reshape(-1, P)
            cal_b = num_col_knn(bank_seg, to_segs([tw[i] for i in calib_idx], lo, hi))
            te_b = num_col_knn(bank_seg, to_segs(test_w, lo, hi))
            mu, sd = cal_b.mean(), cal_b.std()
            if sd >= 1e-3:
                B[c, :len(test_w) * WIN] = np.repeat((te_b - mu) / sd, P, axis=1).reshape(-1)
        np.savez(OUT / f"{e}.npz", D=D, B=B)
        print(f"{e} 완료 (T={T}, 채널 {C})", flush=True)


if __name__ == "__main__":
    main()
