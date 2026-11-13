"""
test 전체에 spike 점수(예전 direction.pkl) + 높이 예측 세 버전(v1 exp_spike_height, v2, v3)을 한 번에 — 그림은 지금 기본 방식 그대로
  눈금: train 최솟값~최댓값 (그림 안에 넘는 값이 있으면 그 그림 최솟값·최댓값까지 넓힘), train 상수인데 test에서 움직이는 채널은 0~1, 둘 다 상수면 건너뜀
  열(14틱)마다: spike = 16행 patch 점수 최댓값 (예전과 같음), height = 열 특징 [최댓값, 평균] → 릿지 높이 예측
결과: spike_height_scores/{entity}.npz — spike (C, T), h1 / h2 / h3 (C, T)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402
from exp_spike_height import col_feat  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spike_height_scores"; OUT.mkdir(exist_ok=True)
P = WIN // GRID


def main():
    sc0, clf0 = pickle.load(open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb"))
    a, _, b = pickle.load(open(HERE / "exp_spike_height" / "model.pkl", "rb"))
    heads = [(a, b)] + [pickle.load(open(HERE / f"exp_spike_height_{v}" / "model.pkl", "rb")) for v in ["v2", "v3"]]
    model = load_model()
    for e in sorted(p.stem for p in (SMD / "test").glob("*.txt")):
        if (OUT / f"{e}.npz").exists():
            continue
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T, C = te.shape; nwin = T // WIN
        sp = np.full((C, T), np.nan, np.float32); ht = [np.full((C, T), np.nan, np.float32) for _ in heads]
        for c in range(C):
            g0, g1 = float(tr[:, c].min()), float(tr[:, c].max())
            if g1 - g0 < 1e-6:
                if np.ptp(te[:, c]) < 1e-6:
                    continue
                g0, g1 = 0.0, 1.0
            imgs = []
            for w in range(nwin):
                x = te[w * WIN:(w + 1) * WIN, c]; imgs.append(ts_to_image_global(x, min(g0, x.min()), max(g1, x.max())))
            for s0 in range(0, nwin, 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for j, Vw in enumerate(pt):
                    w = s0 + j
                    sp[c, w * WIN:(w + 1) * WIN] = np.repeat(clf0.decision_function(sc0.transform(Vw)).reshape(GRID, GRID).max(0), P)
                    F = col_feat(Vw)
                    for k, (sc, reg) in enumerate(heads):
                        ht[k][c, w * WIN:(w + 1) * WIN] = np.repeat(reg.predict(sc.transform(F)), P)
        np.savez(OUT / f"{e}.npz", spike=sp, h1=ht[0], h2=ht[1], h3=ht[2])
        print(f"{e} 완료", flush=True)


if __name__ == "__main__":
    main()
