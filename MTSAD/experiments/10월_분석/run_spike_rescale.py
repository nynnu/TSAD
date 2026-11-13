"""
train 범위를 넘는 값이 있는 그림만 눈금을 넓혀서 다시 그림 (나머지는 spike_scores/ 그대로)
  그림(224틱)마다: 눈금 = [min(train 최솟값, 그 그림 최솟값), max(train 최댓값, 그 그림 최댓값)]  → 넘은 spike가 잘리지 않고 꼭대기에 뾰족하게
  train 상수 채널(0~1 눈금)은 test가 0~1 안이라 해당 없음
결과: spike_scores_rescale/{entity}.npz (scores, combined) + rescaled (C, nwin) 다시 그린 그림 표시
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spike_scores_rescale"
OUT.mkdir(exist_ok=True)
P = WIN // GRID


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = load_model()
    for e in sorted(p.stem for p in (SMD / "test").glob("*.txt")):
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        scores = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].copy()
        nwin = len(te) // WIN; C = te.shape[1]; flag = np.zeros((C, nwin), bool); jobs = []
        for c in range(C):
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            if g1 - g0 < 1e-6:
                continue
            for w in range(nwin):
                x = te[w * WIN:(w + 1) * WIN, c]
                if x.max() > g1 + 1e-9 or x.min() < g0 - 1e-9:
                    flag[c, w] = True; jobs.append((c, w, ts_to_image_global(x, min(g0, x.min()), max(g1, x.max()))))
        for s0 in range(0, len(jobs), 64):
            chunk = jobs[s0:s0 + 64]
            _, pt, _ = extract(model, [j[2] for j in chunk])
            for (c, w, _), Vw in zip(chunk, pt):
                scores[c, w * WIN:(w + 1) * WIN] = np.repeat(clf.decision_function(sc.transform(Vw)).reshape(GRID, GRID).max(0), P)
        np.savez(OUT / f"{e}.npz", scores=scores, combined=np.nanmax(np.where(np.isnan(scores), -np.inf, scores), axis=0), rescaled=flag)
        print(f"{e}: 다시 그림 {len(jobs)}장", flush=True)


if __name__ == "__main__":
    main()
