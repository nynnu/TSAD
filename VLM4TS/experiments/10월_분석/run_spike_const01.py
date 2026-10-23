"""
train 상수 채널을 0~1 눈금(SMD 원래 정규화 값 그대로)으로 그려서 spike 점수 계산
  run_spike_detector.py와 같고, 다른 점: train에서 상수이고 test에서 움직이는 채널만, g_min=0, g_max=1
결과: spike_scores_const01/{entity}.npz (scores: (38, T), 해당 채널 외에는 NaN)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spike_scores_const01"
OUT.mkdir(exist_ok=True)
P = 224 // GRID


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = load_model()
    for e in sorted(p.stem for p in (SMD / "test").glob("*.txt")):
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T, C = te.shape
        nwin = T // WIN
        scores = np.full((C, T), np.nan, dtype=np.float32)
        chs = [c for c in range(C) if np.ptp(tr[:, c]) < 1e-6 and np.ptp(te[:, c]) > 1e-6]
        for ch in chs:
            imgs = [ts_to_image_global(te[w * WIN:(w + 1) * WIN, ch], 0.0, 1.0) for w in range(nwin)]
            for s0 in range(0, nwin, 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for j, Vw in enumerate(pt):
                    col = clf.decision_function(sc.transform(Vw)).reshape(GRID, GRID).max(0)
                    w = s0 + j
                    scores[ch, w * WIN:(w + 1) * WIN] = np.repeat(col, P)
        np.savez(OUT / f"{e}.npz", scores=scores, chs=np.array(chs, dtype=int))
        print(f"{e}: 채널 {chs}", flush=True)


if __name__ == "__main__":
    main()
