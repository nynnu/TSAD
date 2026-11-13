"""
POT용: spike 탐지기 점수를 train 전체에 (run_spike_detector.py와 같은 방식: 224틱 비중첩, train min/max 눈금, 열마다 16행 최고)
  train 상수 채널은 건너뜀 (그 채널 기준은 test 점수로 POT)
결과: spike_scores_train/{entity}.npz — cols (C, nwin*16) 열 점수, 상수 채널은 NaN
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spike_scores_train"
OUT.mkdir(exist_ok=True)


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = load_model()
    for e in sorted(p.stem for p in (SMD / "test").glob("*.txt")):
        if (OUT / f"{e}.npz").exists():
            continue
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        nwin = len(tr) // WIN; C = tr.shape[1]
        cols = np.full((C, nwin * GRID), np.nan, dtype=np.float32)
        for ch in range(C):
            g0, g1 = float(tr[:, ch].min()), float(tr[:, ch].max())
            if g1 - g0 < 1e-6:
                continue
            imgs = [ts_to_image_global(tr[w * WIN:(w + 1) * WIN, ch], g0, g1) for w in range(nwin)]
            for s0 in range(0, nwin, 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for j, Vw in enumerate(pt):
                    w = s0 + j
                    cols[ch, w * GRID:(w + 1) * GRID] = clf.decision_function(sc.transform(Vw)).reshape(GRID, GRID).max(0)
        np.savez(OUT / f"{e}.npz", cols=cols)
        print(f"{e}: train 창 {nwin}개 × {C}채널 완료", flush=True)


if __name__ == "__main__":
    main()
