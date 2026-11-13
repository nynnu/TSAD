"""
ASD test 전체에 spike 점수 + 높이 예측 v3 — SMD의 run_spike_height.py와 같은 그림 방식 (v0)
결과: asd/spike_height_scores/{omi-n}.npz — spike (C, T), h3 (C, T)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from dino_inside import load_model, extract, ts_to_image_global, WIN, GRID  # noqa: E402
from exp_spike_height import col_feat  # noqa: E402
import asd_eval  # noqa: E402

OUT = HERE / "spike_height_scores"; OUT.mkdir(exist_ok=True)
P = WIN // GRID


def main():
    sc0, clf0 = pickle.load(open(HERE.parent / "exp_synthetic_spike" / "direction.pkl", "rb"))
    sc, reg = pickle.load(open(HERE.parent / "exp_spike_height_v3" / "model.pkl", "rb"))
    model = load_model()
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e); T, C = te.shape; nwin = T // WIN
        sp = np.full((C, T), np.nan, np.float32); h3 = np.full((C, T), np.nan, np.float32)
        for c in range(C):
            g0, g1 = float(tr[:, c].min()), float(tr[:, c].max())
            if g1 - g0 < 1e-6:
                if np.ptp(te[:, c]) < 1e-6:
                    continue
                g0, g1 = 0.0, 1.0
            imgs = [ts_to_image_global(te[w * WIN:(w + 1) * WIN, c], min(g0, te[w * WIN:(w + 1) * WIN, c].min()), max(g1, te[w * WIN:(w + 1) * WIN, c].max())) for w in range(nwin)]
            for s0 in range(0, nwin, 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for j, Vw in enumerate(pt):
                    w = s0 + j
                    sp[c, w * WIN:(w + 1) * WIN] = np.repeat(clf0.decision_function(sc0.transform(Vw)).reshape(GRID, GRID).max(0), P)
                    h3[c, w * WIN:(w + 1) * WIN] = np.repeat(reg.predict(sc.transform(col_feat(Vw))), P)
        np.savez(OUT / f"{e}.npz", spike=sp, h3=h3); print(e, "완료", flush=True)


if __name__ == "__main__":
    main()
