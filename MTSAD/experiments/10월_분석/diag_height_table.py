"""
통제 실험 표 (나연 주간 리포트, 2026-10-22): 같은 그림에서 높이만 바꿨을 때 예전 spike 점수 vs 높이 예측 v3
  exp_spike_height.py (a)와 같은 그림: 바닥 0.05 + 잔잡음, 100틱에 3틱 spike, 눈금 0~1 (넘으면 넓힘), spike 열(7)
"""
import pickle
from pathlib import Path

import numpy as np

from dino_inside import load_model, extract, ts_to_image_global, GRID
from exp_spike_height import col_feat

HERE = Path(__file__).resolve().parent


def main():
    model = load_model(); rng = np.random.default_rng(0); H = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 4.0]; imgs = []
    for h in H:
        x = 0.05 + 0.005 * rng.standard_normal(224); x[99:102] += h; imgs.append(ts_to_image_global(x, min(0, x.min()), max(1, x.max())))
    so, co = pickle.load(open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb"))[:2]
    s3, r3 = pickle.load(open(HERE / "exp_spike_height_v3" / "model.pkl", "rb"))
    _, pt, _ = extract(model, imgs); L = ["높이 | 예전 spike 점수 | 높이 예측 v3"]
    for h, Vw in zip(H, pt):
        o = co.decision_function(so.transform(Vw)).reshape(GRID, GRID).max(0)[7]; hp = r3.predict(s3.transform(col_feat(Vw)))[7]
        L.append(f"{h:4} | {o:6.1f} | {hp:5.2f}")
    txt = "\n".join(L); print(txt); (HERE / "coverage_groups" / "height_table.txt").write_text(txt)


if __name__ == "__main__":
    main()
