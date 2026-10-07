"""
가짜 spike로 배운 "spike 방향" 탐지기를 SMD 28개 entity test 전체에 돌림 (zero-shot: GT 라벨 안 씀)
  그래프(224틱, 비중첩, train min/max 정규화) → DINOv2 → patch 256 × 768 → spike 방향 점수
  → 열(14틱)마다 16행 중 최고 점수 → 틱별 점수 → 38채널 중 최고 = 그 시점 점수
결과: spike_scores/{entity}.npz (scores: (38, T) 채널별, combined: (T,))
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spike_scores"
OUT.mkdir(exist_ok=True)
P = 224 // GRID


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = load_model()
    ents = sorted(p.stem for p in (SMD / "test").glob("*.txt"))
    for e in ents:
        if (OUT / f"{e}.npz").exists():
            continue
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T, C = te.shape
        nwin = T // WIN
        scores = np.full((C, T), np.nan, dtype=np.float32)
        for ch in range(C):
            g_min, g_max = float(tr[:, ch].min()), float(tr[:, ch].max())
            if g_max - g_min < 1e-6:                                    # train에서 상수인 채널은 그림이 의미 없음
                continue
            imgs = [ts_to_image_global(te[w * WIN:(w + 1) * WIN, ch], g_min, g_max) for w in range(nwin)]
            for s0 in range(0, nwin, 64):
                _, pt, _ = extract(model, imgs[s0:s0 + 64])
                for j, Vw in enumerate(pt):
                    col = clf.decision_function(sc.transform(Vw)).reshape(GRID, GRID).max(0)   # 열마다 최고
                    w = s0 + j
                    scores[ch, w * WIN:(w + 1) * WIN] = np.repeat(col, P)
        combined = np.nanmax(np.where(np.isnan(scores), -np.inf, scores), axis=0)
        np.savez(OUT / f"{e}.npz", scores=scores, combined=combined)
        print(f"{e}: T={T}, 그래프 {nwin * C}장 완료", flush=True)


if __name__ == "__main__":
    main()
