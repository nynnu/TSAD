"""
ASD: spike 탐지기 관련 3가지 (SMD로 학습한 spike 방향을 다시 학습 없이 그대로 사용)
  1차 탈주기 점수 (24주차 exp_J와 같은 계산) → deseason_scores/
  spike 점수 (원본 224틱 비중첩 그래프 → DINOv2 → spike 방향 → 열 최댓값) → spike_scores/
  결합: 탈주기 OR (spike > dt AND z > zt) / + 채널별 보정 / + 보정 + 갑작스러움(z 대신), (dt, zt/st) 서버당 한 쌍 오라클
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "24주차_원인분석"))
from asd_eval import ENTS, load, best_pred, score, show  # noqa: E402
from exp_J_phase_deseason import channel_score  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_fix_variants import search  # noqa: E402

DES = HERE / "deseason_scores"; SPK = HERE / "spike_scores"
DES.mkdir(exist_ok=True); SPK.mkdir(exist_ok=True)
WIN, GRID, P = 224, 16, 14


def make_scores():
    from dino_inside import load_model, extract, ts_to_image_global
    with open(HERE.parent / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    model = None
    for e in ENTS:
        tr, te, y = load(e)
        if not (DES / f"{e}.npz").exists():
            ch = np.stack([channel_score(tr[:, c], te[:, c]) for c in range(te.shape[1])])
            np.savez(DES / f"{e}.npz", scores=ch, combined=ch.max(0))
        if (SPK / f"{e}.npz").exists():
            continue
        model = model or load_model()
        T, C = te.shape; nw = T // WIN
        S = np.full((C, T), np.nan, dtype=np.float32)
        for c in range(C):
            lo, hi = float(tr[:, c].min()), float(tr[:, c].max())
            if hi - lo < 1e-6:
                continue
            imgs = [ts_to_image_global(te[w * WIN:(w + 1) * WIN, c], lo, hi) for w in range(nw)]
            _, pt, _ = extract(model, imgs)
            for w, Vw in enumerate(pt):
                col = clf.decision_function(sc.transform(Vw)).reshape(GRID, GRID).max(0)
                S[c, w * WIN:(w + 1) * WIN] = np.repeat(col, P)
        np.savez(SPK / f"{e}.npz", scores=S)
        print(e, "spike 점수 완료", flush=True)


def main():
    make_scores()
    names = ["탈주기 단독", "+ spike (보정 없음)", "+ spike + 채널별 보정", "+ spike + 보정 + 갑작스러움"]
    preds = {k: {} for k in names}
    for e in ENTS:
        tr, te, y = load(e)
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(DES / f"{e}.npz")["combined"], y)
        S = np.load(SPK / f"{e}.npz")["scores"].astype(float)
        Sc = calibrate(S)
        sud = np.nan_to_num(suddenness(te, tr), nan=0.0)
        preds[names[0]][e] = p1
        preds[names[1]][e] = search(p1, S, Z, y)
        preds[names[2]][e] = search(p1, Sc, Z, y)
        preds[names[3]][e] = search(p1, Sc, sud, y)
    print("ASD 12대 — spike 탐지기는 SMD로 학습한 것 그대로")
    for i, k in enumerate(names):
        show(k, score(preds[k]), header=(i == 0))


if __name__ == "__main__":
    main()
