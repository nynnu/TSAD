"""
평평한 채널 부풀림 마무리: 1번(작은 움직임 그림 점수 없음) + 눈금 넓히기(train 범위 넘는 그림 다시 그림, spike_scores_rescale)
  spike 쪽만, dt 상위 0.5% · st 상위 1%. 비교: 지금 / 1번 / 눈금 넓힘 / 1번 + 눈금 넓힘
  추가: 남은 MAD<1 채널 수와 상위 0.5% 몫
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat, VIS, FLAT

HERE = Path(__file__).resolve().parent
CASES = [("지금", "spike_scores", False), ("1번", "spike_scores", True), ("눈금 넓힘", "spike_scores_rescale", False), ("1번 + 눈금 넓힘", "spike_scores_rescale", True)]


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    n = len(CASES); cnt = [0] * n; sp = [[0, 0] for _ in range(n)]; lg = [[0, 0] for _ in range(n)]; back = [0] * n; fsh = [[] for _ in range(n)]; nsmall = [0] * n
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        T = min(len(y), te.shape[0]); G = None
        for i, (_, d, drop) in enumerate(CASES):
            raw = np.load(HERE / d / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); yy, raw, tt = y[:T], raw[:, :T], te[:T]
            if G is None:
                G = np.nan_to_num(suddenness(tt, tr), nan=0.0); Gb = G > np.percentile(G, 99)
            S, mad = calib(drop_flat(raw, tt, tr) if drop else raw)
            small = np.isfinite(mad) & (mad < 1.0); nsmall[i] += small.sum()
            fin = np.isfinite(S); top = np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5); p = (top & Gb).any(0)
            fsh[i].append(top[small].sum() / max(top.sum(), 1)); cnt[i] = cnt[i] + ev2_counts(yy, p)
            for a, b in segs(yy):
                t = sp[i] if b - a <= 15 else lg[i]; t[1] += 1; h = p[a:b].any(); t[0] += int(h); back[i] += int(f"{e} [{a},{b})" in vis and h)
    L = [f"spike 쪽만, dt 상위 0.5% · st 상위 1% (SMD 28대)",
         f"{'방식':16s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ | MAD<1 채널 수 | 그 채널들 상위 0.5% 몫 | 보이는데 놓친 15개 되찾음"]
    for i, (nm, _, _) in enumerate(CASES):
        s = ev2_scores(cnt[i])
        L.append(f"{nm:16s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {nsmall[i]:3d} | {np.mean(fsh[i]):.0%} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "flat_rescale_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
