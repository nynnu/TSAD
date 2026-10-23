"""
test 보정 + train 보정 같이 쓰기 (나연 2026-10-20) — 기준선: 1번 + 눈금 넓힘, spike 쪽만, dt = 새 보정 점수 상위 0.5%, st = 갑작 상위 1%
  test 보정 = (점수 − test 중앙값) / test MAD,  train 보정 = (점수 − train 중앙값) / train MAD
  train 중앙값·MAD = spike_scores_train 열 점수 (움직임 1% 미만 train 그림 제외, eval_trainp.train_ref와 같음)
  A: min(test 보정, train 보정)   B: (점수 − test 중앙값) / max(test MAD, train MAD)   C: 평균
  train 기준 없는 채널(train 상수 등)은 test 보정 그대로
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import drop_flat, VIS, FLAT
from eval_trainp import train_ref, NOISE

HERE = Path(__file__).resolve().parent
P = 14
NAMES = ["지금 (test 보정)", "train 보정만", "A min", "B 흔들림 큰 쪽", "C 평균"]


def stats(v):
    med = np.median(v); return med, max(1.4826 * np.median(np.abs(v - med)), 1e-6)


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    n = len(NAMES); cnt = [0] * n; sp = [[0, 0] for _ in range(n)]; lg = [[0, 0] for _ in range(n)]; back = [0] * n; nfp = [0] * n
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        raw = drop_flat(raw, te, tr); ref = train_ref(e, tr); C = raw.shape[0]
        zte = np.full_like(raw, np.nan); zs = [np.full_like(raw, np.nan) for _ in range(4)]
        for c in range(C):
            v = raw[c, ::P]; v = v[np.isfinite(v)]
            if len(v) < 10:
                continue
            mt, dtt = stats(v); a = (raw[c] - mt) / dtt; zte[c] = a
            if ref[c] is None:
                for z in zs:
                    z[c] = a
                continue
            mr, dr = stats(ref[c]); b = (raw[c] - mr) / dr
            zs[0][c] = b; zs[1][c] = np.minimum(a, b); zs[2][c] = (raw[c] - mt) / max(dtt, dr); zs[3][c] = (a + b) / 2
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        for i, S in enumerate([zte] + zs):
            fin = np.isfinite(S); m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & Gb; pr = m.any(0)
            cnt[i] = cnt[i] + ev2_counts(y, pr)
            for s, b in segs(y):
                t = sp[i] if b - s <= 15 else lg[i]; t[1] += 1; h = pr[s:b].any(); t[0] += int(h); back[i] += int(f"{e} [{s},{b})" in vis and h)
            for e2, c in NOISE:
                if e2 == e:
                    nfp[i] += sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
    L = ["spike 쪽만 (1번 + 눈금 넓힘), 새 보정 점수 > 그 점수 상위 0.5% AND 갑작 상위 1% (SMD 28대)",
         f"{'방식':16s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ | 잡음 채널 5개 오탐 | 보이는데 놓친 {len(vis)}개 되찾음"]
    for i, nm in enumerate(NAMES):
        s = ev2_scores(cnt[i])
        L.append(f"{nm:16s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {nfp[i]:3d} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "calib_trte_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
