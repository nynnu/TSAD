"""
평평한 채널 점수 부풀림 고치기 두 가지 vs 지금 — spike 쪽만, dt = 보정 점수 상위 0.5%, st = 갑작스러움 상위 1%
  지금: calibrate (채널 중앙값·MAD, test)
  1. 아주 작은 움직임 그림은 점수 없음: 그림(224틱) 안 움직임(최대−최소) < train 범위의 1%면 그 그림 spike 점수 = 없음(NaN)
     (train 상수 채널은 0~1 눈금이라 범위 1.0). 보정 중앙값·MAD도 점수 있는 그림으로만
  2. MAD 하한 1.0: (점수 − 중앙값) / max(MAD, 1.0)
  채점: event P/R/F1, R50/F1(50%), 후보 수, 짧은 spike 놓침, 16틱+ 잡음, 상위 0.5% 중 평평 채널(지금 기준 MAD<1) 몫, 눈으로 보이는데 놓쳤던 15개 중 되찾음
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores

HERE = Path(__file__).resolve().parent
P, W = 14, 224
VIS = HERE / "stage12_check" / "thr_dt05_st1" / "miss_wide_range" / "list.txt"
FLAT = {"machine-3-10 [2875,2951)", "machine-3-2 [15577,15582)", "machine-3-3 [19081,19088)"}
NAMES = ["지금", "1. 작은 움직임 그림 점수 없음", "2. MAD 하한 1.0"]


def calib(raw, floor=0.0):
    out = np.full_like(raw, np.nan); mad = np.full(raw.shape[0], np.nan)
    for c in range(raw.shape[0]):
        v = raw[c, ::P]; v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        med = np.median(v); mad[c] = 1.4826 * np.median(np.abs(v - med))
        out[c] = (raw[c] - med) / max(mad[c], floor, 1e-6)
    return out, mad


def drop_flat(raw, te, tr):
    r = raw.copy(); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
    for w in range(raw.shape[1] // W):
        x = te[w * W:(w + 1) * W]
        small = np.ptp(x, 0) < 0.01 * rg
        r[small, w * W:(w + 1) * W] = np.nan
    return r


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    cnt = [0] * 3; sp = [[0, 0] for _ in range(3)]; lg = [[0, 0] for _ in range(3)]; back = [0] * 3; fshare = [[] for _ in range(3)]
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        S0, mad0 = calib(raw); flat = np.isfinite(mad0) & (mad0 < 1.0)
        for i, S in enumerate([S0, calib(drop_flat(raw, te, tr))[0], calib(raw, 1.0)[0]]):
            fin = np.isfinite(S); top = np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5); p = (top & Gb).any(0)
            fshare[i].append(top[flat].sum() / max(top.sum(), 1))
            cnt[i] = cnt[i] + ev2_counts(y, p)
            for a, b in segs(y):
                t = sp[i] if b - a <= 15 else lg[i]; t[1] += 1; h = p[a:b].any(); t[0] += int(h)
                back[i] += int(f"{e} [{a},{b})" in vis and h)
    L = [f"spike 쪽만, dt 상위 0.5% · st 상위 1% (SMD 28대, GT 327개: 짧은 spike {sp[0][1]} / 16틱+ {lg[0][1]})",
         f"{'방식':26s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ 잡음 | 상위 0.5% 중 평평 채널 몫(평균) | 보이는데 놓친 {len(vis)}개 되찾음"]
    for i, nm in enumerate(NAMES):
        s = ev2_scores(cnt[i])
        L.append(f"{nm:26s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {np.mean(fshare[i]):.0%} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "flat_fix_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
