"""
spike 채널별 보정 4가지 변형 vs 지금 — spike 쪽만 (탈주기 없음), dt = 보정 점수 상위 0.5%, st = 갑작스러움 상위 1% (machine 안 전 채널·전 틱)
  지금:      (점수 − 채널 중앙값) / 채널 MAD
  1 삭제:    원점수
  2 빼기만:  점수 − 채널 중앙값
  3 MAD 하한: (점수 − 채널 중앙값) / max(채널 MAD, machine 안 채널 MAD들의 중앙값)
  4 중간:    (점수 − 채널 중앙값) / √(채널 MAD × machine 안 채널 MAD들의 중앙값)
  중앙값·MAD는 지금과 같이 test 점수(14틱마다 1개)에서
  채점: event P / R / F1, R50 / F1(50%), 후보 수, 짧은 spike(≤15틱) 잡음·놓침, 16틱+ 잡음, 눈으로 보이는데 놓쳤던 15개 중 되찾은 수
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores

HERE = Path(__file__).resolve().parent
P = 14
VIS = HERE / "stage12_check" / "thr_dt05_st1" / "miss_wide_range" / "list.txt"
FLAT = {"machine-3-10 [2875,2951)", "machine-3-2 [15577,15582)", "machine-3-3 [19081,19088)"}
NAMES = ["지금 (채널별 보정)", "1. 보정 삭제", "2. 빼기만", "3. MAD 하한", "4. 중간"]


def variants(raw):
    C = raw.shape[0]; med = np.full(C, np.nan); mad = np.full(C, np.nan)
    for c in range(C):
        v = raw[c, ::P]; v = v[np.isfinite(v)]
        if len(v) >= 10:
            med[c] = np.median(v); mad[c] = max(1.4826 * np.median(np.abs(v - med[c])), 1e-6)
    gm = np.nanmedian(mad); m, d = med[:, None], mad[:, None]
    return [(raw - m) / d, raw.copy(), raw - m, (raw - m) / np.maximum(d, gm), (raw - m) / np.sqrt(d * gm)]


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    cnt = [0] * 5; sp = [[0, 0] for _ in range(5)]; lg = [[0, 0] for _ in range(5)]; back = [0] * 5
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        for i, S in enumerate(variants(raw)):
            fin = np.isfinite(S); p = ((np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & Gb).any(0)
            cnt[i] = cnt[i] + ev2_counts(y, p)
            for a, b in segs(y):
                t = sp[i] if b - a <= 15 else lg[i]; t[1] += 1; h = p[a:b].any(); t[0] += int(h)
                if f"{e} [{a},{b})" in vis and h:
                    back[i] += 1
    L = [f"spike 쪽만, dt 상위 0.5% · st 상위 1% (SMD 28대, GT 327개: 짧은 spike {sp[0][1]} / 16틱+ {lg[0][1]})",
         f"{'방식':18s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 잡음/놓침 | 16틱+ 잡음 | 보이는데 놓쳤던 {len(vis)}개 중 되찾음"]
    for i, nm in enumerate(NAMES):
        s = ev2_scores(cnt[i])
        L.append(f"{nm:18s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][0]:3d} / {sp[i][1] - sp[i][0]:3d}          | {lg[i][0]:3d}/{lg[i][1]}   | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "calib_variants.txt").write_text(txt)


if __name__ == "__main__":
    main()
