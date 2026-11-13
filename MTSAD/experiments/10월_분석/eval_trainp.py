"""
train spike 점수로 1차 거르기 (나연 2026-10-20) — 기준선: 1번(작은 움직임 그림 점수 없음) + 눈금 넓힘, spike 쪽만
  1. spike 원점수 (test)
  2. 열마다 train p = (그 채널 train 열 중 이 점수 이상인 수 + 1) / (train 열 수 + 1)  → p ≥ α 면 걸러냄 (train에서 흔한 점수)
     train 열 점수 = spike_scores_train (같은 규칙: 움직임이 train 범위 1% 미만인 train 그림은 제외)
     train 점수가 없는 채널(train 상수)이나 train 열이 50개 미만인 채널은 거르지 않음
  3. 채널별 보정 (중앙값·MAD는 거르기 전 test 점수 전체로)
  4. dt = 전체 칸 기준 상위 0.5% (걸러진 칸은 맨 아래로) → 남은 칸이 0.5%보다 적으면 남은 칸 전부 통과
  5. AND 갑작스러움 > 상위 1%
  채점: event P/R/F1, R50/F1(50%), 후보, 짧은 spike 놓침, 16틱+ 잡음, 잡음 채널 5개 오탐, 보이는데 놓친 15개 되찾음
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat, VIS, FLAT

HERE = Path(__file__).resolve().parent
W, GRID, P = 224, 16, 14
ALPHAS = [None, 0.05, 0.02, 0.01]
NOISE = {("machine-1-5", 29), ("machine-2-2", 29), ("machine-2-2", 15), ("machine-3-1", 10), ("machine-3-9", 15)}


def train_ref(e, tr):
    cols = np.load(HERE / "spike_scores_train" / f"{e}.npz")["cols"].astype(float)
    rg = np.ptp(tr, 0); nwin = cols.shape[1] // GRID; ref = []
    for c in range(cols.shape[0]):
        v = cols[c].copy()
        for w in range(nwin):
            if np.ptp(tr[w * W:(w + 1) * W, c]) < 0.01 * max(rg[c], 1e-12):
                v[w * GRID:(w + 1) * GRID] = np.nan
        v = np.sort(v[np.isfinite(v)]); ref.append(v if len(v) >= 50 else None)
    return ref


def train_p(raw, ref):
    p = np.zeros_like(raw)
    for c, v in enumerate(ref):
        if v is None:
            continue
        x = raw[c]; ok = np.isfinite(x)
        p[c, ok] = (len(v) - np.searchsorted(v, x[ok], side="left") + 1) / (len(v) + 1)
    return p


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    n = len(ALPHAS); cnt = [0] * n; sp = [[0, 0] for _ in range(n)]; lg = [[0, 0] for _ in range(n)]; back = [0] * n; nfp = [0] * n
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        raw = drop_flat(raw, te, tr); S, _ = calib(raw); fin = np.isfinite(S)
        pv = train_p(raw, train_ref(e, tr))
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        for i, a in enumerate(ALPHAS):
            Sm = np.where(fin, S, -np.inf) if a is None else np.where(fin & (pv < a), S, -np.inf)
            dt = np.percentile(Sm[fin], 99.5)
            m = (Sm > dt) & Gb; pr = m.any(0)
            cnt[i] = cnt[i] + ev2_counts(y, pr)
            for s, b in segs(y):
                t = sp[i] if b - s <= 15 else lg[i]; t[1] += 1; h = pr[s:b].any(); t[0] += int(h); back[i] += int(f"{e} [{s},{b})" in vis and h)
            for e2, c in NOISE:
                if e2 == e:
                    nfp[i] += sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
    L = ["spike 쪽만 (1번 + 눈금 넓힘), train p로 1차 거르기 → 보정 → dt 상위 0.5% → AND 갑작 상위 1% (SMD 28대)",
         f"{'α':8s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ | 잡음 채널 5개 오탐 | 보이는데 놓친 {len(vis)}개 되찾음"]
    for i, a in enumerate(ALPHAS):
        s = ev2_scores(cnt[i])
        L.append(f"{'안 거름' if a is None else a:8} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {nfp[i]:3d} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "trainp_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
