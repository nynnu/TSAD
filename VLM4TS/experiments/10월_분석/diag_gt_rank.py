"""
지금 모델(1번 + 눈금 넓힘 + 상수 채널 0~1 + 갑작 하한 0.01, λ 없음)에서 GT가 몇 등인가 — machine마다
  GT 점수 = 그 GT의 GT 채널들, GT 구간 안 보정 spike 점수 최댓값
  등수 = 그 machine 전체 칸(모든 채널 × 14틱 열) 중 몇 번째, 상위 % = 등수 / 전체 칸 수
  dt(상위 0.5%)를 넘는지, 그리고 GT 길이(짧은 spike ≤15틱)도 같이
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14


def main():
    L = [f"{'machine':13s} | 전체 칸 | GT 수 | 상위 0.5% 안 | 상위% 중앙 | GT마다 상위% (작은 순, *=짧은 spike)"]; allp = []; allsp = []
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr))
        pool = S[:, ::P]; pool = np.sort(pool[np.isfinite(pool)])[::-1]; N = len(pool)
        items = []
        for a, b in segs(y):
            g = [c for c in interp_channels(e, a, b) if np.isfinite(S[c, a:b]).any()]
            if not g:
                continue
            v = max(np.nanmax(S[c, a:b]) for c in g)
            rank = int(np.searchsorted(-pool, -v, side="left")) + 1; pct = 100 * rank / N
            items.append((pct, b - a <= 15)); allp.append(pct)
            if b - a <= 15:
                allsp.append(pct)
        items.sort()
        L.append(f"{e:13s} | {N:5d} | {len(items):3d} | {sum(p <= 0.5 for p, _ in items):3d} | {np.median([p for p, _ in items]):6.2f}% | "
                 + ", ".join(f"{p:.2f}{'*' if s else ''}" for p, s in items))
    a, s = np.array(allp), np.array(allsp)
    L += ["", f"전체 GT {len(a)}개: 상위 0.1% 안 {np.mean(a <= 0.1):.0%}, 0.5% 안 {np.mean(a <= 0.5):.0%}, 1% 안 {np.mean(a <= 1):.0%}, 5% 안 {np.mean(a <= 5):.0%}, 중앙 {np.median(a):.2f}%",
          f"짧은 spike {len(s)}개: 상위 0.1% 안 {np.mean(s <= 0.1):.0%}, 0.5% 안 {np.mean(s <= 0.5):.0%}, 1% 안 {np.mean(s <= 1):.0%}, 5% 안 {np.mean(s <= 5):.0%}, 중앙 {np.median(s):.2f}%"]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "gt_rank.txt").write_text(txt)


if __name__ == "__main__":
    main()
