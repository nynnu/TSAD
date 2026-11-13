"""
상수 채널 0~1 추가(eval_const01.py) + 그 채널의 갑작스러움 하한을 0~1 범위의 1% = 0.01로
  (기존: train 범위의 1% = 0 → 1e-6)
  출력: 전체 표(틱F1 기준 오라클) + 상수 채널에서 켜진 덩어리 분류(맞춤 / GT구간·라벨X / 오탐) + 상수 GT 채널 놓침
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eval_dsz_allch import segs, best_pred, prf, f1, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_eventf1_threshold import report
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent


def sudden_floor(te, chs, floor=0.01):
    df = pd.DataFrame(te[:, chs])
    med = df.rolling(201, center=True, min_periods=50).median(); dev = (df - med).abs()
    mad = dev.rolling(201, center=True, min_periods=50).median()
    return (dev.values / np.maximum(1.4826 * mad.values, floor)).T


def search_thr(p1, S, G, y):
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); best, bd, bg = f1(*prf(y, p1)), np.inf, np.inf
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = f1(*prf(y, p1 | ((Sv > dt) & Gb).any(0)))
            if v > best:
                best, bd, bg = v, dt, g
    return bd, bg


def main():
    names = ["기존", "상수 0~1 (하한 1e-6)", "상수 0~1 + 하한 0.01"]
    models = {k: [] for k in names}; ys = []; out = []
    cnt = {k: {"맞춤": 0, "GT구간·라벨X": 0, "오탐": 0, "GT채널 놓침": 0} for k in names[1:]}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int); ys.append(y)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)
        d = np.load(HERE / "spike_scores_const01" / f"{e}.npz"); chs = list(d["chs"])
        raw2 = raw.copy(); raw2[chs] = d["scores"][chs]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); G2 = G.copy()
        if chs:
            G2[chs] = np.nan_to_num(sudden_floor(te, chs), nan=0.0)
        for k, S, Gk in [(names[0], calibrate(raw), G), (names[1], calibrate(raw2), G), (names[2], calibrate(raw2), G2)]:
            dt, st = search_thr(p1, S, Gk, y)
            fired = (np.where(np.isfinite(S), S, -np.inf) > dt) & (Gk > st)
            models[k].append(p1 | fired.any(0))
            if k == names[0] or not chs:
                continue
            for c in chs:
                for a, b in segs(fired[c].astype(int)):
                    tag = ("맞춤" if c in interp_channels(e, a, b) else "GT구간·라벨X") if y[a:b].any() else "오탐"
                    cnt[k][tag] += 1
                    if tag != "맞춤":
                        out.append(f"  [{k}] {e} ch{c} [{a},{b}) 값최대 {te[a:b, c].max():.3f} 보정spike {np.nanmax(S[c, a:b]):.1f}(dt {dt:.1f}) 갑작 {Gk[c, a:b].max():.1f}(st {st:.1f}) -> {tag}")
            for a, b in segs(y):
                for c in interp_channels(e, a, b):
                    if c in chs and not fired[c, a:b].any():
                        cnt[k]["GT채널 놓침"] += 1
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report("SMD 28대 (틱F1 기준 오라클)", models, ys)
    txt = buf.getvalue() + "\n상수 채널 17개에서 켜진 덩어리:\n" + "\n".join(f"  {k}: {v}" for k, v in cnt.items()) + "\n" + "\n".join(out)
    print(txt)
    (HERE / "coverage_groups" / "const01_floor_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
