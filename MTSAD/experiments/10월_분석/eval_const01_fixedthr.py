"""
상수 채널 0~1 추가의 직접 효과만 보기: 임계값(탈주기, dt, st)은 기존 모델(상수 채널 없음)에서 고른 오라클 값 그대로 고정
  하한 1e-6 / 하한 0.01 두 가지. 상수 채널에서 켜진 덩어리 분류 + 새 후보 수
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, best_pred, PH, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_const01_floor import sudden_floor, search_thr
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent


def main():
    names = ["하한 1e-6", "하한 0.01"]
    cnt = {k: {"맞춤": 0, "GT구간·라벨X": 0, "오탐": 0, "그중 새 후보": 0, "GT채널 놓침": 0, "새로 걸친 GT 구간": 0} for k in names}
    out = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        d = np.load(HERE / "spike_scores_const01" / f"{e}.npz"); chs = list(d["chs"])
        if not chs:
            continue
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)
        S0 = calibrate(raw); G = np.nan_to_num(suddenness(te, tr), nan=0.0)
        dt, st = search_thr(p1, S0, G, y)                      # 기존 모델 임계값
        old = p1 | ((np.where(np.isfinite(S0), S0, -np.inf) > dt) & (G > st)).any(0)
        raw2 = raw.copy(); raw2[chs] = d["scores"][chs]; S = calibrate(raw2)
        G2 = G.copy(); G2[chs] = np.nan_to_num(sudden_floor(te, chs), nan=0.0)
        out.append(f"  {e}: 기존 임계값 dt {dt:.1f}, st {st:.1f}, 상수 채널 {chs}")
        for k, Gk in [(names[0], G), (names[1], G2)]:
            fired = (np.where(np.isfinite(S[chs]), S[chs], -np.inf) > dt) & (Gk[chs] > st)
            for c, f in zip(chs, fired):
                for a, b in segs(f.astype(int)):
                    tag = ("맞춤" if c in interp_channels(e, a, b) else "GT구간·라벨X") if y[a:b].any() else "오탐"
                    cnt[k][tag] += 1
                    if tag == "오탐" and not old[a:b].any():
                        cnt[k]["그중 새 후보"] += 1
                    if tag != "맞춤":
                        out.append(f"    [{k}] ch{c} [{a},{b}) 값 {te[a:b, c].max():.3f} 보정spike {np.nanmax(S[c, a:b]):.1f} 갑작 {Gk[c, a:b].max():.1f} -> {tag}{' (새 후보)' if tag == '오탐' and not old[a:b].any() else ''}")
            new = old | fired.any(0)
            for a, b in segs(y):
                gc = [c for c in interp_channels(e, a, b) if c in chs]
                if not gc:
                    continue
                cnt[k]["GT채널 놓침"] += sum(not fired[chs.index(c), a:b].any() for c in gc)
                if new[a:b].any() and not old[a:b].any():
                    cnt[k]["새로 걸친 GT 구간"] += 1
    txt = "\n".join(f"{k}: {v}" for k, v in cnt.items()) + "\n" + "\n".join(out)
    print(txt)
    (HERE / "coverage_groups" / "const01_fixedthr_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
