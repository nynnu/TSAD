"""
임계값 원인 확인 (SMD 28대) — 같은 점수, 기준 정하는 방식만 다름. 채점: event F1 / event F1(50%)
  1. 방식별 비교: 오라클(event F1) / 고정 비율(탈주기 3%, spike 0.5%, 갑작 0.05%) / POT 채널별·machine별 (q 1e-4) / Telemanom 채널별
  2. entity별 분포: GT 이상 안 탈주기 최댓값이 그 machine 정상 틱 중 상위 몇 %인지 (작을수록 정상과 잘 갈림)
  3. 이상 길이별 걸침: spike ≤15틱 / 16~224 / 225+
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, PH, SC, SMD  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_event2 import ev2_counts, ev2_scores  # noqa: E402
from oracle_thresholds import oracle  # noqa: E402
from eval_pot import load as pot_load, thresholds as pot_ch, thresholds_machine as pot_m  # noqa: E402
from eval_telemanom import run_entity  # noqa: E402

BINS = [(1, 15, "spike≤15"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def preds(e):
    y, tr, te, D, S, G, Dtr, Str, Gtr, cc = pot_load(e)            # D: 채널별 탈주기, S: spike 원점수(상수 채널 0~1 포함), G
    T = len(y)
    ph = D.max(0); Sraw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)[:, :T]
    Sc = calibrate(Sraw); Gb = np.nan_to_num(suddenness(te, tr), nan=0.0)[:, :T]
    Scv = np.where(np.isfinite(Sc), Sc, -np.inf); fin = np.isfinite(Sc)
    out = {}
    td, bd, bg, *_ = oracle(ph, Sc, Gb, y)
    out["오라클(event F1)"] = (ph > td) | ((Scv > bd) & (Gb > bg)).any(0)
    out["고정 비율 (3, 0.5, 0.05)"] = (ph > np.percentile(ph, 97)) | ((Scv > np.percentile(Sc[fin], 99.5)) & (Gb > np.percentile(Gb, 99.95))).any(0)
    zD, zS, zG = pot_ch(tr, te, D, S, G, Dtr, Str, Gtr, cc, 1e-4)
    out["POT 채널별 (q 1e-4)"] = ((D > zD[:, None]) | ((np.nan_to_num(S, nan=-np.inf) > zS[:, None]) & (G > zG[:, None]))).any(0)
    zD, zS, zG, S2 = pot_m(tr, D, S, G, Dtr, Str, Gtr, cc, 1e-4)
    out["POT machine별 (q 1e-4)"] = ((D > zD[:, None]) | ((np.nan_to_num(S2, nan=-np.inf) > zS[:, None]) & (G > zG[:, None]))).any(0)
    out["Telemanom 채널별"] = run_entity(y, D, Sraw, Gb, "채널별")[1].any(0)
    # 2. GT 이상 안 탈주기 최댓값의 정상 틱 대비 위치
    normal = ph[y == 0]
    pos = [100 * (normal >= ph[a:b].max()).mean() for a, b in segs(y)]
    return y, out, pos


def main():
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    tot, tl, dist = {}, {}, []
    for e in ents:
        y, out, pos = preds(e)
        dist.append((e, pos))
        for k, p in out.items():
            tot[k] = tot.get(k, 0) + ev2_counts(y, p)
            t = tl.setdefault(k, {b[2]: [0, 0] for b in BINS})
            for a, b in segs(y):
                nm = [x for lo, hi, x in BINS if lo <= b - a <= hi][0]; t[nm][1] += 1; t[nm][0] += int(p[a:b].any())
    L = ["1·3. 방식별 결과 (SMD 28대, GT 이상 327개)",
         f"{'방식':24s} | evP   evR   evF1 | evR50 evF1(50%) | 후보 | 걸침 spike≤15 / 16~224 / 225+"]
    for k, c in tot.items():
        s = ev2_scores(c)
        L.append(f"{k:24s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}     | {s['n']:4d} | "
                 + " / ".join(f"{tl[k][b[2]][0]}/{tl[k][b[2]][1]}" for b in BINS))
    L += ["", "2. entity별: GT 이상 안 탈주기 최댓값이 그 machine 정상 틱 중 상위 몇 %인가 (중앙값 / 25~75% / 상위 3% 안에 든 비율)"]
    for e, pos in dist:
        p = np.array(pos)
        L.append(f"  {e:13s} GT {len(p):3d}개 | 중앙 {np.median(p):6.2f}% | {np.percentile(p, 25):6.2f}~{np.percentile(p, 75):6.2f}% | 상위3% 안 {np.mean(p <= 3):.2f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "diag_threshold.txt").write_text(txt)


if __name__ == "__main__":
    main()
