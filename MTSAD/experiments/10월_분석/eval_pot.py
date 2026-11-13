"""
오라클 없이: 채널마다 점수마다 POT(Peaks Over Threshold, SPOT KDD 2017 / OmniAnomaly KDD 2019)로 기준을 정함
  최종 = 채널마다 [ 탈주기 > zD  OR  ( spike > zS  AND  갑작스러움 > zG ) ], 구간 = 어떤 채널이든 켜진 틱
  보통 채널: 그 채널 train 점수로 POT / train 상수 채널: 그 채널 test 점수로 POT (상수 채널 그림 0~1, 갑작스러움 하한 0.01)
  채널마다 기준이 따로라 spike 채널별 보정은 안 씀 (원점수 사용)
  POT: 상위 2% 지점 t → 넘은 만큼에 GPD 맞춤 → 정상에서 넘을 확률 q인 지점. q = 1e-3 / 1e-4 / 1e-5 (기본 1e-4)
  machine 단위 버전: 점수마다 machine에 기준 하나 — train "채널 최댓값" 점수로 POT.
    spike는 채널끼리 비교하려고 채널별 보정 (train 중앙값·MAD, train 상수 채널은 test로). 탈주기·갑작스러움은 원래 "평소의 몇 배" 단위라 그대로
  지표: 틱 P/R/F1, event R/P/F1, 후보 수, 채널 R/P (걸친 GT 이상마다, eval_channel_pr.py와 같음), 채널 F1
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24주차_원인분석"))
from exp_J_phase_deseason import channel_score  # noqa: E402
from eval_dsz_allch import segs, prf, f1, PH, SC, SMD  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_eventf1_threshold import ev_counts  # noqa: E402
from eval_channel_pr_thr import gt_info, ch_counts  # noqa: E402
from eval_const01_floor import sudden_floor  # noqa: E402

LEVEL, QS, P = 0.98, [1e-3, 1e-4, 1e-5], 14


def pot(x, q):
    x = x[np.isfinite(x)]
    if len(x) < 50 or np.ptp(x) < 1e-12:
        return np.inf
    t = np.quantile(x, LEVEL); exc = x[x > t] - t
    if len(exc) < 10 or np.ptp(exc) < 1e-12:
        return float(x.max())
    g, _, s = genpareto.fit(exc, floc=0)
    r = q * len(x) / len(exc)
    z = t - s * np.log(r) if abs(g) < 1e-6 else t + s / g * (r ** (-g) - 1)
    return float(max(z, t))


def load(e):
    y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    D = np.load(PH / f"{e}_phase_deseason.npz")["scores"]
    T = min(len(y), D.shape[1]); y, te, D = y[:T], te[:T], D[:, :T]
    S = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)[:, :T]
    d = np.load(HERE / "spike_scores_const01" / f"{e}.npz"); cc = list(d["chs"]); S[cc] = d["scores"][cc][:, :T]
    G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    if cc:
        G[cc] = np.nan_to_num(sudden_floor(te, cc), nan=0.0)
    Dtr = np.stack([channel_score(tr[:, c], tr[:, c]) for c in range(tr.shape[1])])
    Str = np.load(HERE / "spike_scores_train" / f"{e}.npz")["cols"].astype(float)
    Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0)
    return y, tr, te, D, S, G, Dtr, Str, Gtr, cc


def thresholds(tr, te, D, S, G, Dtr, Str, Gtr, cc, q):
    C = tr.shape[1]; z = np.full((3, C), np.inf)
    for c in range(C):
        if np.ptp(tr[:, c]) >= 1e-6:
            z[:, c] = pot(Dtr[c], q), pot(Str[c], q), pot(Gtr[c], q)
        elif c in cc:                                   # train 상수, test에서 움직임 → test 점수로
            z[1, c], z[2, c] = pot(S[c, ::P], q), pot(G[c], q)
    return z


def calib(S, Str, cc):
    """채널별 (x − 중앙값) / MAD — 기준은 train 열 점수 (train 상수 채널은 test)"""
    S2, T2 = S.copy(), Str.copy()
    for c in range(S.shape[0]):
        ref = S[c, ::P] if c in cc else Str[c]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 10:
            continue
        med = np.median(ref); mad = max(1.4826 * np.median(np.abs(ref - med)), 1e-6)
        S2[c] = (S[c] - med) / mad; T2[c] = (Str[c] - med) / mad
    return S2, T2


def thresholds_machine(tr, D, S, G, Dtr, Str, Gtr, cc, q):
    C = tr.shape[1]; S2, T2 = calib(S, Str, cc)
    zD = pot(Dtr.max(0), q); zS = pot(np.nanmax(np.where(np.isnan(T2), -np.inf, T2), 0), q); zG = pot(Gtr.max(0), q)
    return np.full(C, zD), np.full(C, zS), np.full(C, zG), S2


def main():
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    data = {e: load(e) for e in ents}
    lines = [f"[SMD {len(ents)}대, POT (level {LEVEL})]  {'':4s} 틱P   틱R   틱F1 |  evR   evP   evF1  후보 | 채널R  채널P 채널F1 의심채널중앙"]
    for mode, q in [(m_, q_) for m_ in ["채널별", "machine별"] for q_ in QS]:
        pt = np.zeros(3); ev = np.zeros(4); cht = np.zeros(3); R = []
        for e in ents:
            y, tr, te, D, S, G, Dtr, Str, Gtr, cc = data[e]
            if mode == "채널별":
                zD, zS, zG = thresholds(tr, te, D, S, G, Dtr, Str, Gtr, cc, q)
            else:
                zD, zS, zG, S = thresholds_machine(tr, D, S, G, Dtr, Str, Gtr, cc, q)
            m = (D > zD[:, None]) | ((np.nan_to_num(S, nan=-np.inf) > zS[:, None]) & (G > zG[:, None]))
            p = m.any(0); gt, gch = gt_info(y, SMD / "interpretation_label" / f"{e}.txt")
            pt += prf(y, p); ev += ev_counts(y, p, gt); cht += ch_counts(m, y, gt, gch)
            blocks = segs(p.astype(int))
            for (a, b), g in zip(gt, gch):
                ov = [(u, v) for u, v in blocks if u < b and v > a]
                if ov and g:
                    sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0]); R.append((len(g & sus), len(g), len(sus)))
        tp, fp, fn = pt; eR, eP = ev[0] / ev[1], ev[3] / max(ev[2], 1); A = np.array(R); ct, cf, cn = cht
        lines.append(f"  {mode:6s} q={q:<8g}{'(기본)' if q == 1e-4 else '':6s}{tp / max(tp + fp, 1):.3f} {tp / max(tp + fn, 1):.3f} {f1(tp, fp, fn):.3f} | "
                     f"{eR:.3f} {eP:.3f} {2 * eR * eP / max(eR + eP, 1e-9):.3f} {int(ev[2]):5d} | "
                     f"{np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {2 * ct / (2 * ct + cf + cn):.3f} {np.median(A[:, 2]):5.0f}")
        print(lines[-1], flush=True)
    lines += ["참고 (오라클): 틱F1 기준 0.359 0.646 0.461 | 0.881 0.294 0.441 1849 | 채널R 0.488 P 0.770 F1 0.367",
              "             채널F1 기준 — | 0.939 0.340 — 1261 | 채널R 0.636 P 0.759 F1 0.511"]
    txt = "\n".join(lines); print(txt)
    (HERE / "coverage_groups" / "pot_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
