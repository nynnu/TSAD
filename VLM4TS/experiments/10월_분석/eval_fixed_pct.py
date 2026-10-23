"""
오라클 대신: 세 기준을 모든 machine에 같은 "상위 몇 %"로 고정 (GT로 machine마다 고르지 않음)
  구조는 현재 최선 모델 그대로: 탈주기(채널 최댓값) > 상위 a%  OR  (보정 spike > 상위 b%  AND  갑작스러움 > 상위 c%)
  비율 지점은 기존 오라클과 같은 방식으로 그 machine 점수에서 계산 (탈주기: 채널 최댓값 / spike·갑작: 전 채널·전 틱)
  격자: a ∈ {0.1~20}, b ∈ {0.01~2}, c ∈ {0.01~10}
  채널별 버전: 비율 지점을 채널 하나의 test 점수 안에서 계산 (채널마다·machine마다 기준값이 다름)
  출력: SMD 전 조합 표(틱F1 순) + 기준별 최고 조합을 ASD에 그대로 적용한 결과
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, prf, f1, PH, SC, SMD  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_eventf1_threshold import ev_counts  # noqa: E402
from eval_channel_pr_thr import gt_info, ch_counts  # noqa: E402
import asd_eval  # noqa: E402

A_, B_, C_ = [0.1, 0.5, 1, 2, 5, 10, 20], [0.01, 0.1, 0.5, 1, 2], [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]


def ch_pct(X, q):
    out = np.full((X.shape[0], 1), np.inf)
    for c in range(X.shape[0]):
        v = X[c][np.isfinite(X[c])]
        if len(v):
            out[c, 0] = np.percentile(v, q)
    return out


def prep(y, D, S, G, lab, mode):
    T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    Dm = D.max(0); fin = np.isfinite(S)
    if mode == "machine":
        th = ({a: np.percentile(Dm, 100 - a) for a in A_}, {b: np.percentile(S[fin], 100 - b) for b in B_},
              {c: np.percentile(G, 100 - c) for c in C_})
    else:
        th = ({a: ch_pct(D, 100 - a) for a in A_}, {b: ch_pct(S, 100 - b) for b in B_}, {c: ch_pct(G, 100 - c) for c in C_})
    gt, gch = gt_info(y, lab)
    return y, D, np.where(fin, S, -np.inf), G, th, gt, gch


def evaluate(items, a, b, c):
    pt = np.zeros(3); ev = np.zeros(4); cc = np.zeros(3); R = []
    for y, D, S, G, th, gt, gch in items:
        m = (D > th[0][a]) | ((S > th[1][b]) & (G > th[2][c])); p = m.any(0)
        pt += prf(y, p); ev += ev_counts(y, p, gt); cc += ch_counts(m, y, gt, gch)
        bl = segs(p.astype(int))
        for (s, e), g in zip(gt, gch):
            ov = [(u, v) for u, v in bl if u < e and v > s]
            if ov and g:
                sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0]); R.append((len(g & sus), len(g), len(sus)))
    tp, fp, fn = pt; eR, eP = ev[0] / ev[1], ev[3] / max(ev[2], 1); A = np.array(R); ct, cf, cn = cc
    return dict(tP=tp / max(tp + fp, 1), tR=tp / max(tp + fn, 1), tF=f1(tp, fp, fn), eR=eR, eP=eP, eF=2 * eR * eP / max(eR + eP, 1e-9),
                n=int(ev[2]), cR=np.mean(A[:, 0] / A[:, 1]), cP=np.mean(A[:, 0] / np.maximum(A[:, 2], 1)), cF=2 * ct / max(2 * ct + cf + cn, 1))


def fmt(k, r):
    return (f"{str(k):18s} {r['tP']:.3f} {r['tR']:.3f} {r['tF']:.3f} | {r['eR']:.3f} {r['eP']:.3f} {r['eF']:.3f} {r['n']:5d} | "
            f"{r['cR']:.3f} {r['cP']:.3f} {r['cF']:.3f}")


def main():
    out = []
    for mode, nm in [("machine", "machine별 %"), ("channel", "채널별 %")]:
        out += run(mode, nm) + [""]
    out += ["참고 (machine별 오라클, 틱F1 기준): SMD 0.359 0.646 0.461 | 0.881 0.294 0.441 1849 | 0.488 0.770 0.367",
            "                                  ASD 0.429 0.580 0.494 | 0.539 0.367 0.437  120 | 0.444 0.869 0.357"]
    txt = "\n".join(out); print(txt)
    (HERE / "coverage_groups" / "fixed_pct_result.txt").write_text(txt)


def run(mode, nm):
    smd = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        smd.append(prep(y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0), SMD / "interpretation_label" / f"{e}.txt", mode))
    asd = []
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        asd.append(prep(y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0), asd_eval.ASD / "interpretation_label" / f"{e}.txt", mode))
    head = f"{'(탈주기%, spike%, 갑작%)':18s}  틱P  틱R   틱F1 |  evR   evP  evF1  후보 | 채널R 채널P 채널F1"
    res = {k: evaluate(smd, *k) for k in product(A_, B_, C_)}
    out = [f"===== {nm} =====", "[SMD 28대] 모든 machine 같은 비율 — 틱F1 상위 5개", head] + [fmt(k, r) for k, r in sorted(res.items(), key=lambda x: -x[1]["tF"])[:5]]
    out += ["", "[SMD] event F1 상위 5개", head] + [fmt(k, r) for k, r in sorted(res.items(), key=lambda x: -x[1]["eF"])[:5]]
    out += ["", "[SMD] 채널 F1 상위 5개", head] + [fmt(k, r) for k, r in sorted(res.items(), key=lambda x: -x[1]["cF"])[:5]]
    out += ["", "[ASD 12대] SMD에서 기준별로 가장 좋았던 조합을 그대로", head]
    for key, nm in [("tF", "틱F1"), ("eF", "eventF1"), ("cF", "채널F1")]:
        k = max(res, key=lambda x: res[x][key]); out.append(fmt(k, evaluate(asd, *k)) + f"   ← SMD {nm} 최고")
    return out


if __name__ == "__main__":
    main()
