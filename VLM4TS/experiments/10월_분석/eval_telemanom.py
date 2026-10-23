"""
오라클 없이: Telemanom 동적 임계값 (Hundman et al., KDD 2018, Sec. 3.2)
  점수 e (test)에 대해  ε = μ(e) + z·σ(e),  z ∈ {2, 2.5, ..., 10}
  z* = argmax [ Δμ/μ + Δσ/σ ] / ( |넘은 점| + |넘은 덩어리|² ),  Δ = 넘은 점을 뺐을 때 줄어든 양
  넘는 z가 하나도 없으면 기준 = ∞ (이상 없음). 원 논문의 가지치기(pruning) 단계는 생략
  우리 점수는 음수가 있어서 (탈주기 z, spike logit) e − min(e)로 옮겨서 계산 (기준 위치는 같음)
  채널별: 채널마다 점수마다 기준 / machine별: 38채널 최댓값 점수에 기준 하나 (spike는 test 채널별 보정 후)
  최종 = 채널마다 [ 탈주기 > εD OR ( spike > εS AND 갑작스러움 > εG ) ]
  채점: event F1 / event F1(50%) / 후보 수 (+ 채널 R/P)
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
from eval_channel_pr_thr import gt_info  # noqa: E402
import asd_eval  # noqa: E402

ZS = np.arange(2, 10.01, 0.5)


def telemanom(e):
    e = e[np.isfinite(e)] if np.isnan(e).any() else e
    if len(e) < 50 or np.ptp(e) < 1e-12:
        return np.inf
    x = e - e.min(); mu, sd = x.mean(), x.std()
    if mu <= 0 or sd <= 0:
        return np.inf
    best, bz = 0.0, None
    for z in ZS:
        eps = mu + z * sd; a = x > eps; na = a.sum()
        if na == 0:
            continue
        r = x[~a]; nseq = len(segs(a.astype(int)))
        v = ((mu - r.mean()) / mu + (sd - r.std()) / sd) / (na + nseq ** 2)
        if v > best:
            best, bz = v, eps
    return np.inf if bz is None else bz + e.min()


def run_entity(y, D, S, G, mode):
    T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
    C = D.shape[0]
    if mode == "채널별":
        Sraw = S
        zD = np.array([telemanom(D[c]) for c in range(C)])
        zS = np.array([telemanom(Sraw[c]) for c in range(C)])
        zG = np.array([telemanom(G[c]) for c in range(C)])
        Sv = np.where(np.isfinite(Sraw), Sraw, -np.inf)
    else:
        Sc = calibrate(S); Sv = np.where(np.isfinite(Sc), Sc, -np.inf)
        zD = np.full(C, telemanom(D.max(0))); zS = np.full(C, telemanom(Sv.max(0))); zG = np.full(C, telemanom(G.max(0)))
    m = (D > zD[:, None]) | ((Sv > zS[:, None]) & (G > zG[:, None]))
    return y, m


def score(items, lab_paths):
    c = 0; R = []
    for (y, m), lab in zip(items, lab_paths):
        p = m.any(0); c = c + ev2_counts(y, p)
        gt, gch = gt_info(y, lab); bl = segs(p.astype(int))
        for (a, b), g in zip(gt, gch):
            ov = [(u, v) for u, v in bl if u < b and v > a]
            if ov and g:
                sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0]); R.append((len(g & sus), len(g), len(sus)))
    s = ev2_scores(c); A = np.array(R)
    return (f"evP {s['P']:.3f} | evR {s['R']:.3f} evF1 {s['F']:.3f} | evR50 {s['R50']:.3f} evF1(50%) {s['F50']:.3f} | 후보 {s['n']:5d} | "
            f"채널R {np.mean(A[:, 0] / A[:, 1]):.3f} 채널P {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f}")


def main():
    data = {"SMD 28대": [], "ASD 12대": []}; labs = {"SMD 28대": [], "ASD 12대": []}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        data["SMD 28대"].append((y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0)))
        labs["SMD 28대"].append(SMD / "interpretation_label" / f"{e}.txt")
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        data["ASD 12대"].append((y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0)))
        labs["ASD 12대"].append(asd_eval.ASD / "interpretation_label" / f"{e}.txt")
    out = []
    for name in data:
        for mode in ["채널별", "machine별"]:
            out.append(f"[{name}] Telemanom {mode:6s} {score([run_entity(*d, mode) for d in data[name]], labs[name])}")
            print(out[-1], flush=True)
    out += ["참고 고정 비율 최고 (탈주기 3%, spike 0.5%, 갑작 0.05%): SMD evF1 0.561 / 50% 0.372 / 후보 873,  ASD 0.294 / 0.235 (탈주기만 0.362 / 0.274)",
            "참고 machine별 오라클 (event F1 기준): SMD evF1 0.699, ASD 0.625"]
    txt = "\n".join(out); print(txt)
    (HERE / "coverage_groups" / "telemanom_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
