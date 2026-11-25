"""Label-free alarm threshold for the Z3 score: how much of the gap to the oracle threshold can be closed without labels? Labels only evaluate. Declared before the numbers.
Score rebuilt so that a held-out NORMAL reference exists: profile, sigma and Ledoit-Wolf covariance from the first 2/3 of train ('fit'), the last 1/3 ('cal') scored exactly like the test.
Z3 = (z(log1p J) + z(log1p M3))/sqrt(2), J = Mahalanobis of the clipped residual, M3 = 3rd largest clipped |r|, both 14-tick max filtered; z uses the CAL mean / std, so cal and test share a scale.
Thresholds (all label-free unless marked):
  test mu+z sigma  : z = 1.2816 (current), 2.3263 on the test score itself (VLM4TS)
  test top q%      : q = 1, 2, 4 (fixed alarm budget)
  cal quantile     : 1 - a of the cal scores, a = 0.01, 0.003, 0.001 (conformal false-alarm rate on normal data)
  cal POT          : peaks over the 98% cal quantile, generalized Pareto fit, risk q = 1e-3 (OmniAnomaly / SPOT style)
  ORACLE           : best tick F1 threshold on the test labels (upper bound, not a method)
Metrics per machine: tick F1, point-adjust F1, interval precision / event recall / interval F1 (VLM4TS contextual), alarm share; mean over machines and worst machine;
gap closed = (method - current) / (oracle - current) on mean tick F1.
Usage: python threshold_experiment.py SMD|ASD -> results/threshold_experiment_20261123/"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.stats import genpareto
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"pipeline_final"), str(ROOT/"stage1_final"), str(ROOT/"analysis")]
from residual_stage2 import usual_value
from joint_score import JointModel, CLIP
from evaluate_v2b_asd import counts, metr, events
DS = sys.argv[1] if len(sys.argv) > 1 else "SMD"; D = ROOT.parent/"mv_data"/DS; OUT = ROOT/"analysis/results/threshold_experiment_20261123"; (OUT/"cache").mkdir(parents=True, exist_ok=True)


def z3_scores(m):
    c = OUT/"cache"/f"{DS}_{m}.npz"
    if c.exists(): z = np.load(c); return z["cal"], z["test"]
    tr = np.loadtxt(D/"train"/f"{m}.txt", delimiter=",", ndmin=2); te = np.loadtxt(D/"test"/f"{m}.txt", delimiter=",", ndmin=2); cut = 2*len(tr)//3; fit = tr[:cut]
    ex, sig, ex_fit = usual_value(fit, len(tr)-cut+len(te), with_train=True); Rf = (fit-ex_fit)/sig; Rr = (np.r_[tr[cut:], te]-ex)/sig; model = JointModel(Rf, pairs=False)
    J = maximum_filter1d(model._maha(np.clip(Rr[:, model.keep], -CLIP, CLIP)), 14); M3 = maximum_filter1d(-np.sort(-np.minimum(np.abs(Rr), CLIP), axis=1)[:, 2], 14)
    n = len(tr)-cut; lj, lm = np.log1p(J), np.log1p(M3); zz = lambda v: (v-v[:n].mean())/max(v[:n].std(), 1e-9); Z = (zz(lj)+zz(lm))/np.sqrt(2)
    np.savez(c, cal=Z[:n], test=Z[n:]); return Z[:n], Z[n:]


def pot(cal, q=1e-3, init=.98):
    t = np.quantile(cal, init); ex = cal[cal > t]-t
    if len(ex) < 10: return np.quantile(cal, 1-q)
    c, _, s = genpareto.fit(ex, floc=0); r = q*len(cal)/len(ex)
    return t+(s/c*(r**(-c)-1) if abs(c) > 1e-6 else -s*np.log(r))


def evaluate(pred, y):
    ev = events(y); m = metr(counts(pred, y, ev)); pa = pred.copy()
    for a, b in ev:
        if pa[a:b].any(): pa[a:b] = True
    tp = (pa & y).sum(); p, r = tp/max(pa.sum(), 1), tp/max(y.sum(), 1)
    return dict(tick_f1=m["tick_f1"], pa_f1=round(2*p*r/max(p+r, 1e-9), 3), int_p=m["interval_p"], ev_r=m["event_recall"], int_f1=m["interval_f1"], alarm=m["kept_share"])


def oracle(s, y):
    best, bt = -1, None
    for t in np.unique(np.quantile(s, np.linspace(.5, .9995, 300))):
        f = evaluate(s >= t, y)["tick_f1"]
        if f > best: best, bt = f, t
    return bt


if __name__ == "__main__":
    ms = sorted(p.stem for p in (D/"train").glob("*.txt")); rows = {}
    M = {"test mu+1.28s (current)": lambda c, s: s.mean()+1.2816*s.std(), "test mu+2.33s": lambda c, s: s.mean()+2.3263*s.std(),
         "test top 1%": lambda c, s: np.quantile(s, .99), "test top 2%": lambda c, s: np.quantile(s, .98), "test top 4%": lambda c, s: np.quantile(s, .96),
         "cal q 0.99": lambda c, s: np.quantile(c, .99), "cal q 0.997": lambda c, s: np.quantile(c, .997), "cal q 0.999": lambda c, s: np.quantile(c, .999), "cal POT 1e-3": lambda c, s: pot(c)}
    for m in ms:
        cal, s = z3_scores(m); y = np.loadtxt(D/"test_label"/f"{m}.txt", delimiter=",").astype(bool).ravel()[:len(s)]; s = s[:len(y)]
        rows[m] = {k: evaluate(s >= f(cal, s), y) for k, f in M.items()} | {"ORACLE (test labels)": evaluate(s >= oracle(s, y), y)}; print(m, {k: v["tick_f1"] for k, v in rows[m].items()}, flush=True)
    names = list(rows[ms[0]]); summ = {}
    for k in names:
        v = [rows[m][k] for m in ms]; summ[k] = {f: round(float(np.mean([x[f] for x in v])), 3) for f in v[0]} | {"worst_tick_f1": round(float(min(x["tick_f1"] for x in v)), 3), "worst_int_f1": round(float(min(x["int_f1"] for x in v)), 3)}
    cur, ora = summ["test mu+1.28s (current)"]["tick_f1"], summ["ORACLE (test labels)"]["tick_f1"]
    for k in names: summ[k]["gap_closed"] = round((summ[k]["tick_f1"]-cur)/max(ora-cur, 1e-9), 2)
    print(f"\n=== {DS} ({len(ms)} machines): mean over machines")
    for k, v in summ.items(): print(f"  {k:26s} tick F1 {v['tick_f1']:.3f}  PA-F1 {v['pa_f1']:.3f}  int P/R/F1 {v['int_p']:.3f}/{v['ev_r']:.3f}/{v['int_f1']:.3f}  alarm {v['alarm']:.3f}  worst tick/int {v['worst_tick_f1']:.3f}/{v['worst_int_f1']:.3f}  gap closed {v['gap_closed']:+.2f}")
    (OUT/f"summary_{DS}.json").write_text(json.dumps(dict(summary=summ, per_machine=rows), indent=1, default=float))
