"""Raise the RANKING quality of the zero-shot DINOv2 Stage1 score on SMD (oracle Standard-F1 as in TSB-AD; target: VETime 0.514 on TSB-AD-M-Eva SMD).
Declared before the numbers. Clean protocol: variants are CHOSEN on the 5 SMD machines that TSB-AD does NOT use and that have patch features
(machine-1-8, 2-4, 2-5, 3-3, 3-5), then REPORTED once on the 20 TSB-AD-M-Eva SMD files (labels / offsets as tsbad_smd_eval.py). All scores zero-shot (test only).
Scores: UZ (current Stage1 score), B3 (3rd largest channel of the DINO head scores), MuSc-JB (patch mutual scoring, 30 % / 50 %), and combinations:
  smoothing  moving MAX or MEAN over w ticks (w in 15, 51, 101): an anomaly is an episode, the head score fires on edges / spikes inside it
  fusion     rank average of UZ and MuSc-JB (two DINOv2 views: 'looks like a spike' and 'rare shape on this machine')
Metrics: oracle point F1 (Standard-F1) and AUC-PR, file mean. Usage: python dino_score_boost.py"""
import torch  # noqa: F401
import sys, json, inspect, numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.covariance import LedoitWolf
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent/"pipeline_final")); sys.argv = sys.argv[:1]
import zs_integrated as Zi, musc_dino as MD, tsbad_smd_eval as E
src = inspect.getsource(Zi.stage1).replace('return [(int(e["start"]), int(e["end"])) for e in decide_incidents(Z)]', "return Z, U3, X").replace("def stage1", "def stage1_scores")
ns = dict(Zi.__dict__); exec(src, ns)
CACHE = HERE/"results/resid_dino_20261125/boost_scores.npz"; TUNE = ["machine-1-8", "machine-2-4", "machine-2-5", "machine-3-3", "machine-3-5"]
zs = lambda v: (v-v.mean())/max(v.std(), 1e-9)


def musc_jb(m, T, frac):
    X = MD.tick_scores(m, frac)[:T]; B3 = -np.sort(-X, 1)[:, 2]; live = X.std(0) > 1e-9; Uc = np.clip(X[:, live], -20, 20); lw = LedoitWolf().fit(Uc); d = Uc-lw.location_
    return (zs(np.log1p(np.einsum("ti,ij,tj->t", d, lw.precision_, d)))+zs(B3))/np.sqrt(2)


def machine_scores(m):
    te = np.loadtxt(E.D/"test"/f"{m}.txt", delimiter=",", ndmin=2); T = len(te); y = np.loadtxt(E.D/"test_label"/f"{m}.txt", delimiter=",").ravel().astype(bool)[:T]
    UZ, U3, X = ns["stage1_scores"](m, te); B3 = -np.sort(-X, 1)[:, 2]; out = dict(y=y, UZ=UZ, B3=B3)
    if (MD.OUT/"feats"/f"{m}.npz").exists(): out["M30"] = musc_jb(m, T, .3); out["M50"] = musc_jb(m, T, .5)
    return out


def all_scores():
    if CACHE.exists(): z = np.load(CACHE); d = {}; [d.setdefault(k.split("|")[0], {}).__setitem__(k.split("|")[1], z[k]) for k in z.files]; return d
    ms = TUNE+[E.MAP[i][0] for i in E.MAP if i not in (6, 16)]; d = {}
    for m in ms: d[m] = machine_scores(m); print("scored", m, flush=True)
    np.savez(CACHE, **{f"{m}|{k}": v for m, dd in d.items() for k, v in dd.items()}); return d


def f1max(s, y): p, r, _ = precision_recall_curve(y, s); return float(np.max(2*p*r/(p+r+1e-5)))
rk = lambda v: rankdata(v)/len(v)
D = all_scores()   # cached scores, needed by importers
if __name__ == "__main__":
    VARS = {"UZ (current)": lambda d: d["UZ"], "B3": lambda d: d["B3"], "MuSc30": lambda d: d.get("M30"), "MuSc50": lambda d: d.get("M50"),
            "UZ+MuSc50 rank avg": lambda d: rk(d["UZ"])+rk(d["M50"]) if "M50" in d else None}
    for base in ("UZ (current)", "UZ+MuSc50 rank avg"):
        for w in (15, 51, 101):
            VARS[f"{base} | max{w}"] = (lambda b, w_: lambda d: (lambda s: None if s is None else maximum_filter1d(s, w_))(VARS[b](d)))(base, w)
            VARS[f"{base} | mean{w}"] = (lambda b, w_: lambda d: (lambda s: None if s is None else uniform_filter1d(np.asarray(s, float), w_))(VARS[b](d)))(base, w)
    D = all_scores(); res = {}
    for part, items in (("tune (5 non-TSB-AD SMD)", [(m, 0, None) for m in TUNE]), ("TSB-AD Eva-20", [(E.MAP[i][0], E.MAP[i][1], i) for i in E.MAP if i not in (6, 16)])):
        res[part] = {}
        for n, fn in VARS.items():
            f, a = [], []
            for m, off, i in items:
                s = fn(D[m])
                if s is None: f = None; break
                y = D[m]["y"]
                if i is not None:
                    import glob, pandas as pd
                    n_ = len(pd.read_csv(glob.glob(str(E.TS/f"*_SMD_id_{i}_*.csv"))[0], usecols=[-1] if False else None).iloc[:, -1]); y = y[off:off+n_]; s = np.asarray(s)[off:off+n_]
                f.append(f1max(s, y)); a.append(float(average_precision_score(y, s)))
            res[part][n] = None if f is None else dict(std_f1=round(float(np.mean(f)), 3), auc_pr=round(float(np.mean(a)), 3))
        print("==", part); [print(f"  {n:34s}", v) for n, v in res[part].items()]
    json.dump(res, open(HERE/"results/resid_dino_20261125/boost.json", "w"), indent=1)
