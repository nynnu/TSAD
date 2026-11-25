"""Stage1 WITHOUT any normal (train) segment: fully zero-shot on the test series itself, as VLM4TS / Time-RCD. Anomalies are rare (about 4% of ticks), so robust statistics
of the test series estimate its normal behaviour: per channel, the validated period (stage1.period on the test channel), the usual value = MEDIAN over cycles at each phase
(no period: the channel median), the scale = 1.4826 x MAD of the residual; residual r = (x - usual) / scale. Joint score: Ledoit-Wolf on the residual clipped at 5 (so the
anomalies hardly move the covariance), Mahalanobis of the clipped residual; Z3 = (z(log1p J) + z(log1p M3)) / sqrt(2) with M3 = 3rd largest |r|, both 14-tick max filtered,
exactly the train-based recipe. Then the same decision layer and calm extent. Compared with the train-based Stage1 and with the test-only classical baselines (PCA, IForest,
KNN fitted on the test series itself, TSB-AD style). Labels only evaluate.
Usage: python testonly_stage1.py -> results/testonly_20261124/"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"analysis"), str(ROOT/"pipeline_final"), str(ROOT/"stage1_final")]
from scipy.ndimage import maximum_filter1d
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score
from stage1 import period
from paper_stage2 import decide_incidents, calm_extent, zscore
from iterate_common import data, split, mask, adopted, evaluate, G
from count_extent_bound import runs
OUT = ROOT/"analysis/results/testonly_20261124"; (OUT/"cache").mkdir(parents=True, exist_ok=True); DATA = ROOT.parent/"mv_data"; CLIPJ = 5.


def testonly_score(ds, m):
    c = OUT/"cache"/f"{ds}_{m}.npz"
    if c.exists(): z = np.load(c); return z["R"], z["Z"]
    X = np.loadtxt(DATA/ds/"test"/f"{m}.txt", delimiter=",", ndmin=2); T, C = X.shape; R = np.zeros((T, C))
    for j in range(C):
        x = X[:, j]
        if x.std() < 1e-9: continue
        P = period(x)
        if P:
            prof = np.array([np.median(x[k::P]) for k in range(P)]); u = prof[np.arange(T) % P]
        else: u = np.full(T, np.median(x))
        e = x-u; sc = 1.4826*np.median(np.abs(e-np.median(e)))
        if sc < 1e-9: sc = max(e.std(), 1e-6)
        R[:, j] = e/sc
    live = X.std(0) > 1e-9; Rc = np.clip(R[:, live], -CLIPJ, CLIPJ); lw = LedoitWolf().fit(Rc); d = Rc-lw.location_
    J = maximum_filter1d(np.einsum("ti,ij,tj->t", d, lw.precision_, d), 14); M3 = maximum_filter1d(-np.sort(-np.minimum(np.abs(R), 20.), axis=1)[:, 2], 14)
    Z = (zscore(J)+zscore(M3))/np.sqrt(2); np.savez(c, R=R, Z=Z); return R, Z


def baseline_testonly(ds, m, name):
    c = OUT/"cache"/f"{ds}_{m}_{name}.npy"
    if c.exists(): return np.load(c)
    from sklearn.decomposition import PCA as _P; from sklearn.ensemble import IsolationForest; from sklearn.neighbors import NearestNeighbors
    X = np.loadtxt(DATA/ds/"test"/f"{m}.txt", delimiter=",", ndmin=2); mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1.; X = (X-mu)/sd
    if name == "PCA": p = _P(n_components=.95).fit(X); s = ((X-p.inverse_transform(p.transform(X)))**2).sum(1)
    elif name == "IForest": s = -IsolationForest(n_estimators=200, random_state=0, n_jobs=4).fit(X).score_samples(X)
    else:
        rng = np.random.default_rng(0); ref = X[rng.choice(len(X), min(20000, len(X)), replace=False)]; s = NearestNeighbors(n_neighbors=11, n_jobs=4).fit(ref).kneighbors(X)[0][:, 1:].mean(1)
    s = maximum_filter1d(s, 14); np.save(c, s); return s


def ops(v, R, y):
    T = len(y); inc = decide_incidents(v); calm = [p for ps in calm_extent(R, v, inc) for p in ps]; out = {}
    for op, k in (("old", v > v.mean()+1.2816*v.std()), ("top4", v >= np.quantile(v, .96)), ("v1", mask([(i["start"], i["end"]) for i in inc], T)), ("stage1", mask(calm, T))):
        e = evaluate(k, y); out.update({f"{op} tick": e["tick_f1"], f"{op} int": e["int_f1"], f"{op} pa": e["pa_f1"]})
        if op == "stage1": out["stage1 aff"] = float(G.metric_Affiliation(y.astype(int), v, preds=k.astype(int))); out["stage1 ev_r"] = e["ev_r"]
    return out


if __name__ == "__main__":
    D = data(); rows = {}
    for (ds, m), d in D.items():
        y = d["y"]; T = len(y); R0, Z0 = testonly_score(ds, m); R0, Z0 = R0[:T], Z0[:T]; rows[(ds, m)] = {}
        for name, v, R in (("train-based (ours)", d["s"], d["R"]), ("TEST-ONLY (ours)", Z0, R0)):
            r = dict(auc_roc=float(roc_auc_score(y, v)), auc_pr=float(average_precision_score(y, v))); r.update(ops(v, R, y)); rows[(ds, m)][name] = r
        for b in ("PCA", "IForest", "KNN"):
            s = np.nan_to_num(np.log1p(np.maximum(baseline_testonly(ds, m, b)[:T], 0))); r = dict(auc_roc=float(roc_auc_score(y, s)), auc_pr=float(average_precision_score(y, s))); r.update(ops(s, R0, y)); rows[(ds, m)][f"{b} (test-only)"] = r
        print(ds, m, {k: (round(v["auc_roc"], 3), round(v["stage1 tick"], 3)) for k, v in rows[(ds, m)].items()}, flush=True)
    out = {}
    for sp in ("dev", "held", "ASD"):
        keys = [k for k in D if split(*k) == sp]; out[sp] = {n: {f: round(float(np.mean([rows[k][n][f] for k in keys])), 3) for f in rows[keys[0]][n]} for n in rows[keys[0]]}
    json.dump(dict(summary=out, per_machine={f"{a}/{b}": v for (a, b), v in rows.items()}), open(OUT/"summary.json", "w"), indent=1, default=float)
    for sp, o in out.items():
        print(f"=== {sp}")
        for n, v in o.items(): print(f"  {n:20s} AUC {v['auc_roc']}/{v['auc_pr']}  old {v['old tick']}/{v['old int']}  top4 {v['top4 tick']}/{v['top4 int']}  v1 {v['v1 tick']}/{v['v1 int']}  STAGE1 {v['stage1 tick']}/{v['stage1 int']} pa {v['stage1 pa']} aff {v['stage1 aff']:.3f} evr {v['stage1 ev_r']}")
