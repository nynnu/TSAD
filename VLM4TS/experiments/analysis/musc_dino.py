"""Zero-shot Stage1 with MUTUAL SCORING of DINOv2 patch features across the test windows (MuSc, ICLR 2024, adapted to plots of one channel). SMD, test only, no labels in
the score. Declared before the numbers. Why: 48 % of the zero-shot pipeline's held errors are false intervals, and 'the same shape recurs elsewhere in the test' was the
second-best label-free separator of false vs true intervals (AUROC 0.70). The current DINO heads score each 224-tick plot alone (spike / height), so a shape that is
ordinary for this machine but spiky still scores high. Here a shape is anomalous only if few other windows of the same channel contain a similar shape.
extract: per channel, every non-overlapping 224-tick window is drawn exactly as Stage1 (stage1.ts_to_image, test p0.5-p99.5 scale widened to the window's extremes);
  DINOv2 ViT-B/14 patch tokens (16 x 16) -> column feature = mean over the 16 rows (one 14-tick column), L2-normalised. Flat windows (ptp < 1 % of the usual range) get none.
score: for a column, its cosine distance to the closest column of every OTHER window of the channel; score = mean of the lowest 30 % of these per-window minima
  (MuSc's interval average: a shape shared by >= 30 % of the windows scores low; a rare shape scores high even if it recurs in a few windows). 10 % reported as a check.
  Per channel calib (median / MAD over the channel's columns), flat or missing columns = channel median.
Aggregation + the same machine-level decision layer: B3 (3rd largest channel) and JB ((z(log1p Mahalanobis) + z(3rd)) / sqrt 2), compared with the current DINO head
scores (B3 / UZ) on the same machines.
Usage: python musc_dino.py extract machine-1-5 ... ; python musc_dino.py evaluate machine-1-5 ..."""
import sys, json, time
from pathlib import Path
import torch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"stage1_final"), str(ROOT/"pipeline_final"), str(ROOT/"analysis")]
from stage1 import ts_to_image, load_dino, patches, WIN, GRID, P
DATA = ROOT.parent/"mv_data/SMD"; OUT = ROOT/"analysis/results/musc_dino_20261125"; (OUT/"feats").mkdir(parents=True, exist_ok=True)


def extract(m, dino):
    f = OUT/"feats"/f"{m}.npz"
    if f.exists(): return
    t0 = time.time(); te = np.loadtxt(DATA/"test"/f"{m}.txt", delimiter=",", ndmin=2); T, C = te.shape; W = T//WIN
    g0, g1 = np.quantile(te, .005, 0), np.quantile(te, .995, 0); F = np.zeros((C, W, GRID, 768), np.float16); ok = np.zeros((C, W), bool)
    for c in range(C):
        rg = max(g1[c]-g0[c], 1e-6)
        if np.ptp(te[:, c]) < 1e-6: continue
        ws = [w for w in range(W) if np.ptp(te[w*WIN:(w+1)*WIN, c]) >= .01*rg]
        for i in range(0, len(ws), 64):
            b = ws[i:i+64]; imgs = [ts_to_image(te[w*WIN:(w+1)*WIN, c], min(g0[c], te[w*WIN:(w+1)*WIN, c].min()), max(g1[c], te[w*WIN:(w+1)*WIN, c].max())) for w in b]
            V = patches(dino, imgs).reshape(len(b), GRID, GRID, -1).mean(1); V /= np.linalg.norm(V, axis=-1, keepdims=True)+1e-9
            F[c, b] = V.astype(np.float16); ok[c, b] = True
    np.savez(f, F=F, ok=ok, T=T); print(m, "extracted", f"{time.time()-t0:.0f}s", flush=True)


def mutual(Fc, okc, frac):
    """Fc (W, 16, D) normalised column features of one channel; returns (W, 16) scores, NaN where no feature."""
    W = len(okc); s = np.full((W, GRID), np.nan); idx = np.where(okc)[0]
    if len(idx) < 4: return s
    X = Fc[idx].astype(np.float32).reshape(-1, Fc.shape[-1]); S = X@X.T; n = len(idx)
    S = S.reshape(n, GRID, n, GRID).max(3)                       # (window, column, other window): best match inside the other window
    S[np.arange(n), :, np.arange(n)] = -np.inf                    # not itself
    D = 1-np.sort(S, axis=2)[:, :, ::-1][:, :, :n-1]              # distances to the other windows, closest first
    k = max(1, int(round(frac*(n-1)))); s[idx] = D[:, :, :k].mean(2); return s


def tick_scores(m, frac):
    z = np.load(OUT/"feats"/f"{m}.npz"); F, ok, T = z["F"], z["ok"], int(z["T"]); C, W = ok.shape; X = np.zeros((T, C))
    for c in range(C):
        s = mutual(F[c], ok[c], frac).ravel(); v = s[np.isfinite(s)]
        if len(v) < 10: continue
        med = np.median(v); mad = max(1.4826*np.median(np.abs(v-med)), 1e-6); s = np.where(np.isfinite(s), (s-med)/mad, 0.)
        X[:W*WIN, c] = np.repeat(s, P)
    return X


def evaluate(ms):
    from sklearn.covariance import LedoitWolf
    from paper_stage2 import decide_incidents
    from multi_metric_scan import load
    from iterate_common import mask, G
    from threshold_experiment import evaluate as ev
    import zs_integrated as Zi
    zs = lambda v: (v-v.mean())/max(v.std(), 1e-9)
    def maha(U):
        live = U.std(0) > 1e-9; Uc = np.clip(U[:, live], -20, 20); lw = LedoitWolf().fit(Uc); d = Uc-lw.location_; return np.einsum("ti,ij,tj->t", d, lw.precision_, d)
    _, S, Y = load("SMD"); rows = {}
    for m in ms:
        y = Y[m].astype(bool); T = len(y); te = np.loadtxt(DATA/"test"/f"{m}.txt", delimiter=",", ndmin=2)[:T]
        V = {"current DINO heads, UZ (pipeline Stage1)": Zi.stage1(m, te)}
        for frac in (.3, .1):
            X = tick_scores(m, frac)[:T]; B3 = -np.sort(-X, 1)[:, 2]
            V[f"MuSc {int(frac*100)}% B3"] = [(e["start"], e["end"]) for e in decide_incidents(B3)]
            V[f"MuSc {int(frac*100)}% JB"] = [(e["start"], e["end"]) for e in decide_incidents((zs(np.log1p(maha(X)))+zs(B3))/np.sqrt(2))]
        for n, Q in V.items():
            k = mask(Q, T); e = ev(k, y); e["aff"] = float(G.metric_Affiliation(y.astype(int), S[m]["Z3"], preds=k.astype(int)))
            fl = sum(not y[p:q].any() for p, q in Q); e["intervals"] = len(Q); e["false_intervals"] = fl; e["false_ticks"] = int((k & ~y).sum()); rows.setdefault(n, {})[m] = e
    res = {}
    for n, v in rows.items():
        res[n] = {f: round(float(np.mean([x[f] for x in v.values()])), 3) for f in ("tick_f1", "int_f1", "int_p", "pa_f1", "aff", "ev_r", "alarm")} | {f: int(sum(x[f] for x in v.values())) for f in ("intervals", "false_intervals", "false_ticks")} | dict(per_machine={mm: x["tick_f1"] for mm, x in v.items()})
        print(f"{n:42s}", {k: x for k, x in res[n].items()})
    json.dump(res, open(OUT/"evaluation.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["extract", "evaluate"]); ap.add_argument("--machines", nargs="+", required=True)
    ap.add_argument("--data", type=Path); ap.add_argument("--out", type=Path); a = ap.parse_args(); mode, ms = a.mode, a.machines
    if a.data: DATA = a.data
    if a.out: OUT = a.out; (OUT/"feats").mkdir(parents=True, exist_ok=True)
    if mode == "extract":
        d = load_dino(); torch.set_num_threads(max(1, torch.get_num_threads()))
        for m in ms: extract(m, d)
    else: evaluate(ms)
