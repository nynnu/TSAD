"""Event-time co-movement for channel diagnosis (SMD first, ASD as the generality check). No VLM. Labels only evaluate. Variants declared before the numbers:
Window = the event extended by its own length on both sides (at least 10 ticks), so the onset and the return are inside. Residual r = phase-aligned residual (cache of learnability_ceiling).
  comove_top1  : |corr(r_i, r_anchor)| over the window, anchor = channel with the largest max |r| in the event (amplitude-free)
  comove_top3  : max |corr| with any of the 3 largest channels
  dcomove_top3 : the same on first differences (co-movement of changes)
  resid+comove : rank mean of resid_max and comove_top3 (equal weights, no tuning)
  resid_gate   : resid_max for channels with resid_max >= 3, comove_top3 decides the order of the rest (large movers first, then the group by co-movement)
The anchors themselves score 1 (corr with itself). Flat channels in the window score 0.
Metrics: HitRate / NDCG @100%, @150%; events better / worse than resid_max; missed-GT recovery.
Usage: python diagnosis_comove.py -> results/diagnosis_baseline_20261122/comove.json, comove_cases.png"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from diagnosis_baseline import hit_ndcg, random_expect, segments_gt, DATA, OUT
LC = Path(__file__).resolve().parent/"results/learnability_ceiling_20261119/cache"


def cors(X, cols):
    """|corr| of every column of X with each column in cols -> (C, len(cols)); flat columns 0."""
    Xc = X-X.mean(0); n = np.sqrt((Xc**2).sum(0)); ok = n > 1e-9; out = np.zeros((X.shape[1], len(cols)))
    for j, c in enumerate(cols):
        if ok[c]: out[ok, j] = np.abs(Xc[:, ok].T @ Xc[:, c])/(n[ok]*n[c])
    return out


def score_event(R, a, b):
    L = b-a; p = max(10, L); w = R[max(0, a-p):min(len(R), b+p)]; s = np.abs(R[a:b]).max(0); top = list(np.argsort(-s)[:3])
    c1 = cors(w, top[:1])[:, 0]; c3 = cors(w, top).max(1); d3 = cors(np.diff(w, axis=0), top).max(1)
    rc = rankdata(s)+rankdata(c3); gate = np.where(s >= 3, 1000+s, c3)
    return dict(resid_max=s, comove_top1=c1, comove_top3=c3, dcomove_top3=d3, **{"resid+comove": rc, "resid_gate": gate})


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}; cases = []
    for ds in ("SMD", "ASD"):
        evs = []
        for f in sorted((DATA/ds/"interpretation_label").glob("*.txt")):
            if f.stem == "anomaly_type": continue
            R = np.clip(np.load(LC/f"{ds}_{f.stem}.npz")["R"], -20, 20)
            for a, b, G in segments_gt(f):
                b = min(b, len(R)); sc = score_event(R, a, b); evs.append(dict(m=f.stem, a=a, b=b, G=G, C=R.shape[1], sc=sc))
                if ds == "SMD" and len(cases) < 400: cases.append((f.stem, a, b, G, R))
        names = list(evs[0]["sc"]); res = {}
        for P in (1.0, 1.5):
            for k in names: res[f"{k}@{int(P*100)}"] = [round(float(x), 3) for x in np.mean([hit_ndcg(e["sc"][k], e["G"], P) for e in evs], 0)]
            res[f"random@{int(P*100)}"] = [round(float(x), 3) for x in np.mean([random_expect(e["C"], e["G"], P) for e in evs], 0)]
        base = np.array([hit_ndcg(e["sc"]["resid_max"], e["G"], 1.0)[0] for e in evs])
        res["better/worse vs resid_max @100"] = {k: [int((v > base+1e-9).sum()), int((v < base-1e-9).sum())] for k in names if k != "resid_max" for v in [np.array([hit_ndcg(e["sc"][k], e["G"], 1.0)[0] for e in evs])]}
        # GT vs non-GT co-movement among channels with resid_max < 3 (the ones size cannot rank)
        g, ng = [], []
        for e in evs:
            for c in range(e["C"]):
                if e["sc"]["resid_max"][c] < 3: (g if c in e["G"] else ng).append(e["sc"]["comove_top3"][c])
        from sklearn.metrics import roc_auc_score
        res["small movers (resid<3): comove AUROC GT vs non-GT"] = round(float(roc_auc_score([1]*len(g)+[0]*len(ng), g+ng)), 3); res["small movers n GT / non-GT"] = [len(g), len(ng)]
        out[ds] = res; print(f"\n=== {ds} ({len(evs)} events)")
        for k, v in res.items(): print(" ", k, v)
    (OUT/"comove.json").write_text(json.dumps(out, indent=1))
    # case figure: SMD events where resid_max misses most GT channels
    def miss(c):
        m, a, b, G, R = c; s = np.abs(R[a:b]).max(0); top = set(np.argsort(-s)[:len(G)]); return len(set(G)-top)/len(G)
    pick = sorted(cases, key=miss, reverse=True)[:3]
    fig, axs = plt.subplots(3, 1, figsize=(13, 11))
    for ax, (m, a, b, G, R) in zip(axs, pick):
        p = max(10, b-a); lo, hi = max(0, a-p), min(len(R), b+p); s = np.abs(R[a:b]).max(0); top = list(np.argsort(-s)[:len(G)]); x = np.arange(lo, hi)
        for c in range(R.shape[1]):
            if c in G and c in top: ax.plot(x, R[lo:hi, c]/max(np.abs(R[lo:hi, c]).max(), 1e-9), color="tab:green", lw=1, alpha=.8)
            elif c in G: ax.plot(x, R[lo:hi, c]/max(np.abs(R[lo:hi, c]).max(), 1e-9), color="tab:red", lw=1.4)
            elif c in top: ax.plot(x, R[lo:hi, c]/max(np.abs(R[lo:hi, c]).max(), 1e-9), color="tab:orange", lw=1, ls="--")
        ax.axvspan(a, b, color="0.85"); ax.set_title(f"SMD {m} [{a},{b}): green = GT found by size, red = GT missed, orange dashed = top-k non-GT (each channel scaled to its own max)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"comove_cases.png", dpi=105); print("saved")
