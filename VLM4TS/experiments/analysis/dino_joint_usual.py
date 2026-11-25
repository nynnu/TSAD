"""Give the zero-shot DINOv2 Stage1 the two things that make the test-only residual R0 strong (SMD, no API, test only, labels only evaluate). Declared before the numbers.
(1) JOINT score across channels: Ledoit-Wolf Mahalanobis of the per-channel DINO score vector (how far the machine's pattern of DINO evidence is from its usual
    co-variation), as R0 does on raw residuals.
(2) USUAL value from the test itself: per channel, the DINO score's usual level = median over cycles at each phase (period validated on the RAW test channel,
    stage1.period; no period -> channel median), scale = 1.4826 MAD; residual u = (X - usual) / scale. Recurring normal spikes give recurring DINO scores and cancel.
DINO score per channel X = calib(spike) + calib(height) of the cached zero-shot DINOv2 run (testonly_20261124/cache/SMD_{m}_dino_zs.npz, test-range plots);
flat windows (no score) get the channel median. Only POSITIVE deviations count as evidence (a higher spike / height score = more anomalous).
Variants, each followed by the same machine-level decision layer (paper_stage2.decide_incidents):
  B3  3rd largest channel X (dino_decision_layer.py, reference)            U3  3rd largest channel of max(u, 0)
  J   z(log1p Mahalanobis of X clipped)                                    UZ  R0 recipe on u: (z(log1p J_u) + z(log1p M3_u)) / sqrt 2, J_u on u clipped at 5
  JB  (z(log1p J) + z(B3)) / sqrt 2
Usage: python dino_joint_usual.py"""
import json, sys
from pathlib import Path
import torch  # noqa: F401
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"stage1_final"), str(ROOT/"pipeline_final")]
from sklearn.covariance import LedoitWolf
from iterate_common import mask, split, G, OUT
from threshold_experiment import evaluate
from paper_stage2 import decide_incidents
from stage1 import drop_flat, calib, period
from multi_metric_scan import load
C0 = ROOT/"analysis/results/testonly_20261124/cache"; DATA = ROOT.parent/"mv_data/SMD"
zs = lambda v: (v-v.mean())/max(v.std(), 1e-9)


def maha(U, clip):
    live = U.std(0) > 1e-9; Uc = np.clip(U[:, live], -clip, clip); lw = LedoitWolf().fit(Uc); d = Uc-lw.location_; return np.einsum("ti,ij,tj->t", d, lw.precision_, d)


rows = {}; ms, S, Y = load("SMD")
for m in ms:
    y = Y[m].astype(bool); T = len(y); s = S[m]["Z3"]; te = np.loadtxt(DATA/"test"/f"{m}.txt", delimiter=",", ndmin=2)[:T]
    q = np.load(C0/f"SMD_{m}_dino_zs.npz"); tr = np.vstack([np.quantile(te, .005, 0), np.quantile(te, .995, 0)])
    X = (calib(drop_flat(q["spike"][:, :T], te, tr))+calib(drop_flat(q["height"][:, :T], te, tr))).T   # T x C
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.); X = np.where(np.isfinite(X), X, med)
    U = np.zeros_like(X)
    for c in range(X.shape[1]):
        x = X[:, c]
        if x.std() < 1e-9: continue
        P = period(te[:, c]) if te[:, c].std() > 1e-9 else None
        u = np.array([np.median(x[k::P]) for k in range(P)])[np.arange(T) % P] if P else np.full(T, np.median(x))
        e = x-u; sc = 1.4826*np.median(np.abs(e-np.median(e))); U[:, c] = e/(sc if sc > 1e-9 else max(e.std(), 1e-6))
    B3 = -np.sort(-X, 1)[:, 2]; J = zs(np.log1p(maha(X, 20))); Up = np.maximum(U, 0); U3 = -np.sort(-Up, 1)[:, 2]
    UZ = (zs(np.log1p(maha(U, 5)))+zs(np.log1p(np.minimum(U3, 20))))/np.sqrt(2)
    V = {"B3 3rd channel": B3, "J joint Mahalanobis": J, "JB joint + 3rd": (J+zs(B3))/np.sqrt(2), "U3 usual-residual 3rd": U3, "UZ usual-residual joint (R0 recipe)": UZ}
    for n, v in V.items():
        k = mask([(e["start"], e["end"]) for e in decide_incidents(v)], T); e = evaluate(k, y); e["aff"] = float(G.metric_Affiliation(y.astype(int), s, preds=k.astype(int)))
        rows.setdefault((split("SMD", m), n), {})[m] = e
res = {}
for (sp, n), v in sorted(rows.items()):
    r = {f: round(float(np.mean([x[f] for x in v.values()])), 3) for f in ("tick_f1", "int_f1", "int_p", "pa_f1", "aff", "ev_r", "alarm")} | dict(worst=round(min(x["tick_f1"] for x in v.values()), 3), per_machine={mm: x["tick_f1"] for mm, x in v.items()})
    res[f"{sp} | {n}"] = r; print(f"{sp:5s} {n:36s}", {k: x for k, x in r.items() if k != "per_machine"})
json.dump(res, open(OUT.parent/"resid_dino_20261125/joint_usual.json", "w"), indent=1, default=float)
