"""Does validating a period on held-out train make the detection reliable? Three-way split of each channel's train:
A (first 50%) estimate + profile, B (next 25%) validation, C (last 25%) untouched evaluation. 28 machines, no labels."""
import sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, estimate_reference_period, save
sys.path.insert(0, str(ROOT/"stage1_final"))
from stage1 import period as fft_period

data = ROOT.parent/"mv_data/SMD"
rng = np.random.default_rng(0)


def skill(a, seg, p, offset):
    prof = np.array([a[k::p].mean() for k in range(p)])
    pred = prof[(offset+np.arange(len(seg))) % p]
    return 1-((seg-pred)**2).sum()/max(((seg-a.mean())**2).sum(), 1e-12)


def null95(a, seg, offset):
    return float(np.quantile([skill(a, seg, int(q), offset) for q in rng.integers(5, 1001, 30)], .95))


rows = []
for path in sorted((data/"train").glob("machine-*.txt")):
    tr = np.loadtxt(path, delimiter=",")
    i, j = len(tr)//2, 3*len(tr)//4
    for c in range(tr.shape[1]):
        a, b, cc = tr[:i, c], tr[i:j, c], tr[j:, c]
        if a.std() < 1e-6:
            continue
        lo, hi = np.quantile(a, [.01, .99]); hi = lo+1. if hi-lo < 1e-8 else hi
        p_fft = fft_period(a); p_acf, _ = estimate_reference_period((a-lo)/(hi-lo))
        nB, nC = null95(a, b, i), null95(a, cc, j)
        r = dict(nC=nC)
        for name, p in (("fft", p_fft), ("fft_validated", p_fft), ("acf_halves", p_acf)):
            ok = p is not None
            if name == "fft_validated" and ok:
                ok = skill(a, b, p, i) > max(0., nB)
            r[name] = (skill(a, cc, p, j) if ok else None)
        rows.append(r)
    print(path.stem, flush=True)
out = {"channels": len(rows)}
for name in ("fft", "fft_validated", "acf_halves"):
    got = [r for r in rows if r[name] is not None]
    out[name] = dict(accepted=len(got)/len(rows), skill_C_median=float(np.median([r[name] for r in got])) if got else None,
                     beats_constant_on_C=float(np.mean([r[name] > 0 for r in got])) if got else None,
                     beats_random_95_on_C=float(np.mean([r[name] > r["nC"] for r in got])) if got else None)
save(ROOT/"analysis/results/period_validity_20261111/rules.json", out)
for k, v in out.items():
    print(k, {m: round(x, 3) if isinstance(x, float) else x for m, x in v.items()} if isinstance(v, dict) else v)
