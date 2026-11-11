"""Final stage1.period on real train data: input = first 75% of each channel's train (so its two-thirds cut is the
50% fit / 25% validation split), evaluation on the untouched last 25% (profile from the fit part). 28 machines, no labels."""
import sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
sys.path.insert(0, str(ROOT/"stage1_final"))
from stage1 import period, _season_skill

data = ROOT.parent/"mv_data/SMD"
rng = np.random.default_rng(0)
rows = []
for path in sorted((data/"train").glob("machine-*.txt")):
    tr = np.loadtxt(path, delimiter=",")
    i, j = len(tr)//2, 3*len(tr)//4
    for c in range(tr.shape[1]):
        a, cc = tr[:i, c], tr[j:, c]
        if a.std() < 1e-6:
            continue
        p = period(tr[:j, c])
        null = float(np.quantile([_season_skill(a, cc, int(q), j) for q in rng.integers(5, 1001, 30)], .95))
        rows.append(dict(p=p, skill=_season_skill(a, cc, p, j) if p else None, null=null))
    print(path.stem, flush=True)
got = [r for r in rows if r["p"]]
res = dict(channels=len(rows), accepted=len(got)/len(rows), skill_C_median=float(np.median([r["skill"] for r in got])),
           beats_constant_on_C=float(np.mean([r["skill"] > 0 for r in got])), beats_random_95_on_C=float(np.mean([r["skill"] > r["null"] for r in got])),
           periods_median=float(np.median([r["p"] for r in got])))
save(ROOT/"analysis/results/period_validity_20261111/final.json", res)
print(res)
