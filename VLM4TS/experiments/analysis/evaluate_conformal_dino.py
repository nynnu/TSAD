"""Same conformal structure with DINO spike/height scores as the statistic (only the 3 machines whose train scores are cached).
Compared on identical ticks with the model-free statistic and with the current Stage1 rule (GT only evaluates)."""
import json, sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
from conformal_budget import block_stats, run_series, summarize, show, evaluate, pooled, L
sys.path.insert(0, str(ROOT/"stage1_final"))
from stage1 import drop_flat, P

data = ROOT.parent/"mv_data/SMD"
cache = ROOT/"stage1_final/cache_statistical/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/conformal_budget_20261111"
ALPHAS = (0.003, 0.01, 0.03, 0.1)
AGG = "fisher" if "--fisher" in sys.argv else "min"
assert P == L


def dino_stats(raw_spike, raw_height, series, train):
    cols = []
    for raw in (raw_spike, raw_height):
        r = drop_flat(raw[:, :len(series)], series[:raw.shape[1]], train)[:, ::P]
        cols.append(np.where(np.isfinite(r), r, -np.inf).T)
    return np.concatenate(cols, axis=1)


res = {"dino": [], "statistic": []}; ys = []; stage1 = []
for machine in ("machine-1-1", "machine-1-2", "machine-1-6"):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=","); test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    tr_c, te_c = np.load(cache/f"{machine}_dino_train.npz"), np.load(cache/f"{machine}_dino_test.npz")
    s_tr, s_te = dino_stats(tr_c["spike"], tr_c["height"], train, train), dino_stats(te_c["spike"], te_c["height"], test, train)
    n = len(s_te)
    res["dino"].append(run_series(s_tr, s_te, ALPHAS, agg=AGG))
    res["statistic"].append(run_series(block_stats(train)[:len(s_tr)], block_stats(test)[:n], ALPHAS, agg=AGG))
    doc = json.loads((snap/f"{machine}.json").read_text(encoding="utf8"))
    mask = np.zeros(len(test), bool)
    for e in doc["intervals"]:
        mask[e["start"]:e["end"]] = True
    stage1.append(evaluate(mask[:n*L].reshape(n, L).any(axis=1), y))
    ys.append(y)
    print(machine, "done", n, "blocks", flush=True)
summaries = {k: summarize(v, ys, ALPHAS) for k, v in res.items()}
save(out/("dino_3_machines_fisher.json" if AGG == "fisher" else "dino_3_machines.json"), dict(summaries={k: {str(a): s for a, s in v.items()} for k, v in summaries.items()}, stage1_rule=pooled(stage1),
     limitation="3 development machines (machine-1-1/1-2/1-6); train DINO scores for the other machines are not computed (CPU ~1 image/s)."))
r = pooled(stage1)
print("current Stage1 rule: alarms %.3f segP %.3f eventR %.3f eventF1 %.3f pointF1 %.3f FAR(normal) %.4f" % (r["alarm_fraction"], r["segment_precision"], r["event_recall"], r["event_f1"], r["point_f1"], r["far_on_normal"]))
show(summaries["dino"], "DINO spike+height as the statistic")
show(summaries["statistic"], "model-free statistic on the same machines/ticks")
