"""Extend each Stage1 core to where its channels return to their pre-core level (28 machines, no VLM).

A core is extended to the tick until which at least `need` shown channels are still in an
excursion (need = min(2, shown channels)). Primary setting is fixed a priori; the grid is a
sensitivity report, not a selection. GT is used only for evaluation.
"""
import json, sys
from itertools import product
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
sys.path.insert(0, str(ROOT/"stage2"))
from signal_evidence import excursion_end

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/extent_estimation_20261110"
out.mkdir(parents=True, exist_ok=True)
cores = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]
PRIMARY = dict(k=3.0, quiet=30, horizon=1000, need=2)
GRID = [dict(k=k, quiet=q, horizon=h, need=n) for k, q, h, n in product((3.0, 5.0), (10, 30), (300, 1000), (1, 2))]


def extended_end(test, core, k, quiet, horizon, need, band="mad"):
    a, b = core["core"]
    ends = sorted((e for ch in core["shown"] if (e := excursion_end(test[:, ch], a, b, k=k, quiet=quiet, horizon=horizon, band=band))), reverse=True)
    need = min(need, len(core["shown"]))
    return max(b, ends[need-1]) if len(ends) >= need else b


def segments(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def score(rows):
    tp, fp, fn = (sum(r[x] for r in rows) for x in ("tp", "fp", "fn"))
    p, r = tp/(tp+fp), tp/(tp+fn)
    return dict(point_precision=p, point_recall=r, point_f1=2*p*r/(p+r), fp_ticks=fp,
                gt_events_hit=sum(r["hit"] for r in rows), gt_events=sum(r["events"] for r in rows),
                predicted_segments=sum(r["segments"] for r in rows), segments_on_gt=sum(r["segments_on_gt"] for r in rows))


def machine_row(y, mask):
    segs = segments(mask)
    return dict(tp=int((mask & y).sum()), fp=int((mask & ~y).sum()), fn=int((~mask & y).sum()),
                hit=int(sum(mask[a:b].any() for a, b in segments(y))), events=len(segments(y)),
                segments=len(segs), segments_on_gt=int(sum(y[a:b].any() for a, b in segs)))


RANGE = dict(PRIMARY, band="range")  # second iteration, after the spike-train failure was seen in plots
configs = [None, PRIMARY, RANGE]+[dict(RANGE, horizon=300), dict(RANGE, need=1), dict(RANGE, quiet=10)]+[g for g in GRID if g != PRIMARY]
rows = {i: [] for i in range(len(configs))}
detail = []
for machine in sorted({c["machine"] for c in cores}):
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    mine = [c for c in cores if c["machine"] == machine]
    for i, cfg in enumerate(configs):
        mask = np.zeros(len(y), bool)
        for c in mine:
            end = extended_end(test, c, **cfg) if cfg else c["core"][1]
            mask[c["core"][0]:end] = True
            if i in (1, 2):
                detail.append(dict(machine=machine, band=cfg.get("band", "mad"), core=c["core"], extended_end=int(end), gt_overlap=c["gt_overlap"]))
        rows[i].append(dict(machine=machine, **machine_row(y, mask)))
    print(machine, "done", flush=True)


def f1(r):
    return 2*r["tp"]/(2*r["tp"]+r["fp"]+r["fn"]) if r["tp"] else 0.0


base, prim, rng = rows[0], rows[1], rows[2]
per_machine = [dict(machine=b["machine"], base_f1=f1(b), extent_f1=f1(p), range_f1=f1(r), base_fp=b["fp"], extent_fp=p["fp"], range_fp=r["fp"]) for b, p, r in zip(base, prim, rng)]
dev = lambda r: r["machine"].startswith("machine-1-")  # machines used in earlier development
summary = dict(
    baseline=score(base), primary=dict(config=PRIMARY, **score(prim)), range_band=dict(config=RANGE, **score(rng)),
    range_dev_machines=score([r for r in rng if dev(r)]), range_other_machines=score([r for r in rng if not dev(r)]),
    range_machines_f1_up=sum(m["range_f1"] > m["base_f1"]+1e-9 for m in per_machine), range_machines_f1_down=sum(m["range_f1"] < m["base_f1"]-1e-9 for m in per_machine),
    primary_dev_machines=dict(baseline=score([r for r in base if dev(r)]), extent=score([r for r in prim if dev(r)])),
    primary_other_machines=dict(baseline=score([r for r in base if not dev(r)]), extent=score([r for r in prim if not dev(r)])),
    machines_f1_up=sum(m["extent_f1"] > m["base_f1"]+1e-9 for m in per_machine),
    machines_f1_down=sum(m["extent_f1"] < m["base_f1"]-1e-9 for m in per_machine),
    sensitivity=[dict(config=cfg, **score(rows[i])) for i, cfg in enumerate(configs) if i > 2])
save(out/"summary.json", dict(summary=summary, per_machine=per_machine, cores=detail,
     limitation="Offline: extension looks at future ticks. Core-level union, no VLM verdict applied. "
                "All 28 machines carry GT, so the grid is sensitivity only; the primary setting was fixed before running."))
print(json.dumps({k: v for k, v in summary.items() if k != "sensitivity"}, indent=1))
for s in summary["sensitivity"]:
    print(s["config"], "P %.3f R %.3f F1 %.3f fp %d" % (s["point_precision"], s["point_recall"], s["point_f1"], s["fp_ticks"]))
for m in per_machine:
    print(m["machine"], "%.3f -> mad %.3f / range %.3f" % (m["base_f1"], m["extent_f1"], m["range_f1"]), "fp", m["base_fp"], "->", m["extent_fp"], "/", m["range_fp"])
