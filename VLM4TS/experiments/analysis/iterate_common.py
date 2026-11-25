"""Shared harness for the autonomous iterations (2026-11-24): cached data, the adopted pipeline (decide_incidents + calm_extent), a multi-metric panel per split
(SMD dev machine-1 (8) / SMD held-out machine-2,3 (20) / ASD (12)) and a journal. Rule in every iteration: choose on SMD dev by tick F1 with interval F1 not below the
reference - 0.01; held-out and ASD are reported untouched; adopt only if held-out and ASD lose on none of tick F1, interval F1, PA-F1, affiliation, worst machine
(tolerance 0.005) and win on at least two of them."""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"analysis"), str(ROOT/"pipeline_final")]
from paper_stage2 import decide_incidents, calm_extent
from multi_metric_scan import load, events
from threshold_experiment import evaluate
from diagnosis_comove import LC
from TSB_AD.evaluation.basic_metrics import basic_metricor
G = basic_metricor(); OUT = ROOT/"analysis/results/autonomous_20261124"; OUT.mkdir(parents=True, exist_ok=True); JOURNAL = OUT/"journal.md"
_D = {}


def data():
    if not _D:
        for ds in ("SMD", "ASD"):
            ms, S, Y = load(ds)
            for m in ms:
                s = S[m]["Z3"]; y = Y[m].astype(bool); T = len(s)
                _D[(ds, m)] = dict(ds=ds, m=m, s=s, y=y, R=np.load(LC/f"{ds}_{m}.npz")["R"][:T], M=S[m]["M"], inc=decide_incidents(s))
    return _D


def split(ds, m): return "ASD" if ds == "ASD" else ("dev" if m.startswith("machine-1-") else "held")


def mask(parts, T):
    k = np.zeros(T, bool)
    for a, b in parts: k[a:b] = True
    return k


def adopted(d, **kw):
    return [p for ps in calm_extent(d["R"], d["s"], d["inc"], **kw) for p in ps]


def panel(policies, aff=True):
    D = data(); rows = {}
    for key, d in D.items():
        rows[key] = {}
        for n, f in policies.items():
            k = mask(f(d), len(d["s"])); r = evaluate(k, d["y"]); r["aff"] = float(G.metric_Affiliation(d["y"].astype(int), d["s"], preds=k.astype(int))) if aff else 0.; rows[key][n] = r
    res = {}; ref = list(policies)[0]
    for sp in ("dev", "held", "ASD"):
        keys = [k for k in D if split(*k) == sp]; res[sp] = {}
        for n in policies:
            v = [rows[k][n] for k in keys]; r = {f: round(float(np.mean([x[f] for x in v])), 3) for f in ("tick_f1", "int_f1", "pa_f1", "aff", "ev_r", "alarm")}
            r["worst_tick"] = round(float(min(x["tick_f1"] for x in v)), 3); r["worst_int"] = round(float(min(x["int_f1"] for x in v)), 3)
            r["W/L tick"] = [int(sum(rows[k][n]["tick_f1"] > rows[k][ref]["tick_f1"]+1e-9 for k in keys)), int(sum(rows[k][n]["tick_f1"] < rows[k][ref]["tick_f1"]-1e-9 for k in keys))]
            res[sp][n] = r
    return res


def judge(res, ref, cand):
    def ok(sp):
        a, b = res[sp][cand], res[sp][ref]; m = ("tick_f1", "int_f1", "pa_f1", "aff", "worst_tick")
        return bool(all(a[f] >= b[f]-.005 for f in m) and sum(a[f] > b[f]+.005 for f in m) >= 2)
    return dict(held=ok("held"), ASD=ok("ASD"), adopt=ok("held") and ok("ASD"))


def choose_on_dev(res, ref, cands):
    ok = [c for c in cands if res["dev"][c]["int_f1"] >= res["dev"][ref]["int_f1"]-.01]
    return max(ok, key=lambda c: res["dev"][c]["tick_f1"]) if ok else None


def log(title, text):
    with open(JOURNAL, "a", encoding="utf8") as f: f.write(f"\n## {title}\n{text}\n")


def table(res, names):
    lines = ["| 정책 | 분할 | 틱 F1 | 구간 F1 | PA-F1 | Aff | 사건 재현율 | 경보 | 최악 틱 / 구간 | 틱 승/패 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        for sp in ("dev", "held", "ASD"):
            r = res[sp][n]; lines.append(f"| {n} | {sp} | {r['tick_f1']} | {r['int_f1']} | {r['pa_f1']} | {r['aff']:.3f} | {r['ev_r']} | {r['alarm']} | {r['worst_tick']} / {r['worst_int']} | {r['W/L tick']} |")
    return "\n".join(lines)
